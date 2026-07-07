import os
import sys
import time

import click
import cv2
import numpy as np
import sapien
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager
from transforms3d.quaternions import mat2quat


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from common.z_single_realsense import SingleRealsense
from common.z_single_kinectv2 import SingleKinectV2
from common.z_spacemouse_shared_memory import Spacemouse
from common.z_rtde_interpolation_controller import RTDEInterpolationController
from common.z_keystroke_counter import KeystrokeCounter, KeyCode
from common.z_dualgripper_controller import GripperController
from common.precise_sleep import precise_wait
from common.pose_util import pose10d_to_pose6d
from common.z_tcp_trans import TcpEETrans


DEFAULT_CKPT_DIR = os.environ.get(
    "CHRONOS_CKPT_DIR",
    os.path.join(PACKAGE_ROOT, "checkpoints", "cover_blocks", "S3B_IMAGE_20D_2"),
)
DEFAULT_SCALER_PATH = os.environ.get(
    "CHRONOS_SCALER_PATH",
    os.path.join(PACKAGE_ROOT, "scalers", "scaler_cover_blocks_image_pose10d.pth"),
)

ACT_init = np.array([
    [-0.05084912, -0.34176292, 0.02294293, 1.04612629, -1.12444314, -1.16306351],
    [0.02584831, -0.35633479, 0.08430657, 1.24871465, 1.24880682, 1.20975324],
])

# cover_blocks preset poses
ACT1 = np.array([-0.07658659261957693, -0.41053309691480333, 0.10796969201742433, 1.0462487812117205, -1.124299592815253, -1.1635110341303407])
ACT2 = np.array([-0.07730374942792533, -0.23068046962649186, 0.09086471430893947, 1.0462915900126735, -1.1242582221152426, -1.1635978484081144])
ACT3 = np.array([0.15355951102095025, -0.4245672250489069, 0.05259026362194584, 1.2486060502385146, 1.2489538439341616, 1.209614933821712])
ACT4 = np.array([0.14909295654667545, -0.24913936528500846, 0.0481806797180441, 1.2485798969591728, 1.2489780874389702, 1.2095546578774459])
ACT5 = np.array([0.07342674102429379, -0.4157976410044294, 0.1565522612883653, 1.2489192615480307, 1.2486164428057989, 1.210117629062797])
ACT6 = np.array([0.06885586629910401, -0.2517042821943411, 0.14320938266444927, 1.248795500766106, 1.2487463852487795, 1.209914086778221])

GRIPPER1_RANGE = [1977, 3700]
GRIPPER2_RANGE = [302, 1986]


def select_device_serial(device_name, detected_serials, requested_serial=None, option_name=None):
    if requested_serial:
        if detected_serials and requested_serial not in detected_serials:
            print(
                f"WARNING: requested {device_name} serial {requested_serial} "
                f"was not in detected serials: {detected_serials}"
            )
        return requested_serial

    if not detected_serials:
        option_name = option_name or f"--{device_name.lower().replace(' ', '-')}-serial"
        raise click.ClickException(
            f"No {device_name} device detected. Check USB connection, camera power, "
            f"and device permissions, or pass the serial manually with {option_name}."
        )
    return detected_serials[0]


def clip_gripper(value, value_range):
    return int(np.clip(np.rint(float(value)), value_range[0], value_range[1]))


def scalar_int(value, default=-1):
    if value is None:
        return default
    try:
        return int(np.asarray(value).item())
    except Exception:
        return default


def scalar_float(value, default=float("nan")):
    if value is None:
        return default
    try:
        return float(np.asarray(value).item())
    except Exception:
        return default


def is_gripper_open(value, value_range):
    midpoint = 0.5 * (value_range[0] + value_range[1])
    return scalar_int(value, value_range[0]) < midpoint


def draw_status(img, mode, latency=None):
    vis = cv2.flip(img.copy(), -1)
    text = mode if latency is None else f"{mode} | infer {latency * 1000:.1f} ms"
    cv2.putText(
        vis,
        text,
        (10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.9,
        thickness=2,
        color=(255, 0, 0),
    )
    return vis


def ensure_process_alive(name, process):
    if not process.is_alive():
        detail = None
        if hasattr(process, "_pop_startup_error"):
            try:
                detail = process._pop_startup_error()
            except Exception:
                detail = None
        suffix = f"\n{detail}" if detail else "\nCheck the traceback printed above this message."
        raise RuntimeError(
            f"{name} process died unexpectedly (exitcode={process.exitcode}). "
            f"{suffix}"
        )


def collect_observation_timestamp(*candidates, fallback=None):
    timestamps = []
    now = time.time()
    for value in candidates:
        try:
            timestamp = float(np.asarray(value).item())
        except Exception:
            continue
        if np.isfinite(timestamp) and 0.0 < timestamp <= now + 0.5:
            timestamps.append(timestamp)
    if timestamps:
        return min(timestamps)
    return now if fallback is None else fallback


def compute_aligned_action_step(obs_timestamp, target_time, dt, max_step, action_label_lead_steps):
    delay_steps = int(np.rint((target_time - obs_timestamp) / dt))
    action_step = delay_steps - int(action_label_lead_steps)
    return int(np.clip(action_step, 0, max_step))


@click.command()
@click.option("-r", "--robot_ip", default="192.168.4.63")
@click.option("-r2", "--robot_ip2", default="192.168.4.64")
@click.option("--d435-serial", default=None, help="RealSense D435 serial number to use.")
@click.option("--kinect-serial", default=None, help="Kinect V2 serial number to use.")
@click.option("--ckpt-path", default=DEFAULT_CKPT_DIR, help="Chronos ckpt file or checkpoint directory.")
@click.option("--scaler-path", default=DEFAULT_SCALER_PATH, help="Chronos pose10d image scaler path.")
@click.option("--device", default="auto", help="Inference device: auto, cpu, cuda, cuda:0, ...")
@click.option("--temporal-agg/--no-temporal-agg", default=True, help="Use RMBench-style temporal aggregation.")
@click.option("--policy-action-step", default=0, type=int, help="0 matches RMBench; -1 selects an index from measured latency.")
@click.option("--action-exec-latency", default=0.10, type=float, help="Seconds between sending a policy command and its scheduled execution time.")
@click.option("--action-label-lead-steps", default=1, type=int, help="Dataset target_pose lead in policy steps; data collection schedules target one step after observation.")
@click.option("--max-policy-obs-age", default=0.5, type=float, help="Maximum D435 frame age allowed for policy commands, in seconds.")
@click.option("--max-policy-state-age", default=0.5, type=float, help="Maximum robot/gripper state age allowed for policy commands, in seconds.")
@click.option("--markov-policy/--recurrent-policy", default=False, help="Reset policy hidden state every inference step for a Markov-style timing ablation.")
@click.option("--start-infer", is_flag=True, help="Start in policy mode without pressing i.")
def main(
    robot_ip,
    robot_ip2,
    d435_serial,
    kinect_serial,
    ckpt_path,
    scaler_path,
    device,
    temporal_agg,
    policy_action_step,
    action_exec_latency,
    action_label_lead_steps,
    max_policy_obs_age,
    max_policy_state_age,
    markov_policy,
    start_infer,
):
    cv2.setNumThreads(4)
    frequency = 10
    dt = 1 / frequency
    max_pos_speed = 0.0250
    max_rot_speed = 0.1500
    cube_diag = np.linalg.norm([1, 1, 1])
    offset = 0.0
    j_init = [
        -1.1469181219684046, -2.6888716856585901, -1.3540738264666956,
        -2.6056087652789515, 0.4185419380664825, 0.4437399208545685,
    ]
    j_init2 = [
        1.1009780168533325, -0.7465727964984339, 1.4875335693359375,
        -0.8355210463153284, -0.4877889792071741, 0.0977032706141472,
    ]

    right_base_world = sapien.Pose([0.189821, 0, 1.1822], [0.38269, 0, 0.923877, 0])
    left_base_world = sapien.Pose([-0.189821, 0, 1.1822], [0.38269, 0, -0.923877, 0])
    transformer = TcpEETrans()

    serial_numbers_d435 = SingleRealsense.get_connected_devices_serial(wait_timeout=10.0)
    serial_numbers_kinect = SingleKinectV2.get_connected_devices_serial()
    print(f"Detected D435 serials: {serial_numbers_d435 or 'none'}")
    print(f"Detected Kinect V2 serials: {serial_numbers_kinect or 'none'}")
    d435_serial = select_device_serial("D435", serial_numbers_d435, d435_serial, "--d435-serial")
    kinect_serial = select_device_serial("Kinect V2", serial_numbers_kinect, kinect_serial, "--kinect-serial")

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
                SingleRealsense(
                    shm_manager,
                    d435_serial,
                    resolution=(640, 480),
                    capture_fps=30,
                    put_fps=30,
                    put_downsample=False,
                    enable_depth=True,
                    frame_timeout_ms=1000,
                    max_consecutive_timeouts=None,
                    reset_on_stop=True,
                    reset_wait_s=4.0,
                ) as camera, \
                RTDEInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=robot_ip,
                    lookahead_time=0.1,
                    max_pos_speed=max_pos_speed * cube_diag,
                    max_rot_speed=max_rot_speed * cube_diag,
                    tcp_offset_pose=[0, 0, offset, 0, 0, 0],
                    joints_init=j_init,
                    launch_timeout=60.0,
                    payload_mass=1.0,
                    verbose=False,
                ) as controller, \
                RTDEInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=robot_ip2,
                    lookahead_time=0.1,
                    max_pos_speed=max_pos_speed * cube_diag,
                    max_rot_speed=max_rot_speed * cube_diag,
                    tcp_offset_pose=[0, 0, offset, 0, 0, 0],
                    joints_init=j_init2,
                    launch_timeout=60.0,
                    payload_mass=1.0,
                    verbose=False,
                ) as controller2, \
                SingleKinectV2(
                    shm_manager=shm_manager,
                    serial_number=kinect_serial,
                    resolution=(1280, 720),
                    capture_fps=30,
                    put_fps=30,
                    enable_color=True,
                    enable_depth=True,
                    verbose=False,
                ) as kinect, \
                Spacemouse(shm_manager=shm_manager) as sm, \
                GripperController(
                    shm_manager=shm_manager,
                    port="/dev/ttyUSB0",
                    frequency=30,
                    current_limit=200,
                    state_read_frequency=0,
                    hardware_error_check_period=0,
                    initial_position_1=GRIPPER1_RANGE[0],
                    initial_position_2=GRIPPER2_RANGE[0],
                    tx_only_writes=True,
                    verbose=False,
                ) as gripper:
            print("Loading Chronos inference model after starting robot/camera processes...")
            from inference.inference_choronos import MyInferenceModel

            infer_model = MyInferenceModel(
                checkpoint_path=ckpt_path,
                scaler_path=scaler_path,
                device=device,
                temporal_agg=temporal_agg,
            )
            infer_model.warmup()
            print(
                "Chronos policy config: "
                f"temporal_agg={temporal_agg}, "
                f"policy_action_step={policy_action_step}, "
                f"markov_policy={markov_policy}, "
                f"action_exec_latency={action_exec_latency:.3f}s, "
                f"max_policy_obs_age={max_policy_obs_age:.3f}s, "
                f"max_policy_state_age={max_policy_state_age:.3f}s"
            )
            print(
                "Chronos resolved paths: "
                f"ckpt={infer_model.checkpoint_path}, "
                f"scaler={infer_model.scaler_path}"
            )

            out = None
            out_kinect = None
            state = controller.get_state()
            state2 = controller2.get_state()
            target_pose = state["TargetTCPPose"].copy()
            target_pose2 = state2["TargetTCPPose"].copy()

            stop = False
            start_policy = bool(start_infer)
            active_controller = 1
            gripper1_open = True
            gripper2_open = True
            gripper_goal1 = GRIPPER1_RANGE[0]
            gripper_goal2 = GRIPPER2_RANGE[0]
            last_infer_latency = None
            command_latency = 0.01
            last_gripper_goal = None
            last_gripper_command_t = 0.0
            gripper_command_period = 1.0
            gripper_command_latency = 0.2
            gripper_force_until_t = 0.0
            gripper_force_period = 0.1
            last_gripper_status_print_t = 0.0
            last_timing_warning_t = 0.0
            policy_start_wall_t = None
            policy_step_count = 0
            t_start = time.monotonic()
            iter_idx = 0

            if start_policy:
                infer_model.reset_hiddens()
                policy_start_wall_t = time.time()
                policy_step_count = 0

            print("Ready. Keys: i=start policy, z=stop policy, q=quit, w=switch arm, e=toggle gripper, 0/1/2/3=preset.")
            while not stop:
                ensure_process_alive("D435", camera)
                ensure_process_alive("Kinect", kinect)
                ensure_process_alive("controller1", controller)
                ensure_process_alive("controller2", controller2)
                ensure_process_alive("gripper", gripper)

                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                last_data = controller.get_state()
                last_data2 = controller2.get_state()
                curr_pose = last_data["ActualTCPPose"]
                curr_pose2 = last_data2["ActualTCPPose"]
                robot1_state_age = time.time() - scalar_float(
                    last_data.get("robot_receive_timestamp")
                )
                robot2_state_age = time.time() - scalar_float(
                    last_data2.get("robot_receive_timestamp")
                )
                gripper_stat = gripper.get_state()
                gripper_position_1 = gripper_stat["gripper_position_1"]
                gripper_position_2 = gripper_stat["gripper_position_2"]
                gripper_target_1 = scalar_int(gripper_stat.get("gripper_target_1"))
                gripper_target_2 = scalar_int(gripper_stat.get("gripper_target_2"))
                gripper_write_ok_1 = scalar_int(gripper_stat.get("gripper_write_ok_1"))
                gripper_write_ok_2 = scalar_int(gripper_stat.get("gripper_write_ok_2"))
                gripper_error_1 = scalar_int(gripper_stat.get("gripper_error_1"))
                gripper_error_2 = scalar_int(gripper_stat.get("gripper_error_2"))
                gripper_state_age = time.time() - scalar_float(
                    gripper_stat.get("gripper_receive_timestamp")
                )
                if gripper_state_age > 2.0:
                    raise RuntimeError(
                        "GripperController process is alive but its state is stale "
                        f"({gripper_state_age:.2f}s old). Check serial communication and controller traceback."
                    )

                precise_wait(t_sample)

                if not start_policy:
                    sm_state = sm.get_motion_state_transformed()
                    dpos = sm_state[:3] * (max_pos_speed / frequency)
                    dpos = dpos[[1, 0, 2]] * [-1, 1, 1]
                    drot_xyz = sm_state[3:] * (max_rot_speed / frequency)
                    if not sm.is_button_pressed(0):
                        drot_xyz[:] = 0
                    else:
                        dpos[:] = 0
                    if sm.is_button_pressed(1):
                        dpos[2] = 0
                    drot = st.Rotation.from_euler("xyz", drot_xyz)

                    if active_controller == 1:
                        quat = mat2quat(cv2.Rodrigues(target_pose[3:])[0])
                        tcp1 = sapien.Pose(target_pose[:3], quat)
                        ee1 = transformer.tcp_to_ee(tcp1)
                        ee1_in_world = transformer.ee_in_world(ee1, right_base_world)
                        ee1_in_world.p += dpos
                        ee1 = transformer.ee_in_baselink(ee1_in_world, right_base_world)
                        tcp1 = transformer.ee_to_tcp(ee1)
                        target_pose[:3] = tcp1.p
                        target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()
                    else:
                        quat = mat2quat(cv2.Rodrigues(target_pose2[3:])[0])
                        tcp2 = sapien.Pose(target_pose2[:3], quat)
                        ee2 = transformer.tcp_to_ee(tcp2)
                        ee2_in_world = transformer.ee_in_world(ee2, left_base_world)
                        ee2_in_world.p += dpos
                        ee2 = transformer.ee_in_baselink(ee2_in_world, left_base_world)
                        tcp2 = transformer.ee_to_tcp(ee2)
                        target_pose2[:3] = tcp2.p
                        target_pose2[3:] = (drot * st.Rotation.from_rotvec(target_pose2[3:])).as_rotvec()

                    controller.schedule_waypoint(
                        target_pose,
                        t_command_target - time.monotonic() + time.time(),
                    )
                    controller2.schedule_waypoint(
                        target_pose2,
                        t_command_target - time.monotonic() + time.time(),
                    )

                out = camera.get(out=out)
                img = out["color"]
                d435_frame_age = time.time() - scalar_float(out.get("timestamp"))
                if start_policy and d435_frame_age > max_policy_obs_age:
                    raise RuntimeError(
                        f"D435 frame is stale ({d435_frame_age:.3f}s old). "
                        "Policy commands are stopped to avoid executing against old images."
                    )
                out_kinect = kinect.get(out=out_kinect)
                img_kinect = out_kinect["color"]

                cv2.namedWindow("chronos_d435", 0)
                cv2.imshow(
                    "chronos_d435",
                    draw_status(img, "POLICY" if start_policy else "MANUAL", last_infer_latency),
                )
                cv2.namedWindow("chronos_kinect", 0)
                cv2.imshow("chronos_kinect", img_kinect)
                cv2.pollKey()

                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char="q"):
                        stop = True
                    elif key_stroke == KeyCode(char="w"):
                        active_controller = 2 if active_controller == 1 else 1
                        print(f"Switched to {'controller2' if active_controller == 2 else 'controller'}")
                    elif key_stroke in (KeyCode(char="e"), KeyCode(char="E")):
                        print(
                            "Gripper actual before command: "
                            f"active_arm={active_controller}, "
                            f"pos=({scalar_int(gripper_position_1)}, {scalar_int(gripper_position_2)}), "
                            f"child_target=({gripper_target_1}, {gripper_target_2}), "
                            f"write_ok=({gripper_write_ok_1}, {gripper_write_ok_2}), "
                            f"hw_error=({gripper_error_1}, {gripper_error_2})"
                        )
                        if active_controller == 1:
                            gripper1_open = not gripper1_open
                            gripper_goal1 = GRIPPER1_RANGE[0] if gripper1_open else GRIPPER1_RANGE[1]
                            print(f"Gripper command -> arm1, target={gripper_goal1}")
                        else:
                            gripper2_open = not gripper2_open
                            gripper_goal2 = GRIPPER2_RANGE[0] if gripper2_open else GRIPPER2_RANGE[1]
                            print(f"Gripper command -> arm2, target={gripper_goal2}")
                        gripper_force_until_t = time.time() + 0.8
                        last_gripper_goal = None
                        gripper.schedule_waypoint(
                            [int(gripper_goal1), int(gripper_goal2)],
                            time.time() + 0.05,
                        )
                        print(f"Gripper targets: {int(gripper_goal1)}, {int(gripper_goal2)}")
                    elif key_stroke == KeyCode(char='0'):
                        target_pose = ACT_init[0].copy()
                        target_pose2 = ACT_init[1].copy()
                        print("Moved both arm targets to ACT_init.")
                    elif key_stroke == KeyCode(char='1'):
                        active_controller = 1
                        target_pose[:] = ACT1[:].copy()
                        print("Moved arm1 target to ACT1; active_controller=1")
                    elif key_stroke == KeyCode(char='2'):
                        active_controller = 1
                        target_pose[:] = ACT2[:].copy()
                        print("Moved arm1 target to ACT2; active_controller=1")
                    elif key_stroke == KeyCode(char='3'):
                        active_controller = 2
                        target_pose2[:] = ACT3[:].copy()
                        print("Moved arm2 rotation target to ACT3; active_controller=1")
                    elif key_stroke == KeyCode(char='4'):
                        active_controller = 2
                        target_pose2[:] = ACT4[:].copy()
                        print("Moved arm2 rotation target to ACT4; active_controller=1")
                    elif key_stroke == KeyCode(char='5'):
                        active_controller = 2
                        target_pose2[:] = ACT5[:].copy()
                        print("Moved arm2 rotation target to ACT5; active_controller=1")
                    elif key_stroke == KeyCode(char='6'):
                        active_controller = 2
                        target_pose2[:] = ACT6[:].copy()
                        print("Moved arm2 rotation target to ACT6; active_controller=1")
                    elif key_stroke == KeyCode(char="i"):
                        print("Switching to Chronos policy mode.")
                        start_policy = True
                        infer_model.reset_hiddens()
                        policy_start_wall_t = time.time()
                        policy_step_count = 0
                        last_gripper_goal = None
                        t_start = time.monotonic() + 0.1
                        iter_idx = 0
                    elif key_stroke == KeyCode(char="z"):
                        if start_policy:
                            print("Leaving Chronos policy mode.")
                            start_policy = False
                            target_pose = curr_pose.copy()
                            target_pose2 = curr_pose2.copy()
                            gripper_goal1 = gripper_position_1
                            gripper_goal2 = gripper_position_2
                            gripper1_open = is_gripper_open(gripper_position_1, GRIPPER1_RANGE)
                            gripper2_open = is_gripper_open(gripper_position_2, GRIPPER2_RANGE)
                            policy_start_wall_t = None
                            policy_step_count = 0
                            last_gripper_goal = None
                            t_start = time.monotonic() + 0.1
                            iter_idx = 0

                if not start_policy and not stop:
                    gripper_goal = [int(gripper_goal1), int(gripper_goal2)]
                    now = time.time()
                    if (last_gripper_goal != gripper_goal
                            or now < gripper_force_until_t
                            or now - last_gripper_command_t >= gripper_command_period):
                        gripper.schedule_waypoint(
                            gripper_goal,
                            now + gripper_command_latency,
                        )
                        last_gripper_goal = gripper_goal.copy()
                        last_gripper_command_t = now
                        gripper_command_period = (
                            gripper_force_period if now < gripper_force_until_t else 1.0
                        )
                        if now < gripper_force_until_t and now - last_gripper_status_print_t > 0.3:
                            print(
                                "Gripper status: "
                                f"goal={gripper_goal}, "
                                f"actual=({scalar_int(gripper_position_1)}, {scalar_int(gripper_position_2)}), "
                                f"child_target=({gripper_target_1}, {gripper_target_2}), "
                                f"write_ok=({gripper_write_ok_1}, {gripper_write_ok_2}), "
                                f"hw_error=({gripper_error_1}, {gripper_error_2})"
                            )
                            last_gripper_status_print_t = now

                if start_policy and not stop:
                    if max(robot1_state_age, robot2_state_age, gripper_state_age) > max_policy_state_age:
                        raise RuntimeError(
                            "Policy state is stale: "
                            f"robot1={robot1_state_age:.3f}s, "
                            f"robot2={robot2_state_age:.3f}s, "
                            f"gripper={gripper_state_age:.3f}s. "
                            "Policy commands are stopped to avoid inference on stale lowdim observations."
                        )
                    policy_elapsed = 0.0 if policy_start_wall_t is None else time.time() - policy_start_wall_t
                    obs_timestamp = collect_observation_timestamp(
                        out.get("timestamp") if out is not None else None,
                        last_data.get("robot_timestamp"),
                        last_data2.get("robot_timestamp"),
                        gripper_stat.get("gripper_timestamp"),
                    )

                    infer_start = time.time()
                    qpos = infer_model.build_qpos(
                        curr_pose,
                        curr_pose2,
                        gripper_position_1,
                        gripper_position_2,
                    )
                    if markov_policy:
                        infer_model.reset_hiddens(verbose=False)
                    qpos_norm = infer_model.normalize_qpos(qpos)
                    qpos_norm_np = qpos_norm.detach().cpu().numpy()
                    qpos_z_abs_max = float(np.max(np.abs(qpos_norm_np)))
                    qpos_z_l2 = float(np.linalg.norm(qpos_norm_np))
                    sequence_norm = infer_model.predict_action_sequence_norm(qpos, img)
                    infer_latency = time.time() - infer_start

                    if policy_action_step >= 0:
                        action_idx = policy_action_step
                    else:
                        target_time = time.time() + action_exec_latency
                        action_idx = compute_aligned_action_step(
                            obs_timestamp=obs_timestamp,
                            target_time=target_time,
                            dt=dt,
                            max_step=infer_model.future_steps - 1,
                            action_label_lead_steps=action_label_lead_steps,
                        )

                    action20_seq = infer_model.denormalize(sequence_norm).detach().cpu().numpy()
                    action_idx = int(np.clip(action_idx, 0, action20_seq.shape[0] - 1))
                    if temporal_agg:
                        selected_action20 = infer_model.select_action_from_sequence_norm(
                            sequence_norm,
                            execute_step_offset=action_idx,
                        ).detach().cpu().numpy()
                        action20 = selected_action20[None, :]
                    else:
                        action20 = action20_seq[action_idx:action_idx + 1]

                    pose1 = pose10d_to_pose6d(action20[:, :9])
                    pose2 = pose10d_to_pose6d(action20[:, 10:19])
                    pose1_seq = pose10d_to_pose6d(action20_seq[:, :9])
                    pose2_seq = pose10d_to_pose6d(action20_seq[:, 10:19])
                    left_delta_seq = np.linalg.norm(
                        pose1_seq[:, :3] - curr_pose[:3][None, :],
                        axis=1,
                    )
                    right_delta_seq = np.linalg.norm(
                        pose2_seq[:, :3] - curr_pose2[:3][None, :],
                        axis=1,
                    )
                    grip1_seq = np.array(
                        [clip_gripper(value, GRIPPER1_RANGE) for value in action20_seq[:, 9]],
                        dtype=np.int32,
                    )
                    grip2_seq = np.array(
                        [clip_gripper(value, GRIPPER2_RANGE) for value in action20_seq[:, 19]],
                        dtype=np.int32,
                    )
                    grip1 = clip_gripper(action20[0, 9], GRIPPER1_RANGE)
                    grip2 = clip_gripper(action20[0, 19], GRIPPER2_RANGE)

                    target_time = time.time() + action_exec_latency
                    controller.schedule_waypoint(pose1[0], target_time)
                    controller2.schedule_waypoint(pose2[0], target_time)
                    gripper.schedule_waypoint([grip1, grip2], target_time)

                    target_pose = pose1[0].copy()
                    target_pose2 = pose2[0].copy()
                    gripper_goal1 = grip1
                    gripper_goal2 = grip2
                    gripper1_open = is_gripper_open(grip1, GRIPPER1_RANGE)
                    gripper2_open = is_gripper_open(grip2, GRIPPER2_RANGE)
                    last_infer_latency = infer_latency

                    if policy_step_count % 5 == 0:
                        print(
                            f"Chronos cmd | latency={last_infer_latency:.4f}s | "
                            f"obs_age={time.time() - obs_timestamp:.3f}s | "
                            f"state_age=({robot1_state_age:.3f}/"
                            f"{robot2_state_age:.3f}/"
                            f"{gripper_state_age:.3f}) | "
                            f"qpos_z=({qpos_z_abs_max:.1f}/{qpos_z_l2:.1f}) | "
                            f"policy_step={policy_step_count}({policy_elapsed:.1f}s) | "
                            f"agg={int(temporal_agg)} | "
                            f"idx={action_idx}/{action20_seq.shape[0] - 1} | "
                            f"delta_l/r=({left_delta_seq[action_idx]:.3f}/"
                            f"{right_delta_seq[action_idx]:.3f}) | "
                            f"gripper=({grip1}, {grip2}) | "
                            f"grip_seq=({grip1_seq[0]}/{grip1_seq[action_idx]}/{grip1_seq.max()}, "
                            f"{grip2_seq[0]}/{grip2_seq[action_idx]}/{grip2_seq.max()}) | "
                            f"actual=({scalar_int(gripper_position_1)}, {scalar_int(gripper_position_2)}) | "
                            f"write_ok=({gripper_write_ok_1}, {gripper_write_ok_2}) | "
                            f"pose1={np.array2string(pose1[0], precision=4)} | "
                            f"pose2={np.array2string(pose2[0], precision=4)}"
                        )
                    policy_step_count += 1

                now_mono = time.monotonic()
                overrun = now_mono - t_cycle_end
                if overrun > 3.0 * dt:
                    now = time.time()
                    if now - last_timing_warning_t > 1.0:
                        print(
                            f"Timing warning: control loop overran by {overrun:.3f}s. "
                            "Resetting schedule after a large delay."
                        )
                        last_timing_warning_t = now
                    t_start = time.monotonic()
                    iter_idx = 0
                elif overrun > 0.5 * dt:
                    now = time.time()
                    if now - last_timing_warning_t > 1.0:
                        print(
                            f"Timing warning: control loop overran by {overrun:.3f}s. "
                            "Continuing without an extra wait to preserve policy step rate."
                        )
                        last_timing_warning_t = now
                    iter_idx += 1
                else:
                    precise_wait(t_cycle_end)
                    iter_idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
