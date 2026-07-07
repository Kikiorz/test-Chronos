import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import sapien
import scipy.spatial.transform as st
from transforms3d.quaternions import mat2quat
from tqdm import tqdm

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from common.z_single_realsense import SingleRealsense
from common.z_spacemouse_shared_memory import Spacemouse
from common.z_rtde_interpolation_controller import RTDEInterpolationController
from common.z_keystroke_counter import KeystrokeCounter, KeyCode, Key
from common.z_single_kinectv2 import SingleKinectV2
from common.precise_sleep import precise_wait
from common.z_dualgripper_controller import GripperController
from common.z_tcp_trans import TcpEETrans


def is_empty_or_black_image(image):
    if image is None or image.size == 0:
        return True
    return np.max(image) == 0


def describe_image(image):
    if image is None:
        return "None"
    if not isinstance(image, np.ndarray):
        return f"type={type(image).__name__}"
    if image.size == 0:
        return f"shape={image.shape}, dtype={image.dtype}, empty"
    return f"shape={image.shape}, dtype={image.dtype}, min={np.min(image)}, max={np.max(image)}"


def summarize_process_error(error_text):
    if not error_text:
        return None
    for line in str(error_text).splitlines():
        line = line.strip()
        if line:
            return line
    return None


def get_image_issue(name, image, reject_black=False):
    if image is None:
        return f"{name} is None"
    if not isinstance(image, np.ndarray):
        return f"{name} is {type(image).__name__}, not ndarray"
    if image.size == 0:
        return f"{name} is empty"
    if image.ndim not in (2, 3):
        return f"{name} has unsupported shape {image.shape}"
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        return f"{name} has unsupported channel count: shape={image.shape}"
    if image.dtype not in (np.uint8, np.uint16):
        return f"{name} has unsupported dtype {image.dtype}"
    if reject_black and np.max(image) == 0:
        return f"{name} is black"
    return None


def get_record_image_issues(record_data):
    checks = [
        ("img", "D435 color", True),
        ("rgb", "Kinect color", True),
        ("depth", "Kinect depth", False),
        ("d435_depth", "D435 depth", False),
    ]
    issues = []
    for key, name, reject_black in checks:
        if key not in record_data:
            issues.append(f"{name} missing key '{key}'")
            continue
        issue = get_image_issue(name, record_data[key], reject_black=reject_black)
        if issue is not None:
            issues.append(f"{issue} ({describe_image(record_data[key])})")
    return issues


def get_current_frame_issues(img, img_kinect, img_kinect_dep, img_d435_dep):
    return get_record_image_issues({
        "img": img,
        "rgb": img_kinect,
        "depth": img_kinect_dep,
        "d435_depth": img_d435_dep,
    })


def write_image_checked(path, image, reject_black=False, name="image"):
    issue = get_image_issue(name, image, reject_black=reject_black)
    if issue is not None:
        raise RuntimeError(f"{issue}; skip writing {path} ({describe_image(image)})")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.ascontiguousarray(image)
    ok = cv2.imwrite(path, image)
    if not ok:
        parent = os.path.dirname(path)
        raise RuntimeError(
            f"cv2.imwrite failed for {name}: {path} "
            f"({describe_image(image)}, parent_exists={os.path.isdir(parent)}, "
            f"parent_writable={os.access(parent, os.W_OK)})"
        )


def save_image(i, record_data_buffer, img_path, kinect_rgb_path, kinect_dep_path, d435_dep_path):
    write_image_checked(
        os.path.join(img_path, f"{i:05d}.jpg"),
        record_data_buffer[i]["img"],
        reject_black=True,
        name=f"frame {i} D435 color"
    )
    write_image_checked(
        os.path.join(kinect_rgb_path, f"{i:05d}.jpg"),
        record_data_buffer[i]["rgb"],
        reject_black=True,
        name=f"frame {i} Kinect color"
    )
    write_image_checked(
        os.path.join(kinect_dep_path, f"{i:05d}.png"),
        record_data_buffer[i]["depth"],
        name=f"frame {i} Kinect depth"
    )
    write_image_checked(
        os.path.join(d435_dep_path, f"{i:05d}.png"),
        record_data_buffer[i]["d435_depth"],
        name=f"frame {i} D435 depth"
    )




ACT_init = np.array([[ -0.05084912, -0.34176292, 0.02294293, 1.04612629, -1.12444314, -1.16306351 ],
                     [ 0.02584831, -0.35633479, 0.08430657, 1.24871465, 1.24880682, 1.20975324 ]])

ACT1 = np.array([ -0.16711849, -0.36469953, 0.18070701, 1.04621144, -1.12428807, -1.16322084 ])
ACT2 = np.array([ -0.17381676, -0.29316792, 0.19020146, 1.04624196, -1.12425546, -1.16323566 ])
ACT3 = np.array([ 0.13726749, -0.37045928, 0.23959901, 1.24878388, 1.24875910, 1.20987907 ])

ACT4 = np.array([ 0.12341055, -0.31286883, 0.10750832, 1.20761578, 0.69864733, -0.59031819 ])

ACT5 = np.array([ 0.10016374, -0.38853544, 0.17038664, 1.58483811, -0.00734046, 0.00889610 ])
ACT6 = np.array([ 0.10001195, -0.38858940, 0.17034564, 1.49410165, 0.62650589, -0.61947482 ])
ACT7 = np.array([ 0.10023311, -0.38873217, 0.16997043, 1.28280903, 1.12321898, -1.10316417 ])

ACT8 = np.array([ -0.03756308, -0.36607853, 0.21328162, 1.47487852, 0.58147101, -0.64168096 ])
ACT9 = np.array([ -0.03059125, -0.38504818, 0.22026195, 1.47491811, 0.58154572, -0.64174633 ])

griper1_ranger = [1977, 3700]
griper2_ranger = [302, 1986]

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
            f"and device permissions, or pass the serial manually with "
            f"{option_name}."
        )

    return detected_serials[0]

@click.command()
@click.option('-r', '--robot_ip', default='192.168.4.63')
@click.option('-r2', '--robot_ip2', default='192.168.4.64')
@click.option('--d435-serial', default=None, help='RealSense D435 serial number to use.')
@click.option('--kinect-serial', default=None, help='Kinect V2 serial number to use.')
@click.option(
    '--output-root',
    default=lambda: os.environ.get(
        "CHRONOS_RECORD_ROOT",
        os.path.join(PACKAGE_ROOT, "datasets", "recordings"),
    ),
    show_default="CHRONOS_RECORD_ROOT or ./datasets/recordings",
    help='Directory where new trajectory folders are saved.',
)
def main(robot_ip, robot_ip2, d435_serial, kinect_serial, output_root):
    cv2.setNumThreads(4)
    frequency = 10
    dt = 1 / frequency
    max_pos_speed = 0.0250
    max_rot_speed = 0.1500
    cube_diag = np.linalg.norm([1, 1, 1])
    reset_gripper = False
    offset = 0.0
    j_init = [ -1.1469181219684046, -2.6888716856585901, -1.3540738264666956, -2.6056087652789515, 0.4185419380664825, 0.4437399208545685 ]
    j_init2 = [ 1.1009780168533325, -0.7465727964984339, 1.4875335693359375, -0.8355210463153284, -0.4877889792071741, 0.0977032706141472 ]
    right_base_world = sapien.Pose([0.189821, 0, 1.1822], [0.38269, 0, 0.923877, 0])
    left_base_world = sapien.Pose([-0.189821, 0, 1.1822], [0.38269, 0, -0.923877, 0])
    transformer = TcpEETrans()
    d435_frame_timeout_ms = 1000
    d435_max_consecutive_timeouts = None
    rtde_launch_timeout = 60.0
    serial_numbers_d435 = SingleRealsense.get_connected_devices_serial(wait_timeout=10.0)
    serial_numbers_kinect = SingleKinectV2.get_connected_devices_serial()
    print(f"Detected D435 serials: {serial_numbers_d435 or 'none'}")
    print(f"Detected Kinect V2 serials: {serial_numbers_kinect or 'none'}")
    d435_serial = select_device_serial('D435', serial_numbers_d435, d435_serial, '--d435-serial')
    kinect_serial = select_device_serial('Kinect V2', serial_numbers_kinect, kinect_serial, '--kinect-serial')

    with (SharedMemoryManager() as shm_manager):
        with KeystrokeCounter() as key_counter, \
                SingleRealsense(shm_manager,
                                d435_serial,
                                resolution=(640, 480),
                                capture_fps=30,
                                put_fps=30,
                                enable_depth=True,
                                frame_timeout_ms=d435_frame_timeout_ms,
                                max_consecutive_timeouts=d435_max_consecutive_timeouts,
                                reset_on_stop=True,
                                reset_wait_s=4.0, ) as camera, \
                RTDEInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=robot_ip,
                    lookahead_time=0.1,
                    max_pos_speed=max_pos_speed * cube_diag,
                    max_rot_speed=max_rot_speed * cube_diag,
                    tcp_offset_pose=[0, 0, offset, 0, 0, 0],
                    joints_init=j_init,
                    launch_timeout=rtde_launch_timeout,
                    payload_mass=1.0,
                    verbose=False) as controller, \
                RTDEInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=robot_ip2,
                    lookahead_time=0.1,
                    max_pos_speed=max_pos_speed * cube_diag,
                    max_rot_speed=max_rot_speed * cube_diag,
                    tcp_offset_pose=[0, 0, offset, 0, 0, 0],
                    joints_init=j_init2,
                    launch_timeout=rtde_launch_timeout,
                    payload_mass=1.0,
                    verbose=False) as controller2, \
                SingleKinectV2(
                    shm_manager=shm_manager,
                    serial_number=kinect_serial,
                    resolution=(1280, 720),
                    capture_fps=30,
                    put_fps=30,
                    enable_color=True,
                    enable_depth=True,
                    verbose=False
                ) as kinect, \
                Spacemouse(shm_manager=shm_manager) as sm, \
                GripperController(
                    shm_manager=shm_manager,
                    port='/dev/ttyUSB0',
                    frequency=30,
                    current_limit=200,
                    verbose=False
                ) as gripper:
            out = None
            out_multi = None
            out_kinect = None
            img = None
            img_multi = None
            img_kinect = None
            img_d435_dep = None
            print('Ready!')
            d435_depth_scale = camera.get_depth_scale()
            print(f"D435 depth scale: {d435_depth_scale}")
            record_data_buffer = list()
            state = controller.get_state()
            target_pose = state['TargetTCPPose']
            state2 = controller2.get_state()
            target_pose2 = state2['TargetTCPPose']
            stop = False
            continue_record = False
            t_start = time.monotonic()
            iter_idx = 0
            command_latency = 0.01
            gripper_pos = 0
            last_pose = None
            last_pose2 = None
            active_controller = 1
            gripper1_open = True
            gripper2_open = True
            pos_threshold = 0.0002
            rot_threshold = 0.0005
            gripper_threshold = 50
            gripper_goal1 = griper1_ranger[0]
            gripper_goal2 = griper2_ranger[0]
            gripper_position_1_ = None
            gripper_position_2_ = None
            gripper_process_dead_reported = False
            last_gripper_goal = None
            last_gripper_command_t = 0.0
            gripper_command_period = 1.0
            gripper_command_latency = 0.2
            gripper_force_until_t = 0.0
            gripper_force_period = 0.1
            last_gripper_status_print_t = 0.0
            max_camera_frame_age = max(3.0, d435_frame_timeout_ms / 1000.0 + 2.0)
            last_camera_warning_t = 0.0
            camera_warning_period = 2.0

            while not stop:

                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt
                last_data = controller.get_state()
                last_data2 = controller2.get_state()
                curr_pose = last_data['ActualTCPPose']
                curr_pose2 = last_data2['ActualTCPPose']
                gripper_stat = gripper.get_state()
                gripper_position_1 = gripper_stat['gripper_position_1']
                gripper_position_2 = gripper_stat['gripper_position_2']
                gripper_target_1 = gripper_stat.get('gripper_target_1', -1)
                gripper_target_2 = gripper_stat.get('gripper_target_2', -1)
                gripper_write_ok_1 = gripper_stat.get('gripper_write_ok_1', -1)
                gripper_write_ok_2 = gripper_stat.get('gripper_write_ok_2', -1)
                gripper_error_1 = gripper_stat.get('gripper_error_1', -1)
                gripper_error_2 = gripper_stat.get('gripper_error_2', -1)
                precise_wait(t_sample)
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
                drot = st.Rotation.from_euler('xyz', drot_xyz)
                if active_controller == 1:
                    quat = mat2quat(cv2.Rodrigues(target_pose[3:])[0])
                    tcp1 = sapien.Pose(target_pose[:3], quat)
                    ee1 = transformer.tcp_to_ee(tcp1)
                    ee1_in_world = transformer.ee_in_world(ee1, right_base_world)
                    ee1_in_world.p += dpos
                    ee1 = transformer.ee_in_baselink(ee1_in_world, right_base_world)
                    tcp1 = transformer.ee_to_tcp(ee1)
                    target_pose[:3] = tcp1.p
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()
                else:
                    quat = mat2quat(cv2.Rodrigues(target_pose2[3:])[0])
                    tcp2 = sapien.Pose(target_pose2[:3], quat)
                    ee2 = transformer.tcp_to_ee(tcp2)
                    ee2_in_world = transformer.ee_in_world(ee2, left_base_world)
                    ee2_in_world.p += dpos
                    ee2 = transformer.ee_in_baselink(ee2_in_world, left_base_world)
                    tcp2 = transformer.ee_to_tcp(ee2)
                    target_pose2[:3] = tcp2.p
                    target_pose2[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose2[3:])).as_rotvec()

                controller.schedule_waypoint(target_pose,
                                             t_command_target - time.monotonic() + time.time())
                controller2.schedule_waypoint(target_pose2,
                                              t_command_target - time.monotonic() + time.time())

                frames_valid = True
                camera_warnings = []
                try:
                    out = camera.get(out=out)
                except RuntimeError as e:
                    out = None
                    frames_valid = False
                    camera_warnings.append(f"D435 get failed: {summarize_process_error(e)}")
                if out is not None:
                    img = out['color']
                    img_d435_dep = out['depth']
                    d435_age = time.time() - float(out['timestamp'])
                    if not camera.is_alive():
                        frames_valid = False
                        d435_error = summarize_process_error(camera.get_last_error())
                        d435_warning = f"D435 process is not alive (exitcode={camera.exitcode})"
                        if d435_error:
                            d435_warning += f": {d435_error}"
                        camera_warnings.append(d435_warning)
                    elif d435_age > max_camera_frame_age:
                        frames_valid = False
                        camera_warnings.append(f"D435 frame is stale ({d435_age:.2f}s old)")
                else:
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    img_d435_dep = np.zeros((480, 640), dtype=np.uint16)
                try:
                    out_kinect = kinect.get(out=out_kinect)
                except RuntimeError as e:
                    out_kinect = None
                    frames_valid = False
                    camera_warnings.append(f"Kinect get failed: {e}")
                if out_kinect is not None:
                    img_kinect = out_kinect['color']
                    img_kinect_dep = out_kinect['depth']
                    kinect_age = time.time() - float(out_kinect['timestamp'])
                    if not kinect.is_alive():
                        frames_valid = False
                        camera_warnings.append("Kinect process is not alive")
                    elif kinect_age > max_camera_frame_age:
                        frames_valid = False
                        camera_warnings.append(f"Kinect frame is stale ({kinect_age:.2f}s old)")
                else:
                    img_kinect = np.zeros((720, 1280, 3), dtype=np.uint8)
                    img_kinect_dep = np.zeros((720, 1280), dtype=np.uint16)
                if camera_warnings and time.time() - last_camera_warning_t > camera_warning_period:
                    print("Camera warning: " + " | ".join(camera_warnings))
                    last_camera_warning_t = time.time()
                img_vis = cv2.flip(img.copy(),-1)
                text = f'Num frames saved: {len(record_data_buffer)}'
                cv2.putText(
                    img_vis,
                    text,
                    (10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255, 0, 0)
                )
                cv2.namedWindow('vis_img', 0)
                cv2.imshow('vis_img', img_vis)
               
               
                cv2.namedWindow('kinvect', 0)
                cv2.imshow('kinvect', img_kinect)
                cv2.pollKey()
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    elif key_stroke == KeyCode(char='w'):
                        active_controller = 2 if active_controller == 1 else 1
                        print(f"Switched to {'controller2' if active_controller == 2 else 'controller'}")
                    elif key_stroke == KeyCode(char='e'):
                        print(
                            "Gripper actual before command: "
                            f"pos=({gripper_position_1}, {gripper_position_2}), "
                            f"child_target=({gripper_target_1}, {gripper_target_2}), "
                            f"write_ok=({gripper_write_ok_1}, {gripper_write_ok_2}), "
                            f"hw_error=({gripper_error_1}, {gripper_error_2})"
                        )
                        if active_controller == 1:
                            gripper1_open = not gripper1_open
                            gripper_goal1 = griper1_ranger[0] if gripper1_open else griper1_ranger[1]
                            gripper_force_until_t = time.time() + 0.8
                            print(f"Gripper command -> arm1, target={gripper_goal1}")
                        else:
                            gripper2_open = not gripper2_open
                            gripper_goal2 = griper2_ranger[0] if gripper2_open else griper2_ranger[1]
                            gripper_force_until_t = time.time() + 0.8
                            print(f"Gripper command -> arm2, target={gripper_goal2}")
                        print(f"Gripper1 {'closed' if not gripper1_open else 'opened'}")
                        print(f"Gripper2 {'closed' if not gripper2_open else 'opened'}")
                    elif key_stroke == Key.space:
                        continue_record = not continue_record
                        print(f"continue_record={continue_record}")
                    elif key_stroke == Key.backspace:
                        if len(record_data_buffer) > 0:
                            record_data_buffer = []
                    elif key_stroke == KeyCode(char='0'):
                        target_pose = ACT_init[0].copy()
                        target_pose2 = ACT_init[1].copy()
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
                        print("Moved arm2 rotation target to ACT3; active_controller=2")
                    elif key_stroke == KeyCode(char='p'):
                        if not frames_valid:
                            invalid_images = ['camera frame not fresh']
                        else:
                            invalid_images = get_current_frame_issues(
                                img,
                                img_kinect,
                                img_kinect_dep,
                                img_d435_dep
                            )
                        if invalid_images:
                            print("Skip this frame: " + " | ".join(invalid_images))
                            continue
                        record_data_buffer.append({
                            'img': img.copy(),
                            'tcp_pose': curr_pose.copy(),  # 单臂中已存在的字段
                            'tcp_pose2': curr_pose2.copy(),  # 双臂新增字段
                            'target_pose': target_pose.copy(),
                            'target_pose2': target_pose2.copy(),
                            'gripper1': gripper_goal1,
                            'gripper2': gripper_goal2,
                            'gripper_position_1': gripper_position_1,
                            'gripper_position_2': gripper_position_2,
                            'd435_depth': img_d435_dep.copy(),
                            'rgb': img_kinect.copy(),
                            'depth': img_kinect_dep.copy(),
                        })
                    elif key_stroke == KeyCode(char='s'):
                        if len(record_data_buffer) > 0:
                            valid_record_data_buffer = []
                            dropped_frames = []
                            for frame_idx, frame_data in enumerate(record_data_buffer):
                                frame_issues = get_record_image_issues(frame_data)
                                if frame_issues:
                                    dropped_frames.append((frame_idx, frame_issues))
                                else:
                                    valid_record_data_buffer.append(frame_data)
                            if dropped_frames:
                                print(
                                    f"Skipping {len(dropped_frames)} invalid buffered frames "
                                    f"out of {len(record_data_buffer)} before saving."
                                )
                                for frame_idx, frame_issues in dropped_frames[:5]:
                                    print(f"  dropped frame {frame_idx}: " + " | ".join(frame_issues))
                                if len(dropped_frames) > 5:
                                    print(f"  ... {len(dropped_frames) - 5} more invalid frames")
                            if not valid_record_data_buffer:
                                print("No valid frames to save; buffer kept so you can inspect or clear it.")
                                continue
                            time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                            path = os.path.join(os.path.abspath(output_root), time_now)
                            if not os.path.exists(path):
                                os.makedirs(path)
                            img_path = path + "/img"
                            if not os.path.exists(img_path):
                                os.makedirs(img_path)
                            kinect_rgb_path = path + "/rgb"
                            if not os.path.exists(kinect_rgb_path):
                                os.makedirs(kinect_rgb_path)
                            kinect_dep_path = path + "/depth"
                            if not os.path.exists(kinect_dep_path):
                                os.makedirs(kinect_dep_path)
                            d435_dep_path = path + "/d435_depth"
                            if not os.path.exists(d435_dep_path):
                                os.makedirs(d435_dep_path)
                            pose_list = []
                            pose_list2 = []  # 第二个机械臂的TCP位姿列表
                            gripper_list = []
                            gripper_list2 = []
                            gripper_pos = []
                            gripper_pos2 = []
                            target_pose_list = []
                            target_pose_list2 = []

                            for i in tqdm(range(len(valid_record_data_buffer)), desc='Processing'):
                                gripper_list.append(valid_record_data_buffer[i]["gripper1"])
                                gripper_list2.append(valid_record_data_buffer[i]["gripper2"])
                                pose_list.append(valid_record_data_buffer[i]["tcp_pose"])  # 第一只手臂的TCP
                                pose_list2.append(valid_record_data_buffer[i]["tcp_pose2"])  # 第二只手臂的TCP
                                gripper_pos.append(valid_record_data_buffer[i]["gripper_position_1"])
                                gripper_pos2.append(valid_record_data_buffer[i]["gripper_position_2"])
                                target_pose_list.append(valid_record_data_buffer[i]["target_pose"])
                                target_pose_list2.append(valid_record_data_buffer[i]["target_pose2"])
                            with ThreadPoolExecutor() as executor:
                                futures = [
                                    executor.submit(
                                        save_image,
                                        i,
                                        valid_record_data_buffer,
                                        img_path,
                                        kinect_rgb_path,
                                        kinect_dep_path,
                                        d435_dep_path
                                    )
                                    for i in range(len(valid_record_data_buffer))
                                ]
                                for future in tqdm(as_completed(futures), total=len(futures), desc='Saving images'):
                                    future.result()
                            source_indices = [
                                i for i, frame_data in enumerate(record_data_buffer)
                                if not get_record_image_issues(frame_data)
                            ]
                            np.save(path + "/pose.npy", pose_list)
                            np.save(path + "/pose2.npy", pose_list2)  # 保存第二只手臂的位姿数据
                            np.save(path + "/gripper.npy", gripper_list)
                            np.save(path + "/gripper2.npy", gripper_list2)
                            np.save(path + "/gripper_pos.npy", gripper_pos)
                            np.save(path + "/gripper_pos2.npy", gripper_pos2)
                            np.save(path + "/target_pose.npy", target_pose_list)
                            np.save(path + "/target_pose2.npy", target_pose_list2)
                            np.save(path + "/d435_depth_scale.npy", d435_depth_scale)
                            np.save(path + "/source_frame_indices.npy", source_indices)
                            print(f"Saved data to {path}, clear buffer")
                            t_start = time.monotonic() + 0.1
                            iter_idx = 0
                            t_cycle_end = t_start
                        record_data_buffer.clear()
                        record_data_buffer = []
                        continue_record = False
                gripper_goal = [int(gripper_goal1), int(gripper_goal2)]
                now = time.time()
                if not gripper.is_alive():
                    if not gripper_process_dead_reported:
                        print("ERROR: GripperController process is not alive. Continuing without gripper control.")
                        gripper_process_dead_reported = True
                if (gripper.is_alive()
                        and (last_gripper_goal != gripper_goal
                        or now < gripper_force_until_t
                        or now - last_gripper_command_t >= gripper_command_period)):
                    gripper.schedule_waypoint(
                        gripper_goal,
                        now + gripper_command_latency
                    )
                    last_gripper_goal = gripper_goal.copy()
                    last_gripper_command_t = now
                    if now < gripper_force_until_t:
                        gripper_command_period = gripper_force_period
                    else:
                        gripper_command_period = 1.0
                    if now < gripper_force_until_t and now - last_gripper_status_print_t > 0.3:
                        print(
                            "Gripper status: "
                            f"goal={gripper_goal}, "
                            f"actual=({gripper_position_1}, {gripper_position_2}), "
                            f"child_target=({gripper_target_1}, {gripper_target_2}), "
                            f"write_ok=({gripper_write_ok_1}, {gripper_write_ok_2}), "
                            f"hw_error=({gripper_error_1}, {gripper_error_2})"
                        )
                        last_gripper_status_print_t = now

                arm1_changed = False
                arm2_changed = False
                gripper_changed = False
                if last_pose is not None:
                    arm1_pos_diff = np.max(np.abs((curr_pose - last_pose)[:3]))
                    arm1_rot_diff = np.max(np.abs((curr_pose - last_pose)[3:]))
                    if arm1_pos_diff > pos_threshold or arm1_rot_diff > rot_threshold:
                        arm1_changed = True

                if last_pose2 is not None:
                    arm2_pos_diff = np.max(np.abs((curr_pose2 - last_pose2)[:3]))
                    arm2_rot_diff = np.max(np.abs((curr_pose2 - last_pose2)[3:]))
                    if arm2_pos_diff > pos_threshold or arm2_rot_diff > rot_threshold:
                        arm2_changed = True
                if gripper_position_1_ is not None:
                    er_g1 = np.abs(gripper_position_1 - gripper_position_1_)
                    er_g2 = np.abs(gripper_position_2 - gripper_position_2_)
                    if er_g1 > gripper_threshold or er_g2 > gripper_threshold:
                        gripper_changed = True
                if frames_valid and continue_record and (arm1_changed or arm2_changed or gripper_changed):
                    frame_issues = get_current_frame_issues(
                        img,
                        img_kinect,
                        img_kinect_dep,
                        img_d435_dep
                    )
                    if frame_issues:
                        if time.time() - last_camera_warning_t > camera_warning_period:
                            print("Skip auto-record frame: " + " | ".join(frame_issues))
                            last_camera_warning_t = time.time()
                        last_pose = curr_pose.copy()
                        last_pose2 = curr_pose2.copy()
                        gripper_position_1_ = gripper_position_1
                        gripper_position_2_ = gripper_position_2
                        precise_wait(t_cycle_end)
                        iter_idx += 1
                        continue
                    record_data_buffer.append({
                        'img': img.copy(),
                        'tcp_pose': curr_pose.copy(),  # 单臂中已存在的字段
                        'tcp_pose2': curr_pose2.copy(),  # 双臂新增字段
                        'target_pose': target_pose.copy(),
                        'target_pose2': target_pose2.copy(),
                        'gripper1': gripper_goal1,
                        'gripper2': gripper_goal2,
                        'gripper_position_1': gripper_position_1,
                        'gripper_position_2': gripper_position_2,
                        'd435_depth': img_d435_dep.copy(),
                        'rgb': img_kinect.copy(),
                        'depth': img_kinect_dep.copy(),
                    })
                last_pose = curr_pose.copy()
                last_pose2 = curr_pose2.copy()
                gripper_position_1_ = gripper_position_1
                gripper_position_2_ = gripper_position_2

                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == '__main__':
    main()
