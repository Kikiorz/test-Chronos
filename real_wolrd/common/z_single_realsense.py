from typing import Optional, Callable, Dict
import enum
import time
import json
import queue
import traceback
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from common.shared_memory_dp.shared_ndarray import SharedNDArray
from common.timestamp_accumulator import get_accumulate_timestamp_idxs
from common.shared_memory_dp.shared_memory_ring_buffer import SharedMemoryRingBuffer
from common.shared_memory_dp.shared_memory_queue2 import SharedMemoryQueue, Full, Empty


class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4


class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096  # linux path has a limit of 4096 bytes

    def __init__(
            self,
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(1280, 720),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=30,
            advanced_mode_config=None,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            frame_timeout_ms=5000,
            max_consecutive_timeouts=3,
            launch_timeout=None,
            stop_timeout=5.0,
            reset_on_stop=False,
            reset_wait_s=3.0,
            verbose=False
    ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(
                shape=shape + (3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(
                shape=shape, dtype=np.uint16)
        if enable_infrared:
            examples['infrared'] = np.empty(
                shape=shape, dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None
            else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
            else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': rs.option.exposure.value,
            'option_value': 0.0,
            'video_path': np.array('a' * self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager,
            shape=(7,),
            dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.frame_timeout_ms = frame_timeout_ms
        self.max_consecutive_timeouts = max_consecutive_timeouts
        if launch_timeout is None:
            launch_timeout = max(10.0, frame_timeout_ms * 3 / 1000.0 + 5.0)
        self.launch_timeout = launch_timeout
        self.stop_timeout = stop_timeout
        self.reset_on_stop = reset_on_stop
        self.reset_wait_s = reset_wait_s
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.error_queue = mp.Queue(maxsize=4)
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array

    @staticmethod
    def get_connected_devices_serial(wait_timeout=0.0, retry_period=0.5):
        deadline = time.monotonic() + wait_timeout
        while True:
            serials = list()
            for d in rs.context().devices:
                if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                    serial = d.get_info(rs.camera_info.serial_number)
                    product_line = d.get_info(rs.camera_info.product_line)
                    name = d.get_info(rs.camera_info.name)
                    if product_line == 'D400':
                        if 'D435' in name:
                            # only works with D400 series
                            serials.append(serial)
            serials = sorted(serials)
            if serials or time.monotonic() >= deadline:
                return serials
            time.sleep(retry_period)

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            try:
                self.start_wait()
            except Exception:
                self._cleanup_after_failed_start()
                raise

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        ready = self.ready_event.wait(self.launch_timeout)
        startup_error = self._pop_startup_error()
        if startup_error is not None:
            raise RuntimeError(startup_error)
        if not ready:
            raise RuntimeError(
                f"RealSense {self.serial_number} did not become ready within "
                f"{self.launch_timeout} seconds."
            )
        if self.ring_buffer.count == 0 and not self.is_alive():
            raise RuntimeError(self._startup_failure_message())

    def end_wait(self):
        self.join(timeout=self.stop_timeout)
        if self.is_alive():
            print(
                f"[SingleRealsense {self.serial_number}] process did not stop within "
                f"{self.stop_timeout}s; terminating it.",
                flush=True
            )
            self.terminate()
            self.join(timeout=1.0)
            if self.is_alive() and hasattr(self, "kill"):
                print(
                    f"[SingleRealsense {self.serial_number}] process still alive; killing it.",
                    flush=True
                )
                self.kill()
                self.join(timeout=1.0)
            if self.reset_on_stop:
                self._hardware_reset_device(self.serial_number, self.reset_wait_s)

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def _put_error_message(self, message):
        try:
            self.error_queue.put_nowait(message)
        except queue.Full:
            try:
                self.error_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.error_queue.put_nowait(message)
            except queue.Full:
                pass

    def _pop_startup_error(self):
        try:
            return self.error_queue.get_nowait()
        except queue.Empty:
            return None

    def get_last_error(self):
        return self._pop_startup_error()

    def _startup_failure_message(self):
        return (
            f"RealSense {self.serial_number} exited before the first frame. "
            "Check USB3 connection, camera power/permissions, serial number, "
            "and whether another process is using the camera."
        )

    def _cleanup_after_failed_start(self):
        self.stop_event.set()
        if self.is_alive():
            self.terminate()
        self.join(timeout=1)
        if self.reset_on_stop:
            self._hardware_reset_device(self.serial_number, self.reset_wait_s)

    @staticmethod
    def _hardware_reset_device(serial_number, wait_s=3.0):
        try:
            ctx = rs.context()
            for dev in ctx.devices:
                try:
                    dev_serial = dev.get_info(rs.camera_info.serial_number)
                except Exception:
                    continue
                if dev_serial == serial_number:
                    print(f"[SingleRealsense {serial_number}] hardware reset RealSense device.")
                    dev.hardware_reset()
                    if wait_s > 0:
                        time.sleep(wait_s)
                    return True
            print(f"[SingleRealsense {serial_number}] hardware reset skipped: device not found.")
        except Exception as e:
            print(f"[SingleRealsense {serial_number}] hardware reset failed: {e}")
        return False

    def get(self, k=None, out=None):
        if self.ring_buffer.count == 0:
            if not self.is_alive():
                raise RuntimeError(
                    f"RealSense {self.serial_number} is not running and no frame was captured."
                )
            error = self.get_last_error()
            if error is not None:
                raise RuntimeError(error)
            raise RuntimeError(
                f"RealSense {self.serial_number} has not produced a frame yet."
            )
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)

    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0, 0] = fx
        mat[1, 1] = fy
        mat[0, 2] = ppx
        mat[1, 2] = ppy
        return mat

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale

    def start_recording(self, video_path: str, start_time: float = -1):
        raise NotImplementedError("Recording feature is not available in this version of SingleRealsense.")

    def stop_recording(self):
        raise NotImplementedError("Recording feature is not available in this version of SingleRealsense.")

    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        w, h = self.resolution
        fps = self.capture_fps
        align = rs.align(rs.stream.color)
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(rs.stream.color,
                                    w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth,
                                    w, h, rs.format.z16, fps)
        if self.enable_infrared:
            rs_config.enable_stream(rs.stream.infrared,
                                    w, h, rs.format.y8, fps)

        pipeline = None
        pipeline_started = False
        got_first_frame = False
        consecutive_timeouts = 0
        try:
            rs_config.enable_device(self.serial_number)

            # start pipeline
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)
            pipeline_started = True

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909
            d = pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, 1)

            # setup advanced mode
            if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                device = pipeline_profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # get
            color_stream = pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ['fx', 'fy', 'ppx', 'ppy', 'height', 'width']
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.enable_depth:
                depth_sensor = pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale

            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f'[SingleRealsense {self.serial_number}] Main loop started.')

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                # wait for frames to come in
                try:
                    frameset = pipeline.wait_for_frames(self.frame_timeout_ms)
                except RuntimeError as e:
                    consecutive_timeouts += 1
                    print(
                        f"[SingleRealsense {self.serial_number}] wait_for_frames timeout "
                        f"{consecutive_timeouts}: {e}",
                        flush=True
                    )
                    if (self.max_consecutive_timeouts is not None
                            and consecutive_timeouts >= self.max_consecutive_timeouts):
                        raise RuntimeError(
                            f"RealSense {self.serial_number} failed to deliver frames "
                            f"after {self.max_consecutive_timeouts} consecutive "
                            f"{self.frame_timeout_ms} ms waits."
                        ) from e
                    continue
                receive_time = time.time()
                # align frames to color
                try:
                    frameset = align.process(frameset)
                except RuntimeError as e:
                    consecutive_timeouts += 1
                    print(
                        f"[SingleRealsense {self.serial_number}] dropped unusable frameset "
                        f"{consecutive_timeouts}: align failed: {e}",
                        flush=True
                    )
                    if (self.max_consecutive_timeouts is not None
                            and consecutive_timeouts >= self.max_consecutive_timeouts):
                        raise RuntimeError(
                            f"RealSense {self.serial_number} failed to deliver usable frames "
                            f"after {self.max_consecutive_timeouts} consecutive bad frames."
                        ) from e
                    continue

                color_frame = frameset.get_color_frame() if self.enable_color else None
                depth_frame = frameset.get_depth_frame() if self.enable_depth else None
                infrared_frame = frameset.get_infrared_frame() if self.enable_infrared else None
                missing_streams = []
                if self.enable_color and not color_frame:
                    missing_streams.append("color")
                if self.enable_depth and not depth_frame:
                    missing_streams.append("depth")
                if self.enable_infrared and not infrared_frame:
                    missing_streams.append("infrared")
                if missing_streams:
                    consecutive_timeouts += 1
                    print(
                        f"[SingleRealsense {self.serial_number}] dropped incomplete frameset "
                        f"{consecutive_timeouts}: missing {', '.join(missing_streams)}",
                        flush=True
                    )
                    if (self.max_consecutive_timeouts is not None
                            and consecutive_timeouts >= self.max_consecutive_timeouts):
                        raise RuntimeError(
                            f"RealSense {self.serial_number} failed to deliver complete frames "
                            f"after {self.max_consecutive_timeouts} consecutive bad frames."
                        )
                    continue
                consecutive_timeouts = 0

                # grab data
                data = dict()
                data['camera_receive_timestamp'] = receive_time
                # realsense report in ms
                data['camera_capture_timestamp'] = frameset.get_timestamp() / 1000
                if self.enable_color:
                    data['color'] = np.asarray(color_frame.get_data())
                    t = color_frame.get_timestamp() / 1000
                    data['camera_capture_timestamp'] = t
                    # print('device', time.time() - t)
                    # print(color_frame.get_frame_timestamp_domain())
                if self.enable_depth:
                    data['depth'] = np.asarray(
                        depth_frame.get_data())
                if self.enable_infrared:
                    data['infrared'] = np.asarray(
                        infrared_frame.get_data())

                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                        timestamps=[receive_time],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        # this is non in first iteration
                        # and then replaced with a concrete number
                        next_global_idx=put_idx,
                        # continue to pump frames even if not started.
                        # start_time is simply used to align timestamps.
                        allow_negative=True
                    )

                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        # put_data['timestamp'] = put_start_time + step_idx / self.put_fps
                        put_data['timestamp'] = receive_time
                        # print(step_idx, data['timestamp'])
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)

                # signal ready
                if iter_idx == 0:
                    got_first_frame = True
                    self.ready_event.set()

                # put to vis
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)

                # record frame (dummy implementation without video_recorder)
                # recording_transform is ignored here
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] FPS {frequency}')

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                        # print('auto', sensor.get_option(rs.option.enable_auto_exposure))
                        # print('exposure', sensor.get_option(rs.option.exposure))
                        # print('gain', sensor.get_option(rs.option.gain))
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = pipeline_profile.get_device().first_depth_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                    elif cmd == Command.START_RECORDING.value:
                        raise NotImplementedError(
                            "Recording feature is not available in this version of SingleRealsense.")
                    elif cmd == Command.STOP_RECORDING.value:
                        raise NotImplementedError(
                            "Recording feature is not available in this version of SingleRealsense.")
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                        # self.ring_buffer.clear()

                iter_idx += 1
        except BaseException as e:
            message = (
                f"RealSense {self.serial_number} worker crashed: {e}\n"
                f"{traceback.format_exc()}"
            )
            if not got_first_frame:
                message += f"\n{self._startup_failure_message()}"
            self._put_error_message(message)
            raise
        finally:
            if pipeline_started:
                try:
                    pipeline.stop()
                except Exception as e:
                    if self.verbose:
                        print(f'[SingleRealsense {self.serial_number}] pipeline.stop failed: {e}')
            if self.reset_on_stop:
                self._hardware_reset_device(self.serial_number, self.reset_wait_s)
            rs_config.disable_all_streams()
            if not got_first_frame:
                self.ready_event.set()

        if self.verbose:
            print(f'[SingleRealsense {self.serial_number}] Exiting worker process.')


if __name__ == "__main__":
    # Create a shared memory manager
    with SharedMemoryManager() as smm:
        # Replace 'your_serial_number_here' with an actual serial number of your RealSense device
        serial_numbers = SingleRealsense.get_connected_devices_serial()

        # Create an instance of SingleRealsense
        realsense = SingleRealsense(smm,
                                    serial_numbers[0],
                                    resolution=(640, 480),
                                    capture_fps=30,
                                    put_fps=30, )
        # Start the SingleRealsense process
        realsense.start()

        try:

            # Wait for the process to be ready (you can omit this if not needed)
            realsense.start_wait()

            # Main loop to retrieve frames
            for _ in range(100):  # Example: Retrieve frames for 100 iterations
                # Get frames from the camera
                data = realsense.get()

                frame = data['color']

                # Display the frame
                cv2.imshow('UVC Camera Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.1)  # Example: Simulate processing time

        finally:
            # Stop the camera process
            realsense.stop(wait=True)
