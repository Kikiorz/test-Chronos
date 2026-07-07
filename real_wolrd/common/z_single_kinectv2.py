from typing import Optional, Callable, Dict
import enum
import time
import numpy as np
import multiprocessing as mp
import cv2
from multiprocessing.managers import SharedMemoryManager
from common.shared_memory_dp.shared_ndarray import SharedNDArray
from common.shared_memory_dp.shared_memory_ring_buffer import SharedMemoryRingBuffer
from common.shared_memory_dp.shared_memory_queue2 import SharedMemoryQueue, Full, Empty
from pylibfreenect2 import Freenect2, SyncMultiFrameListener, FrameType, Registration, Frame, OpenGLPacketPipeline
from pylibfreenect2 import FrameType
class Command(enum.Enum):
    START_RECORDING = 0
    STOP_RECORDING = 1
    RESTART_PUT = 2

class SingleKinectV2(mp.Process):
    MAX_PATH_LENGTH = 4096  # Maximum path length

    def __init__(
            self,
            shm_manager: SharedMemoryManager,
            serial_number=None,
            resolution=(1080, 720),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            enable_color=True,
            enable_depth=False,
            get_max_k=30,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            verbose=False
    ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # Create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]  # Reverse the resolution to match (height, width)
        examples = {'color': np.ndarray(shape=shape + (3,), dtype=np.uint8),
                    # 'ir': np.ndarray(shape=shape, dtype=np.uint8),
                    'depth': np.ndarray(shape=shape, dtype=np.uint16),
                    'camera_capture_timestamp': 0.0,
                    'camera_receive_timestamp': 0.0,
                    'timestamp': 0.0,
                    'step_idx': 0}

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

        # Create command queue
        examples = {
            'cmd': Command.START_RECORDING.value,
            'video_path': np.array('a' * self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # Shared variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.verbose = verbose
        self.put_start_time = None

        # Shared events and buffers
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue

    @staticmethod
    def get_connected_devices_serial():
        fn = Freenect2()
        num_devices = fn.enumerateDevices()
        serials = []
        for i in range(num_devices):
            serial = fn.getDeviceSerialNumber(i)
            serials.append(serial)
        serials = sorted(serials)
        return serials


    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # User API methods
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    # Command methods
    def start_recording(self, video_path: str, start_time: float = -1):
        # Implement recording functionality if needed
        pass

    def stop_recording(self):
        # Implement stop recording functionality if needed
        pass

    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    # Process run method
    def run(self):
        from pylibfreenect2 import Freenect2Device

        undistorted = Frame(512, 424, 4)
        registered = Frame(512, 424, 4)
        bigdepth = Frame(1920, 1082, 4)
        color_depth_map = np.zeros((424, 512), np.int32).ravel()
        # Initialize Kinect V2 device
        self.fn = Freenect2()
        self.pipeline = OpenGLPacketPipeline()
        if self.fn.enumerateDevices() == 0:
            print("No Kinect V2 devices connected!")
            return

        if self.serial_number is None:
            self.serial_number = self.fn.getDeviceSerialNumber(0)
        elif isinstance(self.serial_number, str):
            self.serial_number = self.serial_number.encode('utf-8')
        device = self.fn.openDevice(self.serial_number, pipeline=self.pipeline)

        listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
        # self.registration = Registration(device.getIrCameraParams(),
        #                                  device.getColorCameraParams())  # Context manager methods
        device.setColorFrameListener(listener)
        device.setIrAndDepthFrameListener(listener)

        device.start()

        self.registration = Registration(device.getIrCameraParams(),
                                    device.getColorCameraParams())

        # Signal that the device is ready
        self.ready_event.set()

        put_idx = None
        put_start_time = self.put_start_time
        if put_start_time is None:
            put_start_time = time.time()

        iter_idx = 0
        t_start = time.time()

        try:
            while not self.stop_event.is_set():
                frames = listener.waitForNewFrame()

                data = dict()
                receive_time = time.time()
                data['camera_receive_timestamp'] = receive_time

                if self.enable_color:
                    color_frame = frames["color"]
                    data['color'] = color_frame
                    data['camera_capture_timestamp'] = color_frame.timestamp / 1e6  # Convert to seconds

                if self.enable_depth:
                    depth_frame = frames["depth"]
                    data['depth'] = depth_frame
                    if not self.enable_color:
                        data['camera_capture_timestamp'] = depth_frame.timestamp / 1e6  # Convert to seconds
                self.registration.apply(color_frame, depth_frame, undistorted, registered,
                                        bigdepth=bigdepth, color_depth_map=color_depth_map)

                w, h = self.resolution
                data['color'] = cv2.resize(cv2.flip(cv2.cvtColor(color_frame.asarray(),
                                                                     cv2.COLOR_RGBA2RGB), 1), dsize=(w, h))
                depth_data = bigdepth.asarray(np.float32)[:1080, :]
                depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)
                depth_data = depth_data.astype(np.uint16)
                data['depth'] = cv2.resize(cv2.flip(depth_data, 0), self.resolution)
                #
                # data['depth'] = cv2.resize(cv2.flip(bigdepth.asarray(np.float32)[:1080, :].astype(np.uint16),
                #                                         1), dsize=(w, h))
                # Apply transform if any
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                # Frequency regulation
                if self.put_downsample:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)

                # Put to vis ring buffer
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)

                # Verbose output
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[SingleKinectV2 {self.serial_number}] FPS {frequency}')

                # Fetch and execute commands
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.START_RECORDING.value:
                        # Implement recording start if needed
                        pass
                    elif cmd == Command.STOP_RECORDING.value:
                        # Implement recording stop if needed
                        pass
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']

                listener.release(frames)
                iter_idx += 1

        finally:
            device.stop()
            device.close()
            self.ready_event.set()

            if self.verbose:
                print(f'[SingleKinectV2 {self.serial_number}] Exiting worker process.')

if __name__ == "__main__":
    # Create a shared memory manager
    with SharedMemoryManager() as smm:
        # Get the serial numbers of connected Kinect V2 devices
        serial_numbers = SingleKinectV2.get_connected_devices_serial()

        if not serial_numbers:
            print("No Kinect V2 devices connected!")
            exit(1)

        # Create an instance of SingleKinectV2
        kinect = SingleKinectV2(
            shm_manager=smm,
            serial_number=serial_numbers[0],
            resolution=(1080, 720),
            capture_fps=30,
            put_fps=30,
            enable_color=True,
            enable_depth=True,
            verbose=True
        )

        # Start the SingleKinectV2 process
        kinect.start()

        try:
            # Wait for the process to be ready
            kinect.start_wait()

            # Main loop to retrieve frames
            for _ in range(100):  # Retrieve frames for 100 iterations
                # Get frames from the camera
                data = kinect.get()

                frame = data['color']
                depth = data['depth']

                # Display the color frame
                cv2.imshow('Kinect V2 Color Frame', frame)

                # Display the depth frame
                depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_uint8 = depth_normalized.astype(np.uint8)
                cv2.imshow('Kinect V2 Depth Frame', depth_uint8)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.1)  # Simulate processing time

        finally:
            # Stop the camera process
            kinect.stop(wait=True)
            cv2.destroyAllWindows()
