import time
import traceback
import queue
import faulthandler

import cv2
import dynamixel_sdk as dxl
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from common.shared_memory_dp.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from common.shared_memory_dp.shared_memory_ring_buffer import SharedMemoryRingBuffer
from common.precise_sleep import precise_wait


class DynamixelController:
    def __init__(
            self,
            device_name,
            baudrate,
            dxl_id,
            protocol_version=2.0,
            port_handler=None,
            packet_handler=None,
            owns_port=True,
            current_limit=200,
            tx_only_writes=False):
        self.device_name = device_name
        self.baudrate = baudrate
        self.dxl_id = dxl_id
        self.protocol_version = protocol_version

        # DYNAMIXEL SDK 相关初始化
        self.portHandler = port_handler if port_handler is not None else dxl.PortHandler(self.device_name)
        self.packetHandler = packet_handler if packet_handler is not None else dxl.PacketHandler(self.protocol_version)
        self.owns_port = owns_port
        self.tx_only_writes = tx_only_writes

        # 常见控制表地址（以 Protocol 2.0 + X 系列默认控制表为例）
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.ADDR_OPERATING_MODE = 11  # 对于多数 X 系列, Operating Mode 位于地址 11
        self.ADDR_CURRENT_LIMIT = 38  # X 系列大多在地址 38 (2 Bytes)，部分型号可能不同
        self.ADDR_HARDWARE_ERROR_STATUS = 70
        self.ADDR_STATUS_RETURN_LEVEL = 68

        # Torque 相关宏定义
        self.TORQUE_ENABLE = 1
        self.TORQUE_DISABLE = 0

        # 连接标志
        self.connected = False
        if port_handler is None:
            self.connect()
        else:
            self.connected = True
        # self.reboot()  # 根据需要是否开机后重启

        if self.tx_only_writes:
            self.set_status_return_level(1)
        # 设置当前模式为基于电流的控制模式（Current-based Position, mode=5）
        self.set_operating_mode(5)
        # 设置电流限制，可调整夹爪力度，单位具体需查阅控制表
        self.set_current_limit(current_limit)
        # 默认先启用扭矩
        self.enable_torque()
        # 添加 PID 控制表地址（以 X 系列为例）
        self.ADDR_POSITION_P_GAIN = 84  # Position P Gain (2 Bytes)
        self.ADDR_POSITION_I_GAIN = 82  # Position I Gain (2 Bytes)
        self.ADDR_POSITION_D_GAIN = 80  # Position D Gain (2 Bytes)
        self.set_pid_gain(100,1,4700)


    def connect(self):
        if self.portHandler.openPort():
            print("串口打开成功")
            if self.portHandler.setBaudRate(self.baudrate):
                print("波特率设置成功")
                self.connected = True
            else:
                print("无法设置波特率")
                self.portHandler.closePort()
        else:
            print("无法打开串口")

    def enable_torque(self, verbose=True):
        """
        使能电机扭矩
        """
        if self.connected:
            if self.tx_only_writes:
                dxl_comm_result = self.packetHandler.write1ByteTxOnly(
                    self.portHandler,
                    self.dxl_id,
                    self.ADDR_TORQUE_ENABLE,
                    self.TORQUE_ENABLE
                )
                dxl_error = 0
            else:
                dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                    self.portHandler,
                    self.dxl_id,
                    self.ADDR_TORQUE_ENABLE,
                    self.TORQUE_ENABLE
                )
            if dxl_comm_result != dxl.COMM_SUCCESS:
                print("[TxRxResult] %s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif verbose:
                print("电机扭矩启动成功")

    def set_status_return_level(self, level):
        if not self.connected:
            return

        if self.tx_only_writes:
            dxl_comm_result = self.packetHandler.write1ByteTxOnly(
                self.portHandler,
                self.dxl_id,
                self.ADDR_STATUS_RETURN_LEVEL,
                level
            )
            dxl_error = 0
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler,
                self.dxl_id,
                self.ADDR_STATUS_RETURN_LEVEL,
                level
            )
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("[TxRxResult] %s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("[RxPacketError] %s" % self.packetHandler.getRxPacketError(dxl_error))

    def get_hardware_error_status(self):
        if self.connected:
            error_status, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(
                self.portHandler,
                self.dxl_id,
                self.ADDR_HARDWARE_ERROR_STATUS
            )
            if dxl_comm_result == dxl.COMM_SUCCESS and dxl_error == 0:
                return error_status
        return None

    def disable_torque(self):
        """
        关闭电机扭矩
        """
        if self.connected:
            if self.tx_only_writes:
                dxl_comm_result = self.packetHandler.write1ByteTxOnly(
                    self.portHandler,
                    self.dxl_id,
                    self.ADDR_TORQUE_ENABLE,
                    self.TORQUE_DISABLE
                )
                dxl_error = 0
            else:
                dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                    self.portHandler,
                    self.dxl_id,
                    self.ADDR_TORQUE_ENABLE,
                    self.TORQUE_DISABLE
                )
            if dxl_comm_result != dxl.COMM_SUCCESS:
                print("[TxRxResult] %s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("[RxPacketError] %s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("电机扭矩已关闭")

    def set_operating_mode(self, mode):
        """
        设置电机的控制模式（Operating Mode）。
        注意：设置模式前需要先 disable torque，然后再写入模式，最后再 enable torque。

        常见的 Operating Mode（以 X 系列为例）:
            0: 电流控制模式 (Current Control)
            1: 速度控制模式 (Velocity Control)
            3: 位置控制模式 (Position Control)
            4: 扩展位置模式 (Extended Position Control)
            5: 基于电流的 位置控制模式 (Current-based Position Control)
            16: PWM 控制模式 (PWM Control)
        """
        if not self.connected:
            print("尚未连接到舵机，无法设置 Operating Mode")
            return

        # 1. 先关闭扭矩
        self.disable_torque()

        # 2. 写入 Operating Mode 寄存器
        if self.tx_only_writes:
            dxl_comm_result = self.packetHandler.write1ByteTxOnly(
                self.portHandler,
                self.dxl_id,
                self.ADDR_OPERATING_MODE,
                mode
            )
            dxl_error = 0
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler,
                self.dxl_id,
                self.ADDR_OPERATING_MODE,
                mode
            )
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("[TxRxResult] %s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            return
        elif dxl_error != 0:
            print("[RxPacketError] %s" % self.packetHandler.getRxPacketError(dxl_error))
            return
        else:
            print(f"电机 Operating Mode 已设置为 {mode}")

        # 3. 重启扭矩
        self.enable_torque()

    def set_pid_gain(self, p_gain, i_gain, d_gain):
        """
        设置位置控制模式下的 PID 参数（需在 Position Control 模式下生效）
        参数范围参考 Dynamixel 文档（通常 P: 0~32767, I/D: 0~32767）
        """
        if not self.connected:
            print("尚未连接到舵机，无法设置 PID 参数")
            return

        # 1. 关闭扭矩
        self.disable_torque()

        # 2. 写入 PID 参数
        # 设置 P Gain
        if self.tx_only_writes:
            dxl_comm_result = self.packetHandler.write2ByteTxOnly(
                self.portHandler,
                self.dxl_id,
                self.ADDR_POSITION_P_GAIN,
                p_gain
            )
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                self.portHandler,
                self.dxl_id,
                self.ADDR_POSITION_P_GAIN,
                p_gain
            )
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print(f"[TxRxResult] P增益写入失败: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"电机 P 增益已设置为 {p_gain}")

        # 设置 I Gain（可选）
        if self.tx_only_writes:
            dxl_comm_result = self.packetHandler.write2ByteTxOnly(
                self.portHandler,
                self.dxl_id,
                self.ADDR_POSITION_I_GAIN,
                i_gain
            )
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                self.portHandler,
                self.dxl_id,
                self.ADDR_POSITION_I_GAIN,
                i_gain
            )
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print(f"[TxRxResult] I增益写入失败: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"电机 I 增益已设置为 {p_gain}")

        # 设置 D Gain（可选）
        if self.tx_only_writes:
            dxl_comm_result = self.packetHandler.write2ByteTxOnly(
                self.portHandler,
                self.dxl_id,
                self.ADDR_POSITION_D_GAIN,
                d_gain
            )
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                self.portHandler,
                self.dxl_id,
                self.ADDR_POSITION_D_GAIN,
                d_gain
            )
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print(f"[TxRxResult] D增益写入失败: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"电机 D 增益已设置为 {p_gain}")

        # 3. 重新启用扭矩
        self.enable_torque()

    def set_current_limit(self, current_limit):
        """
        设置电流限制（单位：约为 [mA], 具体需查阅对应型号控制表）。
        对于 X 系列大多数型号来说，Current Limit 通常是一个 2 字节的寄存器（地址 38）。

        注意：写入该参数前，也需要先关闭扭矩，然后再写入，再重新开启扭矩。

        current_limit: 整数, 表示最大电流限制值。具体范围取决于舵机型号。
        """
        if not self.connected:
            print("尚未连接到舵机，无法设置 Current Limit")
            return

        # 1. 先关闭扭矩
        self.disable_torque()

        # 2. 写入 Current Limit 寄存器（2 Bytes）
        if self.tx_only_writes:
            dxl_comm_result = self.packetHandler.write2ByteTxOnly(
                self.portHandler,
                self.dxl_id,
                self.ADDR_CURRENT_LIMIT,
                current_limit
            )
            dxl_error = 0
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                self.portHandler,
                self.dxl_id,
                self.ADDR_CURRENT_LIMIT,
                current_limit
            )
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("[TxRxResult] %s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            return
        elif dxl_error != 0:
            print("[RxPacketError] %s" % self.packetHandler.getRxPacketError(dxl_error))
            return
        else:
            print(f"电机 Current Limit 已设置为 {current_limit} (单位请参考控制表)")

        # 3. 重启扭矩
        self.enable_torque()

    def set_goal_position(self, goal_position):
        """
        设置目标位置（在 Position Control / Extended Position Control 模式下有效）。
        """
        if self.connected:
            for attempt in range(3):
                if self.tx_only_writes:
                    dxl_comm_result = self.packetHandler.write4ByteTxOnly(
                        self.portHandler,
                        self.dxl_id,
                        self.ADDR_GOAL_POSITION,
                        goal_position
                    )
                    dxl_error = 0
                else:
                    dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                        self.portHandler,
                        self.dxl_id,
                        self.ADDR_GOAL_POSITION,
                        goal_position
                    )
                if dxl_comm_result == dxl.COMM_SUCCESS and dxl_error == 0:
                    return True
                time.sleep(0.002)
            if dxl_comm_result != dxl.COMM_SUCCESS:
                print(f"[ID {self.dxl_id} TxRxResult] {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"[ID {self.dxl_id} RxPacketError] {self.packetHandler.getRxPacketError(dxl_error)}")
        return False

    def get_present_position(self):
        """
        获取当前实际位置（在 Position Control / Extended Position Control 模式下）或反馈的当前位置值。
        """
        if self.connected:
            dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                self.portHandler,
                self.dxl_id,
                self.ADDR_PRESENT_POSITION
            )
            if dxl_comm_result != dxl.COMM_SUCCESS:
                print("[TxRxResult] %s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            else:
                return dxl_present_position
        return None

    def reboot(self):
        """
        对电机进行重启操作，需要在支持 reboot 指令的 Dynamixel 型号上使用。
        重启后电机将回到初始状态，需要再次 enable torque。
        """
        if self.connected:
            dxl_comm_result, dxl_error = self.packetHandler.reboot(
                self.portHandler,
                self.dxl_id
            )
            if dxl_comm_result != dxl.COMM_SUCCESS:
                print("[TxRxResult] %s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("[RxPacketError] %s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("电机已重启")

    def close(self):
        """
        关闭串口
        """
        if self.connected and self.owns_port:
            self.portHandler.closePort()
            print("串口已关闭")
            self.connected = False
        elif self.connected:
            self.connected = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable_torque()
        self.close()


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2
class GripperController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 port='/dev/ttyUSB0',
                 frequency=30,
                 get_max_k=None,
                 command_queue_size=2048,
                 launch_timeout=3,
                 move_max_speed=200000.0,
                 current_limit=200,
                 state_read_frequency=10,
                 write_repeat_period=0.5,
                 hardware_error_check_period=2.0,
                 initial_position_1=None,
                 initial_position_2=None,
                 tx_only_writes=False,
                 receive_latency=0.0,
                 verbose=False
                 ):
        super().__init__(name="GripperController2")
        self.port = port
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.move_max_speed=move_max_speed
        self.current_limit = current_limit
        self.state_read_frequency = state_read_frequency
        self.write_repeat_period = write_repeat_period
        self.hardware_error_check_period = hardware_error_check_period
        self.initial_position_1 = initial_position_1
        self.initial_position_2 = initial_position_2
        self.tx_only_writes = tx_only_writes
        self.verbose = verbose
        if get_max_k is None:
            get_max_k = int(frequency * 10)

        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos_1': 0,
            'target_pos_2': 0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )

        # build ring buffer
        example = {
            'gripper_position_1': 0,
            'gripper_position_2': 0,
            'gripper_target_1': 0,
            'gripper_target_2': 0,
            'gripper_write_ok_1': 0,
            'gripper_write_ok_2': 0,
            'gripper_error_1': 0,
            'gripper_error_2': 0,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.error_queue = mp.Queue(maxsize=1)
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[GripperController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        if not self.is_alive():
            return
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        ready = self.ready_event.wait(self.launch_timeout)
        startup_error = self._pop_startup_error()
        if startup_error is not None:
            raise RuntimeError(startup_error)
        if not ready:
            raise RuntimeError(
                f"Gripper controller on {self.port} did not become ready "
                f"within {self.launch_timeout} seconds."
            )
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def _pop_startup_error(self):
        try:
            return self.error_queue.get_nowait()
        except queue.Empty:
            return None

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def schedule_waypoint(self, pos: list[int], target_time):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos_1': pos[0],
            'target_pos_2': pos[1],
            'target_time': target_time
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):
        faulthandler.enable()
        try:
            shared_port_handler = dxl.PortHandler(self.port)
            shared_packet_handler = dxl.PacketHandler(2.0)
            if not shared_port_handler.openPort():
                raise RuntimeError(f"无法打开串口 {self.port}")
            if not shared_port_handler.setBaudRate(115200):
                shared_port_handler.closePort()
                raise RuntimeError(f"无法设置 {self.port} 波特率 115200")
            print(f"共享串口 {self.port} 打开成功，波特率 115200")
            try:
                with DynamixelController(
                        self.port, 115200, 1,
                        port_handler=shared_port_handler,
                        packet_handler=shared_packet_handler,
                        owns_port=False,
                        current_limit=self.current_limit,
                        tx_only_writes=self.tx_only_writes) as Gripper1, \
                     DynamixelController(
                        self.port, 115200, 2,
                        port_handler=shared_port_handler,
                        packet_handler=shared_packet_handler,
                        owns_port=False,
                        current_limit=self.current_limit,
                        tx_only_writes=self.tx_only_writes) as Gripper2:
                    curr_t = time.monotonic()
                    if self.initial_position_1 is not None and self.initial_position_2 is not None:
                        last_info_1 = int(self.initial_position_1)
                        last_info_2 = int(self.initial_position_2)
                    else:
                        info_1 = Gripper1.get_present_position()
                        info_2 = Gripper2.get_present_position()
                        if info_1 is None or info_2 is None:
                            raise RuntimeError(
                                f"Failed to read initial gripper positions: id1={info_1}, id2={info_2}"
                            )
                        last_info_1 = int(info_1)
                        last_info_2 = int(info_2)
                    target_pos_1 = last_info_1
                    target_pos_2 = last_info_2
                    last_write_ok_1 = 0
                    last_write_ok_2 = 0
                    last_error_1 = 0
                    last_error_2 = 0
                    last_commanded_1 = None
                    last_commanded_2 = None
                    last_write_t = -float("inf")
                    last_read_t = curr_t
                    last_error_check_t = curr_t
                    pending_target_1 = target_pos_1
                    pending_target_2 = target_pos_2
                    pending_target_time = curr_t
                    read_period = None
                    if self.state_read_frequency and self.state_read_frequency > 0:
                        read_period = 1.0 / self.state_read_frequency
                    keep_running = True
                    t_start = time.monotonic()
                    iter_idx = 0
                    while keep_running:
                        t_now = time.monotonic()
                        try:
                            if t_now >= pending_target_time:
                                target_pos_1 = pending_target_1
                                target_pos_2 = pending_target_2

                            try:
                                commands = self.input_queue.get_all()
                                n_cmd = len(commands['cmd'])
                            except Empty:
                                n_cmd = 0

                            for i in range(n_cmd):
                                command = dict()
                                for key, value in commands.items():
                                    command[key] = value[i]
                                cmd = command['cmd']

                                if cmd == Command.SHUTDOWN.value:
                                    keep_running = False
                                    break
                                elif cmd == Command.SCHEDULE_WAYPOINT.value:
                                    command_target_1 = int(command['target_pos_1'])
                                    command_target_2 = int(command['target_pos_2'])
                                    target_time = float(command['target_time'])
                                    target_time_mono = time.monotonic() - time.time() + target_time
                                    target_time_mono = max(target_time_mono, t_now + 1.0 / self.frequency)
                                    pending_target_1 = command_target_1
                                    pending_target_2 = command_target_2
                                    pending_target_time = target_time_mono
                                    if self.verbose:
                                        print(f"[GripperController] queued target=({pending_target_1}, {pending_target_2})")
                                elif cmd == Command.RESTART_PUT.value:
                                    t_start = command['target_time'] - time.time() + time.monotonic()
                                    iter_idx = 1

                            if t_now >= pending_target_time:
                                target_pos_1 = pending_target_1
                                target_pos_2 = pending_target_2

                            should_write = (
                                last_commanded_1 != target_pos_1
                                or last_commanded_2 != target_pos_2
                                or (t_now - last_write_t) >= self.write_repeat_period
                            )
                            if should_write:
                                last_write_ok_1 = int(Gripper1.set_goal_position(target_pos_1))
                                last_write_ok_2 = int(Gripper2.set_goal_position(target_pos_2))
                                if last_write_ok_1:
                                    last_commanded_1 = target_pos_1
                                    if read_period is None:
                                        last_info_1 = target_pos_1
                                if last_write_ok_2:
                                    last_commanded_2 = target_pos_2
                                    if read_period is None:
                                        last_info_2 = target_pos_2
                                last_write_t = t_now

                            if (
                                    self.hardware_error_check_period
                                    and self.hardware_error_check_period > 0
                                    and (t_now - last_error_check_t) >= self.hardware_error_check_period):
                                error_1 = Gripper1.get_hardware_error_status()
                                error_2 = Gripper2.get_hardware_error_status()
                                last_error_1 = int(error_1) if error_1 is not None else -1
                                last_error_2 = int(error_2) if error_2 is not None else -1
                                if error_1:
                                    print(f"[ID 1 HardwareError] status={error_1}, trying to re-enable torque")
                                    Gripper1.enable_torque(verbose=False)
                                if error_2:
                                    print(f"[ID 2 HardwareError] status={error_2}, trying to re-enable torque")
                                    Gripper2.enable_torque(verbose=False)
                                last_error_check_t = t_now

                            if read_period is not None and (t_now - last_read_t) >= read_period:
                                info_1 = Gripper1.get_present_position()
                                info_2 = Gripper2.get_present_position()
                                if info_1 is not None:
                                    last_info_1 = int(info_1)
                                if info_2 is not None:
                                    last_info_2 = int(info_2)
                                last_read_t = t_now

                            state = {
                                'gripper_position_1': last_info_1,
                                'gripper_position_2': last_info_2,
                                'gripper_target_1': target_pos_1,
                                'gripper_target_2': target_pos_2,
                                'gripper_write_ok_1': last_write_ok_1,
                                'gripper_write_ok_2': last_write_ok_2,
                                'gripper_error_1': last_error_1,
                                'gripper_error_2': last_error_2,
                                'gripper_receive_timestamp': time.time(),
                                'gripper_timestamp': time.time() - self.receive_latency
                            }
                            self.ring_buffer.put(state)
                        except Exception:
                            print("[GripperController2] Non-fatal control-loop exception:")
                            traceback.print_exc()
                            time.sleep(0.05)

                        if iter_idx == 0:
                            self.ready_event.set()
                        iter_idx += 1

                        dt = 1 / self.frequency
                        t_end = t_start + dt * iter_idx
                        precise_wait(t_end=t_end, time_func=time.monotonic)
            finally:
                shared_port_handler.closePort()
                print(f"共享串口 {self.port} 已关闭")
        except Exception:
            message = "[GripperController2] Fatal exception:\n" + traceback.format_exc()
            print(message)
            try:
                self.error_queue.put_nowait(message)
            except queue.Full:
                pass
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[GripperController2] Process exiting.")

if __name__ == "__main__":
    griper1_ranger =[1824,3444]
    griper2_ranger =[686,2396]

    with SharedMemoryManager() as smm:
        controller = GripperController(
            shm_manager=smm,
            port='/dev/ttyUSB0',
            frequency=30,
            verbose=True
        )
        controller.start(wait=True)
        while True:
            print("0")
            controller.schedule_waypoint([griper1_ranger[0],griper2_ranger[0]],  time.time()+ 0.5)
            time.sleep(2.5)
            # print("0")
            # controller.schedule_waypoint([griper1_ranger[1],griper2_ranger[1]],  time.time() + 0.5)
            # time.sleep(2.5)
            # # a = controller.get_state()
            # print("1")
        # controller.stop(wait=True)
