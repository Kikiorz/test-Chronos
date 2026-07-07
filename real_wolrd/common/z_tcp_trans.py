import sapien
import numpy as np
import math
import cv2


class TcpEETrans:
    def __init__(self):
        # Define transformations using SAPIEN Pose
        rt1 = sapien.Pose([0, 0, 0], [0.7071, -0.7071, 0, 0])  # Rotation equivalent to rt1
        rt2 = sapien.Pose([0, 0, 0], [0.7071, 0, 0.7071, 0])  # Rotation equivalent to rt2
        rt3 = sapien.Pose([0, 0, 0], [1, 0, 0, 0])  # Translation along z-axis
        rt4 = sapien.Pose([0, 0, 0], [0, 0, 0, 1])  # Rotation equivalent to rt4

        # Compute the combined transformations
        self.ee_tcp = rt1 * rt2 * rt3
        self.ee_gripper = rt1 * rt2
        self.baselink_base = rt4
        # self.right_base_world = sapien.Pose([0.189821, 0, 1.1822], [0.38269, 0, 0.923877, 0])
        # self.left_base_world = sapien.Pose([-0.189821, 0, 1.1822], [0.38269, 0, -0.923877, 0])

    def ee_to_tcp(self, ee):
        # ee: ee_link in baselink coordinates
        # tcp: tcp in baselink coordinates
        # Perform transformations
        tcp = self.baselink_base * (ee * self.ee_tcp)
        return tcp

    def tcp_to_ee(self, tcp):
        # ee: ee_link in baselink coordinates
        # tcp: tcp in baselink coordinates
        ee = (self.baselink_base.inv() * tcp) * self.ee_tcp.inv()
        return ee

    def ee_in_world(self, ee,base_world):
        # ee: ee in baselink coordinates
        # base_world: base_link in world coordinates
        # ee_world: ee in world coordinates
        ee_world = base_world * ee
        return ee_world

    def ee_in_baselink(self, ee,base_world):
        # ee: ee in world coordinates
        # base_world: base_link in world coordinates
        # ee_baselink: ee in baselink coordinates
        ee_baselink = base_world.inv() * ee
        return ee_baselink