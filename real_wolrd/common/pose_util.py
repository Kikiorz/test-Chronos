import cv2
import numpy as np
import scipy.spatial.transform as st



def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=pos.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot.as_matrix()
    mat[..., 3, 3] = 1
    return mat


def mat_to_pos_rot(mat):
    pos = (mat[..., :3, 3].T / mat[..., 3, 3].T).T
    rot = st.Rotation.from_matrix(mat[..., :3, :3])
    return pos, rot


def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape + (6,), dtype=pos.dtype)
    pose[..., :3] = pos
    pose[..., 3:] = rot.as_rotvec()
    return pose


def pose_to_pos_rot(pose):
    pos = pose[..., :3]
    rot = st.Rotation.from_rotvec(pose[..., 3:])
    return pos, rot


def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))


def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))


def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose


def transform_point(tx, point):
    return point @ tx[:3, :3].T + tx[:3, 3]


def project_point(k, point):
    x = point @ k.T
    uv = x[..., :2] / x[..., [2]]
    return uv


def apply_delta_pose(pose, delta_pose):
    new_pose = np.zeros_like(pose)

    new_pose[:3] = pose[:3] + delta_pose[:3]

    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose


def normalize(vec, tol=1e-7):
    return vec / np.maximum(np.linalg.norm(vec), tol)


def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot


def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out


def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out


def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out


def mat_to_pose10d(mat):
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10


def pose10d_to_mat(d10):
    pos = d10[..., :3]
    d6 = d10[..., 3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1] + (4, 4), dtype=d10.dtype)
    out[..., :3, :3] = rotmat
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1
    return out



def rot3d_to_rot6d(rot3):
    rotmat = Rodrigues(rot3)
    d6 = mat_to_rot6d(rotmat)
    return d6


def rot6d_to_rot3d(rot6):
    rotmat = rot6d_to_mat(rot6)
    rot3 = Rodrigues_inv(rotmat)
    return rot3


def pose6d_to_pose10d(pose6d):
    pos = pose6d[..., :3]
    rot3 = pose6d[..., 3:]
    rotmat = Rodrigues(rot3)
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10


def pose10d_to_pose6d(pose10d):
    pos = pose10d[..., :3]
    d6 = pose10d[..., 3:9]
    rotmat = rot6d_to_mat(d6)
    rot3 = Rodrigues_inv(rotmat)
    pose6d = np.concatenate([pos, rot3], axis=-1)
    return pose6d


def pose6d_to_pose_mat(pose6d):
    pos = pose6d[..., :3]
    rot3 = pose6d[..., 3:]
    rotmat = Rodrigues(rot3)
    shape = pose6d.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=pos.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rotmat
    mat[..., 3, 3] = 1
    return mat

def pose_mat_to_pose6d(mat):
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    rot3 = Rodrigues_inv(rotmat)
    pose6d = np.concatenate([pos,rot3],axis=1)
    return pose6d

def pose6d_to_pose10d_relative(pose6d, base6d):
    pose_mat = pose6d_to_pose_mat(pose6d)
    base_mat = pose6d_to_pose_mat(base6d)
    mat = np.linalg.inv(base_mat) @ pose_mat
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose10d_to_pose10d_relative(pose10d, base10d):
    pose_mat = pose10d_to_pose_mat(pose10d)
    base_mat = pose10d_to_pose_mat(base10d)
    mat = np.linalg.inv(base_mat) @ pose_mat
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose10d_to_pose_mat(pose10d):
    pos = pose10d[..., :3]
    d6 = pose10d[..., 3:9]
    rotmat = rot6d_to_mat(d6)
    shape = pose10d.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=pos.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rotmat
    mat[..., 3, 3] = 1
    return mat


def pose10d_to_pose6d_relative(pose10d, base10d):
    pose_mat = pose10d_to_pose_mat(pose10d)
    base_mat = pose10d_to_pose_mat(base10d)
    mat = base_mat @ pose_mat
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    rot3 = Rodrigues_inv(rotmat)
    pose6d = np.concatenate([pos, rot3], axis=-1)
    return pose6d



def Rodrigues(rot3):
    N = rot3.shape[0]
    rotmat = np.zeros((N,3,3),dtype=np.float32)
    for i in range(N):
        rotmat[i,...] = cv2.Rodrigues(np.array(rot3[i, :]))[0]

    return rotmat

def Rodrigues_inv(rotmat):
    N = rotmat.shape[0]
    rot3 = np.zeros((N,3),dtype=np.float32)
    for i in range(N):
        rot3[i,...] = cv2.Rodrigues(np.array(rotmat[i, :, :]))[0][:,0]

    return rot3




