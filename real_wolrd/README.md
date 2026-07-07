# Real World Chronos Open-Source Code

This folder contains the real-world data collection, training, and inference code for the dual-arm UR3 Chronos image policy.

## Directory Layout

```text
real_wolrd_open/
  common/                    # Shared model, dataset, hardware, pose, and shared-memory utilities
  data_collection/           # Real robot data collection and closed-loop robot scripts
  training/                  # Lightning training entry point
  inference/                 # Chronos inference wrapper
  requirements.txt           # Python dependencies
```

## Important Files

- `data_collection/z_data_collect_chronos.py`: real-world data collection entry. It reads D435, Kinect V2, two UR3 arms, SpaceMouse, and the dual gripper, then saves trajectories to the training dataset format.
- `data_collection/z_chronos.py`: real-world closed-loop execution entry. It loads a trained Chronos checkpoint and runs manual or policy control on the robot.
- `training/train_par_3D_IMLE_UR3.py`: main training script for the image + low-dimensional dual-arm policy.
- `inference/inference_choronos.py`: `MyInferenceModel` wrapper. It loads checkpoint/scaler, preprocesses image and pose observations, and returns 20D pose10d-style actions.
- `common/mamba_policy_par_2D_IMLE.py`: Chronos/Mamba policy network.
- `common/M_dataset_real_UR3.py`: dataset loader and scaler fitting utility.
- `common/scaler_M.py`: low-dimensional observation/action normalizer.
- `common/metric_UR10D.py`: pose10d validation metrics.
- `common/z_*.py`: RealSense, Kinect, RTDE, SpaceMouse, gripper, and TCP/EE transform helpers.
- `common/shared_memory_dp/`: shared-memory queues, ring buffers, and arrays used by hardware subprocesses.

## Dataset Format

Each trajectory folder should contain:

```text
pose.npy              # left arm actual TCP pose, [T, 6]
pose2.npy             # right arm actual TCP pose, [T, 6]
target_pose.npy       # left arm target TCP pose, [T, 6]
target_pose2.npy      # right arm target TCP pose, [T, 6]
gripper.npy           # left gripper target, [T]
gripper2.npy          # right gripper target, [T]
gripper_pos.npy       # left gripper measured position, [T]
gripper_pos2.npy      # right gripper measured position, [T]
img/00000.jpg         # D435 RGB frames
```

The training script expects split folders:

```text
/path/to/cover_blocks/
  train/<trajectory folders>
  test/<trajectory folders>
```

## Setup

```bash
cd tacmi_base/real_wolrd_open
pip install -r requirements.txt
```

Some hardware dependencies need system drivers or vendor SDKs in addition to Python packages: Intel RealSense SDK, Kinect V2/libfreenect2, UR RTDE, SpaceMouse/spacenav, and Dynamixel SDK.

## Data Collection

```bash
python data_collection/z_data_collect_chronos.py \
  --robot_ip 192.168.4.63 \
  --robot_ip2 192.168.4.64 \
  --d435-serial <D435_SERIAL> \
  --kinect-serial <KINECT_SERIAL> \
  --output-root ./datasets/recordings
```

Useful keys:

- `space`: start/stop automatic recording.
- `p`: manually append the current frame.
- `s`: save the current recording buffer.
- `backspace`: clear the current buffer.
- `w`: switch active arm.
- `e`: toggle gripper.
- `0/1/2/3`: move to preset poses.
- `q`: quit.

After collection, move or copy saved trajectory folders into `train/` and `test/` splits.

## Fit Scaler

```bash
python common/M_dataset_real_UR3.py \
  --task-name cover_blocks \
  --data-root /path/to/cover_blocks/train \
  --output-dir ./scalers
```

This writes `scalers/scaler_cover_blocks_image_pose10d.pth`.

## Train

```bash
python training/train_par_3D_IMLE_UR3.py \
  --task-name cover_blocks \
  --data-root /path/to/cover_blocks \
  --output-root . \
  --scaler-path ./scalers/scaler_cover_blocks_image_pose10d.pth \
  --devices 0
```

Checkpoints are saved under `checkpoints/<task-name>/S3B_IMAGE_20D_2/`.

## Run Real-World Policy

```bash
python data_collection/z_chronos.py \
  --robot_ip 192.168.4.63 \
  --robot_ip2 192.168.4.64 \
  --ckpt-path ./checkpoints/cover_blocks/S3B_IMAGE_20D_2/last.ckpt \
  --scaler-path ./scalers/scaler_cover_blocks_image_pose10d.pth \
  --device cuda:0
```

Useful keys:

- `i`: enter policy mode.
- `z`: leave policy mode.
- `w`: switch active arm in manual mode.
- `e`: toggle gripper.
- `0/1/2/3/4/5/6`: move to preset poses.
- `q`: quit.

You can also set paths with environment variables:

```bash
export CHRONOS_CKPT_PATH=/path/to/last.ckpt
export CHRONOS_SCALER_PATH=/path/to/scaler.pth
export CHRONOS_RECORD_ROOT=/path/to/new/recordings
```
