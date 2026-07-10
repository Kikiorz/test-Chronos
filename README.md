# Chronos: A Physics-Informed Full-History Framework for Non-Markovian Long-Horizon Manipulation

This is the official implementation of our paper, which has been submitted to **IEEE Transactions on Robotics (T-RO)**.

[Project Website](https://chronos-manipulation.github.io/) | [arXiv](https://arxiv.org/abs/2606.30318) | [Code](https://github.com/yulinzhouZYL/Chronos)

Chronos is a physics-informed full-history imitation learning framework for memory-dependent long-horizon manipulation. It treats observation history as the latent state of the policy dynamics and refines multimodal action priors through a second-order Schrödinger-inspired acceleration bridge.
<img width="2408" height="837" alt="e39d25de-303e-4b0d-aa4f-c7e9f2f195bf" src="https://github.com/user-attachments/assets/be021f1c-61ca-4cb4-9568-540af040f397" />

In experiments, Chronos achieves **73.6%** average success on RMBench, outperforming Mem-0 by **+22.8 percentage points** with **30x fewer parameters**, and reaches **78%** average success while pi0.5 reaches **7%** in 4 real-world dual-arm experiments.

This repository provides the Chronos implementation for both RMBench simulation experiments and real-world dual-arm robot experiments. The included `RMBench/` folder contains the RMBench benchmark environment, and the Chronos simulation policy is located at `RMBench/policy/Chronos`. The `real_wolrd/` folder contains the real-world dual-arm UR3 code, including data collection, training, and closed-loop inference scripts. Other RMBench policys like pi0.5 and Mem-0 also be included for convenience. For the environment setup and configuration of other RMBench policies, please refer to the official RMBench repository: https://github.com/robotwin-Platform/rmbench.

## News

- **2026-07-10**: RMBench-based Chronos policy ckpt and pth released.
- **2026-07-07**: Real-world dual-arm UR3 Chronos code released, including data collection, training, and closed-loop inference.
- **2026-06-30**: RMBench-based Chronos policy code released.
- **2026-06-29**: Chronos paper released on arXiv.
---

## Repository Structure
```text
Chronos/
├── README.md
├── LICENSE
├── RMBench/
    ├── assets/
    ├── data/
    ├── description/
    ├── envs/
    ├── policy/
    │   └── Chronos/
    │       ├── deploy_policy.py
    │       ├── deploy_policy.yml
    │       ├── eval.sh
    │       ├── mamba_policy_par_3D_IMLE.py
    │       ├── mamba_controller.py
    │       ├── M_dataset_robotwin3D_E.py
    │       ├── train_par_3D_IMLE_EE.py
    │       └── checkpoints/
    ├── script/
    ├── task_config/
    └── collect_data.sh
├──real_wolrd/ 
```

The repository is organized as one unified Chronos codebase. The current release includes both the RMBench simulation implementation and the real-world dual-arm implementation.

---
## Official RMBench Checkpoint

We provide official RMBench Chronos checkpoints and matching normalization files on Hugging Face:

[Chronos-RMBench Checkpoints](https://huggingface.co/yulinzhouZYL/Chronos-RMBench)

This checkpoint uses 16D dual-arm EE pose + gripper actions and should be evaluated with:

```python
TASK_ENV.take_action(action, action_type="ee")
```
This is not a qpos replay checkpoint.
## Code Release Status

### Released

- RMBench-based Chronos policy implementation.
- RMBench data collection, scaler fitting, training, and evaluation workflow.
- Real-world dual-arm UR3 image-policy code.
- Real-world data collection scripts.
- Real-world Chronos training scripts.
- Real-world Chronos closed-loop inference scripts.
- Real-world dataset loader and scaler fitting utility.
- Pose10d-style action preprocessing and validation utilities.
- RealSense, Kinect, RTDE, SpaceMouse, gripper, and TCP/EE helper modules.
- Shared-memory utilities for real-world hardware subprocesses.

### Coming Soon

- Cleaned ALOHA benchmark code.
- Cleaned RoboTwin2.0 benchmark code.
This repository will continue to grow. The current release focuses on making the main RMBench and real-world Chronos pipelines available first.

# Installation

## Base Environment

Create a conda environment:

```bash
conda create -n Chronos python=3.10 -y
conda activate Chronos
```

The RMBench and real-world code have different dependency requirements. Please install dependencies according to the experiment you want to run.

---

# RMBench Simulation Experiments

The `RMBench/` folder contains the RMBench environment and the Chronos policy implementation. The Chronos policy is located under:

```text
RMBench/policy/Chronos
```

This release focuses on Chronos. For other RMBench baseline policies and their environment configurations, please refer to the official RMBench repository:

```text
https://github.com/RoboTwin-Platform/RMBench
```

## Install RMBench Dependencies

Enter the RMBench folder:

```bash
cd RMBench
```

Install basic conda dependencies and CuRobo:

```bash
bash script/_install.sh
```

Download assets:

```bash
bash script/_download_assets.sh
```

Download data if needed:

```bash
bash script/_download_data.sh
```

If you encounter Hugging Face rate limits, log in first:

```bash
huggingface-cli login
```

## Collect RMBench Data

For example, collect demonstrations for `cover_blocks`:

```bash
bash collect_data.sh cover_blocks demo_clean 0
```

After data collection, create a dataset folder with train/test splits:

```text
/path/to/cover_blocks/
  train/
  test/
```

For example, put 50 trajectories into:

```text
/path/to/cover_blocks/train/
```

and 5 trajectories into:

```text
/path/to/cover_blocks/test/
```

## Fit RMBench Normalization Statistics

The current scaler script uses internal configuration fields instead of command-line arguments. Please edit the following fields in `RMBench/policy/Chronos/M_dataset_robotwin3D_E.py`:

```python
TASK_NAME = "cover_blocks"
SSD_ROOT = "/path/to/your/data/root"
```

The expected training data path is:

```text
${SSD_ROOT}/cover_blocks/demo_clean/data/train/
```

Then run:

```bash
cd RMBench/policy/Chronos
python M_dataset_robotwin3D_E.py
```

This generates:

```text
scaler_cover_blocks_ee_3d.pth
```

Please keep it under:

```text
RMBench/policy/Chronos/scaler_cover_blocks_ee_3d.pth
```
## Train Chronos on RMBench

The current training script uses internal configuration fields instead of command-line arguments. Please edit the following fields in `RMBench/policy/Chronos/train_par_3D_IMLE_EE.py`:

```python
TASK_NAME = "cover_blocks"
SSD_ROOT = "/path/to/your/data/root"
CODE_ROOT = "/path/to/your/Chronos/RMBench"
SCALER_FILENAME = f"scaler_{TASK_NAME}_ee_3d.pth"
```

Before training, make sure the scaler exists at:

```text
RMBench/policy/Chronos/scaler_cover_blocks_ee_3d.pth
```

Then run:

```bash
cd RMBench/policy/Chronos
python train_par_3D_IMLE_EE.py
```

Checkpoints will be saved to:

```text
RMBench/policy/Chronos/checkpoints/cover_blocks/EE_16/
```

The expected evaluation checkpoint is:

```text
RMBench/policy/Chronos/checkpoints/cover_blocks/EE_16/last.ckpt
```

## Evaluate Chronos on RMBench

Before evaluation, make sure the checkpoint and scaler are placed as:

```text
RMBench/policy/Chronos/
├── scaler_cover_blocks_ee_3d.pth
├── deploy_policy.yml
└── checkpoints/
    └── cover_blocks/
        └── EE_16/
            └── last.ckpt
```

In `RMBench/policy/Chronos/deploy_policy.yml`, set:

```yaml
ckpt_path: "policy/Chronos/checkpoints/cover_blocks/EE_16/last.ckpt"
scaler_path: "policy/Chronos/scaler_cover_blocks_ee_3d.pth"
temporal_agg: true
gpu_id: 0
```

Then run:

```bash
cd RMBench/policy/Chronos
bash eval.sh cover_blocks demo_clean Chronos 42 0
```

Chronos is evaluated with EE-pose control:

```python
TASK_ENV.take_action(action, action_type="ee")
```

Please make sure the checkpoint and `scaler_cover_blocks_ee_3d.pth` come from the same training run.
# Real-World Dual-Arm Experiments

The `real_wolrd/` folder contains the real-world data collection, training, and inference code for the dual-arm UR3 Chronos image policy.

```text
real_wolrd/
  common/                    # Shared model, dataset, hardware, pose, and shared-memory utilities
  data_collection/           # Real robot data collection and closed-loop robot scripts
  training/                  # Lightning training entry point
  inference/                 # Chronos inference wrapper
  requirements.txt           # Python dependencies
```

## Real-World Important Files

### Data Collection

```text
data_collection/z_data_collect_chronos.py
```

Real-world data collection entry. It reads D435, Kinect V2, two UR3 arms, SpaceMouse, and the dual gripper, then saves trajectories to the training dataset format.

### Closed-Loop Robot Execution

```text
data_collection/z_chronos.py
```

Real-world closed-loop execution entry. It loads a trained Chronos checkpoint and runs manual or policy control on the robot.

### Training

```text
training/train_par_3D_IMLE_UR3.py
```

Main training script for the image + low-dimensional dual-arm policy.

### Inference Wrapper

```text
inference/inference_choronos.py
```

`MyInferenceModel` wrapper. It loads checkpoint/scaler, preprocesses image and pose observations, and returns 20D pose10d-style actions.

### Policy Network

```text
common/mamba_policy_par_2D_IMLE.py
```

Chronos/Mamba policy network.

### Dataset and Scaler

```text
common/M_dataset_real_UR3.py
common/scaler_M.py
```

Dataset loader, scaler fitting utility, and low-dimensional observation/action normalizer.

### Metrics

```text
common/metric_UR10D.py
```

Pose10d validation metrics.

### Hardware Helpers

```text
common/z_*.py
```

RealSense, Kinect, RTDE, SpaceMouse, gripper, and TCP/EE transform helpers.

### Shared Memory

```text
common/shared_memory_dp/
```

Shared-memory queues, ring buffers, and arrays used by hardware subprocesses.

## Install Real-World Dependencies

Enter the real-world code folder:

```bash
cd real_wolrd
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Some hardware dependencies require system drivers or vendor SDKs in addition to Python packages:

- Intel RealSense SDK
- Kinect V2 / libfreenect2
- UR RTDE
- SpaceMouse / spacenav
- Dynamixel SDK

Please verify all hardware devices and emergency stop mechanisms before running closed-loop control.

## Real-World Dataset Format

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

The training script expects train/test split folders:

```text
/path/to/cover_blocks/
  train/<trajectory folders>
  test/<trajectory folders>
```

## Collect Real-World Data

Run:

```bash
python data_collection/z_data_collect_chronos.py \
  --robot_ip 192.168.4.63 \
  --robot_ip2 192.168.4.64 \
  --d435-serial <D435_SERIAL> \
  --kinect-serial <KINECT_SERIAL> \
  --output-root ./datasets/recordings
```

Useful keys:

```text
space      start/stop automatic recording
p          manually append the current frame
s          save the current recording buffer
backspace  clear the current buffer
w          switch active arm
e          toggle gripper
0/1/2/3    move to preset poses
q          quit
```

After collection, move or copy saved trajectory folders into:

```text
/path/to/cover_blocks/train/
```

and:

```text
/path/to/cover_blocks/test/
```

## Fit Real-World Scaler

Run:

```bash
python common/M_dataset_real_UR3.py \
  --task-name cover_blocks \
  --data-root /path/to/cover_blocks/train \
  --output-dir ./scalers
```

This writes:

```text
scalers/scaler_cover_blocks_image_pose10d.pth
```

## Train Real-World Chronos Policy

Run:

```bash
python training/train_par_3D_IMLE_UR3.py \
  --task-name cover_blocks \
  --data-root /path/to/cover_blocks \
  --output-root . \
  --scaler-path ./scalers/scaler_cover_blocks_image_pose10d.pth \
  --devices 0
```

Checkpoints are saved under:

```text
checkpoints/<task-name>/S3B_IMAGE_20D_2/
```

## Run Real-World Policy

Run:

```bash
python data_collection/z_chronos.py \
  --robot_ip 192.168.4.63 \
  --robot_ip2 192.168.4.64 \
  --ckpt-path ./checkpoints/cover_blocks/S3B_IMAGE_20D_2/last.ckpt \
  --scaler-path ./scalers/scaler_cover_blocks_image_pose10d.pth \
  --device cuda:0
```

Useful keys:

```text
i              enter policy mode
z              leave policy mode
w              switch active arm in manual mode
e              toggle gripper
0/1/2/3/4/5/6  move to preset poses
q              quit
```

You can also set paths with environment variables:

```bash
export CHRONOS_CKPT_PATH=/path/to/last.ckpt
export CHRONOS_SCALER_PATH=/path/to/scaler.pth
export CHRONOS_RECORD_ROOT=/path/to/new/recordings
```

---

# Recommended Workflows

## RMBench Workflow

```text
RMBench data collection
→ train/test split
→ scaler fitting
→ Chronos training
→ RMBench evaluation
```

## Real-World Workflow

```text
hardware setup
→ real-world data collection
→ train/test split
→ scaler fitting
→ Chronos training
→ checkpoint validation
→ closed-loop robot execution
```

---

# Real-Robot Safety Notes

Before running real-world closed-loop policy execution:

1. Verify robot IP addresses and RTDE connections.
2. Check camera calibration and image stream stability.
3. Confirm gripper direction and command range.
4. Start with low speed and conservative workspace limits.
5. Keep an emergency stop available.
6. Test policy behavior in manual or replay mode before enabling autonomous execution.
7. Inspect predicted poses and gripper commands before long-horizon deployment.

The real-world code is released for research use. Please use it carefully on physical robots.

---

# Troubleshooting

## Video or Image Input Does Not Load

Check camera serial numbers, driver installation, and permissions. For RealSense devices, verify that the Intel RealSense SDK is installed and that the camera can be opened by the official viewer.

## Robot Does Not Move

Check robot IP addresses, RTDE connectivity, robot mode, protective stop state, and workspace/speed limits.

## Policy Output Looks Unstable

Verify that the scaler path matches the training dataset, the observation format matches the training format, and the checkpoint corresponds to the same task.

## Dataset Cannot Be Loaded

Check that each trajectory contains the required `.npy` files and image frames, and that the train/test split follows the expected folder structure.

---

# Citation

If you find Chronos useful, please cite:

```bibtex
@article{zhou2026chronos,
  title={Chronos: A Physics-Informed Full-History Framework for Non-Markovian Long-Horizon Manipulation},
  author={Zhou, Yulin and Wang, Yimeng and Wang, Nengyu and Xing, Shaojia and Tu, Shiyun and Li, Xiang and Zhang, Jingkai and Jiang, Ningbo and Lin, Yuankai and Yang, Hua and Zeng, Xiangrui and Yin, Zhouping},
  journal={arXiv preprint arXiv:2606.30318},
  year={2026}
}
```

---

## Acknowledgement

This repository builds on the RMBench and RoboTwin 2.0 simulation ecosystem. We thank the authors of RMBench and RoboTwin for providing open-source robotic manipulation environments.

For the setup and configuration of other RMBench policies, please refer to the official RMBench repository:

```text
https://github.com/robotwin-Platform/rmbench
```

## License

This repository includes code adapted for RMBench-based Chronos experiments. RMBench is released under the MIT License. Please follow the original RMBench license terms when using or redistributing benchmark components.

Chronos policy code is released for research use. A formal license statement will be updated in a later release.
