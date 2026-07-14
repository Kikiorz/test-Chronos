# Chronos_RGB_Joint for RMBench

This is a separate RMBench policy for:

- visual observation: one `head_camera` RGB stream;
- visual backbone: frozen DINOv3 ViT-B/16;
- proprioception: native dual-arm 14-D joint drive-target vector;
- prediction: 16 future 14-D absolute joint targets;
- execution: `TASK_ENV.take_action(action, action_type="qpos")`.

It does not modify or reuse the scaler/checkpoint of `Chronos_RGB`, which is
the 16-D end-effector variant.

## Exact RMBench joint contract

The data comes from the original 50 `cover_blocks/demo_clean` HDF5 episodes.
No converted or third-party dataset is used.  Every file contains:

```text
observation/head_camera/rgb
joint_action/left_arm       # 6 values
joint_action/left_gripper   # 1 value
joint_action/right_arm      # 6 values
joint_action/right_gripper  # 1 value
joint_action/vector         # 14 values
```

The 14-D order is fixed:

```text
[fl_joint1..6, left_gripper_normalized,
 fr_joint1..6, right_gripper_normalized]
```

The dataset verifies every episode's `joint_action/vector` against the four
component arrays.  These values are RMBench joint `drive_target`s, not measured
physics qpos.  Online RMBench observations expose the same convention, so
simulation training and evaluation are consistent.  A real robot must choose
and document whether its state is measured qpos or commanded targets; they must
not be mixed silently.

Targets start at the next recorded drive target:

```text
target[t, h] = joint[min(t + 1 + h, episode_length - 1)]
h = 0..15
```

The released offset-0 dataset code copies the current drive target into the
first action.  For joint control, a perfect offset-0 policy therefore commands
the robot to stay where it is; temporal candidates also point to the current
target.  This RGB-Joint variant intentionally fixes that causal alignment while
keeping the 16-step horizon and all optimization parameters unchanged.

## Normalization

The scaler is fitted from the 45 training episodes only.  The five validation
episodes selected by split seed 42 never contribute statistics.

- Each of the 14 current-state dimensions has its own mean/std/min/max.
- Each action dimension has separate statistics for every one of the 16 future
  horizons (`16 x 14` statistics), rather than reusing current-state values.
- Horizon 0 is the next recorded drive target, so it has independently fitted
  action statistics rather than copied current-state statistics.
- Standard deviation uses population variance and is lower-bounded by `1e-6`.
- Fitting rejects NaN/Inf.
- Deployment denormalizes every predicted horizon before temporal aggregation.
- Final gripper targets (indices 6 and 13) are clipped to `[0,1]`.
- By default arm targets are bounded to the training range plus a 5% margin to
  reduce silent RMBench TOPP failures.  This can be disabled in the YAML.

At every training start, statistics are recomputed from the current 45 training
episodes and compared with an existing scaler before it may be reused.  A
canonical scaler SHA-256 is written into the checkpoint; deployment compares
that fingerprint with both the external scaler and the scaler state embedded in
the checkpoint.  A same-shaped scaler from another task or split is rejected.

Use only the new scaler:

```text
scaler_cover_blocks_joint_rgb.pth
```

An EE or 3D scaler fails strict loading and must not be renamed into this path.

## DINOv3 image path

The encoder is `vit_base_patch16_dinov3.lvd1689m` (ViT-B/16, 768-D tokens).
Native RMBench images are `240x320`.  The shared training/deployment encoder
resizes them to `336x448` with bicubic antialiased interpolation, matching the
timm pretrained interpolation family while preserving 4:3 and divisibility by
patch size 16, then applies exactly:

```text
RGB [0,1]
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

The larger tensor is an interpolation and does not create new camera detail.
True higher-resolution input requires recollecting demonstrations and matching
evaluation images.

One unavoidable source-domain gap remains: RMBench stores demonstration frames
as default OpenCV JPEG, while live SAPIEN evaluation supplies uncompressed
uint8 RGB.  The preprocessing is bit-exact for identical input pixels, but JPEG
has already changed some demonstration pixels.  We document this instead of
deliberately degrading live images with another codec pass; DINOv3 is expected
to tolerate the small compression noise.

One CLS token and a row-major adaptive `4x5` grid of patch features are kept,
giving `[21,768]` per frame.  This retains coarse spatial layout; it is not a
single global patch mean.

DINOv3 weights use Meta's gated DINOv3 license.  Accept the terms and keep the
authorized model outside Git.  The model page is:

```text
https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
```

The loader validates every backbone tensor strictly.  For a checkpoint that
already embeds a timm DINOv3 backbone, extract a standalone file with:

```bash
conda run --no-capture-output -n RoboTwin \
  python RMBench/policy/Chronos_RGB_Joint/extract_dinov3_backbone.py \
  --source /external/path/model.safetensors \
  --output /external/path/dinov3_vitb16_lvd1689m.safetensors
```

Neither source nor output weights are committed.

## Frozen-feature cache

The backbone is frozen.  Recomputing it for all 51,127 frames on every epoch
would change no feature while wasting most of the local training time.  Create
the exact external cache once:

```bash
conda run --no-capture-output -n RoboTwin \
  python RMBench/policy/Chronos_RGB_Joint/precompute_dinov3_features.py \
  --data-root /home/zeno-rp/2026test/rmbench_rgb_dataset/data/cover_blocks/demo_clean/data \
  --feature-root /home/zeno-rp/2026test/rmbench_rgb_dataset/features/cover_blocks_dinov3b_336x448_grid4x5 \
  --weights /home/zeno-rp/2026test/models/dinov3_vitb16_lvd1689m.safetensors \
  --expected-episodes 50 \
  --batch-images 1 \
  --amp-dtype none
```

`--amp-dtype none` computes DINOv3 in FP32 and cached values remain FP32
(roughly 3.2 GB).  Batch size 1 matches online deployment and avoids the tiny
GPU-kernel rounding difference observed between batch-16 cache extraction and
batch-1 control.  Cache extraction, training, and deployment all use strict
FP32 (`matmul_precision=highest`, CUDA matmul TF32 off, cuDNN TF32 off).
The formal cache is CUDA-only because CPU and CUDA uint8 division do not have
the same float32 bits for every pixel value.  Formal controller inference is
therefore CUDA-only as well.
Metadata records a versioned preprocessing fingerprint,
model SHA-256, camera, resolution, token layout, extraction batch size, and
SHA-256 for every source HDF5 and feature file.  Existing `.npy` files are
reused only when all these fields and checksums match.

## Local formal training

The supported local environment on this machine is `RoboTwin`.  The wrapper
uses external data/model/run directories and keeps the released training
recipe:

```bash
bash RMBench/policy/Chronos_RGB_Joint/local/train.sh
```

Formal parameters:

```text
seed / split seed / validation seed = 42
episodes = 45 train + 5 validation
batch size = 1
complete ~1000-frame episode history
all valid timesteps supervised
future steps = 16
epochs = 600
gradient accumulation = 3
learning rate = 1.7e-4
weight decay = 1e-4
warmup = 15 epochs
eta_min = 2e-5
precision = FP32
float32 matmul precision = highest; CUDA/cuDNN TF32 = disabled
gradient clip = 1.0
EMA = enabled, max decay 0.9999
deployment symplectic solver = 5-step semi-implicit Euler
```

Default outputs are external to Git:

```text
/home/zeno-rp/2026test/chronos_rgb_joint_runs/cover_blocks/
```

The demonstrations and evaluation config are specifically bound to RMBench's
`aloha-agilex` embodiment (AgileX Aloha dual ARX5), not UR5 or Piper.  A Piper
deployment needs a separate joint mapping, data collection, and retraining; a
matching vector length alone is not a valid robot contract.

`--resume auto` resumes `last.ckpt`.  A full checkpoint is about 5.4 GiB because
it contains raw policy, EMA policy, and Adam moments.  The default writes
one rotating resume checkpoint (with an atomic `last.ckpt` symlink) plus the
best two full checkpoints every five epochs, and disables extra periodic
copies. Checkpoint temporary files are staged in the short
`<output-parent>/.tmp` directory on the same filesystem rather than in a
potentially small `/tmp`. The short path also leaves room for Linux DataLoader
`AF_UNIX` worker sockets; an unsafe `--temp-dir` is rejected before training.
This changes checkpoint cadence only, not optimization, validation, or model
behavior.

Resume is preflighted before Lightning restores any tensor.  Backbone SHA,
cache preprocessing/data fingerprints, scaler fingerprint, joint contract, and
train/validation episode lists must all match.  This prevents an old frozen
backbone from being restored while training its adapter against a new cache.

On this machine, a measured full 1,005-frame forward/backward used about 5.4
GiB; adding AdamW state and full EMA peaked near 8.2 GiB.  A 24 GB RTX 4090 is
sufficient.  An initial step was roughly nine seconds, but the final strict
FP32 setting can be slower; use the first complete epochs for the reliable
600-epoch ETA.

For a short diagnostic only, append:

```bash
--overfit-batches 1 --epochs 3 --accumulate-grad-batches 1 \
--warmup-epochs 0 --checkpoint-every-n-epochs 1 --resume none --no-ema
```

Formal training must leave `--overfit-batches` at zero.

For evaluation, strip raw optimizer state and keep only the complete EMA policy,
contract, and scaler state (about 1.5 GiB):

```bash
conda run --no-capture-output -n RoboTwin \
  python RMBench/policy/Chronos_RGB_Joint/export_deploy_checkpoint.py \
  --input /external/run/Joint_14/last.ckpt \
  --output /external/run/Joint_14/last-ema-deploy.pth
```

The compact file is still a model artifact and must not be committed.

## Evaluation

From `RMBench/`:

```bash
conda activate RoboTwin
bash policy/Chronos_RGB_Joint/eval.sh cover_blocks demo_clean_rgb_joint \
  rgb_head_joint14_dinov3b 42 0 5 \
  /external/run/Joint_14/last.ckpt \
  /external/run/scaler_cover_blocks_joint_rgb.pth
```

After a five-episode smoke test, replace `5` with `100` for each formal seed and
report several seeds separately.
Checkpoint loading is strict and requires the policy contract
`chronos_rgb_joint14_dinov3b`; EE/point-cloud checkpoints fail immediately.

`demo_clean_rgb_joint` keeps the same robot, camera, scene, and domain settings
as `demo_clean`, but disables unused depth/point-cloud collection and evaluation
video.  The released `demo_clean` combines `third_view: false` with
`eval_video_log: true`, which otherwise raises a `third_view_rgb` KeyError on
the first action.

RMBench can silently swallow a TOPP exception, leaving an arm stationary while
the gripper still moves.  The deploy adapter logs a suspected failure when a
nontrivial requested arm target produces no drive-target change, and also reads
real physics qpos for diagnostics.  Closed-loop validation must still inspect
these warnings, actual state change, and task success—not merely the absence of
a Python exception.

The training-range clip is a simulation output guard, not a Piper/real-robot
safety layer.  It does not implement URDF limits, rate/acceleration bounds,
collision checking, watchdogs, or an emergency stop.

## Git hygiene

HDF5 data, DINOv3 weights, feature caches, scalers, checkpoints, TensorBoard
events, and logs are ignored.  Only source code and documentation belong on
GitHub.
