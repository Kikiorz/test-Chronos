# Chronos RGB + Joint for RMBench

This policy tests whether Chronos can learn RMBench from one RGB camera without
point clouds. It deliberately combines two released code paths:

- RGB modality: the official `real_wolrd` ResNet-18 encoder and preprocessing;
- benchmark dynamics: the official RMBench Chronos Mamba/IMLE/symplectic core;
- control: RMBench's native dual-arm 14-D joint drive targets.

It is therefore an RGB modality experiment on the original RMBench data, not a
claim that the released RMBench result used RGB. The released RMBench policy
uses point clouds; the released real-world policy uses RGB.

## Exact RGB contract

Every HDF5 frame under `observation/head_camera/rgb` is decoded as RGB uint8 at
`240x320`. The RMBench writer passed its RGB array directly through OpenCV JPEG
encoding, so decode preserves those numeric channel positions; this loader must
not add a BGR-to-RGB swap.

The shared training/deployment preprocessing exactly follows the released
real-world image path after channel acquisition:

```text
OpenCV INTER_AREA resize: (width,height) = (640,480)
float32 RGB = uint8 / 255
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
output = CHW float32 [3,480,640]
```

The encoder is torchvision `resnet18(ResNet18_Weights.DEFAULT)`. Its avgpool
and classifier are removed; all 20 `BatchNorm2d` modules are replaced by fresh
`GroupNorm(32,C)` modules exactly as in the official code, and the whole trunk
is frozen. The official weights file is kept outside Git and must have SHA-256:

```text
f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
```

For a `480x640` image the trunk output is `[512,15,20]`. The trainable official
adapter is:

```text
Conv 512->256, GN32, SiLU
Conv 256->128 stride 2, GN16, SiLU
Flatten 128*8*10 -> Linear 1024 -> LayerNorm -> SiLU -> Dropout(0.10)
```

The 14-D proprioception projector is `14->128->512`; concatenated visual and
proprio features are projected `1536->1024` before the released Mamba core.

## Joint/action contract

The original 50 `cover_blocks/demo_clean` HDF5 episodes are used directly. The
fixed 14-D order is:

```text
[fl_joint1..6, left_gripper_normalized,
 fr_joint1..6, right_gripper_normalized]
```

The loader cross-checks `joint_action/vector` against the four arm/gripper
datasets in every episode. Targets are 16 future joint drive targets:

```text
target[t,h] = joint[min(t + 1 + h, episode_length - 1)], h=0..15
```

`t+1` is intentional for RMBench joint control: `joint_action[t]` is the
current drive target, so offset zero would teach the first command to stay at
the current target. This differs from the indexing text in the real-world
end-effector loader because the stored action semantics differ.

## Normalization

The scaler is fitted only on the 45 training episodes selected with split seed
42; the five validation episodes never contribute statistics.

- Current state: independent statistics for all 14 dimensions.
- Action: independent statistics for each of 16 horizons and 14 dimensions.
- Standard deviation: official `torch.std(..., unbiased=True)` sample std.
- Epsilon: official `1e-8` lower bound.
- NaN/Inf, shape, dtype and fingerprint checks are additional guards.
- All 16 horizons are denormalized before temporal aggregation at deployment.

The run recomputes the scaler and dataset SHA manifest before training. Resume
is rejected if the backbone, data, split, scaler or policy contract differs.

## Formal training recipe

The RGB-specific path above follows official real-world code. Because the data
and evaluator are RMBench, the temporal objective and training horizon follow
the released RMBench policy:

```text
seed / split seed / validation seed = 42
episodes = 45 train + 5 held-out validation
batch size = 2
gradient accumulation = 3 (effective 6 episodes/update)
complete episode history; all valid frames supervised
future steps = 16
IMLE samples = 5
Mamba embed/d_model = 1024, blocks = 6
epochs = 600
AdamW lr = 1.7e-4, weight_decay = 1e-4
warmup = 15 epochs, cosine eta_min = 2e-5
precision = FP32 (official PyTorch defaults: matmul TF32 off, cuDNN TF32 on)
gradient clip = 1.0
frozen-image chunk size = 128 frames on the local 24 GB GPU
EMA power = 2/3, max decay = 0.9999
inference = 5-step semi-implicit Euler
```

Start a guaranteed fresh run from repository root:

```bash
bash RMBench/policy/Chronos_RGB_Joint/local/train.sh
```

Defaults on this workstation:

```text
data: /home/zeno-rp/2026test/rmbench_rgb_dataset/data/cover_blocks/demo_clean/data
run:  /home/zeno-rp/2026test/chronos_rgb_joint_runs/cover_blocks
weights: ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
environment: RoboTwin
```

`local/train.sh` passes `--resume none`, so it always starts at epoch 0. To
resume intentionally, append `--resume /absolute/path/to/last.ckpt`.

The released real-world default is a 256-frame image chunk. The local wrapper
uses 128 because gradients remain resident across the official three-batch
accumulation window on a 24 GB GPU. Chunking only partitions the frozen
ResNet/adapter computation; batch size, loss, optimizer step, effective batch
and every supervised frame remain unchanged. On a larger GPU, pass
`--vision-chunk-size 256` to use the released throughput setting.

Training performs held-out offline validation to select the best EMA weights.
It does not run RMBench closed-loop task evaluation. Closed-loop evaluation is
deferred until training has finished.

The output directory keeps one atomic `best-ema-deploy.pth`, two full top-k
checkpoints, and a rotating resume checkpoint every five epochs. Data, model
weights, scalers, checkpoints, logs and TensorBoard events are ignored by Git.

## Evaluation after training

From `RMBench/`, use the best EMA artifact:

```bash
bash policy/Chronos_RGB_Joint/eval.sh cover_blocks demo_clean_rgb_joint \
  rgb_head_joint14_resnet18 42 0 5 \
  /external/run/Joint_14/best-ema-deploy.pth \
  /external/run/scaler_cover_blocks_joint_rgb.pth
```

After the five-episode smoke test succeeds, increase the episode count and
report multiple seeds. Checkpoint loading is strict; DINO, point-cloud and EE
checkpoints are incompatible.

## Real-robot warning

This checkpoint is bound to RMBench's Aloha dual-arm 14-D drive-target order.
A Piper controller is not compatible merely because it also exposes joints.
Real deployment needs a Piper-specific state/action mapping, limits, control
rate, data collection and retraining, plus collision checking, watchdog and
emergency-stop layers.
