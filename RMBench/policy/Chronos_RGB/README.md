# Chronos RGB for RMBench

This folder trains Chronos on RMBench `cover_blocks` demonstrations using one
head-camera RGB stream instead of the released point cloud.

The comparison has an explicit two-part contract:

- RMBench stays authoritative for the 16-D dual-arm EE state/action layout,
  16-step action horizon, Mamba/action heads, loss and 600-epoch training setup.
- `real_wolrd` stays authoritative for the RGB preprocessing and visual
  encoder.  This is the same image front end used by the released real-robot
  code, with the proprioceptive input adapted from real-world 20D to RMBench
  16D.

Data, scalers, checkpoints, logs and evaluation videos are runtime artifacts
and must not be committed to Git.

## Exact model and tensor contract

One training sample is a complete episode:

```text
head RGB:       [L, 3, 480, 640], uint8 on CPU
EE state:       [L, 16]
action target:  [L, 16 future steps, 16]
padding mask:   [L]
```

The 16-D EE order is:

```text
left  [x, y, z, qw, qx, qy, qz, gripper]
right [x, y, z, qw, qx, qy, qz, gripper]
```

Targets match the released RMBench dataset code:

```text
observation[t] -> [state[t], state[t+1], ..., state[t+15]]
```

Indices beyond the episode are clamped to its final state.  The controller
therefore executes horizon offset 0.  This explicit array offset is different
from the real-world recorder: its array also starts at index `t`, but the
recorded target command is scheduled one control cycle into the future.

Low-dimensional state and actions use the official per-key z-score scaler.
Each action horizon has its own mean/std, `torch.std(dim=0)` uses its default
sample standard deviation, and the clamp epsilon is `1e-8`.  Statistics are fit
only on the training episodes.

RMBench writes simulator RGB directly through OpenCV JPEG encoding.  OpenCV
decode consequently restores the original numeric RGB channels, so the loader
must not apply another BGR/RGB swap.  Frames are resized to 640x480 with
`INTER_AREA`, converted to `[0,1]`, and normalized with ImageNet values:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The visual encoder exactly follows `real_wolrd/common/mamba_policy_par_2D_IMLE.py`:

```text
RGB [3,480,640]
-> ImageNet ResNet18 children()[:-2]
-> replace every BatchNorm2d with GroupNorm(32)
-> freeze the complete trunk
-> [512,15,20]
-> Conv 512->256, GN32, SiLU
-> Conv 256->128 stride 2, GN16, SiLU
-> [128,8,10] -> flatten -> Linear 10240->1024
-> LayerNorm, SiLU, Dropout(0.1)
```

The 16-D EE state is projected `16->128->512`.  Visual 1024 and EE 512 are
concatenated, then projected `1536->1024` before the released RMBench Chronos
temporal/action model.

## Official training settings

The formal run fixes the released RMBench settings:

- seed 42;
- 600 epochs;
- batch size 2 and gradient accumulation 3;
- full episode history and full padded per-frame loss;
- AdamW, one learning rate `1.7e-4`, weight decay `1e-4`;
- 15-epoch linear warm-up from 1% LR;
- cosine schedule with `eta_min=2e-5`;
- FP32 and gradient clipping 1.0;
- frozen ResNet18 trunk, trainable visual adapter/fusion and Chronos model;
- warm-up EMA with maximum decay 0.9999;
- best five, last, and every-100-epoch resumable checkpoints.

The real-world script itself defaults to 200 epochs and `eta_min=3e-5`; those
two settings are not used because this is an RMBench experiment.  Both official
paths otherwise agree on batch 2, accumulation 3, LR `1.7e-4`, weight decay,
warm-up and FP32.

The official repository README recommends 50 training plus 5 test episodes.
The current supplied dataset has only 50 flat episodes, so the trainer records
and uses a deterministic seed-42 45/5 episode holdout.  This is a data
availability difference, not a hidden hyperparameter change.  If explicit
`train/` and `test/` directories are later supplied, they take priority.

From the repository root:

```bash
conda run -n RoboTwin python RMBench/policy/Chronos_RGB/train_par_2D_IMLE_EE.py \
  --data-root /path/to/cover_blocks/demo_clean/data \
  --output-dir /path/to/run/EE_16_official_rgb \
  --scaler-path /path/to/run/scaler_cover_blocks_ee_official_rgb.pth \
  --expected-episodes 50 \
  --resume none
```

The command refuses changes to the defining official batch, accumulation,
resolution, label offset and 600-epoch contract.  `--vision-chunk-size` only
controls bounded RGB transfer/feature extraction; it does not shorten episode
history or reduce supervised timesteps.

## Vast training

`vast/train.sh` launches the same contract and resumes only a matching
`last.ckpt`.  It intentionally has no warm-start path.  Runtime paths can be
set with:

```text
CHRONOS_REPO_ROOT
CHRONOS_VENV_ROOT
CHRONOS_DATA_ROOT
CHRONOS_RUN_ROOT
```

The default remote artifacts are:

```text
/workspace/chronos_rgb_runs/cover_blocks/EE_16_official_rgb/
/workspace/chronos_rgb_runs/cover_blocks/scaler_cover_blocks_ee_official_rgb.pth
```

`run_manifest.json` records the exact split, tensor contract and training
parameters.  `--resume auto` restores model, raw optimizer state, scheduler and
EMA state from `last.ckpt`; a fresh run must use `--resume none` and a new/empty
artifact directory.

## Evaluation

From `RMBench/`:

```bash
bash policy/Chronos_RGB/eval.sh cover_blocks demo_clean official_rgb_head_ee16 42 0 5 \
  policy/Chronos_RGB/checkpoints/cover_blocks/EE_16_official_rgb/last.ckpt \
  policy/Chronos_RGB/scaler_cover_blocks_ee_official_rgb.pth
```

Deployment resizes the live head-camera RGB to 480x640, applies the same
normalization inside the shared fusion module, strictly loads every policy
tensor, and emits a 16-D EE action through
`TASK_ENV.take_action(action, action_type="ee")`.  `reset_model()` clears Mamba,
latent-noise and bounded temporal-ensemble state at every episode.  As in the
released RMBench point-cloud and real-world controllers, temporal ensembling is
performed in normalized action space and the selected current action is then
denormalized with the horizon-zero statistics.
