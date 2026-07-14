# Chronos_RGB V2 for RMBench

This is the single-head RGB version of Chronos for the RMBench/RoboTwin
`cover_blocks` simulation:

- observation: `head_camera` RGB plus dual-arm 16-D EE state;
- output: 16 future dual-arm 16-D EE targets;
- execution: `TASK_ENV.take_action(action, action_type="ee")`;
- default image size: native RMBench 240x320;
- backbone: ImageNet ResNet18.

Data, checkpoints, scalers, evaluation videos, and TensorBoard runs are runtime
artifacts and must not be committed to Git.

## V2 contract

V1 built its target sequence as `[state[t], ..., state[t+15]]`.  Its first
target merely copied the current observation, while the simulator needed the
next EE target.  V2 explicitly trains:

```text
observation[t] -> [state[t+1], state[t+2], ..., state[t+16]]
```

Indices beyond the episode are clamped to its final state.  Scaler fitting uses
the same shifted targets.  The split manifest records `action_target_offset: 1`.
At deployment, a V2 checkpoint therefore uses `execution_horizon_offset: 0`.
The controller retains the offset option only to diagnose a legacy V1
checkpoint with offset `1` and explicit `visual_architecture=v1`; do not use
both the V2 target shift and deployment offset `1`.

V2 also preserves the ResNet output's 8x10 spatial grid.  The second adapter
convolution is stride 1 with 64 channels, followed by an 8x10 pool and a
256-to-512 projection.  Its total adapter parameter count remains within 2% of
V1, so the comparison does not simply add a much larger head.

## Training defaults

The formal defaults are a 50-epoch FP32 fine-tune:

- batch size 1, full episode history, full timestep supervision;
- visual adapter LR `1e-4`;
- ResNet layer4 LR `1e-5`;
- remaining Chronos/fusion LR `3e-5`;
- ResNet stem through layer3 frozen;
- every ResNet BatchNorm fixed in eval mode with frozen parameters;
- EMA enabled and deterministic 45/5 episode split with seed 42;
- best-five, last, and every-10-epoch resumable checkpoints.

From the repository root:

```bash
conda run -n RoboTwin python RMBench/policy/Chronos_RGB/train_par_2D_IMLE_EE.py \
  --data-root /path/to/cover_blocks/demo_clean/data \
  --task-name cover_blocks \
  --batch-size 1 \
  --vision-chunk-size 32 \
  --precision 32-true \
  --epochs 50
```

`--vision-chunk-size 32` is conservative; use `64` when GPU memory permits.
Changing the chunk does not shorten Mamba history.  A positive
`--supervision-frames` samples expensive loss timesteps while retaining full
history; the formal default `0` supervises every valid frame.

The V2 artifact defaults are deliberately separate from V1:

```text
policy/Chronos_RGB/checkpoints/cover_blocks/EE_16_v2/last.ckpt
policy/Chronos_RGB/scaler_cover_blocks_ee_rgb_v2.pth
```

`--resume auto` resumes `last.ckpt` only when it exists in the V2 output
directory.  Use `--resume none` for an intentional fresh run.  Exact resume
restores optimizer, scheduler, EMA, and epoch state.

## Warm-start from V1 epoch 60

V1 and V2 share ResNet18, Mamba, proprioception, and action heads.  The V2
trainer can initialize all shape-compatible policy tensors from a V1 policy or
Lightning checkpoint:

```bash
conda run -n RoboTwin python RMBench/policy/Chronos_RGB/train_par_2D_IMLE_EE.py \
  --data-root /path/to/cover_blocks/demo_clean/data \
  --output-dir /path/to/new_v2_run/EE_16_v2 \
  --scaler-path /path/to/new_v2_run/scaler_cover_blocks_ee_rgb_v2.pth \
  --warm-start /path/to/v1_epoch60.ckpt \
  --resume none \
  --refit-scaler \
  --epochs 50 \
  --vision-chunk-size 32 \
  --precision 32-true
```

Warm-start prefers `ema_policy_state_dict` when present.  It loads every
same-shaped tensor and prints each changed/new V2 visual tensor left randomly
initialized.  Missing or mismatched tensors outside the V2 visual adapter are
a hard error.  `--warm-start` starts a new optimizer run and cannot be combined
with a resolved `--resume` checkpoint.  Refit/use the V2 scaler because its
action targets use the new offset.

## Vast

`vast/train.sh` uses the same 50-epoch FP32 V2 defaults and accepts additional
CLI arguments.  Override its paths with:

```text
CHRONOS_REPO_ROOT
CHRONOS_VENV_ROOT
CHRONOS_DATA_ROOT
CHRONOS_RUN_ROOT
CHRONOS_V1_WARM_START
```

When `CHRONOS_V1_WARM_START` is non-empty and no V2 `last.ckpt` exists, the
script performs the one-time V1 warm-start and fits the V2 scaler.  Supervisor
restarts resume the V2 `last.ckpt` instead of starting over.  Checkpoints and
scalers live below the external run root, not in Git.
`vast/chronos_rgb.conf.example` can run the script under Supervisor.

## Evaluation

From `RMBench/`, a V2 one-seed smoke test is:

```bash
conda activate RoboTwin
bash policy/Chronos_RGB/eval.sh cover_blocks demo_clean rgb_v2_head_ee16 42 0 1
```

Arguments seven and eight override checkpoint and scaler.  Argument nine is
the diagnostic execution offset and argument ten is the visual architecture.
They must remain `0` and `v2` for V2:

```bash
bash policy/Chronos_RGB/eval.sh cover_blocks demo_clean my_v2 42 0 5 \
  policy/Chronos_RGB/checkpoints/my_v2/last.ckpt \
  policy/Chronos_RGB/scalers/my_v2.pth \
  0 \
  v2
```

Deployment never guesses architecture from a checkpoint: it constructs V2 by
default and strictly loads the complete state.  Passing a V1 checkpoint to
that default fails on the changed visual tensors.  The explicit legacy
diagnostic requires both final arguments `1 v1`; ordinary V2 remains `0 v2`.

```bash
bash policy/Chronos_RGB/eval.sh cover_blocks demo_clean legacy_v1 42 0 1 \
  /path/to/v1_epoch60.pth /path/to/v1_scaler.pth 1 v1
```

`reset_model()` clears Mamba, latent-noise, and temporal-ensemble state at every
episode.  Temporal voting uses a bounded `16 x 16 x 16` ring and averages
denormalized physical EE targets.
