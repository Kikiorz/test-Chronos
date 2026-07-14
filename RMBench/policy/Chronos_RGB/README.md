# Chronos_RGB for RMBench

This policy is the controlled RGB version of RMBench Chronos:

- visual input: one `head_camera` RGB image;
- proprioception: dual-arm 16-D EE state (`xyz + quaternion + gripper` per arm);
- output: a 16-step sequence of 16-D EE targets;
- execution: `TASK_ENV.take_action(action, action_type="ee")`.

Only the visual modality changes.  Keeping the EE state, action representation,
task, demonstrations, and evaluation seeds aligned with the 3D baseline makes
the RGB-versus-point-cloud result interpretable.

## Important differences from the released checkpoint

The official `policy/Chronos` checkpoint was trained with colored point clouds
and is **not compatible** with this RGB network.  Train a new RGB checkpoint and
use the RGB scaler saved by that same training run.  Deployment loads every
policy tensor with `strict=True`; a 3D or otherwise mismatched checkpoint fails
immediately instead of silently running with random layers.

RoboTwin/SAPIEN supplies true RGB arrays.  `deploy_policy.py` intentionally does
not apply the OpenCV BGR-to-RGB channel swap used by the real-robot camera code.

## Expected artifacts

The default evaluation paths are:

```text
policy/Chronos_RGB/checkpoints/cover_blocks/EE_16/last.ckpt
policy/Chronos_RGB/scaler_cover_blocks_ee_rgb.pth
```

Both files are produced by RGB training; neither should be copied from the 3D
policy directory.

## Train

The downloaded RMBench demonstrations are external data and are intentionally
kept outside this Git repository.  On this machine the complete Cover Blocks
dataset is located at:

```text
/home/zeno-rp/2026test/rmbench_rgb_dataset/data/cover_blocks/demo_clean/data
```

From the repository root, start the default 600-epoch run with:

```bash
conda run -n RoboTwin python RMBench/policy/Chronos_RGB/train_par_2D_IMLE_EE.py \
  --data-root /home/zeno-rp/2026test/rmbench_rgb_dataset/data/cover_blocks/demo_clean/data \
  --task-name cover_blocks
```

The formal default keeps the complete roughly 1,000-frame episode for RGB
fusion and Mamba history.  `--supervision-frames 0` also applies the original
IMLE/symplectic loss at every valid timestep.  A positive value such as `64`
retains full history but samples only that many expensive loss timesteps if a
smaller GPU needs it.

The loader verifies that exactly 50 HDF5 episodes exist before fitting or
training (`--expected-episodes 50`).  With `--split-seed 42` it makes a
deterministic episode-level 45/5 train/validation split and writes the exact
filenames to
`RMBench/policy/Chronos_RGB/checkpoints/cover_blocks/EE_16/split_manifest.json`.
No frames
from a held-out episode enter the training split.

Training creates or loads:

```text
RMBench/policy/Chronos_RGB/scaler_cover_blocks_ee_rgb.pth
RMBench/policy/Chronos_RGB/checkpoints/cover_blocks/EE_16/last.ckpt
RMBench/policy/Chronos_RGB/checkpoints/cover_blocks/EE_16/mamba-best-*.ckpt
RMBench/policy/Chronos_RGB/checkpoints/cover_blocks_rgb/version_*/
```

`--resume auto` is the default: if `last.ckpt` exists, training resumes it;
otherwise it starts a fresh run.  Use `--resume none` only when an intentional
fresh run is wanted.  The scaler is fitted from all frames of the 45 training
episodes only.  Use `--refit-scaler` after deliberately changing the split or
dataset.

Warmup EMA is enabled by default to match the released 3D training recipe
(`--no-ema` disables it).  Checkpoints keep raw weights for exact optimizer
resume and a complete `ema_policy_state_dict` for strict deployment; evaluation
automatically prefers EMA.  Validation uses an isolated fixed RNG seed, so its
stochastic IMLE/bridge loss is repeatable and does not alter training RNG state.

The local smoke test used a real 1,005-frame episode, the frozen ImageNet
ResNet18, all timesteps supervised, and one Lightning optimizer plus validation
step.  It completed on the RTX 4090 D with a 6.044 GiB CUDA peak.  This checks
the data, full-history forward/backward, optimizer, and validation paths; it is
not a task-success result.

For a short one-episode optimization diagnostic, `--overfit-batches 1` makes
Lightning reuse one complete trajectory.  This flag is only for confirming
that loss can decrease and must remain `0` for the formal 45/5 run.

When moving the run to Vast or another machine, sync the external dataset and
the desired scaler/checkpoints separately (for example with `rsync`).  Do not
add the multi-gigabyte HDF5 data, checkpoints, or TensorBoard logs to Git.

For a Vast base-image instance, `vast/train.sh` contains the full-supervision
600-epoch command and configurable workspace defaults.  Long-running training
can be managed by copying `vast/chronos_rgb.conf.example` to
`/etc/supervisor/conf.d/chronos_rgb.conf`, then running `supervisorctl reread`,
`supervisorctl update`, and `supervisorctl start chronos_rgb`.  The instance
workspace must be checked for volume persistence; when it is not a mounted
volume, copy important checkpoints off the instance before recycling it.

## Evaluate

From `RMBench/`, after activating the RoboTwin environment:

```bash
conda activate RoboTwin
bash policy/Chronos_RGB/eval.sh cover_blocks demo_clean rgb_head_ee16 42 0 5
```

The final argument above evaluates five expert-solvable seeds as a smoke test.
After the pipeline is verified, use `100` for the formal fixed-seed result:

```bash
bash policy/Chronos_RGB/eval.sh cover_blocks demo_clean rgb_head_ee16 42 0 100
```

Custom checkpoint and scaler paths can be supplied as arguments seven and
eight:

```bash
bash policy/Chronos_RGB/eval.sh cover_blocks demo_clean my_run 42 0 5 \
  policy/Chronos_RGB/checkpoints/my_run/last.ckpt \
  policy/Chronos_RGB/scalers/my_run.pth
```

## Deployment safeguards

`reset_model()` reinitializes the Mamba cache and latent state at the start of
every episode.  Temporal aggregation uses a bounded
`future_steps x future_steps x action_dim` ring (16 x 16 x 16 by default), not
the original 5000-squared allocation.  Actions are denormalized before temporal
averaging so that predictions made at different future horizons are combined
in physical EE coordinates.
