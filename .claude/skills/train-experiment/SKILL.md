---
name: train-experiment
description: Train (or resume) the ensemble members of one or more experiments with `uq train`, respecting the three-GPU limit and parent dependencies, and confirm completeness with `uq status`. Use when asked to train, retrain missing members, or add splits/members to an experiment.
---

# Train an experiment

1. `uq show <exp>` and `uq status <exp> -v`: know what exists; never retrain complete
   members unless the user explicitly asks (they are provenance for recorded results).
2. Pick GPUs: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`.
   **At most three GPUs in total**, counting every job already running for this project.
3. Dry run: `uq train <exp> [<exp> ...] --gpus G1 [G2] [--splits 0-4] [--members 0-9] --dry-run`.
   Check the member list and that `lf-to-hf` members wait for their parent member.
4. Launch detached so it survives the session, e.g.
   `nohup uq train <exp> --gpus G1 G2 --jobs-per-gpu N > runs/_logs/train_<name>.log 2>&1 &`
   (small models such as water fit 3 jobs per GPU; ANI one or two). Per-member logs are
   in `runs/_logs/<id>__..._s<s>_m<m>.log`; W&B logs to `runs/wandb` (needs W&B login).
5. Monitor with a background wait on the scheduler PID or its log ("finished; N failed"),
   not with repeated sleeps. On failure, read the member log, fix, and rerun the same
   command (complete members are skipped).
6. `uq status <exp>` must show 10/10 for every split before evaluating
   (`evaluate-experiment` skill). Report which members were trained, on which GPUs, and
   the commit.
