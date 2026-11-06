# CINECA Leonardo: access, rules, our layout

Facts checked on 2026-11-06 against the CINECA user guide ([Leonardo][leo],
[filesystems][fs], [accounting and linearization][intro], [scheduler][slurm], [access][acc])
and, where the guide is silent, on the cluster itself (marked *observed*). Items marked
**verify** are not established yet.

[leo]: https://docs.hpc.cineca.it/hpc/leonardo.html
[fs]: https://docs.hpc.cineca.it/hpc/hpc_data_storage.html
[intro]: https://docs.hpc.cineca.it/hpc/hpc_intro.html
[slurm]: https://docs.hpc.cineca.it/hpc/hpc_scheduler.html
[acc]: https://docs.hpc.cineca.it/general/access.html

## 1. Account and access

- User `mtolloso`, registered email `matteo.tolloso@phd.unipi.it`, login host
  `login01-ext.leonardo.cineca.it` (login01-ext ... login07-ext exist).
- **Project account `IscrC_UQ-MF`** (2026-07-29 to 2027-04-29): 80,000 local h = ~10,000 GPU-h.
  Always pass `--account=IscrC_UQ-MF`. The user also has `IscrB_ProAmmo`, another project: never
  charge it, never touch its files (`$HOME/miniconda3` belongs to it).
- 2FA: SSH certificates from `step ssh login`, valid **12 h**. The browser step (password + OTP)
  is always the user's.

**From the MacBook (user):**
```bash
step ssh login 'matteo.tolloso@phd.unipi.it' --provisioner cineca-hpc
ssh mtolloso@login01-ext.leonardo.cineca.it
```
(A colleague's `ProxyCommand` that runs `step ssh login` before each connection is handy on a
laptop with a browser; not used on the DGX.)

**From the DGX, for agents (`ssh leonardo`).** All in user space: `~/.local/bin/step`
(Smallstep CLI v0.31.0, sha256 checked), CINECA CA in `~/.step/` (`step ca bootstrap
--ca-url=https://sshproxy.hpc.cineca.it --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd`),
an ssh-agent as systemd user unit `leonardo-ssh-agent` (socket `~/.ssh/agent-leonardo.sock`,
certificate only in memory), `Host leonardo` in `~/.ssh/config` (shared connection 4 h).
Renewal every 12 h, by the user, in a terminal **on the DGX**:
```bash
leonardo-login      # prints a CINECA URL: open it in the Mac browser within ~5 min
```
The browser is redirected to `http://127.0.0.1:10000` of the DGX, so that port must reach the
DGX: a VS Code Remote terminal usually forwards it (Ports panel); otherwise first, on the Mac,
`ssh -N -L 10000:127.0.0.1:10000 m.tolloso@<dgx address>` and only then run `leonardo-login`
(a port that nobody listens on gives "channel 2: open failed: connect failed"). Success ends with
`connected to <host> as mtolloso`. The terminal-only flow (`--console`) is disabled by CINECA.

## 2. Filesystems: what goes where

| Area | Quota | Kept | Backup | Use for |
|---|---|---|---|---|
| `$HOME` `/leonardo/home/userexternal/mtolloso` | 50 GB | while the user exists | daily, 3 versions | dot-files, ssh, small files only |
| `$WORK` `/leonardo_work/IscrC_UQ-MF` | 1 TB per project | up to 6 months after the project ends; no purge | **none** | **persistent project data**: code, conda env, datasets, run outputs ("the main area for maintaining scratch files resulting from batch processing") |
| `$FAST` `/leonardo_scratch/fast/IscrC_UQ-MF` | 1 TB per project, no extensions | up to 6 months after the project ends | none | I/O-heavy working data; our overflow for epoch checkpoints |
| `$CINECA_SCRATCH` `/leonardo_scratch/large/userexternal/mtolloso` | no quota | **files not accessed for 40 days are deleted daily** | none | temporary files of batch jobs only; never the only copy of anything |
| `$TMPDIR` (on a Booster node: 10 GB in RAM) | 10 GB | **deleted when the job ends** | none | per-job caches (pip, torch, matplotlib) |
| `$PUBLIC` | 50 GB | while the user exists | none | sharing with other users (not needed) |

Nothing is backed up except `$HOME`: anything citable must also come back to the DGX
(`RUNS.md` protocol). Usage: `cindata`; budget: `saldo -b`.

**Our layout** (`$UQ` = `$WORK/uq-mf`):

| Path | Content |
|---|---|
| `$UQ/miniconda3` | our own Miniconda (separate from the other project's), env `mace` from `environment.yml` |
| `$UQ/mace` | git clone of `iclr_rebuttal` (clean checkouts only; `data/` inside it, copied from the DGX) |
| `$UQ/runs` | `UQ_RUNS` for all experiments (same layout as the DGX `runs/`) |
| `$UQ/slurm` | sbatch scripts and SLURM stdout/stderr (`%x_%j.out`) |
| `$FAST/uq-mf/runs` | overflow `UQ_RUNS` for LF-scale experiments if `$WORK` runs short |
| `$CINECA_SCRATCH/uq-mf` | W&B offline directories and other throwaway files |

Storage arithmetic: an LF-scale experiment keeps ~255 GB of epoch checkpoints (17 MB x 300 x 50)
until it is evaluated and pruned (approved: every 5th epoch + best + `.model`, ~50 GB after).
Never more than ~3 unpruned LF-scale experiments on `$WORK` at once.

## 3. Jobs (SLURM)

Booster node: 4x A100 64 GB, 32 cores, 512 GB RAM. QOS on partition `boost_usr_prod`:

| QOS | Limits | Use |
|---|---|---|
| `normal` (default) | <= 64 nodes, 24 h | production; one member (or 2-3 packed) per GPU |
| `boost_qos_dbg` | <= 8 nodes, 30 min, max 2 jobs running+pending per user, priority 80 | smoke tests |
| `boost_qos_lprod` | <= 8 nodes (32 GPUs) per project, 4 days | runs longer than 24 h (e.g. pretraining with forces) |
| `boost_qos_bprod` | >= 65 full nodes | not for us |

Partition `lrd_all_serial`: budget-free, <= 4 cores, 4 h, no GPU (compilation, file checks).

**Accounting.** Charge = elapsed h x nodes x R x 32 cores, with R the *largest* fraction of any
resource reserved on the node (cores, GPUs, memory). One GPU = 1/4 node = 8 local h per hour, *if*
the job asks for <= 8 cores and <= 1/4 of the memory. Template for one GPU:
```bash
#SBATCH --account=IscrC_UQ-MF
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal                 # boost_qos_dbg for tests
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # more cores => charged as more GPUs
#SBATCH --mem=120G                   # <= 128 GB keeps R = 1/4
#SBATCH --time=24:00:00
```
Packing several members in one 1-GPU job costs the same budget as one member: set
`OMP_NUM_THREADS` so the members share the 8 cores (measure throughput first, WP11.8).

**Monthly quota (linearization).** `saldo -b` shows `monthTotal` (budget / months, 8,759 local h
in Nov 2026). Exceeding it does **not** stop jobs: "it will take more time to start since its
priority will be lowered". Plan big runs early in the month when possible.

**Login nodes:** only light work; interactive processes are killed after **10 min of CPU time**
(CINECA course material; the user guide is silent). Long installs or file operations go into a job
(`lrd_all_serial`) — **verify** whether those nodes have internet. rsync from the DGX to a login
node is network-bound and fine.

**Internet:** login nodes yes (*observed*: PyPI, GitHub). Compute nodes: **verify** in the smoke
test; plan for none (`WANDB_MODE=offline`, everything installed beforehand).

Useful: `squeue --me`, `scontrol show job <id>` (estimated start), `sprio -j <id>`, `sacct -j <id>`,
`scancel <id>`, `saldo -b`, `cindata`.
