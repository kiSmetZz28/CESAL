# Requirements

## Hardware

| Purpose | Requirement |
| --- | --- |
| Edge stage, dashboard, unit tests | x86-64 CPU, 16 GB RAM. No GPU needed. |
| Cloud BAT ensemble (81 EM-AT models) | One CUDA GPU. In the paper this ran on the Talon HPC cluster; a single 16 GB GPU is enough to re-run inference. |
| LLM incident classification | One CUDA GPU, 16 GB VRAM sufficient (verified on an RTX 2000 Ada). Models larger than VRAM are partly offloaded to CPU RAM, which is slower but works. |
| Edge resource measurements (paper Table 6) | Physical **Raspberry Pi 3B+, 4B and 5** devices. Cannot be reproduced without that hardware. |

### Reference testbed (paper Section 4.1)

The published results were produced on the following testbed. None of it is required to run the
artifact — the table records what the reported numbers were measured on.

| Platform | Hardware profile | OS |
| --- | --- | --- |
| Talon cluster node | 2 × 18-core Intel Xeon Gold 6140; 8 × NVIDIA Tesla V100; 1.5 TB RAM | Red Hat Enterprise Linux 9.2 |
| Dell PowerEdge R650 | 36-core Intel Xeon Platinum; Mellanox ConnectX-6 100 Gb NIC; 256 GB RAM | Ubuntu 24.04.2 LTS |
| Data processing server | Intel Core i7-14700 (28 cores / 56 threads); NVIDIA RTX 2000 Ada; 32 GB RAM | Windows 11 |
| Data analytics server | Intel Core i7-14700 (28 cores / 56 threads); NVIDIA RTX 2000 Ada; 32 GB RAM | Ubuntu 24.04.2 LTS |
| Raspberry Pi 5 | Cortex-A76, 4 cores / 4 threads; 8 GB RAM | Ubuntu 20.04.5 LTS |
| Raspberry Pi 4B | Cortex-A72, 4 cores / 4 threads; 8 GB RAM | Ubuntu 20.04.5 LTS |
| Raspberry Pi 3B+ | Cortex-A53, 4 cores / 4 threads; 1 GB RAM | Ubuntu 20.04.5 LTS |

Roles: the Talon cluster handles BAT training and cloud-side verification; the R650 handles log
collection and storage; the two i7 servers handle data processing, EM-AT analysis, Q-BAT
preparation and model conversion; the Raspberry Pi cluster is the edge deployment target.

### Disk

| Item | Size |
| --- | --- |
| Repository clone | ~31 MB |
| BAT checkpoints (81 `.pth` per dataset) | ~3.5 GB per dataset |
| Q-BAT checkpoints (3 `.pte` per dataset) | small, bundled in the download |
| ExecuTorch runtime + build tree | ~1.4 GB |
| Raw HDFS logs (only for the dashboard's log browser) | ~1.6 GB |
| Dashboard SQLite database (built at runtime, optional) | ~5.4 GB |
| Prediction outputs per dataset | ~1.2 GB |

Budget about **40 GB** free for a full HDFS + OpenStack reproduction.

## Software

Linux with `conda`. Two environments, created by [install.sh](install.sh):

| Environment | Python | Key packages | Used for |
| --- | --- | --- | --- |
| `cesal-edge` | 3.10 | PyTorch 2.6 (CPU), ExecuTorch 0.5.0, FastAPI | Edge Q-BAT inference, routing, dashboard, orchestration |
| `cesal-cloud` | 3.10 | PyTorch 2.4.0+cu124, transformers 4.47.1, accelerate 1.2.1 | BAT cloud ensemble, LLM incident classification |

Exact pinned lists: [environment/edge/requirements.txt](environment/edge/requirements.txt), [environment/cloud/requirements.txt](environment/cloud/requirements.txt).

The two environments are deliberately separate: they install different PyTorch builds (CPU vs CUDA). The inference pipeline relies on the split — the edge stage orchestrates and spawns cloud inference as a subprocess using the cloud environment's interpreter, so BAT checkpoints are never loaded inside the edge environment.

## Accounts and external services

- **Google Drive** — `run.py download` fetches pre-trained checkpoints. No account needed, but Drive rate-limits per IP; downloads resume if interrupted.
- **Hugging Face** — the LLM stage pulls four backbones. `Meta-Llama-3.1-8B-Instruct` and `gemma-2-9b-it` are **gated**: accept their licenses on the model pages and run `hf auth login` first. The two Qwen models are ungated.

## Expected runtimes

Measured on the reference machine (RTX 2000 Ada, 16-core CPU):

| Task | Time |
| --- | --- |
| `install.sh` (both envs, no assets) | ~5-10 min |
| Unit tests | ~3 s |
| **Scaled-down end-to-end run** (300 windows, all 4 stages) | **~2 min** |
| Full HDFS edge stage (221,540 windows × 3 Q-BAT models) | Many hours — see note |
| Full HDFS cloud stage (81 BAT checkpoints) | ~1-2 h on GPU |
| LLM classification, one backbone, 4,124 sequences | ~1-3 h depending on backbone |

> **Note on the full edge run.** If the ExecuTorch **Python bindings** are not installed in `cesal-edge`, the edge stage falls back to invoking the pre-built C++ `executor_runner` binary **once per window per model**, which extrapolates to roughly a day for full HDFS. Reviewers who only need to exercise the pipeline should use the scaled-down script, which takes ~2 minutes.
