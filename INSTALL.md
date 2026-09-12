# Installation and reproduction

Tested on Ubuntu with conda. See [REQUIREMENTS.md](REQUIREMENTS.md) for hardware, disk and
runtime expectations.

## 1. Install

```bash
git clone https://github.com/kiSmetZz28/CESAL.git
cd CESAL
./install.sh                 # both environments + unit tests (~5-10 min)
```

`install.sh` creates `cesal-edge` and `cesal-cloud`, installs the pinned requirements into
each, installs CESAL itself with `--no-deps` (so the pinned PyTorch builds are not
re-resolved), and finishes by running the test suite. Expected: **24 passed**.

Options:

```bash
./install.sh --edge-only     # no CUDA GPU on this machine
./install.sh --with-assets   # also download checkpoints + ExecuTorch runtime (~5 GB)
EDGE_ENV=my-edge CLOUD_ENV=my-cloud ./install.sh    # custom environment names
```

## 2. Fetch checkpoints

```bash
conda activate cesal-edge
python run.py download            # all datasets, BAT + Q-BAT + ExecuTorch runtime
python run.py download hdfs       # one dataset only
```

Downloads come from Google Drive, which rate-limits per IP; re-run to resume.

## 3. Tell the pipeline which interpreters to use

The edge stage spawns the cloud stage as a subprocess, so it needs both paths:

```bash
export EDGE_PYTHON=$(conda run -n cesal-edge  which python)
export CLOUD_PYTHON=$(conda run -n cesal-cloud which python)
```

Defaults are `~/miniconda3/envs/cesal-edge/bin/python` and `~/miniconda3/envs/cesal-cloud/bin/python`;
set the variables if your environment names or conda root differ.

## 4. Kick the tires (~2 minutes)

```bash
./artifact/claims/claim4_scaled_down.sh
```

Runs all four stages — edge Q-BAT scan, Mahalanobis routing, cloud BAT verification over all
81 checkpoints, hybrid evaluation — on a 300-window subsample, writing to a scratch
directory. This is the fastest way to confirm the install works end to end.

## 5. Reproduce the paper claims

| Claim | Paper | Script |
| --- | --- | --- |
| Log-based incident detection | Section 4.2, Table 3 | [`artifact/claims/claim1_detection.sh`](artifact/claims/claim1_detection.sh) |
| Open-set incident classification | Section 4.6, Table 7 | [`artifact/claims/claim2_classification.sh`](artifact/claims/claim2_classification.sh) |
| Controlled response workflows | Section 3.7, Table 1 | [`artifact/claims/claim3_response.sh`](artifact/claims/claim3_response.sh) |
| Scaled-down end-to-end run | — | [`artifact/claims/claim4_scaled_down.sh`](artifact/claims/claim4_scaled_down.sh) |

Each script prints the expected values next to what it measured.

> **Runtime warning.** `claim1` on full HDFS is long. If the ExecuTorch **Python bindings**
> are missing from `cesal-edge`, the edge stage shells out to the C++ `executor_runner` once
> per window per model — roughly a day for HDFS's 221,540 windows. OpenStack is far smaller
> and finishes quickly. Use `claim4` for a fast check of the same code path.

## 6. Optional — the dashboard

```bash
conda activate cesal-edge
export EDGE_PYTHON=$(which python)
export CLOUD_PYTHON=$(conda run -n cesal-cloud which python)
python dashboard/app.py            # http://localhost:8765
```

`PORT=8799 python dashboard/app.py` to use another port. The dashboard browses the parsed
logs, runs the pipeline, and shows detection plus incident-classification results. It is not
required for any claim.

`launch_dashboard.py` is an alternative entry point that first downloads assets — note it
re-runs the ExecuTorch Python-bindings cmake build on every launch whenever
`from executorch.runtime import Runtime` fails, so prefer `python dashboard/app.py` for
day-to-day use.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Edge threshold file not found` | Inference reads pre-computed thresholds; they are committed under `outputs/<ds>/thresholds_*.yaml`. If you point `output_dir` somewhere new, copy both threshold YAMLs into it first — only `run.py train` regenerates them. |
| `conda: command not found` in `install.sh` | Install Miniconda, then re-run. |
| Google Drive returns HTTP 403 | Per-IP rate limiting. Wait and re-run `run.py download`; it resumes. |
| LLM stage fails with a 401/403 from Hugging Face | Gated model. Accept the license on the model page, then `hf auth login`. |
| Dashboard starts but panels are empty | The SQLite database is still importing on first launch; the indicator shows **Loading**. |
