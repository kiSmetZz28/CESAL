# CESAL

**Cloud–Edge Security Analytics for Log-Based Incident Detection, Classification, and Controlled Response**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![HF Demo](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face-yellow)](https://kismetzz-ceco-lad.hf.space/)

**[🤗 Try the live demo on Hugging Face Spaces](https://kismetzz-ceco-lad.hf.space/)**

---

## How It Works

CESAL is a security-aware cloud–edge framework for log-based incident detection, classification, and controlled response. It follows an **edge-first, cloud-assisted** strategy:

- **Edge-side detection** — Q-BAT, an ensemble of quantized EM-AT models, scores log sequences locally on resource-constrained devices.
- **Uncertainty-guided routing** — a Mahalanobis distance-based routing policy keeps confident samples at the edge and escalates only the most uncertain ones (10% by default) to the cloud, reducing edge-to-cloud transmission of security-sensitive logs.
- **Cloud-side verification** — BAT, a high-capacity ensemble of 81 EM-AT models, re-evaluates the escalated samples.
- **Incident classification and controlled response** — detected abnormal sequences are queued and classified by a retrieval-augmented LLM into a known anomaly type or an unknown anomaly; known types are mapped to predefined response workflows, and unknown ones are flagged for human investigation.

<p align="center">
  <img src="pictures/framework.png" width="800">
</p>

### Models

| Component          | Where        | What                                                                                                                                                                   |
| ------------------ | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EM-AT**          | Base learner | Anomaly Transformer that scores log sequences by reconstruction error and association discrepancy, with EM-GMM-based automated thresholding                            |
| **BAT**            | Cloud        | Bagging-style ensemble of 81 EM-AT models, each trained on a bootstrap sample with its own epochs, loss weight, batch size, and encoder depth; majority voting         |
| **Q-BAT**          | Edge         | Ensemble of 3 EM-AT models quantized with TorchAO (int8 dynamic activations, int4 weights) and exported as ExecuTorch `.pte` programs                                  |
| **LLM classifier** | Cloud        | Qwen2.5-14B-Instruct by default; combines retrieved reference sequences with open-set decision rules to output one of 10 known HDFS anomaly types or the unknown class |

#### EM-AT — the base learner (Sec. 3.5.1)

EM-AT extends the Anomaly Transformer by replacing manual threshold selection with **EM-GMM-based automated thresholding**, so no device-specific tuning is needed across heterogeneous cloud–edge deployments. A Transformer encoder learns normal log behaviour from normal training sequences only; each encoder layer contains an **Anomaly Attention** module and a feed-forward network. Anomaly Attention models two associations — the **series association** `S` (global log dependencies) and the **prior association** `P` (local adjacency patterns). Their difference is the **association discrepancy**, defined as the symmetrized KL divergence (Eq. 1):

```
AD(P, S; C) = [ (1/N) · Σ_{n=1..N} ( KL(P_i,:^n ‖ S_i,:^n) + KL(S_i,:^n ‖ P_i,:^n) ) ]_{i=1..L}
```

The anomaly score combines reconstruction error with the normalized association discrepancy (Eq. 2), where `⊙` is element-wise multiplication and `Ĉ` the reconstruction of `C`:

```
E(C) = softmax( −AD(P, S; C) ) ⊙ ‖ C_i,: − Ĉ_i,: ‖²₂ ,   i = 1..L
```

**Automated thresholding.** EM-AT fits a Gaussian Mixture Model to the anomaly score distribution, choosing the number of components by BIC. Components are sorted by increasing mean `μ₁ ≤ μ₂ ≤ … ≤ μ_H` with mixture weights `ξ₁…ξ_H`. Because low scores usually correspond to normal behaviour, EM-AT treats the lowest-mean components as the normal region: it finds the smallest `m` with `Σ_{k=1..m} ξ_k ≥ τ`, estimates the anomaly proportion `r = 1 − Σ_{k=1..m} ξ_k`, and sets the threshold `λ` as the `(1−r)`-quantile of the score distribution. A base learner is therefore the pair `(f, λ)`.

Configuration: 512 hidden channels, 8 attention heads, Adam with initial learning rate 1e-4.

Code: [cesal_core/models/](cesal_core/models/), scoring in [cesal_core/utils/energy.py](cesal_core/utils/energy.py), thresholding in [training_pipeline/solver.py](training_pipeline/solver.py) and [cesal_inference_pipeline/lad_qbat_edge.py](cesal_inference_pipeline/lad_qbat_edge.py).

#### BAT — cloud-side bagging ensemble (Sec. 3.5.2)

A single EM-AT is sensitive to the sampled training data and to hyperparameters, especially when rare or context-dependent anomalies appear only in limited patterns. BAT alleviates this by **bagging**: each base learner trains on a bootstrap subset sampled with replacement from the normal training set, and — to increase ensemble diversity — with a configuration drawn from a predefined hyperparameter pool.

```
Algorithm 1 — BAT training
Input : normal training set C_train; ensemble size I; bootstrap size n; hyperparameter pool H
Output: trained BAT ensemble B = {(b_i, λ_i^B)}_{i=1..I}
for i = 1 … I:
    h_i    ← SelectConfig(H, i)          # epochs, batch size, encoder depth, loss weight
    C_i    ← BootstrapSample(C_train, n) # sample n with replacement
    b_i    ← TrainEMAT(C_i; h_i)         # train the i-th EM-AT base learner
    E_i    ← Score(b_i, C_i)             # training anomaly scores
    λ_i^B  ← FitThreshold_EM-GMM(E_i)    # per-learner threshold
    B      ← B ∪ {(b_i, λ_i^B)}
```

At inference each base learner converts its anomaly scores to binary decisions using its own threshold, and BAT aggregates them by **majority voting**. The ensemble size is **I = 81** — the full grid of epochs `{3, 6, 10}` × loss weight η `{3, 4, 5}` × encoder depth `{3, 6, 8}` × batch size `{32, 64, 96}`. The ensemble-size analysis (Fig. 5) shows performance stabilizes after roughly 65 base learners, so 81 sits comfortably in the stable region.

Code: [training_pipeline/](training_pipeline/), cloud inference in [cesal_inference_pipeline/lad_bat_cloud.py](cesal_inference_pipeline/lad_bat_cloud.py), voting in [cesal_core/utils/voting.py](cesal_core/utils/voting.py).

#### Q-BAT — quantized edge ensemble (Sec. 3.5.3)

BAT's compute and memory cost makes it impractical to deploy directly on edge hardware — in Table 6 the full BAT ensemble is marked *Not Executable* on all three Raspberry Pi models, because it exceeds available memory. Q-BAT is a compact ensemble of **K = 3** quantized EM-AT base learners, `Q = {(q_k, λ_k^Q)}_{k=1..K}` — the largest ensemble that runs reliably on a Raspberry Pi without excessive resource consumption.

Each base learner is trained in PyTorch, then optimized for edge execution with two PyTorch-native tools:

- **ExecuTorch** gives a path from model export to on-device execution without cross-framework conversion, removing the risk of inconsistency between the trained model and the deployed edge binary. Ahead-of-time export, a reduced operator representation and memory planning yield predictable runtime behaviour and low memory overhead.
- **TorchAO** applies the `int8_dynamic_activation_int4_weight()` scheme: activations quantized to 8-bit integers and weights to 4-bit integers, reducing model size, memory footprint and compute cost while maintaining detection accuracy.

Deployment has two phases: **program preparation** (export → quantize → lower through the ExecuTorch pipeline → serialize to a `.pte` program) and **program execution** (the `.pte` is loaded on the edge device and run to produce anomaly predictions).

Code: [quantization/qbat_export.py](quantization/qbat_export.py) (export), [cesal_inference_pipeline/lad_qbat_edge.py](cesal_inference_pipeline/lad_qbat_edge.py) (edge inference).

### Framework Overview

CESAL is a security-aware cloud-edge framework for log-based incident detection, classification, and controlled response. Raw logs generated by cloud servers, gateways, and edge devices are collected near their sources, parsed, and converted into structured log sequences. The collaborative LAD pipeline then applies Q-BAT for lightweight edge-side inference and BAT for high-capacity cloud-side verification of selected uncertain samples. A Mahalanobis distance-based routing policy estimates edge-side prediction uncertainty and forwards only samples near the decision boundary to the cloud. This edge-first design keeps confident samples local, reduces unnecessary edge-to-cloud transmission of security-sensitive logs, and preserves strong detection performance. After abnormal sequences are detected, CESAL can optionally invoke a cloud-side LLM-based open-set classification and controlled response module. With retrieval-augmented generation (RAG), the module classifies each detected abnormal sequence into a predefined anomaly type or assigns it to the "Unknown Anomaly Types" class when no known category is sufficiently supported. Known anomaly types are mapped to predefined response workflows to support controlled mitigation through the LLM agent, while unknown anomaly types are preserved and flagged for human investigation.

### Paper ↔ Code Mapping

| Paper component                                                                                                           | Code location                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Log data processing: sequences → sliding context windows (Sec. 3.4; datasets are provided already parsed into log events) | [cesal_core/data/](cesal_core/data/)                                                                                                                             |
| EM-AT base learner (Sec. 3.5.1)                                                                                           | [cesal_core/models/](cesal_core/models/)                                                                                                                         |
| EM-GMM automated thresholding (Sec. 3.5.1)                                                                                | [training_pipeline/solver.py](training_pipeline/solver.py) (BAT), [cesal_inference_pipeline/lad_qbat_edge.py](cesal_inference_pipeline/lad_qbat_edge.py) (Q-BAT) |
| BAT training (Sec. 3.5.2, Algorithm 1)                                                                                    | [training_pipeline/](training_pipeline/)                                                                                                                         |
| Q-BAT quantization and ExecuTorch export (Sec. 3.5.3)                                                                     | [quantization/qbat_export.py](quantization/qbat_export.py)                                                                                                       |
| Edge-side inference with Q-BAT (Sec. 3.6, Algorithm 2)                                                                    | [cesal_inference_pipeline/lad_qbat_edge.py](cesal_inference_pipeline/lad_qbat_edge.py)                                                                           |
| Mahalanobis distance-based routing policy (Sec. 3.6.1, Algorithm 3)                                                       | [cesal_inference_pipeline/routing.py](cesal_inference_pipeline/routing.py)                                                                                       |
| Cloud-side verification with BAT (Sec. 3.6, Algorithm 2)                                                                  | [cesal_inference_pipeline/lad_bat_cloud.py](cesal_inference_pipeline/lad_bat_cloud.py)                                                                           |
| Cloud–edge collaborative inference pipeline (Sec. 3.6)                                                                    | [cesal_inference_pipeline/run.py](cesal_inference_pipeline/run.py), [dashboard/cloud_runner.py](dashboard/cloud_runner.py)                                       |
| Edge-side and cloud-side anomaly queues Q_E / Q_C (Sec. 3.1)                                                              | [incident_response/queues.py](incident_response/queues.py)                                                                                                       |
| RAG-based evidence retrieval and LLM open-set incident classification (Sec. 3.7)                                          | [incident_response/classifier.py](incident_response/classifier.py)                                                                                               |
| Predefined response workflows and escalation policy (Sec. 3.7, Table 1)                                                   | [incident_response/workflows.py](incident_response/workflows.py)                                                                                                 |
| Queued incidents → classification → response workflow (Sec. 3.1, 3.7)                                                     | [incident_response/process_queues.py](incident_response/process_queues.py)                                                                                       |
| Open-set incident classification evaluation (Sec. 4.6, Table 7)                                                           | [incident_response/evaluate.py](incident_response/evaluate.py), [incident_response/data_prep.py](incident_response/data_prep.py)                                 |

---

## Repository Structure

`run.py` is the primary entry point — a single CLI that dispatches every pipeline stage (download, train, eval, convert, infer, classify, respond). `launch_dashboard.py` is an optional local helper that bundles asset download with launching the web UI. Everything else falls into one of three groups — **shared library**, **pipeline stages**, or **supporting assets**.

<pre>
CESAL/
│
│ ── Entry points (run from project root) ─────────────────────────────────
├── run.py                         # Main CLI: download | train | eval | convert | infer | classify | respond
├── launch_dashboard.py            # Optional local helper: fetch assets + start the web UI
│
│ ── Shared library ───────────────────────────────────────────────────────
├── cesal_core/                    # Imported by every pipeline below
│   ├── models/                    #   EM-AT architecture (attention, embedding)
│   ├── data/                      #   Dataset loaders + log preprocessor
│   └── utils/                     #   Energy scoring, voting, config I/O, metrics
│
│ ── Pipeline stages ──────────────────────────────────────────────────────
├── training_pipeline/             # 1. Train BAT ensemble (81 EM-AT models)
│   ├── train.py                   #    Hyperparameter sweep
│   ├── solver.py                  #    EM-AT base model training
│   └── evaluate.py                #    Per-model evaluation
│
├── quantization/                  # 2. Convert BAT → Q-BAT
│   └── qbat_export.py             #    A8W4 quantize, export to ExecuTorch .pte
│
├── <b>🔴 cesal_inference_pipeline/</b>      # 3. CESAL collaborative inference pipeline: Edge → routing → cloud
│   ├── lad_qbat_edge.py           #    Edge-side LAD using Q-BAT via ExecuTorch
│   ├── routing.py                 #    Mahalanobis distance-based routing (uncertain → cloud)
│   ├── lad_bat_cloud.py           #    Cloud-side LAD using BAT for reevaluation
│   ├── run.py                     #    Pipeline driver: runs stages 1–4
│   └── executorch/                #    Pre-built ExecuTorch runtime (fetched by `run.py download`)
│
├── dashboard/                     # 4. Front-end UI for visualization
│
├── incident_response/             # 5. LLM-based open-set incident classification + controlled response
│   ├── classifier.py              #    Lexical RAG retriever, open-set decision rules, LLM prompt/parsing
│   ├── evaluate.py                #    Evaluate LLM backbones on HDFS abnormal sequences (Table 7)
│   ├── data_prep.py               #    Build open-set test set + RAG knowledge base
│   ├── workflows.py               #    Predefined response workflows (Table 1)
│   ├── queues.py                  #    HDFS detections → edge/cloud anomaly queues
│   └── process_queues.py          #    Classify queued incidents + select workflows
│
│ ── Configuration & data ─────────────────────────────────────────────────
├── configs/
│   ├── training/{hdfs,os}.yaml
│   ├── inference/{hdfs,os}.yaml
│   └── llm/hdfs.yaml
│
├── data/                          # Pre-processed log event sequences (+ HDFS/open_set/ for the LLM module)
├── outputs/                       # Thresholds, predictions (+ hdfs/llm/table7_reference_metrics.csv)
├── checkpoints/                   # bat/*.pth + qbat/*.pte — fetched by `run.py download`
├── logs/                          # Training & inference logs
│
├── environment/                   # Python dependency lists
│   ├── cloud/requirements.txt     #   Training, eval, cloud inference, LLM incident response
│   └── edge/requirements.txt      #   ExecuTorch edge inference, dashboard
│
│ ── Tooling ──────────────────────────────────────────────────────────────
└── tools/
    ├── download_checkpoints.py    # Fetch BAT / Q-BAT checkpoints
    ├── download_data.py           # Fetch ExecuTorch runtime + raw logs
    └── deploy/                    # Maintainer-only — Hugging Face Space deployment
</pre>

---

## Full Setup

All commands run from the project root. CESAL uses **two Conda environments**, one for each inference tier:

| Environment   | Tier      | Stack                                       | What runs in it                                                                                                                                                                                                             |
| ------------- | --------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cesal-edge`  | **Edge**  | PyTorch 2.6 (CPU) + ExecuTorch 0.5          | Dashboard, Q-BAT edge inference (`.pte` via ExecuTorch), pipeline orchestration. CPU is sufficient.                                                                                                                         |
| `cesal-cloud` | **Cloud** | PyTorch 2.4 + CUDA 12.4 + transformers 4.47 | BAT ensemble training (81 models), cloud re-check inference, and LLM-based incident classification and response (`incident_response/`). GPU strongly recommended; the inference pipeline launches this env as a subprocess. |

**Why two environments?** Edge and cloud have different runtime needs. Edge uses ExecuTorch (compact, CPU-only, runs `.pte` quantized models), while cloud uses full-precision PyTorch with CUDA. Splitting them keeps each install minimal and avoids version conflicts between ExecuTorch and CUDA PyTorch.

### Step 1 — Set up environments

#### Edge environment (`cesal-edge`)

```bash
conda create -yn cesal-edge python=3.10.0
conda activate cesal-edge
pip install -r environment/edge/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

ExecuTorch **0.5.0** ([docs](https://docs.pytorch.org/executorch/0.5/)) and its bundled `torchao` build are downloaded and installed automatically in Step 2 below (or by `launch_dashboard.py`) — no manual compilation needed (they carry PEP 440 local version labels and are not on PyPI, hence commented out in `requirements.txt`). **Linux/macOS only**; Windows users: use WSL2.

#### Cloud environment (`cesal-cloud`)

```bash
conda create -yn cesal-cloud python=3.10.0
conda activate cesal-cloud
pip install -r environment/cloud/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

> Edge and cloud install **different** requirements files and PyTorch builds (CPU vs CUDA), so they must be separate envs. The inference pipeline also relies on this split: edge orchestrates and spawns cloud inference as a subprocess pointed at the cloud env's interpreter.

**Verify both environments exist:**

```bash
conda env list   # should list both 'cesal-edge' and 'cesal-cloud'
```

### Step 2 — Download checkpoints and the edge runtime

`run.py` is the project's main entry point. Use `run.py download` to fetch the BAT (`.pth`) and Q-BAT (`.pte`) checkpoints required by the inference pipeline. Whenever Q-BAT checkpoints are part of the download, `run.py` _also_ installs the **ExecuTorch 0.5.0 runtime** (and its bundled `torchao` build) — these are required by `run.py infer` (edge stage) and `run.py convert`.

```bash
conda activate cesal-edge
python run.py download                # all datasets, both checkpoint types + ExecuTorch runtime
python run.py download hdfs           # one dataset (BAT + Q-BAT) + ExecuTorch runtime
python run.py download hdfs bat       # HDFS full-precision only (no ExecuTorch needed)
python run.py download hdfs qbat      # HDFS quantized Q-BAT + ExecuTorch runtime
```

> **Note:** raw log files are _not_ fetched here — they are only needed for the optional web dashboard's log-browsing panels and are downloaded by `launch_dashboard.py` (see "Optional — Launch the local dashboard" below).

### Step 3 — Run the full pipeline from the CLI

`run.py` dispatches every pipeline stage. Pre-computed thresholds for both datasets are bundled with the repository, so you can run inference immediately:

```bash
conda activate cesal-edge
python run.py infer os                # edge scan → routing → cloud re-check → final prediction
python run.py infer hdfs
```

For other stages — `train`, `convert` — see [Advanced Options](#advanced-options).

### Optional — Launch the local dashboard

The dashboard is a web UI at **http://localhost:8765** that runs the inference pipeline and browses the parsed logs. There are two ways to start it.

**First time — fetch the assets, then launch.** `launch_dashboard.py` performs the same checkpoint + ExecuTorch download as `run.py download`, and additionally fetches the **raw log files** needed for the log-browsing panels:

```bash
conda activate cesal-edge
python launch_dashboard.py                # download missing assets (ExecuTorch + Q-BAT + raw logs + BAT) + launch UI
python launch_dashboard.py --setup-only   # download assets, do not launch
python launch_dashboard.py --no-bat       # skip BAT checkpoints (~3.5 GB × dataset)
python launch_dashboard.py --status       # print what is present / missing, then exit (no downloads, no setup)
```

**Every run after that — start the UI directly.** This skips all setup and comes up in a few seconds:

```bash
conda activate cesal-edge
export EDGE_PYTHON=$(which python)                            # interpreter for the edge stage
export CLOUD_PYTHON=~/miniconda3/envs/cesal-cloud/bin/python  # interpreter for the BAT and LLM stages
python dashboard/app.py                                       # PORT=8799 python dashboard/app.py for another port
```

`python dashboard/app.py` is exactly what `launch_dashboard.py` execs once its asset checks pass, so it serves the same UI — it simply never downloads or installs anything. Prefer it for day-to-day runs, because `launch_dashboard.py` re-runs the ExecuTorch **Python-bindings** install (a cmake build) on every launch whenever `from executorch.runtime import Runtime` fails in the active environment. Those bindings are optional: the edge stage falls back to the pre-built C++ `executor_runner` that ships in the ExecuTorch download, which is the path the pipeline uses by default.

`EDGE_PYTHON` and `CLOUD_PYTHON` tell the dashboard which interpreters to spawn for the pipeline stages (defaults: `~/miniconda3/envs/cesal-edge/bin/python` and `~/miniconda3/envs/cesal-cloud/bin/python`); set them when your environment names differ. A built-in **? Help** button guides you through the panels. On first launch the **Database** indicator shows **Loading** while log data is imported. The HDFS log-browsing panel also needs a few minutes the first time each process serves it, while it builds an ordering cache in memory; the results, config and prediction panels respond immediately.

### Optional — LLM-based incident classification and controlled response

`incident_response/` classifies abnormal HDFS log sequences into the 10 known anomaly types or `Other anomaly type` (unknown) with a RAG-enhanced LLM, then maps each label to a predefined response workflow (Table 1). It runs in the cloud environment (`cesal-cloud`) and needs a CUDA GPU; models that exceed GPU memory are partly offloaded to CPU RAM.

<p align="center">
  <img src="pictures/llm_module.png" width="850">
</p>

#### How the module works (Sec. 3.7, Fig. 4)

Abnormal sequences detected by the collaborative LAD pipeline are buffered in an **anomaly queue** — edge-side `Q_E` or cloud-side `Q_C` — instead of being classified inline. This decouples detection from classification and response, letting CESAL defer analysis until connectivity is sufficient, cloud resources are available, or an operator requests it. Each queued sequence `c_j` then passes through three stages:

1. **RAG evidence retrieval.** Reference abnormal log sequences for each known anomaly type are indexed offline as a knowledge base. The detected sequence is used as the query, and the most relevant reference entries `R_j` are retrieved as evidence. Grounding the decision in observed log evidence reduces reliance on unconstrained generation and improves consistency of type-level classification.
2. **Open-set classification.** The retrieved evidence, the input sequence, and the candidate label set `A = {a₁ … a_S, a_unk}` are assembled into a structured prompt, and the LLM must output **exactly one** label: `â_j = G(c_j, R_j, A)`. A known type is selected only when the retrieved evidence and the full input sequence provide sufficient support; otherwise the sequence is assigned to **"Unknown Anomaly Types"**. This open-set design avoids forcing unfamiliar abnormal patterns into known categories and preserves unknown cases for later human investigation and knowledge-base refinement.
3. **Workflow selection.** The predicted label is mapped to its predefined workflow, `w_j = Ω(â_j)` — see Table 1 below.

Default backbone: **Qwen2.5-14B-Instruct** (14.7B parameters), chosen for long-context support, instruction following, and structured output that the downstream workflow-selection module can parse directly.

**Execution and escalation policy.** The LLM agent never generates arbitrary response plans: its output only *selects* among predefined workflows, and execution is constrained by policy checks and approval gates. Low-impact actions — evidence collection, status validation, diagnostic command execution, safe retry, metadata-view refresh, controlled re-replication — can execute automatically when policy conditions are satisfied. High-impact actions — metadata modification, permanent block cleanup, destructive cleanup, service-level restart — remain subject to safeguards or **administrator approval**. Sequences classified as "Unknown Anomaly Types" trigger **no** automated mitigation: the abnormal sequence, retrieved evidence and system context are preserved and the case is flagged for human investigation.

> **Implementation note.** The paper describes the knowledge base as indexed in a vector database. This implementation retrieves lexically — TF-IDF cosine similarity combined with multiset and bigram Jaccard overlap over event tokens, plus exact/containment/prefix/suffix bonuses — in [incident_response/classifier.py](incident_response/classifier.py). No embedding model or vector store is required, which keeps the module dependency-light and deterministic.

#### Table 1 — Predefined response workflows and escalation policy (HDFS)

`🔒` marks a workflow containing an administrator-approval step, `⬆` one containing an escalation step. Verbatim text lives in [`incident_response/workflows.py`](incident_response/workflows.py); print any of them with `python -m incident_response.workflows --label "<type>"`.

| Anomaly type | Gate | Predefined controlled response workflow |
| --- | :---: | --- |
| Namenode not updated after deleting block | 🔒 | Collect NameNode edit logs, namespace snapshots, and DataNode block reports; run HDFS fsck to verify namespace and block consistency; refresh NameNode/DataNode metadata views through approved interfaces; if inconsistency persists, submit an approval-gated metadata synchronization task. |
| Write exception client give up | ⬆ | Collect client logs, DataNode status, and network diagnostics; identify the failed stage in the write pipeline; verify target DataNode availability and pipeline health; trigger a safe write retry after the pipeline is recovered; escalate repeated failures with the collected evidence. |
| Write failed at beginning | — | Check safe mode, permission, quota, and block allocation status; inspect client-side and NameNode initialization logs; correct safe configuration issues through approved management actions when permitted; trigger a controlled write retry after the initial write path is validated. |
| Replica immediately deleted | 🔒 | Inspect replication policy, block state, and replica placement records; determine whether the replica is invalid, corrupt, excessive, or prematurely removed; trigger re-replication when policy conditions are satisfied; submit administrator approval for destructive replica cleanup or metadata correction. |
| Received block that does not belong to any file | 🔒 | Compare DataNode block records with NameNode namespace metadata; run HDFS fsck to identify orphan, stale, or inconsistent blocks; quarantine suspicious block records through approved interfaces; submit administrator approval before permanent block cleanup or metadata modification. |
| Redundant addStoredBlock | ⬆ | Inspect duplicate block reports and repeated addStoredBlock events; verify whether the replica has already been registered in NameNode metadata; refresh block mappings through approved interfaces; suppress duplicate update handling when safe; escalate persistent metadata inconsistency. |
| Delete a block that no longer exists on data node | ⬆ | Compare the deletion request with the local DataNode block state; refresh DataNode block reports and NameNode metadata views; reconcile stale deletion requests through approved metadata update procedures; escalate repeated stale deletion events that indicate NameNode/DataNode state divergence. |
| Empty packet for block | ⬆ | Inspect block transfer logs, client status, and network conditions; determine whether the event is caused by timeout, transfer interruption, or client disconnect; restart or retry the block transfer after the connection and pipeline state are recovered; escalate repeated transfer failures. |
| Receive block exception | ⬆ | Collect receiver DataNode logs, disk I/O status, permissions, and network diagnostics; identify storage, permission, or communication causes; execute safe node-level recovery or retry actions when permitted; escalate failures requiring manual intervention. |
| Replication Monitor timeout | 🔒 | Check under-replicated block queues, NameNode workload, and live DataNode status; inspect delayed or blocked replication tasks; trigger approved rebalancing or restart delayed replication tasks when safe; submit administrator approval for service-level recovery operations. |
| **Unknown Anomaly Types** | 👤 | Flag the sequence as an unknown anomaly; preserve the full abnormal log sequence, retrieved evidence, and system context; notify engineers or domain experts for manual investigation. **No automated mitigation.** |

Meta-Llama-3.1-8B-Instruct and gemma-2-9b-it are gated on Hugging Face: accept their licenses and run `hf auth login` first. The open-set test set (4,124 unique abnormal sequences) and the knowledge base (top-100 sequences per known type) are bundled in `data/HDFS/open_set/`; rebuild them from loghub's `HDFS_v1/preprocessed/Event_traces.csv` with `python -m incident_response.data_prep`.

```bash
conda activate cesal-cloud
python run.py classify                          # evaluate all four LLMs in configs/llm/hdfs.yaml
python run.py classify qwen2.5-14b-instruct     # CESAL's default backbone only
python -m incident_response.workflows --label "Replica immediately deleted"
python -m incident_response.workflows --results outputs/hdfs/llm/results_Qwen_Qwen2.5-14B-Instruct.csv
```

Results go to `outputs/hdfs/llm/` (per-sequence predictions with retrieval evidence, `model_summary.csv`, `per_class_metrics_long.csv`); compare them with `outputs/hdfs/llm/table7_reference_metrics.csv`. Settings that differ per backbone (Qwen2.5-14B uses a length-aware retrieval score, lower thresholds, and 32 new tokens) are in `model_overrides` of `configs/llm/hdfs.yaml`.

#### From detection to response

`python run.py respond [MODEL]` connects the collaborative LAD pipeline to the module. It reads the outputs of `python run.py infer hdfs`, marks a test session as detected when CESAL's final prediction (edge Q-BAT, with routed events replaced by cloud BAT, before point adjustment) flags any of its events, and writes one incident record per detected session to `outputs/hdfs/llm/queues/`: `queue_cloud.csv` when cloud BAT verified an anomalous event, `queue_edge.csv` otherwise. It then classifies every queued sequence with `respond_model` (default Qwen2.5-14B-Instruct), attaches the selected workflow, and scores detected abnormal sessions whose anomaly type is known.

```bash
conda activate cesal-edge
python run.py infer hdfs      # detection outputs (skip if outputs/hdfs/*.npy already exist)
conda activate cesal-cloud
python run.py respond         # queues → classification → workflows
```

The LAD test data carries no block IDs or timestamps, so incident records are identified by session index. Its HDFS sessions come from a different log-key extraction than loghub's `Event_traces.csv`: most exception events (e.g. E7) are absent, so about 20% of abnormal sessions do not match a knowledge-base or test sequence exactly, and those sessions are left out of the classification score. Classifications are cached per unique sequence in `classified_<model>.csv`; re-running resumes where an interrupted run stopped.

---

## Advanced Options

### Cloud-side re-check only

If the edge phase has already been run and you only want to re-run the cloud-side BAT re-prediction step:

```bash
conda activate cesal-cloud
python dashboard/cloud_runner.py --config configs/inference/os.yaml
```

### Train from scratch

```bash
conda activate cesal-cloud
python run.py train os
python run.py train hdfs
```

Each `train` invocation runs a hyperparameter sweep over `(num_epochs, k, e_layer_num, batch_size)` and writes **81 BAT checkpoints** to `checkpoints/bat/<dataset>/`.

### Convert to edge models

```bash
conda activate cesal-edge
python run.py convert os
python run.py convert hdfs
```

Applies quantization techniques and exports `.pte` files to `checkpoints/qbat/{dataset}/`. Skip if you already downloaded Q-BAT checkpoints via `python run.py download <dataset> qbat`.

---

## Results

Numbers from the paper (Section 4), which also reports baselines, the BAT ensemble-size analysis, routing ablations, and edge resource measurements on Raspberry Pi 3B+, 4B, and 5.

### Hardware setup (Section 4.1)

The experimental environment spans a cloud platform, log collection and processing servers, and edge devices:

| Platform | Hardware profile | Operating system |
| --- | --- | --- |
| **Talon cluster node** | 2 × 18-core Intel Xeon Gold 6140; 8 × NVIDIA Tesla V100; 1.5 TB memory | Red Hat Enterprise Linux 9.2 |
| **Dell PowerEdge R650** | 36-core Intel Xeon Platinum; Mellanox ConnectX-6 100 Gb NIC; 256 GB memory | Ubuntu 24.04.2 LTS |
| **Data processing server** | Intel Core i7-14700 (28 cores / 56 threads); NVIDIA RTX 2000 Ada Generation; 32 GB memory | Windows 11 |
| **Data analytics server** | Intel Core i7-14700 (28 cores / 56 threads); NVIDIA RTX 2000 Ada Generation; 32 GB memory | Ubuntu 24.04.2 LTS |
| **Raspberry Pi 5** | Cortex-A76, 4 cores / 4 threads; 8 GB memory | Ubuntu 20.04.5 LTS |
| **Raspberry Pi 4B** | Cortex-A72, 4 cores / 4 threads; 8 GB memory | Ubuntu 20.04.5 LTS |
| **Raspberry Pi 3B+** | Cortex-A53, 4 cores / 4 threads; 1 GB memory | Ubuntu 20.04.5 LTS |

The Talon high-performance computing cluster handles BAT training and cloud-side verification; the Dell PowerEdge R650 handles log collection and storage; the two i7 servers support data processing, EM-AT analysis, Q-BAT preparation and model conversion; and edge deployment is evaluated on the Raspberry Pi cluster.

Reproducing the artifact needs far less than this — see [REQUIREMENTS.md](REQUIREMENTS.md). Only Table 6 (edge resource consumption) requires the physical Raspberry Pi devices.

### Log-based incident detection (Table 3)

| Method                              | HDFS P | HDFS R | HDFS F1 | OpenStack P | OpenStack R | OpenStack F1 |
| ----------------------------------- | -----: | -----: | ------: | ----------: | ----------: | -----------: |
| Q-BAT (edge only)                   |  99.06 | 100.00 |   99.53 |       98.09 |      100.00 |        99.03 |
| BAT (cloud only)                    |  99.99 | 100.00 |   99.99 |       99.99 |      100.00 |        99.99 |
| **CESAL** (10% routed to the cloud) |  99.96 | 100.00 |   99.98 |       99.90 |      100.00 |        99.95 |

> **Evaluation protocol.** These are **point-adjusted** scores, following the Anomaly Transformer convention this code inherits: [`training_pipeline/solver.py`](training_pipeline/solver.py#L394-L412) marks an entire ground-truth anomaly segment as detected once any single line inside that segment is detected. Both test splits are assembled as all-normal lines followed by all-abnormal lines ([`cesal_core/data/loaders.py:45-53`](cesal_core/data/loaders.py#L45-L53)), so each test set contains exactly **one** anomaly segment. [`artifact/claims/claim1_detection.sh`](artifact/claims/claim1_detection.sh) reports raw and point-adjusted scores side by side so the effect of the protocol is explicit.
>
> The per-model vote arrays `outputs/<dataset>/edge_preds_per_model.npy` and `cloud_preds_per_model.npy` are **recorded results downloaded with the other assets**, not regenerated by the pipeline — no code in this repository writes them. The dashboard reads them to display per-model votes.

### Open-set incident classification on HDFS (Table 7, macro average)

| LLM backbone             | Precision | Recall |    F1 |
| ------------------------ | --------: | -----: | ----: |
| Llama-3.1-8B-Instruct    |     77.36 |  91.53 | 81.19 |
| Gemma-2-9B-IT            |     78.21 |  91.21 | 81.71 |
| Qwen2.5-7B-Instruct      |     78.05 |  91.23 | 81.52 |
| **Qwen2.5-14B-Instruct** |     79.71 |  92.07 | 83.03 |

Per-class values are in [outputs/hdfs/llm/table7_reference_metrics.csv](outputs/hdfs/llm/table7_reference_metrics.csv).

### Sample inference output (edge-only vs. collaborative)

OpenStack:

<p align="center">
  <img src="pictures/openstack_results.png" width="700">
</p>

HDFS:

<p align="center">
  <img src="pictures/hdfs_results.png" width="700">
</p>

---

## Artifact evaluation

CESAL is packaged for artifact evaluation. Start with [INSTALL.md](INSTALL.md).

| Document | Purpose |
| --- | --- |
| [install.sh](install.sh) | Scripted install of both environments, ending in the test suite |
| [INSTALL.md](INSTALL.md) | Step-by-step install and reproduction |
| [REQUIREMENTS.md](REQUIREMENTS.md) | Hardware, software, disk budget, expected runtimes |
| [STATUS.md](STATUS.md) | Badges claimed, evaluation protocol, known issues |
| [metadata.toml](metadata.toml) | Artifact metadata (pre-filled; regenerate with [artmeta](https://github.com/jelenamirkovic/artmeta) before submitting) |
| [CITATION.cff](CITATION.cff) | How to cite CESAL |

### Claim → script mapping

| Claim | Paper | Script | Runtime |
| --- | --- | --- | --- |
| Log-based incident detection | Sec. 4.2, Table 3 | [`claim1_detection.sh`](artifact/claims/claim1_detection.sh) | OpenStack minutes; HDFS long (see below) |
| Open-set incident classification | Sec. 4.6, Table 7 | [`claim2_classification.sh`](artifact/claims/claim2_classification.sh) | 1–3 h per backbone, GPU |
| Controlled response workflows | Sec. 3.7, Table 1 | [`claim3_response.sh`](artifact/claims/claim3_response.sh) | seconds, no GPU |
| Scaled-down end-to-end run | — | [`claim4_scaled_down.sh`](artifact/claims/claim4_scaled_down.sh) | **~2 min** |

Quickest way to confirm a working install, exercising all four pipeline stages:

```bash
./artifact/claims/claim4_scaled_down.sh
```

> **Runtime note.** A full HDFS edge pass scores 221,540 windows with 3 Q-BAT models. When the ExecuTorch **Python bindings** are unavailable, the edge stage shells out to the pre-built C++ `executor_runner` once per window per model, which extrapolates to roughly a day. OpenStack is far smaller. Use the scaled-down script to exercise the same code path in about two minutes.

### Unit tests

```bash
conda activate cesal-edge
python -m pytest tests -q        # 24 passed
```
