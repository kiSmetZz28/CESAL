"""Convert trained EMAT checkpoints to quantized ExecuTorch .pte files (Q-BAT).

Each .pte file is a self-contained, A8W4-quantized model that runs on the
ExecuTorch CPU backend. The exported model takes a single window tensor of
shape [1, win_size, input_c] and returns a 1-D energy tensor of length win_size.

Usage — convert one specific model:
    python quantization/qbat_export.py --config configs/training/os.yaml \\
        --num_epochs 3 --k 3 --e_layer_num 3 --batch_size 32

Usage — convert every combination in the sweep (all Q-BAT models):
    python quantization/qbat_export.py --config configs/training/os.yaml --all
"""
import argparse
import logging
import os
import sys
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import export, ExportedProgram
from executorch.exir import EdgeProgramManager, ExecutorchBackendConfig, to_edge
from torchao.quantization import quantize_, int8_dynamic_activation_int4_weight
from torchao.utils import unwrap_tensor_subclass

# Run directly as a script (as run.py does), so put the project root on the path
# the way the other entry points do rather than relying on `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cesal_core.models.EMAT import EMAT
from cesal_core.utils import steps
from cesal_core.utils.energy import my_kl_loss
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.io import mkdir
from cesal_core.utils.steps import StepReporter
from cesal_core.utils.reproducibility import model_seed, seed_training, environment, sha256, write_json


_ABOUT = """
Quantize the trained BAT learners and export them for the edge device.

Each selected EM-AT checkpoint is quantized to int8 activations with int4
weights and exported as an ExecuTorch program (.pte), which the edge tier
executes through the ExecuTorch C++ runtime. The cloud tier keeps the
full-precision checkpoints; only the edge tier uses these.
"""


class _ExportableEMAT(nn.Module):
    """EMAT wrapper that fuses energy scoring into the forward pass.

    Returns a 1-D float32 tensor of energy values (length win_size for
    batch=1), making the full scoring pipeline exportable as a single
    ExecuTorch program.  All operations are tensor-only — no numpy, no
    in-place Python control flow that cannot be traced.
    """

    def __init__(self, emat: nn.Module, win_size: int):
        super().__init__()
        self.model = emat
        self.win_size = win_size
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, series, prior, _ = self.model(x)
        loss = torch.mean(self.criterion(x, output), dim=-1)

        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            norm = (
                prior[u]
                / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                .repeat(1, 1, 1, self.win_size)
            )
            if u == 0:
                series_loss = my_kl_loss(series[u], norm.detach()) * 50
                prior_loss  = my_kl_loss(norm, series[u].detach()) * 50
            else:
                series_loss = series_loss + my_kl_loss(series[u], norm.detach()) * 50
                prior_loss  = prior_loss  + my_kl_loss(norm, series[u].detach()) * 50

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        energy = metric * loss
        return energy.reshape(-1)



def convert_one(
    src_ckpt: str,
    dst_pte: str,
    win_size: int,
    input_c: int,
    output_c: int,
    e_layer_num: int,
    seed: int = 42,
) -> None:
    """Export and quantize one EMAT .pth checkpoint into a .pte file.

    Reports the four sub-stages through the active step (see
    cesal_core/utils/steps.py), so a long conversion shows what it is doing
    rather than appearing to hang.
    """
    seed_training(seed)
    device = torch.device("cpu")
    dtype = torch.float32
    step = steps.current()

    step.progress_note(f"{Path(dst_pte).stem} — loading")
    emat = EMAT(win_size=win_size, enc_in=input_c, c_out=output_c, e_layers=e_layer_num)
    emat.load_state_dict(torch.load(src_ckpt, map_location=device, weights_only=True))

    model = _ExportableEMAT(emat, win_size=win_size)
    model.eval().to(dtype=dtype, device=device)

    # A8W4: int8 dynamic activations, int4 weights. This is the step that
    # shrinks the model: weights drop from 32-bit floats to 4-bit integers,
    # trading a little numeric precision for ~8x less space.
    #
    # Uses torchao's quantize_ API. The older Int8DynActInt4WeightQuantizer
    # cannot be used here: its _create_quantized_state_dict asserts
    # `not mod.bias`, and every one of EMAT's 16 Linear layers carries a bias
    # (torchao drops bias entirely — see its own "TODO: support bias?").
    step.progress_note(f"{Path(dst_pte).stem} — quantizing")
    quantize_(model, int8_dynamic_activation_int4_weight())
    # quantize_ replaces weights with tensor subclasses; torch.export needs
    # them unwrapped back into plain tensors first.
    model = unwrap_tensor_subclass(model)

    sample_input = (torch.randn(1, win_size, input_c, dtype=dtype, device=device),)

    step.progress_note(f"{Path(dst_pte).stem} — exporting")
    exported: ExportedProgram = export(model, sample_input, strict=True)
    edge: EdgeProgramManager = to_edge(exported)
    et_program = edge.to_executorch(ExecutorchBackendConfig(passes=[]))

    step.progress_note(f"{Path(dst_pte).stem} — writing")
    mkdir(os.path.dirname(dst_pte))
    with open(dst_pte, "wb") as f:
        f.write(et_program.buffer)


def _mb(path: str) -> float:
    """Size of a file in MB, or 0.0 when it is missing."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EMAT .pth checkpoints to ExecuTorch .pte files."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the training YAML config.")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--k",          type=int, default=None)
    parser.add_argument("--e_layer_num",type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to write .pte files (default: checkpoints/qbat/<dataset>).")
    parser.add_argument("--all", action="store_true",
                        help="Convert every combination in the hyperparameter sweep.")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg      = load_config(args.config)
    win_size = cfg["win_size"]
    input_c  = cfg["input_c"]
    output_c = cfg.get("output_c", input_c)
    dataset  = cfg["dataset"]
    src_dir  = cfg["model_save_path"]
    src_path = Path(src_dir)
    dst_dir  = args.output_dir or str(src_path.parent.parent / "qbat" / src_path.name)

    if args.all:
        combos = list(product(cfg["num_epochs"], cfg["k"], cfg["e_layer_num"], cfg["batch_size"]))
    else:
        combos = [(
            args.num_epochs  or cfg["num_epochs"][0],
            args.k           or cfg["k"][0],
            args.e_layer_num or cfg["e_layer_num"][0],
            args.batch_size  or cfg["batch_size"][0],
        )]

    rep = StepReporter("convert", dataset=dataset, steps=steps.CONVERT_STEPS,
                       about=_ABOUT)

    # ── Step 1: see what is available to convert ──────────────────────────
    with rep.step("locate") as st:
        st.detail("reading from", src_dir)
        st.detail("writing to", dst_dir)
        pending, missing = [], []
        for num_epochs, k, e_layers, batch_size in combos:
            fileparam = f"e{num_epochs}_k{k}_l{e_layers}_b{batch_size}"
            src = os.path.join(src_dir, f"{dataset}_{fileparam}_checkpoint.pth")
            dst = os.path.join(dst_dir, f"{dataset}_{fileparam}.pte")
            (pending if os.path.exists(src) else missing).append(
                (fileparam, src, dst, e_layers)
            )
        if missing:
            st.warn(f"{len(missing)} of {len(combos)} trained learners are not on disk "
                    f"and will be skipped — run 'train' first to create them.")
        st.outcome(**{
            "learners requested": len(combos),
            "ready to convert": len(pending),
            "missing": len(missing),
            "size on disk now": f"{sum(_mb(p[1]) for p in pending):.1f} MB",
        })

    if not pending:
        rep.skip("convert", "None of the requested learners are on disk, so there is "
                            "nothing to convert.")
        rep.finish(outputs=dst_dir)
        sys.exit(1)

    # ── Step 2: quantize and export each one ──────────────────────────────
    with rep.step("convert") as st:
        st.expect("learners", len(pending))
        converted, failed = 0, 0
        src_mb = dst_mb = 0.0

        for fileparam, src, dst, e_layers in pending:
            try:
                if os.path.exists(dst):
                    raise FileExistsError(f'Export already exists: {dst}; choose a new output directory.')
                parameters = tuple(int(part[1:]) for part in fileparam.split('_'))
                seed = model_seed(args.seed, dataset, parameters)
                convert_one(src, dst, win_size, input_c, output_c, e_layers, seed)
                write_json(dst + '.json', dict(seed=seed, master_seed=args.seed,
                    source_checkpoint=src, source_sha256=sha256(src), pte_sha256=sha256(dst),
                    exporter_sha256=sha256(__file__), environment=environment(),
                    input_shape=[1, win_size, input_c], quantization='int8_dynamic_activation_int4_weight'))
                before, after = _mb(src), _mb(dst)
                src_mb += before
                dst_mb += after
                converted += 1
                logging.info(
                    "   %-22s %6.1f MB → %5.1f MB   %3.0f%% smaller",
                    fileparam, before, after,
                    (1 - after / before) * 100 if before else 0.0,
                )
            except Exception as exc:  # one failure must not abort the batch
                logging.debug('Learner conversion failure', exc_info=True)
                failed += 1
                st.warn(f"{fileparam} could not be quantized — {exc}")
            st.tick("learners")

        st.outcome(**{
            "learners converted": f"{converted}/{len(pending)}",
            "learners failed": failed,
            "total size before": f"{src_mb:.1f} MB",
            "total size after": f"{dst_mb:.1f} MB",
            "space saved": (f"{src_mb - dst_mb:.1f} MB "
                            f"({(1 - dst_mb / src_mb) * 100:.0f}% smaller)"
                            if src_mb else "—"),
        })
        if converted == 0:
            st.fail(
                f"No model could be converted ({failed} failed). The messages "
                f"above name the cause; a mismatch between the installed torchao "
                f"and the model is the usual one."
            )
        elif failed or missing:
            st.fail(f"Conversion is incomplete: {failed} failed and {len(missing)} missing "
                    "learners. Successfully exported models have been kept.")

    rep.finish(outputs=dst_dir)
    sys.exit(0 if converted and not failed and not missing else 1)


if __name__ == "__main__":
    setup_logging("convert_qbat")
    main()
