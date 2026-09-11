"""Convert trained EMAT checkpoints to quantized ExecuTorch .pte files (Q-BAT).

Each .pte file is a self-contained, A8W4-quantized model that runs on the
ExecuTorch CPU backend. The exported model takes a single window tensor of
shape [1, win_size, input_c] and returns a 1-D energy tensor of length win_size.

Usage — convert one specific model:
    python quantization/qbat_export.py --config configs/training/bgl.yaml \\
        --num_epochs 3 --k 3 --e_layer_num 3 --batch_size 32

Usage — convert every combination in the sweep (all Q-BAT models):
    python quantization/qbat_export.py --config configs/training/bgl.yaml --all
"""
import argparse
import os
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import export, ExportedProgram
from executorch.exir import EdgeProgramManager, ExecutorchBackendConfig, to_edge
from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

from ceco_core.models.EMAT import EMAT
from ceco_core.utils.energy import my_kl_loss
from ceco_core.utils.config import load_config
from ceco_core.utils.io import mkdir


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
) -> None:
    """Export and quantize one EMAT .pth checkpoint into a .pte file."""
    device = torch.device("cpu")
    dtype = torch.float32

    emat = EMAT(win_size=win_size, enc_in=input_c, c_out=output_c, e_layers=e_layer_num)
    emat.load_state_dict(torch.load(src_ckpt, map_location=device, weights_only=True))

    model = _ExportableEMAT(emat, win_size=win_size)
    model.eval().to(dtype=dtype, device=device)

    # A8W4: int8 dynamic activations, int4 weights (ExecuTorch 0.3 API)
    model = Int8DynActInt4WeightQuantizer(
        precision=dtype,
        groupsize=-1,
    ).quantize(model)

    sample_input = (torch.randn(1, win_size, input_c, dtype=dtype, device=device),)

    exported: ExportedProgram = export(model, sample_input, strict=True)
    edge: EdgeProgramManager = to_edge(exported)
    et_program = edge.to_executorch(ExecutorchBackendConfig(passes=[]))

    mkdir(os.path.dirname(dst_pte))
    with open(dst_pte, "wb") as f:
        f.write(et_program.buffer)
    print(f"  Saved: {dst_pte}")


if __name__ == "__main__":
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

    print(f"Converting {len(combos)} model(s) — dataset: {dataset}")
    skipped = 0
    for num_epochs, k, e_layers, batch_size in combos:
        fileparam = f"e{num_epochs}_k{k}_l{e_layers}_b{batch_size}"
        src = os.path.join(src_dir,  f"{dataset}_{fileparam}_checkpoint.pth")
        dst = os.path.join(dst_dir,  f"{dataset}_{fileparam}.pte")

        if not os.path.exists(src):
            print(f"  Skipping (checkpoint not found): {src}")
            skipped += 1
            continue

        print(f"  [{fileparam}] ...")
        convert_one(src, dst, win_size, input_c, output_c, e_layers)

    converted = len(combos) - skipped
    print(f"\nDone: {converted} converted, {skipped} skipped.")
