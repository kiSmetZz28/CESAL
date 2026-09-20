"""Explicit RNG control and provenance for new training/export runs."""
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch


def model_seed(seed, dataset, parameters):
    """Stable per-learner seed, independent of sweep order and Python hashing."""
    payload = json.dumps([int(seed), dataset, list(parameters)], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], 'big')


def seed_training(seed):
    """Require deterministic kernels; fail rather than silently relax them."""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def environment():
    packages = {}
    for name in ('torch', 'numpy', 'scikit-learn', 'executorch', 'torchao'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages=packages, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                threads=torch.get_num_threads(),
                cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                deterministic=torch.are_deterministic_algorithms_enabled())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)
