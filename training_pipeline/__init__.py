"""Training pipeline: BAT ensemble training and evaluation.

  python run.py train os          # python -m training_pipeline.train --config configs/training/os.yaml
  python run.py eval os majority  # python -m training_pipeline.evaluate --config configs/training/os.yaml

Quantization and export: quantization.qbat_export (python run.py convert os).
"""
