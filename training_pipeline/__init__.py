"""Training pipeline — BAT ensemble training and point-adjusted evaluation.

Entry points (run from project root)
-------------------------------------
python -m training_pipeline.train    --config configs/training/os.yaml
python -m training_pipeline.evaluate --config configs/training/os.yaml

Or via the main CLI:
  python run.py train os
  python run.py eval os majority

Quantization and export are handled separately by quantization.qbat_export:
  python run.py convert os
"""
