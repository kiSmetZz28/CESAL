"""Training pipeline — BAT ensemble training, evaluation, and model export.

Entry points (run from project root)
-------------------------------------
python -m training_pipeline.train    --config configs/training/os.yaml
python -m training_pipeline.evaluate --config configs/training/os.yaml
python -m training_pipeline.export   --dataset os

Or via run.sh / Makefile:
  bash run.sh train os
  make train-os
"""
