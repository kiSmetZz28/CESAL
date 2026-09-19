"""Inference pipeline — collaborative edge+cloud anomaly detection.

Q-BAT (edge) and BAT (cloud) run in separate environments and are
orchestrated by cesal_inference_pipeline.run. Published calibration thresholds
are bundled for both datasets; training/evaluation can recalibrate them.

Entry point (run from project root)
-------------------------------------
python -m cesal_inference_pipeline.run --config configs/inference/os.yaml

Or via the main CLI:
  python run.py infer os

Pipeline stages
---------------
  1. edge   -- Q-BAT test inference: load pre-computed thresholds, score
               test windows, majority-vote → edge predictions
  2. route  -- Mahalanobis routing: select event vectors for cloud
  3. cloud  -- BAT scores routed events packed into complete windows
  4. hybrid -- Merge edge + cloud predictions, report point-adjusted metrics
"""
