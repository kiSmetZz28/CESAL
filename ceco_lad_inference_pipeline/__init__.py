"""Inference pipeline — collaborative edge+cloud anomaly detection.

Q-BAT (edge) and BAT (cloud) run in separate environments and are
orchestrated by ceco_lad_inference_pipeline.run.  Thresholds must be pre-computed
by the training/evaluation pipeline before inference is started.

Entry point (run from project root)
-------------------------------------
python -m ceco_lad_inference_pipeline.run --config configs/inference/bgl.yaml

Or via run.sh:
  bash run.sh infer bgl

Pipeline stages
---------------
  1. edge   -- Q-BAT test inference: load pre-computed thresholds, score
               test windows, majority-vote → edge predictions
  2. route  -- Mahalanobis routing: select uncertain windows for cloud
  3. cloud  -- BAT ensemble re-predicts routed windows (optional)
  4. hybrid -- Merge edge + cloud predictions, evaluate final metrics
"""
