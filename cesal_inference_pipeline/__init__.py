"""Inference pipeline: collaborative edge (Q-BAT) + cloud (BAT) anomaly detection.

1. edge   -- Q-BAT scores the test windows; majority vote gives the edge predictions
2. route  -- Mahalanobis routing selects event vectors for the cloud
3. cloud  -- BAT scores the routed events packed into complete windows
4. hybrid -- merge edge + cloud predictions and report metrics

Run: python run.py infer os
  (or python -m cesal_inference_pipeline.run --config configs/inference/os.yaml)
"""
