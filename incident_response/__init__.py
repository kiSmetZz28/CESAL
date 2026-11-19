"""LLM-based open-set incident classification and controlled response (paper Sec. 3.7).

Detected abnormal HDFS log sequences are classified by a RAG-enhanced LLM into one of 10
known anomaly types or "Other anomaly type", and each label is mapped to its response
workflow (paper Table 1).

Entry points (from project root):
  python -m incident_response.data_prep                                     # open-set test set + RAG knowledge base
  python -m incident_response.evaluate --config configs/llm/hdfs.yaml       # LLM backbones (Table 7)
  python -m incident_response.workflows --label "<predicted label>"         # selected response workflow
  python -m incident_response.queues --config configs/llm/hdfs.yaml         # HDFS detections -> anomaly queues
  python -m incident_response.process_queues --config configs/llm/hdfs.yaml # classify queued incidents + workflows
"""
