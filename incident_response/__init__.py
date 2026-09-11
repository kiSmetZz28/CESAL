"""LLM-based open-set incident classification and controlled response (paper Sec. 3.7).

Detected abnormal HDFS log sequences are classified by a RAG-enhanced LLM into one of 10 known
anomaly types or "Other anomaly type" (unknown), and each label is mapped to a predefined
response workflow (paper Table 1).

Entry points (run from project root)
-------------------------------------
python -m incident_response.data_prep                              # build open-set test set + RAG knowledge base
python -m incident_response.evaluate --config configs/llm/hdfs.yaml  # evaluate LLM backbones (Table 7)
python -m incident_response.workflows --label "<predicted label>"   # show the selected response workflow
python -m incident_response.queues --config configs/llm/hdfs.yaml         # HDFS detections → anomaly queues
python -m incident_response.process_queues --config configs/llm/hdfs.yaml # classify queued incidents + workflows

Modules
-------
  classifier     -- lexical retriever, open-set decision rules, LLM prompt and label parsing
  evaluate       -- evaluation loop, metrics, and result files
  data_prep      -- builds data/HDFS/open_set/ from loghub HDFS_v1 Event_traces.csv
  workflows      -- Table 1 response workflows and label → workflow selection
  queues         -- edge/cloud anomaly queues from the collaborative LAD outputs
  process_queues -- classifies queued incidents and attaches their response workflows
"""
