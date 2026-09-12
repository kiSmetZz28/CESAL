"""HDFS anomaly type label set shared by the classifier and the response workflows.

Kept free of heavy imports so that light-weight consumers (e.g. the dashboard) can map
labels to workflows without loading torch or scikit-learn.
"""

# HDFS anomaly type IDs (loghub HDFS_v1 Event_traces.csv "Type") of the 10 known classes
TYPE_TO_TEXT = {
    5: "Namenode not updated after deleting block",
    31: "Write exception client give up",
    3: "Write failed at beginning",
    0: "Replica immediately deleted",
    4: "Received block that does not belong to any file",
    1: "Redundant addStoredBlock",
    21: "Delete a block that no longer exists on data node",
    7: "Empty packet for block",
    12: "Receive block exception",
    8: "Replication Monitor timeout",
}

KNOWN_LABELS = list(TYPE_TO_TEXT.values())
OTHER_LABEL = "Other anomaly type"
ALL_EVAL_LABELS = KNOWN_LABELS + [OTHER_LABEL]
