"""Build the HDFS open-set classification inputs from loghub's HDFS_v1 Event_traces.csv.

Outputs (default: data/HDFS/open_set/):
  open_set_test.csv                    every unique (Type, template sequence) of abnormal blocks;
                                       types outside the 10 known ones keep a blank label and are
                                       evaluated as the open-set "Other anomaly type" class
  top100_split_template_sequences.csv  RAG knowledge base: the 100 most frequent template sequences
                                       of each of the 10 most frequent anomaly types

Usage (from project root):
  python -m incident_response.data_prep
  python -m incident_response.data_prep --event_traces /path/to/Event_traces.csv --top_n 100
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from incident_response.classifier import TYPE_TO_TEXT

LOG_ROOT = Path(os.environ.get("CESAL_LOG_ROOT", Path.home() / "Desktop" / "Log Data"))
DEFAULT_EVENT_TRACES = LOG_ROOT / "HDFS_v1" / "preprocessed" / "Event_traces.csv"


def _abnormal_rows(traces: pd.DataFrame) -> pd.DataFrame:
    """Abnormal blocks (Type is set) with integer Type IDs."""
    df = traces[traces["Type"].notna()].copy()
    df["Type"] = df["Type"].astype(int)
    return df


def build_open_set_test(traces: pd.DataFrame) -> pd.DataFrame:
    df = _abnormal_rows(traces)
    df = df[df["Features"].notna()].copy()
    # Known 10 anomaly types → label text; every other type → blank label (open-set Other)
    df["label"] = df["Type"].map(TYPE_TO_TEXT).fillna("")
    result = (
        df[["Type", "label", "Features"]]
        .drop_duplicates()
        .sort_values(["Type", "Features"])
        .reset_index(drop=True)
    )
    return result.rename(columns={"Features": "template_seq"})


def build_topn_reference(traces: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = _abnormal_rows(traces)
    top_types = set(df["Type"].value_counts().head(len(TYPE_TO_TEXT)).index)
    df = df[df["Type"].isin(top_types)].copy()

    seq_counts = df.groupby(["Type", "Features"]).size().reset_index(name="SeqCount")
    result = (
        seq_counts.sort_values(["Type", "SeqCount"], ascending=[True, False])
        .groupby("Type")
        .head(top_n)
        .copy()
    )
    result["label"] = result["Type"].map(TYPE_TO_TEXT)
    result = result.rename(columns={"Features": "template_seq"})
    return result[["Type", "label", "template_seq"]].drop_duplicates().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HDFS open-set test set and RAG knowledge base.")
    parser.add_argument("--event_traces", default=str(DEFAULT_EVENT_TRACES),
                        help="loghub HDFS_v1 preprocessed/Event_traces.csv")
    parser.add_argument("--out_dir", default=str(ROOT / "data" / "HDFS" / "open_set"))
    parser.add_argument("--top_n", type=int, default=100,
                        help="Most frequent sequences kept per known type in the knowledge base.")
    args = parser.parse_args()

    traces = pd.read_csv(args.event_traces)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = build_open_set_test(traces)
    test_path = out_dir / "open_set_test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"{test_path}: {len(test_df)} unique sequences "
          f"({(test_df['label'] == '').sum()} from {test_df.loc[test_df['label'] == '', 'Type'].nunique()} unknown types)")

    kb_df = build_topn_reference(traces, args.top_n)
    kb_path = out_dir / f"top{args.top_n}_split_template_sequences.csv"
    kb_df.to_csv(kb_path, index=False)
    print(f"{kb_path}: {len(kb_df)} reference sequences over {kb_df['Type'].nunique()} known types")


if __name__ == "__main__":
    main()
