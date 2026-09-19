"""Dataset accounting for terminal/log output; never transforms model inputs."""

import os
from functools import lru_cache

from cesal_core.utils import steps


# ACSAC_2026.pdf, Section 4.1. Source-log totals and parsed-file counts are
# different stages. OpenStack message counts use the paper references;
# generated context rows and windows are counted from the actual inputs.
_PAPER = {
    'HDFS': (11_175_629, 4_855, 16_838, 'sessions'),
    'Openstack': (207_820, 52_312, 18_434, 'log messages'),
}
# Counted in the original HDFS_v1/anomaly_label.csv: 558,223 normal + 16,838
# abnormal sessions. This is a source-dataset reference, not a paper claim or
# a runtime dependency on the optional raw-log files.
_HDFS_SOURCE_SESSIONS = 575_061


@lru_cache(maxsize=32)
def _session_count(path, mtime_ns, size):
    """Count nonempty sequence lines once, even when loading many learners.

    File metadata is part of the cache key so an edited split is recounted.
    Event counts below come from the actual context tensors, not this scan.
    """
    with open(path, 'rb') as source:
        return sum(1 for line in source if line.strip())


def report_preprocessing(dataset, paths, event_counts, context_length, mode):
    """Explain the loaded splits and the existing context/scaling operations."""
    st = steps.current()
    if st.state == 'inactive':
        return

    sessions = []
    for path in paths:
        stat = os.stat(path)
        sessions.append(_session_count(os.path.abspath(path), stat.st_mtime_ns, stat.st_size))

    name = 'OpenStack' if dataset == 'Openstack' else dataset
    paper_total, paper_train, paper_abnormal, reference_unit = _PAPER[dataset]
    train, normal, abnormal = event_counts
    total = sum(event_counts)
    sequence_unit = 'sessions' if dataset == 'HDFS' else 'sequence groups'
    st.detail('preprocessing 1/5', 'source-log reference and parsing (already prepared, not rerun)')
    st.detail('dataset reference', f'{name}, paper Section 4.1')
    st.detail('paper source logs', f'{paper_total:,} messages (paper reference, not a raw-file recount)')
    if dataset == 'Openstack':
        st.detail('paper training messages', f'{paper_train:,} normal messages')
        st.detail('paper normal test messages', f'{paper_total - paper_train - paper_abnormal:,} messages (total minus training and abnormal)')
        st.detail('paper abnormal messages', f'{paper_abnormal:,} messages')
        st.detail('runtime accounting', 'paper counts describe source messages; context rows/windows below use the current bundled inputs')
    st.detail('log parsing', 'message templates -> event IDs; this run reads the bundled parsed files')
    st.detail('preprocessing 2/5', 'log sequence generator output: one event-ID sequence per nonempty line')
    for label, path, n_sessions, n_events in zip(
        ('train (normal)', 'test (normal)', 'test (abnormal)'), paths, sessions, event_counts,
    ):
        if dataset == 'Openstack':
            st.detail(label, f'{os.path.basename(path)}: {n_sessions:,} {sequence_unit}')
        else:
            st.detail(label, f'{os.path.basename(path)}: {n_sessions:,} {sequence_unit} contain {n_events:,} event IDs')
    if dataset == 'Openstack':
        st.detail('bundled sequences', f'{sum(sessions):,} sequence groups; generated context shapes follow below')
    else:
        st.detail('bundled parsed total', f'{sum(sessions):,} {sequence_unit} / {total:,} event IDs')
        st.detail('counting units', 'one event-ID occurrence represents one parsed log message; a sequence contains multiple events')
        st.detail('parsed vs paper total', f'{total:,} parsed / {paper_total:,} paper messages; difference {total - paper_total:+,}')
        for label, reference, count in (
            ('training count vs paper', paper_train, sessions[0]),
            ('abnormal count vs paper', paper_abnormal, sessions[2]),
        ):
            comparison = 'MATCH' if count == reference else f'MISMATCH: difference {count - reference:+,}'
            st.detail(label, f'{count:,} loaded / {reference:,} paper {reference_unit} ({comparison})')
        st.detail('paper count check', 'compare HDFS training/abnormal sessions; Section 4.1 does not give a total session count')
        source_difference = sum(sessions) - _HDFS_SOURCE_SESSIONS
        comparison = 'MATCH' if source_difference == 0 else f'MISMATCH: difference {source_difference:+,}'
        st.detail('HDFS source sessions', f'{sum(sessions):,} bundled / {_HDFS_SOURCE_SESSIONS:,} original anomaly_label.csv sessions ({comparison})')
        if total != paper_total:
            st.detail('source-count difference', 'cause not established by bundled files; inputs are used as supplied, without count correction')

    st.detail('preprocessing 3/5', 'sliding context sequence generator: expand each sequence into one row per event')
    st.detail('sequence -> context rows', f'{sum(sessions):,} {sequence_unit} -> {total:,} rows x {context_length} features; no input events removed')
    st.detail('event-ID encoding', 'IDs mapped within each input file; reserved NO_EVENT supplies missing history')
    st.detail('context construction', f'1 row/event; {context_length} preceding events within its source sequence')
    st.detail('short sequence history', 'NO_EVENT padding keeps every event; no rows removed')
    st.detail('train context matrix', f'[{train:,}, {context_length}] (rows, context features)')
    st.detail('normal context matrix', f'[{normal:,}, {context_length}] -> {normal:,} normal labels')
    st.detail('abnormal context matrix', f'[{abnormal:,}, {context_length}] -> {abnormal:,} abnormal labels')
    st.detail('test concatenation', f'{normal:,} normal + {abnormal:,} abnormal = {normal + abnormal:,} context rows')
    st.detail('test context matrix', f'[{normal + abnormal:,}, {context_length}]; labels 0=normal, 1=abnormal')
    st.detail('preprocessing 4/5', 'standardization: change feature values, preserve row counts and labels')
    st.detail('standardization', f'fit {context_length} feature means/stds on {train:,} training rows only')
    st.detail('scaled matrix sizes', f'train [{train:,}, {context_length}] -> [{train:,}, {context_length}]; test [{normal + abnormal:,}, {context_length}] -> [{normal + abnormal:,}, {context_length}]')
    st.detail('test standardization', 'reuse training means/stds; row counts and labels unchanged')
    if mode == 'train':
        st.detail('training bootstrap', f'{train:,} rows -> {train:,} sampled rows, with replacement')


def report_windows(ds, batch_size):
    """Describe the exact stride, coverage and batch shape used by this loader."""
    st = steps.current()
    if st.state == 'inactive':
        return

    mode = ds.mode
    rows = ds.train if mode == 'train' else ds.val if mode == 'val' else ds.test
    stride = ds.step if mode in ('train', 'val', 'test') else ds.win_size
    n_rows, features = rows.shape
    windows = max(0, (n_rows - ds.win_size) // stride + 1)
    last_end = (windows - 1) * stride + ds.win_size if windows else 0
    covered = ds.win_size + (windows - 1) * min(stride, ds.win_size) if windows else 0
    gaps = max(0, windows - 1) * max(0, stride - ds.win_size)

    st.detail('preprocessing 5/5', 'model-window generator and batching (complete windows only)')
    st.detail(f'{mode} row -> window count', f'{n_rows:,} context rows -> {windows:,} windows; each has {ds.win_size} events x {features} features')
    st.detail(f'{mode} windows', f'{windows:,} x [{ds.win_size}, {features}]; stride {stride}')
    st.detail(f'{mode} event coverage', f'{covered:,} / {n_rows:,} rows; {n_rows - last_end:,} unused tail rows')
    if gaps:
        st.detail(f'{mode} stride gaps', f'{gaps:,} rows between windows are unused')
    if windows and stride < ds.win_size:
        st.detail(f'{mode} overlap', f'{windows * ds.win_size - covered:,} repeated row appearances')
    st.detail('window construction', 'slice concatenated context rows, possibly across sequence boundaries; no window padding')
    st.detail(f'{mode} count balance', f'{n_rows:,} rows = {covered:,} covered + {n_rows - last_end:,} unused tail + {gaps:,} stride gaps')
    if batch_size is not None:
        batches = (windows + batch_size - 1) // batch_size
        last_batch = windows - (batches - 1) * batch_size if batches else 0
        st.detail(f'{mode} loader batches', f'{batches:,}; up to {batch_size} windows/batch; last batch {last_batch}')
        st.detail(f'{mode} tensor layout', f'[batch, {ds.win_size}, {features}], float32')
