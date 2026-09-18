"""Dataset accounting for terminal/log output; never transforms model inputs."""

import os
from functools import lru_cache

from cesal_core.utils import steps


# ACSAC_2026.pdf, Section 4.1. Source-log totals and parsed-file counts are
# different stages: report both, without attributing an undocumented difference.
_PAPER = {
    'HDFS': (11_175_629, 4_855, 16_838, 'sessions'),
    'Openstack': (207_820, 52_312, 18_434, 'events'),
}


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
    st.detail('dataset reference', f'{name}, paper Section 4.1')
    st.detail('paper source logs', f'{paper_total:,} messages before the bundled parsed splits')
    for label, path, n_sessions, n_events in zip(
        ('train (normal)', 'test (normal)', 'test (abnormal)'), paths, sessions, event_counts,
    ):
        st.detail(label, f'{os.path.basename(path)}: {n_sessions:,} sessions -> {n_events:,} events')
    st.detail('bundled parsed total', f'{sum(sessions):,} sessions / {total:,} events')
    st.detail('parsed vs paper total', f'{total - paper_total:+,} events relative to the source-log total')

    observed = sessions if reference_unit == 'sessions' else event_counts
    for label, reference, count in (
        ('training count vs paper', paper_train, observed[0]),
        ('abnormal count vs paper', paper_abnormal, observed[2]),
    ):
        comparison = 'matches' if count == reference else f'difference {count - reference:+,}'
        st.detail(label, f'{count:,} loaded / {reference:,} paper {reference_unit} ({comparison})')

    st.detail('context construction', f'1 row/event; {context_length} preceding events within its session')
    st.detail('short session history', 'NO_EVENT padding keeps every event; no rows removed')
    st.detail('train context matrix', f'[{train:,}, {context_length}] (events, context features)')
    st.detail('test concatenation', f'{normal:,} normal + {abnormal:,} abnormal = {normal + abnormal:,} events')
    st.detail('test context matrix', f'[{normal + abnormal:,}, {context_length}]; labels 0=normal, 1=abnormal')
    st.detail('standardization', f'fit {context_length} feature means/stds on {train:,} training rows only')
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

    st.detail(f'{mode} windows', f'{windows:,} x [{ds.win_size}, {features}]; stride {stride}')
    st.detail(f'{mode} event coverage', f'{covered:,} / {n_rows:,} rows; {n_rows - last_end:,} unused tail rows')
    if gaps:
        st.detail(f'{mode} stride gaps', f'{gaps:,} rows between windows are unused')
    if windows and stride < ds.win_size:
        st.detail(f'{mode} overlap', f'{windows * ds.win_size - covered:,} repeated row appearances')
    st.detail('window construction', 'slice concatenated context rows; keep full windows only, no window padding')
    if batch_size is not None:
        batches = (windows + batch_size - 1) // batch_size
        last_batch = windows - (batches - 1) * batch_size if batches else 0
        st.detail(f'{mode} loader batches', f'{batches:,}; up to {batch_size} windows/batch; last batch {last_batch}')
        st.detail(f'{mode} tensor layout', f'[batch, {ds.win_size}, {features}], float32')
