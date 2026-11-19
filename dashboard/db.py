"""SQLite database layer for the CESAL dashboard."""
import asyncio
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "cesal.db"

# HDFS raw-log line counts from the pre-split files at LOG_ROOT.
# Line numbers are continuous across all three files (train first, then test).
HDFS_N_TRAIN_LINES         = 95_125        # lines in train.log
HDFS_N_TEST_NORMAL_LINES   = 10_792_213    # test_normal.log (one blank line skipped at ingest)
HDFS_N_TEST_ABNORMAL_LINES = 284_786       # lines in test_abnormal.log

# OpenStack event-space boundaries (outputs/os/ground_truth.npy: test_normal, then test_abnormal).
OS_N_NORMAL_EVENTS       = 136_913  # npy entries from test_normal
OS_N_ABNORMAL_EVENTS     = 18_387   # npy entries from test_abnormal
OS_N_TEST_NORMAL_LINES   = 137_074  # raw log lines in test_normal block (DB count)
OS_N_TEST_ABNORMAL_LINES = 18_434   # raw log lines in test_abnormal block (DB count)
# Legacy sequence-level constants kept for reference
OS_N_NORMAL_SESSIONS   = 1_248
OS_N_ABNORMAL_SESSIONS = 138
OS_AVG_NORMAL_EVENTS   = OS_N_NORMAL_EVENTS   // OS_N_NORMAL_SESSIONS
OS_AVG_ABNORMAL_EVENTS = OS_N_ABNORMAL_EVENTS // OS_N_ABNORMAL_SESSIONS


# ── Connection factory ────────────────────────────────────────────────────────

def _conn(fast: bool = False) -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), timeout=30.0)
    c.row_factory = sqlite3.Row
    if fast:
        c.execute("PRAGMA journal_mode=OFF")
        c.execute("PRAGMA synchronous=OFF")
    else:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA cache_size=-65536")   # 64 MB page cache
    c.execute("PRAGMA foreign_keys=OFF")
    return c


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS raw_logs (
                id          INTEGER PRIMARY KEY,
                dataset     TEXT    NOT NULL,
                line_number INTEGER NOT NULL,
                label       TEXT,
                timestamp   TEXT,
                component   TEXT,
                level       TEXT,
                content     TEXT,
                block_id    TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_rl_key
                ON raw_logs(dataset, line_number);
            CREATE INDEX IF NOT EXISTS idx_rl_ds
                ON raw_logs(dataset);
            CREATE INDEX IF NOT EXISTS idx_rl_blk
                ON raw_logs(block_id) WHERE block_id IS NOT NULL;
            -- Counting how many entries of a block precede a line is the inner
            -- loop of the pipeline line ordering. Without the line_number column
            -- here SQLite falls back to idx_rl_key and scans a huge line_number
            -- range instead of the block's few dozen rows.
            CREATE INDEX IF NOT EXISTS idx_rl_ds_blk_ln
                ON raw_logs(dataset, block_id, line_number);
            CREATE INDEX IF NOT EXISTS idx_rl_lbl
                ON raw_logs(dataset, label);
            CREATE INDEX IF NOT EXISTS idx_rl_lbl_ln
                ON raw_logs(dataset, label, line_number);

            CREATE TABLE IF NOT EXISTS windows (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset        TEXT    NOT NULL,
                split          TEXT    NOT NULL,
                window_index   INTEGER NOT NULL,
                block_id       TEXT,
                label          INTEGER,
                session_length INTEGER,
                content        TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_win_key
                ON windows(dataset, split, window_index);
            CREATE INDEX IF NOT EXISTS idx_win_ds
                ON windows(dataset, split);
            CREATE INDEX IF NOT EXISTS idx_win_blk
                ON windows(block_id) WHERE block_id IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_win_lbl
                ON windows(dataset, split, label);

            CREATE TABLE IF NOT EXISTS ingest_status (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        """)


# ── Status ────────────────────────────────────────────────────────────────────

def _sync_get_status() -> dict:
    try:
        with _conn() as c:
            kv = {r["key"]: r["value"]
                  for r in c.execute("SELECT key, value FROM ingest_status")}
            for tbl in ("raw_logs", "windows"):
                row = c.execute(f"SELECT COUNT(*) n FROM {tbl}").fetchone()
                kv[f"{tbl}_total"] = row["n"] if row else 0
            for ds in ("hdfs", "os"):
                row = c.execute(
                    "SELECT COUNT(*) n FROM raw_logs WHERE dataset=?", (ds,)
                ).fetchone()
                kv[f"raw_{ds}"] = row["n"] if row else 0
                for sp in ("train", "test"):
                    row = c.execute(
                        "SELECT COUNT(*) n FROM windows WHERE dataset=? AND split=?",
                        (ds, sp),
                    ).fetchone()
                    kv[f"win_{ds}_{sp}"] = row["n"] if row else 0
            return kv
    except Exception:
        return {}


async def get_status() -> dict:
    return await asyncio.to_thread(_sync_get_status)


# ── Raw-log queries ───────────────────────────────────────────────────────────

def _sync_query_raw_logs(
    dataset: str, page: int, per_page: int, search: str, label: str
) -> dict:
    with _conn() as c:
        where = ["dataset=?"]
        params: list = [dataset]
        if search:
            where.append("content LIKE ?")
            params.append(f"%{search}%")
        if label:
            where.append("label=?")
            params.append(label)
        w = " AND ".join(where)
        total = c.execute(
            f"SELECT COUNT(*) FROM raw_logs WHERE {w}", params
        ).fetchone()[0]
        rows = c.execute(
            f"SELECT line_number, label, timestamp, component, level, "
            f"CASE WHEN length(content)>300 THEN substr(content,1,300)||'…' "
            f"     ELSE content END AS content, block_id "
            f"FROM raw_logs WHERE {w} ORDER BY line_number LIMIT ? OFFSET ?",
            params + [per_page, page * per_page],
        ).fetchall()
        return {
            "total": total, "page": page, "per_page": per_page,
            "rows": [dict(r) for r in rows],
        }


async def query_raw_logs(
    dataset: str = "os", page: int = 0, per_page: int = 100,
    search: str = "", label: str = "",
) -> dict:
    return await asyncio.to_thread(
        _sync_query_raw_logs, dataset, page, per_page, search, label
    )


# ── Window queries ────────────────────────────────────────────────────────────

def _sync_query_windows(
    dataset: str, split: str, page: int, per_page: int, label: str
) -> dict:
    with _conn() as c:
        where = ["dataset=?", "split=?"]
        params: list = [dataset, split]
        if label != "":
            where.append("label=?")
            params.append(int(label))
        w = " AND ".join(where)
        total = c.execute(
            f"SELECT COUNT(*) FROM windows WHERE {w}", params
        ).fetchone()[0]
        rows = c.execute(
            f"SELECT id, window_index, block_id, label, session_length, "
            f"CASE WHEN length(content)>250 THEN substr(content,1,250)||'…' "
            f"     ELSE content END AS content_preview "
            f"FROM windows WHERE {w} ORDER BY window_index LIMIT ? OFFSET ?",
            params + [per_page, page * per_page],
        ).fetchall()
        return {
            "total": total, "page": page, "per_page": per_page,
            "rows": [dict(r) for r in rows],
        }


async def query_windows(
    dataset: str = "os", split: str = "test", page: int = 0,
    per_page: int = 50, label: str = "",
) -> dict:
    return await asyncio.to_thread(
        _sync_query_windows, dataset, split, page, per_page, label
    )


# ── OS Pipeline view ──────────────────────────────────────────────────────────

# Approximate raw log lines per OpenStack log sequence (file lines / sequences).
OS_RAW_PER_SESSION = {
    "train_normal":  136,
    "test_normal":   110,
    "test_abnormal": 134,
}


def _sync_get_os_session(session_idx: int, split: str) -> dict:
    """Processed window and raw log lines for one OpenStack log sequence. Sequence i
    starts near i * lines_per within its block_id (LIMIT/OFFSET per block)."""
    with _conn() as c:
        win = c.execute(
            "SELECT * FROM windows WHERE dataset='os' AND split=? AND window_index=?",
            (split, session_idx),
        ).fetchone()
        if not win:
            return {}
        d = dict(win)

        blk = d.get("block_id") or ""
        lines_per = OS_RAW_PER_SESSION.get(blk, 110)

        # Index local to the source file (test_abnormal sequences start at 1248).
        local_idx = session_idx - 1248 if blk == "test_abnormal" else session_idx
        offset    = max(0, local_idx * lines_per)

        # OFFSET is relative to rows matching block_id=blk, so no global bias
        raw_rows = c.execute(
            "SELECT line_number, label, timestamp, component, level, content "
            "FROM raw_logs "
            "WHERE dataset='os' AND block_id=? "
            "ORDER BY line_number LIMIT ? OFFSET ?",
            (blk, lines_per + 20, offset),
        ).fetchall()

        d["raw_logs"]         = [dict(r) for r in raw_rows]
        d["approx_raw_offset"] = offset
        return d


async def get_os_session(session_idx: int, split: str = "test") -> dict:
    return await asyncio.to_thread(_sync_get_os_session, session_idx, split)


def _sync_query_os_sessions(split: str, page: int, per_page: int, label: str) -> dict:
    with _conn() as c:
        where = ["dataset='os'", "split=?"]
        params: list = [split]
        if label != "":
            where.append("label=?")
            params.append(int(label))
        w = " AND ".join(where)
        total = c.execute(f"SELECT COUNT(*) FROM windows WHERE {w}", params).fetchone()[0]
        rows = c.execute(
            f"SELECT window_index, block_id, label, session_length, "
            f"content AS content_preview "
            f"FROM windows WHERE {w} ORDER BY window_index LIMIT ? OFFSET ?",
            params + [per_page, page * per_page],
        ).fetchall()
        return {
            "total": total, "page": page, "per_page": per_page,
            "rows": [dict(r) for r in rows],
        }


async def query_os_sessions(
    split: str = "test", page: int = 0, per_page: int = 50, label: str = ""
) -> dict:
    return await asyncio.to_thread(_sync_query_os_sessions, split, page, per_page, label)


def _sync_query_os_raw(source: str, page: int, per_page: int, search: str) -> dict:
    """Query raw OpenStack logs filtered by source file (block_id)."""
    with _conn() as c:
        where = ["dataset='os'", "block_id=?"]
        params: list = [source]
        if search:
            where.append("content LIKE ?")
            params.append(f"%{search}%")
        w = " AND ".join(where)
        total = c.execute(f"SELECT COUNT(*) FROM raw_logs WHERE {w}", params).fetchone()[0]
        rows = c.execute(
            f"SELECT line_number, label, timestamp, component, level, content "
            f"FROM raw_logs WHERE {w} ORDER BY line_number LIMIT ? OFFSET ?",
            params + [per_page, page * per_page],
        ).fetchall()
        return {
            "total": total, "page": page, "per_page": per_page,
            "rows": [dict(r) for r in rows],
        }


async def query_os_raw(
    source: str = "test_normal", page: int = 0,
    per_page: int = 100, search: str = "",
) -> dict:
    return await asyncio.to_thread(_sync_query_os_raw, source, page, per_page, search)


def _sync_query_os_raw_split(split: str, page: int, per_page: int, search: str) -> dict:
    """Query OS raw logs for a whole split: 'train' or 'test' (combined normal+abnormal)."""
    with _conn() as c:
        if split == "train":
            block_clause = "block_id='train_normal'"
        else:
            block_clause = "block_id IN ('test_normal','test_abnormal')"
        where = [f"dataset='os'", block_clause]
        params: list = []
        if search:
            where.append("content LIKE ?")
            params.append(f"%{search}%")
        w = " AND ".join(where)
        total = c.execute(f"SELECT COUNT(*) FROM raw_logs WHERE {w}", params).fetchone()[0]
        rows = c.execute(
            f"SELECT line_number, label, timestamp, component, level, content, block_id "
            f"FROM raw_logs WHERE {w} ORDER BY line_number LIMIT ? OFFSET ?",
            params + [per_page, page * per_page],
        ).fetchall()
        return {
            "total": total, "page": page, "per_page": per_page,
            "rows": [dict(r) for r in rows],
        }


async def query_os_raw_split(
    split: str = "test", page: int = 0, per_page: int = 100, search: str = ""
) -> dict:
    return await asyncio.to_thread(_sync_query_os_raw_split, split, page, per_page, search)


# ── Window detail (with raw-log trace) ───────────────────────────────────────

def _sync_get_window_detail(dataset: str, split: str, window_index: int) -> dict:
    with _conn() as c:
        win = c.execute(
            "SELECT * FROM windows WHERE dataset=? AND split=? AND window_index=?",
            (dataset, split, window_index),
        ).fetchone()
        if not win:
            return {}
        d = dict(win)

        # Resolve matching raw-log rows
        if d.get("block_id"):
            logs = c.execute(
                "SELECT line_number, label, timestamp, component, level, content, block_id "
                "FROM raw_logs "
                "WHERE dataset=? AND block_id=? "
                "ORDER BY line_number LIMIT 200",
                (dataset, d["block_id"]),
            ).fetchall()
        else:
            logs = []

        d["raw_logs"] = [dict(r) for r in logs]
        return d


async def get_window_detail(dataset: str, split: str, window_index: int) -> dict:
    return await asyncio.to_thread(
        _sync_get_window_detail, dataset, split, window_index
    )


# ── Pipeline raw-log query (all datasets, split-aware) ───────────────────────

def _sync_query_pipeline_raw(
    dataset: str, split: str, page: int, per_page: int, search: str, label: str = "",
    interesting_lines: "list[int]" = [],
) -> dict:
    """Return raw log lines for any dataset/split pair."""
    with _conn() as c:
        # Build WHERE clause depending on dataset split semantics
        if dataset == "hdfs":
            # line_numbers: 0..TRAIN-1 = train, TRAIN..TRAIN+TEST_NORM-1 = test_normal,
            # TRAIN+TEST_NORM.. = test_abnormal
            if split == "train":
                where  = ["dataset='hdfs'",
                          f"line_number < {HDFS_N_TRAIN_LINES}"]
            else:
                where  = ["dataset='hdfs'",
                          f"line_number >= {HDFS_N_TRAIN_LINES}"]
            params: list = []
        elif dataset == "os":
            if split == "train":
                where = ["dataset='os'", "block_id='train_normal'"]
            else:
                where = ["dataset='os'", "block_id IN ('test_normal','test_abnormal')"]
            params = []
        else:
            where = ["dataset=?"]
            params = [dataset]

        if search:
            where.append("content LIKE ?")
            params.append(f"%{search}%")

        if label == "1":
            where.append("label='1'")
        elif label == "0":
            where.append("(label='0' OR label IS NULL OR label='')")

        w = " AND ".join(where)
        total = c.execute(f"SELECT COUNT(*) FROM raw_logs WHERE {w}", params).fetchone()[0]

        # ORDER BY tiers for test views: 0 = interesting normal line (edge learners
        # disagree), 1 = other normal, 2 = anomaly (test-all only). test-normal uses the
        # same tiers, so its first pages match test-all; other views sort by line_number.
        _SEL = (
            "SELECT line_number, label, timestamp, component, level, "
            "CASE WHEN length(content)>300 THEN substr(content,1,300)||'…' "
            "     ELSE content END AS content, block_id "
            "FROM raw_logs"
        )

        # _pad = 1 for the first _CTX entries of each sub-split file (always NO_EVENT-padded);
        # a secondary sort that pushes them back within each tier.
        _CTX = 10
        if dataset == "hdfs":
            _tn0 = HDFS_N_TRAIN_LINES                                # 95_125
            _ta0 = HDFS_N_TRAIN_LINES + HDFS_N_TEST_NORMAL_LINES    # 10_887_338
            _pad = (f"CASE WHEN line_number BETWEEN {_tn0} AND {_tn0+_CTX-1} THEN 1 "
                    f"     WHEN line_number BETWEEN {_ta0} AND {_ta0+_CTX-1} THEN 1 "
                    f"     ELSE 0 END")
        elif dataset == "os":
            _tn0 = 52_312                            # after 52 312 train entries
            _ta0 = 52_312 + OS_N_TEST_NORMAL_LINES  # 189_386
            _pad = (f"CASE WHEN block_id='test_normal'   AND line_number < {_tn0+_CTX} THEN 1 "
                    f"     WHEN block_id='test_abnormal' AND line_number < {_ta0+_CTX} THEN 1 "
                    f"     ELSE 0 END")
        else:
            _pad = "0"

        # Test views with interesting lines use two queries: a CASE ORDER BY filesorts the
        # HDFS table (minutes) and would lose the padding order of _interesting_lines.
        # Layout: interesting lines, then other normal rows, then anomalies (pad-back, ln ASC).
        if (interesting_lines and split == "test" and label in ("", "0")
                and not search):
            in_vals   = ",".join(str(x) for x in interesting_lines[:500])
            ln_to_pos = {ln: i for i, ln in enumerate(interesting_lines[:500])}

            # Query 1: interesting normal rows (fast IN lookup via index)
            int_rows = c.execute(
                f"{_SEL} WHERE {w} AND line_number IN ({in_vals}) "
                f"AND (label='0' OR label IS NULL OR label='')"
            ).fetchall()
            # Restore _interesting_lines order (no-padding first, already sorted)
            int_rows = sorted(int_rows, key=lambda r: ln_to_pos.get(r[0], 9999))

            n_int = len(int_rows)
            start = page * per_page
            end   = start + per_page

            if end <= n_int:
                # Whole page fits inside interesting lines — no second query needed
                rows = int_rows[start:end]
            else:
                # Query 2: the remaining rows (normal before anomaly, padding last).
                ni_offset = max(0, start - n_int)
                ni_limit  = per_page - max(0, n_int - start)
                if label == "":
                    ni_order = (f"CASE WHEN label='1' THEN 1 ELSE 0 END ASC, "
                                f"{_pad} ASC, line_number ASC")
                else:
                    ni_order = f"{_pad} ASC, line_number ASC"
                ni_rows = c.execute(
                    f"{_SEL} WHERE {w} AND line_number NOT IN ({in_vals}) "
                    f"ORDER BY {ni_order} LIMIT ? OFFSET ?",
                    params + [ni_limit, ni_offset],
                ).fetchall()
                rows = (int_rows[start:] if start < n_int else []) + ni_rows

            return {"total": total, "page": page, "per_page": per_page,
                    "rows": [dict(r) for r in rows]}

        # HDFS test-anomaly fast path: two range queries on idx_rl_lbl_ln avoid a filesort
        # of ~284k rows (unpadded rows first, then the <= 10 padded ones).
        if dataset == "hdfs" and split == "test" and label == "1" and not search:
            _ta0_v  = HDFS_N_TRAIN_LINES + HDFS_N_TEST_NORMAL_LINES  # 10_887_338
            _ta_end = _ta0_v + _CTX - 1                               # 10_887_347
            _base   = "dataset='hdfs' AND label='1'"
            _non_q  = (f"{_SEL} WHERE {_base} AND line_number > {_ta_end}"
                       f" ORDER BY line_number ASC")
            pad_rows = c.execute(
                f"{_SEL} WHERE {_base} AND line_number BETWEEN {_ta0_v} AND {_ta_end}"
                f" ORDER BY line_number ASC"
            ).fetchall()
            n_pad = len(pad_rows)
            n_non = total - n_pad
            start = page * per_page
            end   = start + per_page
            if end <= n_non:
                rows = c.execute(_non_q + " LIMIT ? OFFSET ?",
                                 [per_page, start]).fetchall()
            elif start >= n_non:
                rows = list(pad_rows)[start - n_non : end - n_non]
            else:
                head = c.execute(_non_q + " LIMIT ? OFFSET ?",
                                 [n_non - start, start]).fetchall()
                rows = list(head) + list(pad_rows)[:end - n_non]
            return {"total": total, "page": page, "per_page": per_page,
                    "rows": [dict(r) for r in rows]}

        # Standard path; test views push the padding zone back within each tier.
        if split == "test" and not label:
            # test-all without interesting_lines: anomaly last, padding back
            priority = f"CASE WHEN label='1' THEN 1 ELSE 0 END ASC, {_pad} ASC"
        elif split == "test" and label == "1":
            # test-anomaly: only anomaly rows — push padding zone to back
            priority = f"{_pad} ASC"
        else:
            priority = None

        order = f"{priority+', ' if priority else ''}line_number ASC"

        rows = c.execute(
            f"{_SEL} WHERE {w} ORDER BY {order} LIMIT ? OFFSET ?",
            params + [per_page, page * per_page],
        ).fetchall()
        return {"total": total, "page": page, "per_page": per_page,
                "rows": [dict(r) for r in rows]}


async def query_pipeline_raw(
    dataset: str = "os", split: str = "train",
    page: int = 0, per_page: int = 100, search: str = "", label: str = "",
    interesting_lines: "list[int] | None" = None,
) -> dict:
    return await asyncio.to_thread(
        _sync_query_pipeline_raw, dataset, split, page, per_page, search, label,
        interesting_lines or [],
    )


def get_test_line_numbers(dataset: str) -> list:
    """Return all test line_numbers for a dataset in their default query order."""
    with _conn() as c:
        if dataset == "hdfs":
            where  = f"dataset='hdfs' AND line_number >= {HDFS_N_TRAIN_LINES}"
            order  = "line_number ASC"
        elif dataset == "os":
            where  = "dataset='os' AND block_id IN ('test_normal','test_abnormal')"
            order  = "line_number ASC"
        else:
            return []
        rows = c.execute(f"SELECT line_number FROM raw_logs WHERE {where} ORDER BY {order}").fetchall()
    return [r[0] for r in rows]


# ── Pipeline stats ────────────────────────────────────────────────────────────

def _sync_get_pipeline_stats(dataset: str) -> dict:
    with _conn() as c:
        raw_total = c.execute(
            "SELECT COUNT(*) n FROM raw_logs WHERE dataset=?", (dataset,)
        ).fetchone()["n"]
        raw_anom = c.execute(
            "SELECT COUNT(*) n FROM raw_logs WHERE dataset=? AND label NOT IN ('-','0','normal','INFO','')",
            (dataset,),
        ).fetchone()["n"]

        def _win(split, lbl=None):
            q = "SELECT COUNT(*) n FROM windows WHERE dataset=? AND split=?"
            p = [dataset, split]
            if lbl is not None:
                q += " AND label=?"; p.append(lbl)
            return c.execute(q, p).fetchone()["n"]

        return {
            "raw_total":    raw_total,
            "raw_anomaly":  raw_anom,
            "train_total":  _win("train"),
            "train_anomaly":_win("train", 1),
            "test_total":   _win("test"),
            "test_anomaly": _win("test", 1),
        }


async def get_pipeline_stats(dataset: str) -> dict:
    return await asyncio.to_thread(_sync_get_pipeline_stats, dataset)
