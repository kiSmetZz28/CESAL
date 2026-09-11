"""SQLite database layer for the CECO-LAD dashboard."""
import asyncio
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "ceco_lad.db"

# BGL raw-log line counts from the pre-split files (LOG_ROOT/BGL/split/).
# Used to convert a global line_number back to a per-split local index.
BGL_N_TRAIN_LINES       = 3_439_321   # lines in bgl_train.log
BGL_N_TEST_NORMAL_LINES = 925_712     # lines in bgl_test_normal.log
# Derived: test-abnormal lines start at BGL_N_TRAIN_LINES + BGL_N_TEST_NORMAL_LINES

# Legacy alias kept for any callers that reference it directly.
BGL_N_TRAIN = BGL_N_TRAIN_LINES // 100   # ≈ 34393 windows

# Approximate raw-log lines per session for each BGL split — same role as
# OS_RAW_PER_SESSION for OpenStack.  Computed as log_lines / txt_sessions.
BGL_LINES_PER_SESSION = {
    "bgl_train":        41,   # 3_439_321 / 83_359
    "bgl_test_normal":  44,   # 925_712   / 20_840
    "bgl_test_abnormal": 27,  # 348_460   / 13_042
}
# Number of normal test sessions (= lines in bgl_test_normal.txt).
# test_abnormal sessions start at this index in the combined test window space.
BGL_N_TEST_NORMAL_SESSIONS = 20_840

# Pipeline event-space boundaries for BGL (from outputs/bgl/ground_truth.npy).
# Each raw-log LINE maps proportionally to one npy entry — no session grouping.
#   test_normal  line k  →  npy[ k * N_NORMAL_EVENTS  / N_TEST_NORMAL_LINES  ]
#   test_abnormal line k →  npy[ N_NORMAL_EVENTS + k * N_ABNORMAL_EVENTS / N_TEST_ABNORMAL_LINES ]
BGL_N_TEST_ABNORMAL_LINES = 348_460   # lines in bgl_test_abnormal.log
BGL_N_NORMAL_EVENTS       = 854_957   # events from bgl_test_normal  (= first npy entries)
BGL_N_ABNORMAL_EVENTS     = 419_143   # events from bgl_test_abnormal (= last  npy entries)

# HDFS raw-log line counts from the pre-split files at LOG_ROOT.
# Line numbers are continuous across all three files (train first, then test).
HDFS_N_TRAIN_LINES         = 95_125        # lines in train.log
HDFS_N_TEST_NORMAL_LINES   = 10_792_213    # lines in test_normal.log (actual stored count; one blank line skipped during ingest)
HDFS_N_TEST_ABNORMAL_LINES = 284_786       # lines in test_abnormal.log

# Pipeline event-space boundaries for OpenStack (from outputs/os/ground_truth.npy).
# npy is per-line: test_normal lines first, then test_abnormal.
OS_N_NORMAL_EVENTS       = 136_913  # npy entries from test_normal
OS_N_ABNORMAL_EVENTS     = 18_387   # npy entries from test_abnormal
OS_N_TEST_NORMAL_LINES   = 137_074  # raw log lines in test_normal block (DB count)
OS_N_TEST_ABNORMAL_LINES = 18_434   # raw log lines in test_abnormal block (DB count)
# Legacy session-level constants kept for reference
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
            for ds in ("bgl", "hdfs", "os"):
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
    dataset: str = "bgl", page: int = 0, per_page: int = 100,
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
    dataset: str = "bgl", split: str = "test", page: int = 0,
    per_page: int = 50, label: str = "",
) -> dict:
    return await asyncio.to_thread(
        _sync_query_windows, dataset, split, page, per_page, label
    )


# ── OS Pipeline view ──────────────────────────────────────────────────────────

# Approximate raw log lines per session for each OS source file
# Computed from: file_line_count / session_count
# normal1.log: 52312 lines / 386 sessions ≈ 136
# normal2.log: 137074 lines / 1248 sessions ≈ 110
# abnormal.log: 18434 lines / 138 sessions ≈ 134
OS_RAW_PER_SESSION = {
    "train_normal":  136,
    "test_normal":   110,
    "test_abnormal": 134,
}


def _sync_get_os_session(session_idx: int, split: str) -> dict:
    """
    Return the processed window and corresponding raw log lines for one
    OpenStack session.

    Raw log mapping strategy:
      Each source file is stored in raw_logs with a consistent block_id tag.
      Lines are ordered by global line_number within each block_id group.
      Session i within that group starts at approximately OFFSET = i * lines_per.
      We use LIMIT/OFFSET rather than line_number ranges to avoid global-offset
      confusion (test_normal lines start at global line 52312, not 0).
    """
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

        # Local index within the source file group
        #   train_normal:  session_idx is already local (0-based within train.txt)
        #   test_normal:   session_idx 0-1247 → local 0-1247 (within test_normal.txt)
        #   test_abnormal: session_idx 1248-1385 → local 0-137 (within test_abnormal.txt)
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
        if dataset == "bgl":
            # Mirror OpenStack's OFFSET-based approach: use block_id + OFFSET
            # within that block so we don't need exact line_number boundaries.
            if split == "train":
                blk = "bgl_train"
                lines_per = BGL_LINES_PER_SESSION.get(blk, 41)
                local_idx = window_index
            elif window_index >= BGL_N_TEST_NORMAL_SESSIONS:
                blk = "bgl_test_abnormal"
                lines_per = BGL_LINES_PER_SESSION.get(blk, 27)
                local_idx = window_index - BGL_N_TEST_NORMAL_SESSIONS
            else:
                blk = "bgl_test_normal"
                lines_per = BGL_LINES_PER_SESSION.get(blk, 44)
                local_idx = window_index
            offset = max(0, local_idx * lines_per)
            logs = c.execute(
                "SELECT line_number, label, timestamp, component, level, content "
                "FROM raw_logs "
                "WHERE dataset='bgl' AND block_id=? "
                "ORDER BY line_number LIMIT ? OFFSET ?",
                (blk, lines_per + 10, offset),
            ).fetchall()
        elif d.get("block_id"):
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
        if dataset == "bgl":
            # Use block_id (set by the split-file ingest) — same pattern as OS.
            if split == "train":
                where = ["dataset='bgl'", "block_id='bgl_train'"]
            else:
                where = ["dataset='bgl'",
                         "block_id IN ('bgl_test_normal','bgl_test_abnormal')"]
            params: list = []
        elif dataset == "hdfs":
            # line_numbers: 0..TRAIN-1 = train, TRAIN..TRAIN+TEST_NORM-1 = test_normal,
            # TRAIN+TEST_NORM.. = test_abnormal
            if split == "train":
                where  = ["dataset='hdfs'",
                          f"line_number < {HDFS_N_TRAIN_LINES}"]
            else:
                where  = ["dataset='hdfs'",
                          f"line_number >= {HDFS_N_TRAIN_LINES}"]
            params = []
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

        # Build ORDER BY for test views:
        #   test-all:    interesting first (0), normal middle (1), anomaly last (2)
        #   test-normal: interesting first (0), rest by line_number — matches normal
        #                part of test-all ordering
        #   test-anomaly / train: plain line_number ASC
        # test-all and test-normal use identical priority logic for normal lines so
        # the first pages of both views show the same rows in the same order.
        #
        #   0 = interesting normal line (disagree edge models, not label=1)
        #   1 = non-interesting normal line
        #   2 = anomaly line (label=1) — always at the end in test-all
        #
        # In test-normal (label="0") only tiers 0 and 1 exist, which exactly
        # matches what you'd see filtering test-all to normal lines.
        _SEL = (
            "SELECT line_number, label, timestamp, component, level, "
            "CASE WHEN length(content)>300 THEN substr(content,1,300)||'…' "
            "     ELSE content END AS content, block_id "
            "FROM raw_logs"
        )

        # ── Padding-zone SQL expression ────────────────────────────────────────
        # Returns 1 when an entry is in the first _CTX (=10) positions of its
        # sub-split file, meaning it sits at a session start and always has
        # NO_EVENT padding in its forward sequence.  Used as secondary sort so
        # these entries are pushed to the back of every tier while all other
        # interesting-case ordering is preserved.
        _CTX = 10
        if dataset == "bgl":
            _tn0 = BGL_N_TRAIN_LINES                             # 3_439_321
            _ta0 = BGL_N_TRAIN_LINES + BGL_N_TEST_NORMAL_LINES  # 4_365_033
            _pad = (f"CASE WHEN line_number BETWEEN {_tn0} AND {_tn0+_CTX-1} THEN 1 "
                    f"     WHEN line_number BETWEEN {_ta0} AND {_ta0+_CTX-1} THEN 1 "
                    f"     ELSE 0 END")
        elif dataset == "hdfs":
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

        # ── Two-query path: ALL datasets, test split, with interesting lines ───
        # Solves two problems at once:
        #  1. HDFS (11 M rows): CASE ORDER BY triggers a full filesort (minutes).
        #  2. All datasets: SQL ORDER BY within "interesting" tier uses
        #     line_number ASC, silently discarding the has_padding sort already
        #     applied to _interesting_lines.  Two-query restores that order so
        #     complete-context cases truly come first.
        # Layout:  interesting lines (no-padding first per _interesting_lines)
        #       →  remaining rows: non-interesting normal (pad-back, ln ASC)
        #                       then anomaly (pad-back, ln ASC)
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
                # Query 2: remaining rows — exclude interesting set, then:
                #   - for test-all (label=""): normal first, anomaly last within
                #     each group → pad-back secondary sort preserves context quality
                #   - for test-normal (label="0"): only normal rows, pad-back
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

        # ── HDFS test-anomaly fast path ───────────────────────────────────────
        # The standard CASE-expression sort triggers a filesort of ~284k rows.
        # Two range queries on idx_rl_lbl_ln (dataset, label, line_number) avoid
        # any filesort: non-padded rows served first in line_number order, then
        # the ≤10 padded rows appended at the back.
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

        # ── Standard path: train, filtered/search views, and test-anomaly ─────
        # Add _pad as secondary sort for test views so first-session padding
        # entries are pushed to the back without changing the primary tier order.
        if split == "test" and not label:
            # test-all without interesting_lines: anomaly last, padding back
            priority = f"CASE WHEN label='1' THEN 1 ELSE 0 END ASC, {_pad} ASC"
        elif split == "test" and label == "1":
            # test-anomaly: only anomaly rows — push padding zone to back
            priority = f"{_pad} ASC"
        else:
            priority = None

        if dataset == "bgl" and split == "test":
            bgl_fix = "CASE WHEN line_number IN (3439321,3439322) THEN 1 ELSE 0 END ASC"
            order = f"{priority+', ' if priority else ''}{bgl_fix}, line_number ASC"
        else:
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
        if dataset == "bgl":
            where  = "dataset='bgl' AND block_id IN ('bgl_test_normal','bgl_test_abnormal')"
            order  = "CASE WHEN line_number IN (3439321,3439322) THEN 1 ELSE 0 END ASC, line_number ASC"
        elif dataset == "hdfs":
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
