"""
Ingest log datasets into the CECO-LAD SQLite database.

Imports (in order):
  1. BGL processed windows   (train.csv + test.csv)
  2. HDFS processed windows  (train.csv + test.csv)
  3. BGL raw structured logs (BGL.log_structured.csv — all rows)
  4. HDFS raw structured logs (HDFS.log_structured.csv — test-window rows only)
  5. OpenStack raw logs       (openstack_*.log)

Run standalone:  python dashboard/ingest.py
"""
import csv
import hashlib
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import os

DB_PATH  = Path(__file__).parent / "ceco_lad.db"
LOG_ROOT = Path(os.environ.get("CECO_LOG_ROOT", Path.home() / "Desktop" / "Log Data"))

BLK_RE = re.compile(r"blk_-?\d+")
BATCH  = 25_000   # rows per executemany call


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _conn(fast: bool = False) -> sqlite3.Connection:
    # Always use WAL mode — it allows concurrent reads while bulk-writing,
    # preventing "database is locked" errors from the status-polling loop.
    c = sqlite3.connect(str(DB_PATH), timeout=60.0)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=OFF" if fast else "PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA busy_timeout=30000")   # wait up to 30 s on a locked DB
    c.execute("PRAGMA cache_size=-131072")   # 128 MB page cache
    c.execute("PRAGMA foreign_keys=OFF")
    return c


def _set_status(key: str, value: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO ingest_status(key, value) VALUES (?,?)",
            (key, value),
        )


def _insert_windows(rows: list) -> None:
    with _conn(fast=True) as c:
        c.executemany(
            "INSERT OR REPLACE INTO windows"
            "(dataset, split, window_index, block_id, label, session_length, content)"
            " VALUES (?,?,?,?,?,?,?)",
            rows,
        )


def _insert_raw(rows: list) -> None:
    with _conn(fast=True) as c:
        c.executemany(
            "INSERT OR IGNORE INTO raw_logs"
            "(dataset, line_number, label, timestamp, component, level, content, block_id)"
            " VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )


def _rebuild_indexes() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_rl_key  ON raw_logs(dataset, line_number);
            CREATE INDEX IF NOT EXISTS idx_rl_ds          ON raw_logs(dataset);
            CREATE INDEX IF NOT EXISTS idx_rl_blk         ON raw_logs(block_id) WHERE block_id IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_rl_lbl         ON raw_logs(dataset, label);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_win_key ON windows(dataset, split, window_index);
            CREATE INDEX IF NOT EXISTS idx_win_ds         ON windows(dataset, split);
            CREATE INDEX IF NOT EXISTS idx_win_blk        ON windows(block_id) WHERE block_id IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_win_lbl        ON windows(dataset, split, label);
            CREATE INDEX IF NOT EXISTS idx_rl_lbl_ln      ON raw_logs(dataset, label, line_number);
        """)


# ── Window imports ────────────────────────────────────────────────────────────

def _ingest_windows(dataset: str, data_dir: Path, label_col: str, id_col: Optional[str]) -> None:
    rows: list = []
    for split in ("train", "test"):
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            for i, row in enumerate(csv.DictReader(f)):
                raw_lbl = row.get(label_col, "0")
                try:
                    lbl = int(float(raw_lbl))
                except (ValueError, TypeError):
                    lbl = 1 if str(raw_lbl) not in ("0", "-", "normal") else 0
                rows.append((
                    dataset, split, i,
                    row.get(id_col) or None if id_col else None,
                    lbl,
                    int(float(row.get("session_length") or 0)),
                    row.get("Content", ""),
                ))
                if len(rows) >= BATCH:
                    _insert_windows(rows); rows = []
    if rows:
        _insert_windows(rows)


def _already_done(status_key: str, count_query: str) -> bool:
    """Return True if this table already has data from a completed import."""
    with _conn() as c:
        st = c.execute(
            "SELECT value FROM ingest_status WHERE key=?", (status_key,)
        ).fetchone()
        if st and str(st[0]).startswith("done:"):
            n = c.execute(count_query).fetchone()[0]
            if n > 0:
                return True
    return False


def ingest_bgl_windows(cb: Optional[Callable] = None) -> None:
    """Import BGL windows from the processed txt files.

    Each line in the txt file is one machine-session (space-separated event
    template IDs), exactly mirroring how OpenStack windows are stored.
    This aligns window_index with the BGLSegLoader machine ordering so that
    single-log prediction ground-truth and _fresh_predict content match.

    A versioned key ("bgl_windows_txt") forces a clean re-ingest when
    upgrading from the legacy CSV-based import.
    """
    data_dir = Path(__file__).parent.parent / "data" / "BGL"
    txt_ok = (
        (data_dir / "bgl_train.txt").exists()
        and (data_dir / "bgl_test_normal.txt").exists()
        and (data_dir / "bgl_test_abnormal.txt").exists()
    )
    if not txt_ok:
        # Fallback: old CSV path
        if _already_done("bgl_windows", "SELECT COUNT(*) FROM windows WHERE dataset='bgl'"):
            cb and cb("BGL windows already in DB (CSV) — skipping."); return
        cb and cb("BGL txt files missing — importing windows from CSV…")
        _set_status("bgl_windows", "running")
        _ingest_windows("bgl", LOG_ROOT / "BGL", "Label", None)
        _set_status("bgl_windows", "done:")
        cb and cb("BGL windows done (from CSV).")
        return

    if _already_done("bgl_windows_txt", "SELECT COUNT(*) FROM windows WHERE dataset='bgl'"):
        cb and cb("BGL windows already in DB (txt) — skipping."); return

    cb and cb("Importing BGL windows from txt files (event IDs)…")
    _set_status("bgl_windows_txt", "running")
    with _conn(fast=True) as c:
        c.execute("DELETE FROM windows WHERE dataset='bgl'")

    rows: list = []

    with open(data_dir / "bgl_train.txt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            seq = line.strip()
            if seq:
                rows.append(("bgl", "train", i, "bgl_train", 0, len(seq.split()), seq))
                if len(rows) >= BATCH:
                    _insert_windows(rows); rows = []

    test_offset = 0
    for fname, lbl, blk in [
        ("bgl_test_normal.txt",   0, "bgl_test_normal"),
        ("bgl_test_abnormal.txt", 1, "bgl_test_abnormal"),
    ]:
        p = data_dir / fname
        if not p.exists():
            continue
        count = 0
        with open(p, encoding="utf-8", errors="replace") as f:
            for j, line in enumerate(f):
                seq = line.strip()
                if seq:
                    rows.append(("bgl", "test", test_offset + j, blk, lbl,
                                 len(seq.split()), seq))
                    if len(rows) >= BATCH:
                        _insert_windows(rows); rows = []
                count = j + 1
        test_offset += count

    if rows:
        _insert_windows(rows)
    _set_status("bgl_windows_txt", f"done:{test_offset}")
    cb and cb(f"BGL windows done ({test_offset} test sessions from txt files).")


def ingest_hdfs_windows(cb: Optional[Callable] = None) -> None:
    if _already_done("hdfs_windows", "SELECT COUNT(*) FROM windows WHERE dataset='hdfs'"):
        cb and cb("HDFS windows already in DB — skipping."); return
    cb and cb("Importing HDFS windows…")
    _set_status("hdfs_windows", "running")
    _ingest_windows("hdfs", LOG_ROOT / "HDFS_v1", "Label", "BlockId")
    _set_status("hdfs_windows", "done:")
    cb and cb("HDFS windows done.")


# ── Raw-log imports ───────────────────────────────────────────────────────────

def ingest_bgl_raw(cb: Optional[Callable] = None) -> None:
    """Ingest BGL raw logs from the pre-split files in LOG_ROOT/BGL/split/.

    Three files mirror the OpenStack layout:
      bgl_train.log          → block_id='bgl_train',         label=None
      bgl_test_normal.log    → block_id='bgl_test_normal',   label=None
      bgl_test_abnormal.log  → block_id='bgl_test_abnormal', label='1'

    Global line_number is assigned continuously: train first, then test_normal,
    then test_abnormal.  The db.py constants BGL_N_TRAIN_LINES and
    BGL_N_TEST_NORMAL_LINES record the per-file line counts so the rest of the
    dashboard can reconstruct range queries without a block_id DB lookup.
    """
    if _already_done("bgl_raw_v2", "SELECT COUNT(*) FROM raw_logs WHERE dataset='bgl'"):
        cb and cb("BGL raw logs already in DB — skipping."); return

    split_dir = LOG_ROOT / "BGL" / "split"
    file_meta = [
        (split_dir / "bgl_train.log",        "bgl_train",        False),
        (split_dir / "bgl_test_normal.log",   "bgl_test_normal",  False),
        (split_dir / "bgl_test_abnormal.log", "bgl_test_abnormal", True),
    ]
    if not all(p.exists() for p, _, _ in file_meta):
        _set_status("bgl_raw_v2", "skipped:missing_split_files")
        cb and cb("BGL split files not found in LOG_ROOT/BGL/split/ — skipping raw import.")
        return

    # BGL format: label unix_ts date node datetime node_alias type component level content...
    # label="-" means normal; any other value means anomaly.
    cb and cb("Importing BGL raw logs from split files…")
    _set_status("bgl_raw_v2", "running:0")
    with _conn(fast=True) as c:
        c.execute("DELETE FROM raw_logs WHERE dataset='bgl'")

    rows: list = []
    n = 0   # global line_number counter (continuous across all files)
    for log_path, block_id, all_anomaly in file_meta:
        file_n = 0
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split(None, 9)
                if len(parts) < 9:
                    continue
                lbl       = parts[0]
                timestamp = parts[4]
                component = parts[7]
                level     = parts[8]
                content   = line.rstrip()
                rows.append((
                    "bgl", n,
                    "1" if all_anomaly else (None if lbl == "-" else "1"),
                    timestamp,
                    component,
                    level,
                    content,
                    block_id,
                ))
                n += 1
                file_n += 1
                if len(rows) >= BATCH:
                    _insert_raw(rows); rows = []
                    if n % 500_000 == 0:
                        _set_status("bgl_raw_v2", f"running:{n}")
                        cb and cb(f"  BGL raw: {n:,} rows…")
        cb and cb(f"  {log_path.name}: {file_n:,} lines ingested.")
    if rows:
        _insert_raw(rows)
    _set_status("bgl_raw_v2", f"done:{n}")
    cb and cb(f"BGL raw logs done ({n:,} rows).")


def ingest_hdfs_raw(cb: Optional[Callable] = None) -> None:
    """Ingest HDFS raw logs from the pre-split files in LOG_ROOT.

    Files (one-to-one with the processed txt-file sessions):
      train.log         → line_numbers 0..N_TRAIN-1,               label=None
      test_normal.log   → line_numbers N_TRAIN..,                  label=None
      test_abnormal.log → line_numbers N_TRAIN+N_TEST_NORMAL..,    label='1'

    Each line is stored with its extracted block ID so that
    _find_window_for_raw can map any raw line to its session.
    """
    if _already_done("hdfs_raw_v2", "SELECT COUNT(*) FROM raw_logs WHERE dataset='hdfs'"):
        cb and cb("HDFS raw logs already in DB — skipping."); return

    file_meta = [
        (LOG_ROOT / "train.log",        False),  # all normal — no anomaly label
        (LOG_ROOT / "test_normal.log",  False),
        (LOG_ROOT / "test_abnormal.log", True),
    ]
    missing = [str(p) for p, _ in file_meta if not p.exists()]
    if missing:
        _set_status("hdfs_raw_v2", "skipped:missing_split_files")
        cb and cb(f"HDFS split files not found ({', '.join(missing)}) — skipping raw import.")
        return

    # HDFS format: YYMMDD HHMMSS pid level component: content
    cb and cb("Importing HDFS raw logs from split files…")
    _set_status("hdfs_raw_v2", "running:0")
    with _conn(fast=True) as c:
        c.execute("DELETE FROM raw_logs WHERE dataset='hdfs'")

    rows: list = []
    n = 0
    for log_path, all_anomaly in file_meta:
        file_n = 0
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                parts = line.split(None, 4)
                timestamp = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else ""
                level     = parts[3] if len(parts) > 3 else ""
                component = parts[4].partition(": ")[0] if len(parts) > 4 else ""
                blk       = BLK_RE.search(line)
                blk       = blk.group() if blk else None
                rows.append((
                    "hdfs", n,
                    "1" if all_anomaly else None,
                    timestamp,
                    component,
                    level,
                    line,      # full raw line as content
                    blk,
                ))
                n += 1
                file_n += 1
                if len(rows) >= BATCH:
                    _insert_raw(rows); rows = []
                    if n % 1_000_000 == 0:
                        _set_status("hdfs_raw_v2", f"running:{n}")
                        cb and cb(f"  HDFS raw: {n:,} rows…")
        cb and cb(f"  {log_path.name}: {file_n:,} lines ingested.")
    if rows:
        _insert_raw(rows)
    _set_status("hdfs_raw_v2", f"done:{n}")
    cb and cb(f"HDFS raw logs done ({n:,} rows).")


# ── Synthetic OpenStack log generator ────────────────────────────────────────
# Realistic log message templates based on the Nova / Keystone / Neutron
# patterns found in the LogHub OpenStack dataset.  Used when the original
# openstack_*.log files are not available (demo / container deployments).

_OS_COMPONENTS = [
    "nova.osapi_compute.wsgi.server",
    "nova.compute.manager",
    "nova.scheduler.manager",
    "nova.conductor.manager",
    "nova.network.manager",
    "keystonemiddleware.auth_token",
    "nova.api.openstack.requestlog",
]

# Real tenant/user IDs from the LogHub OpenStack dataset
_OS_TENANT_ID = "54fadb412c4e40cdbaed9335e4c35a9e"
_OS_USER_ID   = "113d3a99c3da401fbd62cc2caa5b96d2"
_OS_SRC_IP    = "10.11.10.1"

# Templates match the exact format of the real openstack_*.log files:
#   [req-{uuid} {user_id} {tenant_id} - - -] {ip} "METHOD /v2/{tenant}/... HTTP/1.1" status: N len: N time: N
_NORMAL_MSGS = [
    ('INFO', "nova.osapi_compute.wsgi.server",
     '[req-{req} {user} {proj} - - -] {ip} "GET /v2/{proj}/servers/detail HTTP/1.1" status: 200 len: {sz} time: {t}'),
    ('INFO', "nova.osapi_compute.wsgi.server",
     '[req-{req} {user} {proj} - - -] {ip} "GET /v2/{proj}/servers/{srv} HTTP/1.1" status: 200 len: {sz} time: {t}'),
    ('INFO', "nova.osapi_compute.wsgi.server",
     '[req-{req} {user} {proj} - - -] {ip} "GET /v2/{proj}/flavors HTTP/1.1" status: 200 len: {sz} time: {t}'),
    ('INFO', "nova.osapi_compute.wsgi.server",
     '[req-{req} {user} {proj} - - -] {ip} "POST /v2/{proj}/servers HTTP/1.1" status: 202 len: {sz} time: {t}'),
    ('INFO', "nova.osapi_compute.wsgi.server",
     '[req-{req} {user} {proj} - - -] {ip} "DELETE /v2/{proj}/servers/{srv} HTTP/1.1" status: 204 len: 0 time: {t}'),
    ('INFO', "nova.compute.manager",
     '[req-{req} {user} {proj} - - -] [instance: {inst}] VM started successfully.'),
    ('INFO', "nova.compute.manager",
     '[req-{req} {user} {proj} - - -] [instance: {inst}] Took {t} seconds to deallocate network for instance.'),
    ('INFO', "nova.compute.manager",
     '[req-{req} {user} {proj} - - -] [instance: {inst}] Updating instance state to active.'),
    ('INFO', "nova.virt.libvirt.driver",
     '[req-{req} {user} {proj} - - -] [instance: {inst}] Creating image'),
    ('INFO', "nova.compute.manager",
     '[-] [instance: {inst}] VM Stopped (Lifecycle Event)'),
]

_ABNORMAL_MSGS = [
    ('ERROR',   "nova.compute.manager",
     '[req-{req} {user} {proj} - - -] [instance: {inst}] Build of instance failed: No valid host was found.'),
    ('ERROR',   "nova.compute.manager",
     '[req-{req} {user} {proj} - - -] [instance: {inst}] Error destroying instance on host {host}: Timeout.'),
    ('ERROR',   "nova.osapi_compute.wsgi.server",
     '[req-{req} {user} {proj} - - -] {ip} "POST /v2/{proj}/servers HTTP/1.1" status: 500 len: {sz} time: {t}'),
    ('WARNING', "nova.virt.libvirt.imagecache",
     '[req-{req} - - - - -] Unknown base file: /var/lib/nova/instances/_base/{inst}'),
    ('WARNING', "nova.compute.manager",
     '[req-{req} {user} {proj} - - -] [instance: {inst}] Timeout waiting for instance to become active.'),
    ('ERROR',   "nova.network.manager",
     '[req-{req} {user} {proj} - - -] Failed to allocate network for instance {inst}: Connection refused.'),
    ('ERROR',   "nova.compute.manager",
     '[req-{req} {user} {proj} - - -] [instance: {inst}] Instance failed to spawn: Insufficient resources.'),
    ('WARNING', "nova.conductor.manager",
     '[req-{req} {user} {proj} - - -] [instance: {inst}] Instance entered ERROR state after {n} retries.'),
]


def _synth_val(seed: int, key: str) -> str:
    """Generate deterministic values matching real OpenStack log format."""
    h = int(hashlib.md5(f"{seed}{key}".encode()).hexdigest(), 16)
    if key == "ip":
        return _OS_SRC_IP          # real source IP from dataset
    if key == "proj":
        return _OS_TENANT_ID       # real tenant ID from dataset
    if key == "user":
        return _OS_USER_ID         # real user ID from dataset
    if key == "req":
        # UUID format: 8-4-4-4-12
        hx = hashlib.md5(f"req{seed}".encode()).hexdigest()
        return f"{hx[:8]}-{hx[8:12]}-{hx[12:16]}-{hx[16:20]}-{hx[20:32]}"
    if key == "srv":
        hx = hashlib.md5(f"srv{seed}".encode()).hexdigest()
        return f"{hx[:8]}-{hx[8:12]}-{hx[12:16]}-{hx[16:20]}-{hx[20:32]}"
    if key == "inst":
        hx = hashlib.md5(f"inst{seed}".encode()).hexdigest()
        return f"{hx[:8]}-{hx[8:12]}-{hx[12:16]}-{hx[16:20]}-{hx[20:32]}"
    if key == "host":
        return f"compute-node-{h % 4:02d}"
    if key == "sz":
        sizes = [1583, 1708, 1759, 1893, 2048, 1234, 3607, 661, 270]
        return str(sizes[h % len(sizes)])
    if key == "t":
        return f"{0.05 + (h % 400) / 1000:.7f}"
    if key == "n":
        return str(1 + h % 5)
    return str(h % 100)


def _make_log_line(seed: int, level: str, component: str, template: str,
                   ts: datetime, pid: int) -> tuple:
    """Render one synthetic log line and return a raw_logs insert tuple."""
    # Fill in all placeholders deterministically from the seed
    import re as _re
    def _sub(m):
        return _synth_val(seed, m.group(1))
    content = _re.sub(r"\{(\w+)(?::[^}]*)?\}", _sub, template)
    ts_str  = ts.strftime("%Y-%m-%d %H:%M:%S.") + f"{seed % 1000:03d}"
    return ts_str, component, level, content


def _ingest_os_synthetic_raw(cb: Optional[Callable] = None) -> None:
    """Generate realistic OpenStack log messages when source .log files are absent.

    Produces one log row per event in each processed session, using realistic
    Nova / Keystone message templates so the Raw Log Lines panel shows actual
    log-style content instead of event IDs.

    Detection marker: timestamp is a real datetime string (NOT NULL), so the
    _find_window_for_raw mapping still uses the lines-per-session estimates.
    """
    data_dir = Path(__file__).parent.parent / "data" / "OpenStack"
    if not data_dir.exists():
        _set_status("os_raw_v2", "skipped:missing_data_dir")
        cb and cb("data/OpenStack not found — cannot generate synthetic raw logs.")
        return

    cb and cb("Generating synthetic OpenStack raw logs from event sequences…")
    _set_status("os_raw_v2", "running")
    with _conn(fast=True) as c:
        c.execute("DELETE FROM raw_logs WHERE dataset='os'")

    file_meta = [
        ("train.txt",         "0", "train_normal",   _NORMAL_MSGS),
        ("test_normal.txt",   "0", "test_normal",    _NORMAL_MSGS),
        ("test_abnormal.txt", "1", "test_abnormal",  _ABNORMAL_MSGS),
    ]

    rows: list = []
    global_n  = 0   # global line_number counter
    base_time = datetime(2017, 5, 16, 0, 0, 0)

    for fname, label, block_id, templates in file_meta:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8", errors="replace") as f:
            for sess_idx, line in enumerate(f):
                events = line.split()
                if not events:
                    continue
                pid = 1000 + (sess_idx % 9000)
                for ev_pos, ev_id in enumerate(events):
                    # Pick a template based on event ID for deterministic variety
                    ev_int  = int(ev_id) if ev_id.isdigit() else 0
                    tmpl    = templates[ev_int % len(templates)]
                    level, component, msg_template = tmpl
                    # Advance timestamp slightly per event
                    ts = base_time + timedelta(
                        hours=sess_idx // 60,
                        minutes=sess_idx % 60,
                        seconds=ev_pos,
                        milliseconds=(ev_int * 37) % 1000,
                    )
                    seed    = global_n
                    ts_str, comp, lvl, content = _make_log_line(
                        seed, level, component, msg_template, ts, pid
                    )
                    rows.append((
                        "os", global_n, label,
                        ts_str, comp, lvl, content, block_id,
                    ))
                    global_n += 1
                    if len(rows) >= BATCH:
                        _insert_raw(rows)
                        rows = []
    if rows:
        _insert_raw(rows)
    _set_status("os_raw_v2", f"done:{global_n}")
    cb and cb(f"Synthetic OpenStack raw logs done ({global_n:,} rows).")


def ingest_os_raw(cb: Optional[Callable] = None) -> None:
    """
    Import OpenStack raw log files into raw_logs.

    Split labelling based on source file:
      normal1.log   → split=train,      label=0  (block_id='train_normal')
      normal2.log   → split=test/normal, label=0  (block_id='test_normal')
      abnormal.log  → split=test/abnorm, label=1  (block_id='test_abnormal')

    Falls back to synthetic entries from data/OpenStack/*.txt when the source
    log files are not present (e.g. demo deployments without full data).
    """
    if _already_done("os_raw_v2", "SELECT COUNT(*) FROM raw_logs WHERE dataset='os'"):
        cb and cb("OpenStack raw logs already in DB — skipping."); return
    os_dir = LOG_ROOT / "OpenStack"
    if not os_dir.exists():
        # Fallback: bundled sample inside the project (data/OpenStack/raw/)
        _project_raw = Path(__file__).parent.parent / "data" / "OpenStack" / "raw"
        if _project_raw.exists():
            cb and cb("Using bundled OpenStack log sample from data/OpenStack/raw/")
            os_dir = _project_raw
        else:
            cb and cb("OpenStack log directory not found — using synthetic raw logs from processed data.")
            _ingest_os_synthetic_raw(cb)
            return

    # file → (label, block_id tag)
    file_meta = {
        "openstack_normal1.log":   ("0", "train_normal"),
        "openstack_normal2.log":   ("0", "test_normal"),
        "openstack_abnormal.log":  ("1", "test_abnormal"),
    }
    cb and cb("Importing OpenStack raw logs…")
    _set_status("os_raw_v2", "running")
    with _conn(fast=True) as c:
        c.execute("DELETE FROM raw_logs WHERE dataset='os'")

    # Pattern: source_file  YYYY-MM-DD HH:MM:SS.fff  PID  LEVEL  component  rest
    pat = re.compile(
        r"^(\S+)\s+"
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+"
        r"(\d+)\s+(\w+)\s+(\S+)\s+(.*)"
    )
    rows: list = []
    n = 0
    for fname, (file_lbl, block_tag) in file_meta.items():
        lf = os_dir / fname
        if not lf.exists():
            cb and cb(f"  {fname} not found — skipping.")
            continue
        with open(lf, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                m = pat.match(line)
                if m:
                    rows.append((
                        "os", n, file_lbl,
                        m.group(2), m.group(5), m.group(4),
                        line, block_tag,
                    ))
                else:
                    rows.append(("os", n, file_lbl, None, None, None, line, block_tag))
                n += 1
                if len(rows) >= BATCH:
                    _insert_raw(rows); rows = []
    if rows:
        _insert_raw(rows)
    _set_status("os_raw_v2", f"done:{n}")
    cb and cb(f"OpenStack raw logs done ({n:,} rows).")


def ingest_os_windows(cb: Optional[Callable] = None) -> None:
    """
    Import processed OpenStack sequences from data/OpenStack/*.txt into windows.

    Each line in the txt file = one session (sequence of event template IDs).
    - train.txt          → split='train', label=0,  window_index 0..N-1
    - test_normal.txt    → split='test',  label=0,  window_index 0..M-1
    - test_abnormal.txt  → split='test',  label=1,  window_index M..M+K-1

    The concatenated test window order (normal first, then abnormal) matches
    the order used by OpenStackSegLoader for ground_truth / predictions.
    """
    if _already_done("os_windows", "SELECT COUNT(*) FROM windows WHERE dataset='os'"):
        cb and cb("OpenStack windows already in DB — skipping."); return
    data_dir = Path(__file__).parent.parent / "data" / "OpenStack"
    if not data_dir.exists():
        _set_status("os_windows", "skipped:missing_data_dir")
        cb and cb("data/OpenStack not found — skipping OS windows.")
        return

    cb and cb("Importing OpenStack processed windows…")
    _set_status("os_windows", "running")
    with _conn(fast=True) as c:
        c.execute("DELETE FROM windows WHERE dataset='os'")

    rows: list = []

    # Train windows (all normal)
    train_path = data_dir / "train.txt"
    if train_path.exists():
        with open(train_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                seq = line.strip()
                if seq:
                    rows.append(("os", "train", i, "train_normal", 0,
                                  len(seq.split()), seq))

    # Test windows — normal first, then abnormal (matches loader concat order)
    test_offset = 0
    for fname, lbl, blk_tag in [
        ("test_normal.txt",   0, "test_normal"),
        ("test_abnormal.txt", 1, "test_abnormal"),
    ]:
        p = data_dir / fname
        if not p.exists():
            continue
        count = 0
        with open(p, encoding="utf-8", errors="replace") as f:
            for j, line in enumerate(f):
                seq = line.strip()
                if seq:
                    rows.append(("os", "test", test_offset + j, blk_tag, lbl,
                                  len(seq.split()), seq))
                count = j + 1  # track last index even if line was blank
        test_offset += count  # next file's indices start after this file's last

    if rows:
        _insert_windows(rows)
    n = len(rows)
    _set_status("os_windows", f"done:{n}")
    cb and cb(f"OpenStack windows done ({n:,} sessions).")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_full_ingest(cb: Optional[Callable] = None) -> None:
    """Run all ingestion steps. Safe to call multiple times (idempotent per step)."""
    _set_status("overall", "running")
    t0 = time.time()

    steps = [
        ingest_bgl_windows,
        ingest_hdfs_windows,
        ingest_os_windows,
        ingest_bgl_raw,
        ingest_hdfs_raw,
        ingest_os_raw,
    ]

    for fn in steps:
        try:
            fn(cb)
        except Exception as exc:
            msg = f"[ERROR] {fn.__name__}: {exc}"
            cb and cb(msg)
            _set_status(fn.__name__, f"error:{exc}")

    cb and cb("Rebuilding indexes…")
    _rebuild_indexes()

    elapsed = time.time() - t0
    _set_status("overall", f"done:{elapsed:.1f}s")
    cb and cb(f"Import complete in {elapsed:.1f}s")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from db import init_db
    init_db()
    run_full_ingest(print)
