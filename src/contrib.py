"""
contrib.py
Opt-in community contributions: photos uploaded through the web app, with
the uploader's own label, stored in Turso for a FUTURE training set. The
same database also records who signs in (name, role, email, institution and
survey site entered on the welcome screen of either interface); photos are
never stored for a sign-in alone.

Nothing here touches data/splits or any number in the paper. A stored image
only becomes training material after a reviewer confirms its label
(status 'confirmed'); `python -m src.contrib export` then writes those to
data/contributed/<label>/, a separate folder a later study can use.

Safeguards:
  - images identical or near-identical (dHash) to any image of the research
    dataset are refused, so held-out test photos cannot leak into training;
  - the stored copy is resized to INFER_MAX_SIDE and re-encoded, which also
    strips EXIF (including GPS);
  - the consent wording shown to the uploader is stored with each row.

Usage:
    python -m src.contrib stats
    python -m src.contrib visitors    # everyone who signed in
    python -m src.contrib export      # confirmed images -> data/contributed/
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import secrets
import threading
from datetime import datetime, timezone

import cv2
import numpy as np

try:
    from . import config as C
except ImportError:                     # pragma: no cover
    import config as C

CONSENT_TEXT = ("I took this photo or have the right to share it. I agree that the photo, "
                "my label and the details I entered may be stored and used to build a coral "
                "research dataset, and that I can ask for removal by quoting the contribution ID.")

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS contributors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, role TEXT, email TEXT, institution TEXT,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS contributions (
        id TEXT PRIMARY KEY,
        contributor_id INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        dhash TEXT NOT NULL,
        filename TEXT, width INTEGER, height INTEGER,
        image BLOB NOT NULL,
        user_label TEXT NOT NULL,
        label_after_prediction INTEGER NOT NULL,
        model_run TEXT, model_label TEXT, model_confidence REAL,
        site TEXT,
        consent_text TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'unverified',
        verified_label TEXT, reviewer TEXT, reviewed_at TEXT, review_note TEXT)""",
    """CREATE TABLE IF NOT EXISTS visitors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, role TEXT, email TEXT, institution TEXT,
        interface TEXT NOT NULL, site TEXT,
        first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
        visits INTEGER NOT NULL DEFAULT 1)""",
    "CREATE INDEX IF NOT EXISTS idx_contrib_sha ON contributions(sha256)",
    "CREATE INDEX IF NOT EXISTS idx_contrib_status ON contributions(status)",
]
LIST_COLS = ("c.id, c.created_at, c.filename, c.width, c.height, c.user_label, "
             "c.model_run, c.model_label, c.model_confidence, c.site, c.status, "
             "c.verified_label, c.reviewer, c.reviewed_at, c.review_note, "
             "p.name, p.role, p.email, p.institution")
VISITOR_COLS = ("id", "name", "role", "email", "institution", "interface", "site",
                "first_seen", "last_seen", "visits")
INTERFACES = ("web", "streamlit")
STATUSES = ("unverified", "confirmed", "rejected")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ContributionError(ValueError):
    """A contribution was refused; the message is safe to show the user."""


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
def load_env() -> None:
    """Read KEY=VALUE lines from .env into os.environ (existing vars win)."""
    if not C.ENV_FILE.exists():
        return
    for line in C.ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def admin_token() -> str | None:
    load_env()
    return os.environ.get("CORAL_ADMIN_TOKEN") or None


def check_admin(token: str | None) -> bool:
    want = admin_token()
    return bool(want and token and secrets.compare_digest(want, token))


# ----------------------------------------------------------------------
# Image fingerprints
# ----------------------------------------------------------------------
def dhash(bgr: np.ndarray) -> int:
    """64-bit difference hash: robust to resizing and re-compression."""
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _popcount(x: np.ndarray) -> np.ndarray:
    return np.array([bin(int(v)).count("1") for v in x])


class DatasetFingerprints:
    """md5 + dHash of every image in the research dataset (raw and splits)."""

    def __init__(self):
        self._md5: set[str] = set()
        self._dh = np.zeros(0, dtype=np.uint64)
        self._ready = False
        self._lock = threading.Lock()

    def _build(self):
        md5s, dhs = set(), []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for root in (C.RAW_DIR, C.SPLIT_RAW_DIR):
            for f in root.rglob("*"):
                if f.suffix.lower() not in exts:
                    continue
                data = f.read_bytes()
                md5s.add(hashlib.md5(data).hexdigest())
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    dhs.append(dhash(img))
        for f in C.SPLIT_DIR.rglob("*"):          # enhanced copies: exact match only
            if f.suffix.lower() in exts:
                md5s.add(hashlib.md5(f.read_bytes()).hexdigest())
        self._md5, self._dh = md5s, np.array(sorted(set(dhs)), dtype=np.uint64)
        self._ready = True

    def match(self, data: bytes, bgr: np.ndarray) -> str | None:
        """Why this image is part of the dataset, or None if it is new."""
        with self._lock:
            if not self._ready:
                self._build()
        if hashlib.md5(data).hexdigest() in self._md5:
            return "it is an image from the research dataset"
        if len(self._dh):
            d = _popcount(self._dh ^ np.uint64(dhash(bgr))).min()
            if d <= C.CONTRIB_NEAR_DUP_BITS:
                return "it looks like a copy of an image from the research dataset"
        return None


FINGERPRINTS = DatasetFingerprints()


def training_copy(bgr: np.ndarray) -> tuple[bytes, int, int]:
    """The stored version: longest side INFER_MAX_SIDE, JPEG, no metadata."""
    h, w = bgr.shape[:2]
    if max(h, w) > C.INFER_MAX_SIDE:
        s = C.INFER_MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, C.CONTRIB_JPEG_QUALITY])
    if not ok:
        raise ContributionError("could not encode the image")
    return buf.tobytes(), bgr.shape[1], bgr.shape[0]


# ----------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(v, n: int) -> str | None:
    v = (v or "").strip()
    return v[:n] or None


class ContribDB:
    """Turso database with a local replica. Writes are pushed immediately;
    one lock serialises access because the API serves requests from a
    thread pool."""

    def __init__(self, replica: str = "replica.db"):
        # Each process needs its own replica file: the engine locks it, so
        # the CLI cannot share the running API server's copy.
        self._replica = replica
        self._conn = None
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        load_env()
        return bool(os.environ.get("TURSO_DATABASE_URL") and os.environ.get("TURSO_AUTH_TOKEN"))

    def _connect(self):
        if self._conn is None:
            import turso.sync
            C.CONTRIB_DIR.mkdir(parents=True, exist_ok=True)
            self._conn = turso.sync.connect(
                str(C.CONTRIB_DIR / self._replica),
                remote_url=os.environ["TURSO_DATABASE_URL"],
                auth_token=os.environ["TURSO_AUTH_TOKEN"],
                client_name="coral-reef-monitor")
            cur = self._conn.cursor()
            for stmt in SCHEMA:
                cur.execute(stmt)
            self._conn.commit()
            self._conn.push()
        return self._conn

    def _run(self, fn, write=False):
        if not self.configured:
            raise ContributionError("contributions are not configured on this server")
        with self._lock:
            conn = self._connect()
            if not write:
                conn.pull()
            out = fn(conn.cursor())
            if write:
                conn.commit()
                conn.push()
            return out

    # -- writes --------------------------------------------------------
    def add(self, *, contributor: dict, pending: dict, user_label: str,
            site: str | None, label_after_prediction: bool) -> str:
        if user_label not in C.CONTRIB_LABELS:
            raise ContributionError(f"label must be one of {C.CONTRIB_LABELS}")
        name = _clean(contributor.get("name"), 60)
        if not name:
            raise ContributionError("a name is required")
        email = _clean(contributor.get("email"), 120)
        if email and not EMAIL_RE.match(email):
            raise ContributionError("that email address does not look valid")
        who = (name, _clean(contributor.get("role"), 60), email,
               _clean(contributor.get("institution"), 120))

        def tx(cur):
            if cur.execute("SELECT 1 FROM contributions WHERE sha256 = ?",
                           (pending["sha256"],)).fetchall():
                raise ContributionError("this photo has already been contributed")
            row = cur.execute(
                "SELECT id FROM contributors WHERE name IS ? AND role IS ? AND email IS ? "
                "AND institution IS ?", who).fetchall()
            if row:
                pid = row[0][0]
            else:
                cur.execute("INSERT INTO contributors (name, role, email, institution, created_at) "
                            "VALUES (?, ?, ?, ?, ?)", (*who, _now()))
                pid = cur.lastrowid
            cid = "CC-" + secrets.token_hex(4).upper()
            cur.execute(
                "INSERT INTO contributions (id, contributor_id, created_at, sha256, dhash, filename, "
                "width, height, image, user_label, label_after_prediction, model_run, model_label, "
                "model_confidence, site, consent_text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (cid, pid, _now(), pending["sha256"], f"{pending['dhash']:016x}",
                 _clean(pending["filename"], 200), pending["width"], pending["height"],
                 pending["jpeg"], user_label, int(label_after_prediction), pending["model_run"],
                 pending["model_label"], float(pending["model_confidence"]),
                 _clean(site, 120), CONSENT_TEXT))
            return cid
        return self._run(tx, write=True)

    def visit(self, user: dict, interface: str) -> int:
        """Record a sign-in. The same person on the same interface is one row
        whose visit count and last_seen are updated. Returns the row id."""
        if interface not in INTERFACES:
            raise ContributionError(f"interface must be one of {INTERFACES}")
        name = _clean(user.get("name"), 60)
        if not name:
            raise ContributionError("a name is required")
        email = _clean(user.get("email"), 120)
        if email and not EMAIL_RE.match(email):
            raise ContributionError("that email address does not look valid")
        who = (name, _clean(user.get("role"), 60), email,
               _clean(user.get("institution"), 120), interface)
        site = _clean(user.get("site"), 120)

        def tx(cur):
            row = cur.execute(
                "SELECT id FROM visitors WHERE name IS ? AND role IS ? AND email IS ? "
                "AND institution IS ? AND interface IS ?", who).fetchall()
            if row:
                cur.execute("UPDATE visitors SET visits = visits + 1, last_seen = ?, "
                            "site = COALESCE(?, site) WHERE id = ?", (_now(), site, row[0][0]))
                return row[0][0]
            now = _now()
            cur.execute("INSERT INTO visitors (name, role, email, institution, interface, site, "
                        "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (*who, site, now, now))
            return cur.lastrowid
        return self._run(tx, write=True)

    def set_visitor_site(self, vid: int, site: str | None) -> None:
        site = _clean(site, 120)
        if site:
            self._run(lambda cur: cur.execute("UPDATE visitors SET site = ? WHERE id = ?",
                                              (site, int(vid))), write=True)

    def review(self, cid: str, status: str, verified_label: str | None,
               reviewer: str | None, note: str | None) -> None:
        if status not in STATUSES:
            raise ContributionError(f"status must be one of {STATUSES}")
        if status == "confirmed" and verified_label not in C.CLASS_NAMES:
            raise ContributionError(f"a confirmed image needs a label from {C.CLASS_NAMES}")
        if status != "confirmed":
            verified_label = None

        def tx(cur):
            if not cur.execute("SELECT 1 FROM contributions WHERE id = ?", (cid,)).fetchall():
                raise ContributionError(f"no contribution {cid}")
            cur.execute("UPDATE contributions SET status=?, verified_label=?, reviewer=?, "
                        "reviewed_at=?, review_note=? WHERE id=?",
                        (status, verified_label, _clean(reviewer, 60),
                         _now() if status != "unverified" else None, _clean(note, 300), cid))
        self._run(tx, write=True)

    def delete(self, cid: str) -> None:
        """Withdrawal on request. The contributor row stays if they have others."""
        def tx(cur):
            row = cur.execute("SELECT contributor_id FROM contributions WHERE id = ?",
                              (cid,)).fetchall()
            if not row:
                raise ContributionError(f"no contribution {cid}")
            cur.execute("DELETE FROM contributions WHERE id = ?", (cid,))
            if not cur.execute("SELECT 1 FROM contributions WHERE contributor_id = ?",
                               (row[0][0],)).fetchall():
                cur.execute("DELETE FROM contributors WHERE id = ?", (row[0][0],))
        self._run(tx, write=True)

    # -- reads ---------------------------------------------------------
    def list(self, status: str | None = None, limit: int = 200) -> list[dict]:
        keys = [k.split(".")[1] for k in LIST_COLS.split(", ")]
        q = f"SELECT {LIST_COLS} FROM contributions c JOIN contributors p ON p.id = c.contributor_id"
        args: tuple = ()
        if status:
            q += " WHERE c.status = ?"
            args = (status,)
        q += " ORDER BY c.created_at DESC LIMIT ?"
        rows = self._run(lambda cur: cur.execute(q, (*args, int(limit))).fetchall())
        return [dict(zip(keys, r)) for r in rows]

    def image(self, cid: str) -> bytes | None:
        rows = self._run(lambda cur: cur.execute(
            "SELECT image FROM contributions WHERE id = ?", (cid,)).fetchall())
        return bytes(rows[0][0]) if rows else None

    def stats(self) -> dict:
        def q(cur):
            out = {s: 0 for s in STATUSES}
            for s, n in cur.execute("SELECT status, COUNT(*) FROM contributions GROUP BY status"):
                out[s] = n
            out["confirmed_by_label"] = {l: n for l, n in cur.execute(
                "SELECT verified_label, COUNT(*) FROM contributions WHERE status='confirmed' "
                "GROUP BY verified_label")}
            out["contributors"] = cur.execute("SELECT COUNT(*) FROM contributors").fetchall()[0][0]
            out["visitors"] = cur.execute("SELECT COUNT(*) FROM visitors").fetchall()[0][0]
            return out
        return self._run(q)

    def visitors(self, limit: int = 500) -> list[dict]:
        rows = self._run(lambda cur: cur.execute(
            f"SELECT {', '.join(VISITOR_COLS)} FROM visitors ORDER BY last_seen DESC LIMIT ?",
            (int(limit),)).fetchall())
        return [dict(zip(VISITOR_COLS, r)) for r in rows]

    def confirmed(self):
        return self._run(lambda cur: cur.execute(
            "SELECT id, verified_label, image, created_at, user_label, model_label, reviewer, "
            "reviewed_at FROM contributions WHERE status='confirmed' ORDER BY id").fetchall())


DB = ContribDB()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def export() -> None:
    """Write confirmed images to data/contributed/<label>/ plus a manifest.
    The manifest carries no personal details (no names or emails)."""
    rows = DB.confirmed()
    out = C.CONTRIB_EXPORT_DIR
    for label in C.CLASS_NAMES:
        (out / label).mkdir(parents=True, exist_ok=True)
    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "file", "contributed_at", "uploader_label",
                    "model_label", "reviewer", "reviewed_at"])
        for cid, label, img, at, ul, ml, rev, rat in rows:
            f = out / label / f"{cid}.jpg"
            f.write_bytes(bytes(img))
            w.writerow([cid, label, f.relative_to(out).as_posix(), at, ul, ml, rev, rat])
    print(f"exported {len(rows)} confirmed images to {out}")


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["stats", "visitors", "export"])
    a = ap.parse_args()
    DB = ContribDB("replica-cli.db")
    if a.command == "stats":
        print(json.dumps(DB.stats(), indent=2))
    elif a.command == "visitors":
        print(json.dumps(DB.visitors(), indent=2))
    else:
        export()
