import sqlite3
import numpy as np
import json
import os
from datetime import datetime

DB_FILE = "spectraid_library.db"

def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS spectra (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT DEFAULT '',
            subcategory TEXT DEFAULT '',
            cas_number TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            wavenumber_min REAL,
            wavenumber_max REAL,
            n_points INTEGER,
            wavenumber BLOB NOT NULL,
            spectrum BLOB NOT NULL,
            added_by TEXT DEFAULT 'admin',
            added_at TEXT DEFAULT '',
            updated_at TEXT DEFAULT ''
        )
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_name ON spectra(name)
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_category ON spectra(category)
    """)
    conn.commit()
    conn.close()

def add_spectrum(name, category, subcategory, cas_number, notes,
                 wavenumber, spectrum, added_by="admin"):
    conn = get_conn()
    c = conn.cursor()
    wn = np.array(wavenumber, dtype=np.float32)
    sp = np.array(spectrum, dtype=np.float32)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO spectra
        (name, category, subcategory, cas_number, notes,
         wavenumber_min, wavenumber_max, n_points,
         wavenumber, spectrum, added_by, added_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        name, category, subcategory, cas_number, notes,
        float(wn.min()), float(wn.max()), len(wn),
        wn.tobytes(), sp.tobytes(),
        added_by, now, now
    ))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id

def delete_spectrum(spec_id):
    conn = get_conn()
    conn.execute("DELETE FROM spectra WHERE id=?", (spec_id,))
    conn.commit()
    conn.close()

def update_spectrum_meta(spec_id, name, category, subcategory, cas_number, notes):
    conn = get_conn()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        UPDATE spectra SET name=?,category=?,subcategory=?,
        cas_number=?,notes=?,updated_at=?
        WHERE id=?
    """, (name, category, subcategory, cas_number, notes, now, spec_id))
    conn.commit()
    conn.close()

def get_all_meta():
    conn = get_conn()
    rows = conn.execute("""
        SELECT id, name, category, subcategory, cas_number,
               notes, wavenumber_min, wavenumber_max, n_points,
               added_by, added_at
        FROM spectra ORDER BY category, name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_spectrum_by_id(spec_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM spectra WHERE id=?", (spec_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["wavenumber"] = np.frombuffer(d["wavenumber"], dtype=np.float32)
    d["spectrum"]   = np.frombuffer(d["spectrum"],   dtype=np.float32)
    return d

def get_all_spectra_for_matching():
    """Load all spectra as numpy arrays for fast batch matching."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, category, wavenumber, spectrum FROM spectra"
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "name": r["name"],
            "category": r["category"],
            "wavenumber": np.frombuffer(r["wavenumber"], dtype=np.float32),
            "spectrum":   np.frombuffer(r["spectrum"],   dtype=np.float32),
        })
    return result

def count_spectra():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0]
    conn.close()
    return n

def get_categories():
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT category FROM spectra ORDER BY category"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows if r[0]]

def import_from_json(json_path, added_by="admin"):
    with open(json_path) as f:
        entries = json.load(f)
    added = 0
    for e in entries:
        try:
            add_spectrum(
                name=e.get("name",""),
                category=e.get("category",""),
                subcategory=e.get("subcategory",""),
                cas_number=e.get("cas_number",""),
                notes=e.get("notes",""),
                wavenumber=e["wavenumber"],
                spectrum=e["spectrum"],
                added_by=added_by
            )
            added += 1
        except Exception:
            continue
    return added
