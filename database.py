"""
database.py  —  SpectraVision Pro v3.0
Persistent storage via Supabase PostgreSQL.
Replaces SQLite + users.json (both were ephemeral on Streamlit Cloud).

Setup:
  1. Add to Streamlit Cloud Secrets (Settings → Secrets):
       [supabase]
       url = "https://xxxx.supabase.co"
       key = "your-anon-public-key"

  2. First run will auto-create the required tables.
"""

import json
import time
import functools
import hashlib
from datetime import datetime

import httpx
import streamlit as st
from supabase import create_client, Client


# ── Supabase client (cached, dengan batas waktu) ──────────────
#
# ttl=600 (10 menit): klien lama (beserta connection pool httpx di
# dalamnya) dibuang dan dibuat ulang secara berkala. Ini mencegah
# "koneksi basi" — soket yang sudah menganggur lama lalu ditutup diam-
# diam oleh sisi server/proxy, yang kalau dipakai ulang tanpa sadar
# akan memicu httpx.ReadError: [Errno 11] Resource temporarily
# unavailable. Tanpa ttl, klien yang sama bisa hidup selama proses
# Streamlit berjalan (bisa berhari-hari), sehingga makin lama makin
# rawan menyentuh koneksi basi.

@st.cache_resource(ttl=600)
def get_supabase() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


def _retry_on_network_error(max_attempts=3, base_delay=0.8):
    """
    Decorator untuk fungsi yang memanggil Supabase. Menangkap error
    JARINGAN TRANSIEN saja (httpx.TransportError — mencakup ReadError,
    ConnectError, WriteError, PoolTimeout, dll — lihat error yang
    dilaporkan: "ReadError: [Errno 11] Resource temporarily unavailable"),
    lalu mencoba ulang fungsi tsb dari awal setelah membuang klien lama
    (get_supabase.clear()) sehingga percobaan berikutnya memakai koneksi
    baru, bukan koneksi basi yang sama.

    SENGAJA TIDAK menangkap error lain (mis. constraint/validasi data
    dari Postgres) — itu harus tetap langsung gagal dan terlihat,
    supaya bug data tidak ikut tersembunyi oleh mekanisme retry ini.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except httpx.TransportError as e:
                    last_err = e
                    if attempt < max_attempts:
                        try:
                            get_supabase.clear()
                        except Exception:
                            pass
                        time.sleep(base_delay * attempt)
                    else:
                        raise
            raise last_err  # pragma: no cover
        return wrapper
    return decorator


# ── Table initialisation ──────────────────────────────────────

@_retry_on_network_error()
def init_db():
    """
    Create tables if they don't exist.
    Called once at app startup — safe to call repeatedly.
    """
    sb = get_supabase()

    # spectra_library table
    sb.rpc("create_spectra_table_if_not_exists", {}).execute()

    # derivative spectra library table (1st/2nd/... derivative reference
    # spectra — kept SEPARATE from sv_spectra because a derivative
    # spectrum is only valid to match against another derivative
    # spectrum processed with the same SG parameters; mixing domains
    # produces meaningless similarity scores. See sv_spectra_derivative.
    sb.rpc("create_derivative_spectra_table_if_not_exists", {}).execute()

    # users table
    sb.rpc("create_users_table_if_not_exists", {}).execute()

    # Seed default admin if users table is empty
    _seed_default_admin()


@_retry_on_network_error()
def _seed_default_admin():
    """Insert admin/admin123 if no users exist yet."""
    sb = get_supabase()
    resp = sb.table("sv_users").select("username").limit(1).execute()
    if not resp.data:
        sb.table("sv_users").insert({
            "username":     "admin",
            "password":     hashlib.sha256("admin123".encode()).hexdigest(),
            "role":         "admin",
            "display_name": "Administrator"
        }).execute()


# ── User management ───────────────────────────────────────────

@_retry_on_network_error()
def load_users() -> dict:
    sb = get_supabase()
    resp = sb.table("sv_users").select("*").execute()
    users = {}
    for row in resp.data:
        users[row["username"]] = {
            "password": row["password"],
            "role":     row["role"],
            "name":     row["display_name"],
        }
    return users


@_retry_on_network_error()
def save_users(users: dict):
    """
    Full sync: upsert all users in dict.
    Called by auth.py the same way it called the old JSON version.
    """
    sb = get_supabase()
    for username, data in users.items():
        sb.table("sv_users").upsert({
            "username":     username,
            "password":     data["password"],
            "role":         data["role"],
            "display_name": data.get("name", username),
        }, on_conflict="username").execute()


@_retry_on_network_error()
def delete_user(username: str):
    sb = get_supabase()
    sb.table("sv_users").delete().eq("username", username).execute()


# ── Spectral library ──────────────────────────────────────────

# CATATAN: fungsi insert (add_spectrum, add_derivative_spectrum) di bawah
# ini juga dibungkus retry. Ada satu skenario tepi yang TIDAK sepenuhnya
# hilang: kalau insert SEBENARNYA sudah sukses di server tapi respons-nya
# gagal terbaca (persis error yang dilaporkan), retry akan mengirim insert
# kedua → entri terduplikasi. Ini jauh lebih jarang terjadi setelah
# perbaikan ttl cache di atas, tapi kalau Anda ingin risiko ini nol sama
# sekali, beri tahu saya — bisa ditambahkan pengecekan nama duplikat
# sebelum insert (trade-off: mencegah entri dengan nama sama meski
# disengaja).

@_retry_on_network_error()
def add_spectrum(name, category, subcategory, cas_number,
                 notes, wavenumber, spectrum, added_by="admin") -> int:
    sb = get_supabase()
    wn = [float(x) for x in wavenumber]
    sp = [float(x) for x in spectrum]
    resp = sb.table("sv_spectra").insert({
        "name":          name,
        "category":      category or "",
        "subcategory":   subcategory or "",
        "cas_number":    cas_number or "",
        "notes":         notes or "",
        "wavenumber":    json.dumps(wn),
        "spectrum":      json.dumps(sp),
        "n_points":      len(wn),
        "wavenumber_min": min(wn),
        "wavenumber_max": max(wn),
        "added_by":      added_by,
        "added_at":      datetime.utcnow().isoformat(),
    }).execute()
    return resp.data[0]["id"]


@_retry_on_network_error()
def delete_spectrum(spectrum_id: int):
    sb = get_supabase()
    sb.table("sv_spectra").delete().eq("id", spectrum_id).execute()


@_retry_on_network_error()
def update_spectrum_meta(spectrum_id, name, category, subcategory,
                         cas_number, notes):
    sb = get_supabase()
    sb.table("sv_spectra").update({
        "name":        name,
        "category":    category or "",
        "subcategory": subcategory or "",
        "cas_number":  cas_number or "",
        "notes":       notes or "",
    }).eq("id", spectrum_id).execute()


@_retry_on_network_error()
def get_all_meta() -> list:
    sb = get_supabase()
    resp = sb.table("sv_spectra").select(
        "id, name, category, subcategory, cas_number, notes, "
        "n_points, wavenumber_min, wavenumber_max, added_by, added_at"
    ).order("id").execute()
    return resp.data


@_retry_on_network_error()
def get_spectrum_by_id(spectrum_id: int) -> dict | None:
    sb = get_supabase()
    resp = sb.table("sv_spectra").select(
        "id, name, category, wavenumber, spectrum"
    ).eq("id", spectrum_id).single().execute()
    if not resp.data:
        return None
    row = resp.data
    return {
        "id":         row["id"],
        "name":       row["name"],
        "category":   row["category"],
        "wavenumber": json.loads(row["wavenumber"]),
        "spectrum":   json.loads(row["spectrum"]),
    }


@_retry_on_network_error()
def get_all_spectra_for_matching() -> list:
    sb = get_supabase()
    resp = sb.table("sv_spectra").select(
        "id, name, category, wavenumber, spectrum"
    ).execute()
    results = []
    for row in resp.data:
        results.append({
            "id":         row["id"],
            "name":       row["name"],
            "category":   row["category"],
            "wavenumber": json.loads(row["wavenumber"]),
            "spectrum":   json.loads(row["spectrum"]),
        })
    return results


@_retry_on_network_error()
def count_spectra() -> int:
    sb = get_supabase()
    resp = sb.table("sv_spectra").select("id", count="exact").execute()
    return resp.count or 0


@_retry_on_network_error()
def get_categories() -> list:
    sb = get_supabase()
    resp = sb.table("sv_spectra").select("category").execute()
    cats = sorted(set(r["category"] for r in resp.data if r["category"]))
    return cats


# ── Derivative spectral library ───────────────────────────────
# Table sv_spectra_derivative — SEPARATE dari sv_spectra. Setiap entri
# menyimpan spektrum turunan (bukan mentah) beserta metadata parameter
# SG yang dipakai membuatnya (deriv_order, sg_window, sg_poly), dan
# opsional referensi ke entri sv_spectra asal (source_spectrum_id) untuk
# keterlacakan/audit trail. Matching HANYA valid antar entri dengan
# deriv_order yang sama — lihat batch_match_derivative() di mcr_engine.py.

@_retry_on_network_error()
def add_derivative_spectrum(name, category, subcategory, deriv_order,
                            sg_window, sg_poly, wavenumber, spectrum,
                            source_spectrum_id=None, notes="",
                            added_by="admin") -> int:
    sb = get_supabase()
    wn = [float(x) for x in wavenumber]
    sp = [float(x) for x in spectrum]
    resp = sb.table("sv_spectra_derivative").insert({
        "name":               name,
        "category":           category or "",
        "subcategory":        subcategory or "",
        "deriv_order":        int(deriv_order),
        "sg_window":          int(sg_window),
        "sg_poly":            int(sg_poly),
        "source_spectrum_id": source_spectrum_id,
        "notes":              notes or "",
        "wavenumber":         json.dumps(wn),
        "spectrum":           json.dumps(sp),
        "n_points":           len(wn),
        "wavenumber_min":     min(wn),
        "wavenumber_max":     max(wn),
        "added_by":           added_by,
        "added_at":           datetime.utcnow().isoformat(),
    }).execute()
    return resp.data[0]["id"]


@_retry_on_network_error()
def delete_derivative_spectrum(spectrum_id: int):
    sb = get_supabase()
    sb.table("sv_spectra_derivative").delete().eq("id", spectrum_id).execute()


@_retry_on_network_error()
def get_all_derivative_meta(deriv_order: int | None = None) -> list:
    sb = get_supabase()
    q = sb.table("sv_spectra_derivative").select(
        "id, name, category, subcategory, deriv_order, sg_window, sg_poly, "
        "source_spectrum_id, notes, n_points, wavenumber_min, wavenumber_max, "
        "added_by, added_at"
    )
    if deriv_order is not None:
        q = q.eq("deriv_order", int(deriv_order))
    resp = q.order("id").execute()
    return resp.data


@_retry_on_network_error()
def get_derivative_spectrum_by_id(spectrum_id: int) -> dict | None:
    sb = get_supabase()
    resp = sb.table("sv_spectra_derivative").select(
        "id, name, category, deriv_order, sg_window, sg_poly, "
        "source_spectrum_id, wavenumber, spectrum"
    ).eq("id", spectrum_id).single().execute()
    if not resp.data:
        return None
    row = resp.data
    return {
        "id":                 row["id"],
        "name":               row["name"],
        "category":           row["category"],
        "deriv_order":        row["deriv_order"],
        "sg_window":          row["sg_window"],
        "sg_poly":            row["sg_poly"],
        "source_spectrum_id": row["source_spectrum_id"],
        "wavenumber":         json.loads(row["wavenumber"]),
        "spectrum":           json.loads(row["spectrum"]),
    }


@_retry_on_network_error()
def get_all_derivative_spectra_for_matching(deriv_order: int) -> list:
    """
    Hanya mengembalikan entri dengan deriv_order yang SAMA PERSIS dengan
    yang diminta — pemanggil (batch_match_derivative) tidak perlu
    memfilter ulang, tapi filter di sini dipertahankan sebagai lapisan
    keamanan kedua supaya domain turunan yang berbeda tidak pernah
    tercampur secara tidak sengaja.
    """
    sb = get_supabase()
    resp = sb.table("sv_spectra_derivative").select(
        "id, name, category, deriv_order, sg_window, sg_poly, wavenumber, spectrum"
    ).eq("deriv_order", int(deriv_order)).execute()
    results = []
    for row in resp.data:
        results.append({
            "id":          row["id"],
            "name":        row["name"],
            "category":    row["category"],
            "deriv_order": row["deriv_order"],
            "sg_window":   row["sg_window"],
            "sg_poly":     row["sg_poly"],
            "wavenumber":  json.loads(row["wavenumber"]),
            "spectrum":    json.loads(row["spectrum"]),
        })
    return results


@_retry_on_network_error()
def count_derivative_spectra(deriv_order: int | None = None) -> int:
    sb = get_supabase()
    q = sb.table("sv_spectra_derivative").select("id", count="exact")
    if deriv_order is not None:
        q = q.eq("deriv_order", int(deriv_order))
    resp = q.execute()
    return resp.count or 0


def import_from_json(filepath: str, added_by="admin") -> int:
    with open(filepath) as f:
        entries = json.load(f)
    count = 0
    for e in entries:
        try:
            add_spectrum(
                name        = e.get("name", "Unknown"),
                category    = e.get("category", ""),
                subcategory = e.get("subcategory", ""),
                cas_number  = e.get("cas_number", ""),
                notes       = e.get("notes", ""),
                wavenumber  = e["wavenumber"],
                spectrum    = e["spectrum"],
                added_by    = added_by,
            )
            count += 1
        except Exception:
            continue
    return count
