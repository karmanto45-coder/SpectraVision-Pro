"""
robustness_sweep_engine.py — SpectraVision Pro
=================================================
Modul ADITIF untuk Tier 4 brief riset JM/JE/MA (Section 14): studi
robustness TERSTRUKTUR yang menjalankan pipeline MCR-ALS -> ambiguity ->
SIRI -> hierarchical decision (identity_decision_engine.py) berulang
kali di bawah variasi:

  1. Metode inisialisasi MCR-ALS (pca/simplisma/nmf) — "init grid"
  2. Preprocessing (raw/baseline+norm/smooth+baseline+norm/SNV)
  3. Region spektral (window wavenumber penuh vs dipersempit)
  4. Noise perturbation berulang (bootstrap sederhana: tambah noise
     kecil pada data riil dan ulangi n_reps kali)

...lalu melaporkan APAKAH VERDICT IDENTITAS (bukan cuma LOF/fit) tetap
stabil di seluruh variasi — hasil dari sini dipakai sebagai nilai
`robustness_ok` untuk evaluate_component_identity() di
identity_decision_engine.py (parameter itu sebelumnya berupa
placeholder None sampai modul ini dibangun).

BEDA dengan run_mcr_multi_k() yang SUDAH ADA di mcr_engine.py:
run_mcr_multi_k menguji STABILITAS ANTAR-k (2,3,4,...) lewat Spectral
Reproducibility Index murni pada level spektra. Modul INI menguji
apakah KEPUTUSAN IDENTITAS AKHIR (bukan cuma kemiripan spektra mentah)
tetap konsisten di bawah variasi konfigurasi pipeline yang JAUH lebih
luas dari sekadar k (init, preprocessing, region, noise) — dua alat ini
saling melengkapi, bukan duplikat.

PENTING - keterbatasan yang harus dinyatakan di manuskrip jika dipakai:
  - Preprocessing di modul ini SENGAJA ditulis ulang di konvensi array
    (n_sampel x n_wavenumber) yang dipakai run_mcr_als(), BUKAN dengan
    memanggil preprocess() yang sudah ada di mcr_engine.py — karena
    preprocess() memakai konvensi TRANSPOSE (n_wavenumber x n_sampel,
    sama seperti konvensi session_state yang sudah dipakai di app.py).
    Operasinya (baseline min-subtract, area-norm, Savitzky-Golay
    smoothing) sama persis, hanya beda layout array, supaya tidak perlu
    transpose bolak-balik dan menghindari bug orientasi.
  - Threshold stability_threshold_pct (default 80%) adalah KONVENSI
    INTERPRETATIF proyek ini, sama seperti threshold-threshold lain di
    ambiguity_engine.py/identity_decision_engine.py — bukan standar baku.
  - Grid sweep berasumsi n_components (jumlah komponen) SUDAH diketahui
    benar dari domain knowledge (JM/JE/MA = 3). Sensitivitas terhadap
    kesalahan spesifikasi k sendiri sebaiknya dicek terpisah lewat
    run_mcr_multi_k() yang sudah ada, bukan diulang di modul ini.
"""

import numpy as np
from scipy.signal import savgol_filter

from mcr_engine import run_mcr_als, pearson_corr, cosine_sim
from ambiguity_engine import compute_rotational_ambiguity
from identity_decision_engine import (
    compute_identity_robustness_index,
    evaluate_component_identity,
)
from similarity_simulator import match_components_to_truth

DEFAULT_STABILITY_THRESHOLD_PCT = 80.0

# Label kategori keputusan yang dianggap "baik" untuk perhitungan stabilitas
GOOD_LABELS = {"reliably_identified", "provisional"}
# Label yang HARUS ditandai eksplisit sebagai temuan serius kalau pernah
# muncul walau cuma sekali dalam sweep (bukan cuma dirata-rata)
SERIOUS_LABELS = {"possible_false_resolution"}


# ═══════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING VARIANTS — konvensi (n_sampel x n_wavenumber),
#    operasi sama dengan preprocess() di mcr_engine.py, layout beda.
# ═══════════════════════════════════════════════════════════════════════

def _trapz(y, x):
    """Fallback konsisten dengan pola yang sudah dipakai di mcr_engine.py
    (np.trapz dihapus di NumPy 2.0+, diganti np.trapezoid)."""
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)


def _preprocess_raw(D, wavenumber):
    return D.copy()


def _preprocess_baseline_norm(D, wavenumber):
    D = D.copy()
    D = D - D.min(axis=1, keepdims=True)
    wn = np.asarray(wavenumber, dtype=float)
    for i in range(D.shape[0]):
        area = _trapz(np.abs(D[i]), wn)
        if area > 0:
            D[i] = D[i] / area
    return D


def _preprocess_smooth_baseline_norm(D, wavenumber):
    D = D.copy()
    win = min(11, D.shape[1] - (1 - D.shape[1] % 2))
    if win >= 5:
        win = win if win % 2 == 1 else win - 1
        poly = min(3, win - 2)
        for i in range(D.shape[0]):
            D[i] = savgol_filter(D[i], win, poly)
    return _preprocess_baseline_norm(D, wavenumber)


def _preprocess_snv(D, wavenumber):
    """Standard Normal Variate — per-sampel: kurangi mean, bagi std.
    Preprocessing ini yang sebelumnya BELUM ADA di preprocess()
    mcr_engine.py, padahal brief Section 14 eksplisit minta SNV
    sebagai salah satu preprocessing yang dibandingkan.

    PERINGATAN METODOLOGIS (ditemukan lewat smoke test modul ini,
    penting dinyatakan di manuskrip kalau varian ini dipakai): SNV
    mengurangi MEAN PER-BARIS (per sampel) dan membagi STD PER-BARIS —
    kedua operasi ini TIDAK berkorespondensi ke transformasi linear
    Beer-Lambert manapun, sehingga MENERAPKAN SNV PADA SPEKTRUM
    CAMPURAN (SEBELUM MCR-ALS) berpotensi merusak struktur bilinear
    D = C @ S yang menjadi fondasi seluruh model MCR. Ini BEDA dari
    normalisasi area (baseline_norm) yang murni skala per sampel
    (kompatibel dengan model, karena C bisa menyerap faktor skala).
    Varian ini disediakan karena brief eksplisit memintanya sebagai
    pembanding robustness, TAPI hasil dari varian 'snv' sebaiknya
    dibaca sebagai 'apa yang terjadi jika preprocessing ini (secara
    keliru) diterapkan', bukan sebagai preprocessing yang direkomendasikan
    untuk data campuran nyata. SNV lebih tepat dipakai untuk
    membandingkan SPEKTRUM MURNI (post-MCR atau library-to-library),
    bukan pra-proses data mixture.
    """
    D = D.copy()
    mean = D.mean(axis=1, keepdims=True)
    std = D.std(axis=1, keepdims=True)
    std[std < 1e-12] = 1.0
    return (D - mean) / std


PREPROCESSING_VARIANTS = {
    "raw": _preprocess_raw,
    "baseline_norm": _preprocess_baseline_norm,
    "smooth_baseline_norm": _preprocess_smooth_baseline_norm,
    "snv": _preprocess_snv,
}

# s_nonneg yang secara domain BENAR untuk tiap preprocessing — SNV
# menghasilkan nilai negatif secara struktural (sama seperti domain
# turunan yang sudah didokumentasikan di run_mcr_als()), jadi
# s_nonneg WAJIB False untuk domain ini supaya tidak di-clip paksa
# dan mendorong ALS ke solusi degenerate (lihat peringatan di atas).
PREPROCESSING_S_NONNEG_DEFAULT = {
    "raw": True,
    "baseline_norm": True,
    "smooth_baseline_norm": True,
    "snv": False,
}


# ═══════════════════════════════════════════════════════════════════════
# 2. SATU KONFIGURASI: jalankan MCR-ALS + evaluasi identitas
# ═══════════════════════════════════════════════════════════════════════

def _crop_to_window(D, wavenumber, S_reference, window):
    if window is None:
        return D, np.asarray(wavenumber, dtype=float), S_reference
    wn = np.asarray(wavenumber, dtype=float)
    lo, hi = window
    mask = (wn >= lo) & (wn <= hi)
    if mask.sum() < 4:
        raise ValueError(f"Window {window} menyisakan <4 titik data — terlalu sempit.")
    return D[:, mask], wn[mask], S_reference[:, mask]


def _run_and_evaluate(D, wavenumber, n_components, S_reference, component_labels,
                       init_method="simplisma", mcr_kwargs=None, seed=None,
                       n_directions_ambiguity=15):
    """
    Jalankan SATU kombinasi (D sudah dipreprocess & di-crop) lewat
    MCR-ALS, cocokkan ke reference lewat Hungarian assignment (reuse
    match_components_to_truth dari similarity_simulator.py), lalu
    hitung similarity/ambiguity/SIRI/decision per komponen.

    Returns list of dict, satu per component_labels, dengan
    field-field yang sama dengan output run_single_replicate() di
    similarity_simulator.py supaya konsisten dibaca lintas modul.
    """
    mcr_kwargs = dict(mcr_kwargs or {})
    # closure=False (default project run_mcr_als) — lihat catatan di
    # blind_validation_engine.py soal closure=True menjebak ALS di
    # local optimum saat skala antar komponen jauh berbeda.
    mcr_defaults = dict(max_iter=300, tol=1e-6, closure=False, init_method=init_method)
    mcr_defaults.update(mcr_kwargs)

    C, S, lof_history, r2, converged, diagnostics = run_mcr_als(
        D, n_components=n_components, **mcr_defaults
    )
    assignment, cosine_matrix = match_components_to_truth(S_reference, S)
    constraints_used = diagnostics.get("constraints_used", {
        "s_nonneg": mcr_defaults.get("s_nonneg", True),
        "closure": mcr_defaults.get("closure", False),
    })
    fit_ok = bool(converged) and float(diagnostics.get("lof_final", 100.0)) < 10.0

    try:
        ambiguity_result = compute_rotational_ambiguity(
            C, S, constraints_used, method="auto",
            n_directions=n_directions_ambiguity, seed=seed,
        )
    except Exception as e:
        ambiguity_result = {"error": str(e), "components": []}
    amb_list = ambiguity_result.get("components", [])

    out = []
    for i, label in enumerate(component_labels):
        j = assignment[i]
        component_collapsed = bool(np.linalg.norm(S[j]) < 1e-8)
        pear = pearson_corr(S_reference[i], S[j])
        cos = cosine_sim(S_reference[i], S[j])
        amb_comp = amb_list[j] if j < len(amb_list) else None

        siri_result = None
        if amb_comp is not None:
            siri_result = compute_identity_robustness_index(
                S[j], amb_comp["S_band_min"], amb_comp["S_band_max"], S_reference[i],
            )

        decision = evaluate_component_identity(
            component_label=label, fit_ok=fit_ok, pearson=pear, cosine=cos,
            ambiguity_width_pct=amb_comp["ambiguity_width_pct"] if amb_comp else None,
            ambiguity_reliability_label=amb_comp["reliability"]["label"] if amb_comp else None,
            siri_result=siri_result,
        )
        out.append({
            "true_label": label,
            "component_collapsed": component_collapsed,
            "pearson": pear, "cosine": cos,
            "ambiguity_width_pct": amb_comp["ambiguity_width_pct"] if amb_comp else None,
            "siri_pct": siri_result["siri_pct"] if siri_result else None,
            "final_label": decision["final_label"],
            "fit_ok": fit_ok, "converged": bool(converged),
            "lof_final": float(diagnostics.get("lof_final", float("nan"))),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════
# 3. GRID SWEEP: init x preprocessing x region
# ═══════════════════════════════════════════════════════════════════════

def run_robustness_grid(D_raw, wavenumber, S_reference, component_labels,
                         n_components=3,
                         init_methods=("pca", "simplisma", "nmf"),
                         preprocessing_variants=("raw", "baseline_norm",
                                                  "smooth_baseline_norm", "snv"),
                         wavenumber_windows=(None,),
                         mcr_kwargs=None, seed=0):
    """
    Jalankan seluruh kombinasi (init_method x preprocessing x window)
    dan kumpulkan hasil evaluate_component_identity untuk tiap kombinasi.

    D_raw          : (n_samples x n_wavenumber) — data mixture MENTAH
                     (sebelum preprocessing apa pun).
    S_reference    : (n_komponen x n_wavenumber) — spektrum reference
                     murni untuk tiap label di component_labels, PADA
                     GRID WAVENUMBER YANG SAMA dengan D_raw.
    component_labels : list[str] — nama komponen sesuai urutan baris
                     S_reference (mis. ["JM","JE","MA"]).

    Returns
    -------
    dict:
      combos  : list of dict {init_method, preprocessing, window,
                results (list per komponen dari _run_and_evaluate)}
      summary : dict per label — stability_pct, labels_seen (set),
                any_serious_flag (bool), n_combos_run, n_combos_collapsed
    """
    combos = []
    for window in wavenumber_windows:
        for prep_name in preprocessing_variants:
            prep_fn = PREPROCESSING_VARIANTS[prep_name]
            # s_nonneg domain-appropriate per preprocessing (lihat catatan
            # _preprocess_snv), kecuali user eksplisit mengoverride sendiri.
            combo_mcr_kwargs = dict(mcr_kwargs or {})
            combo_mcr_kwargs.setdefault("s_nonneg", PREPROCESSING_S_NONNEG_DEFAULT[prep_name])
            for init_method in init_methods:
                try:
                    D_c, wn_c, S_ref_c = _crop_to_window(D_raw, wavenumber, S_reference, window)
                    D_proc = prep_fn(D_c, wn_c)
                    results = _run_and_evaluate(
                        D_proc, wn_c, n_components, S_ref_c, component_labels,
                        init_method=init_method, mcr_kwargs=combo_mcr_kwargs, seed=seed,
                    )
                    combos.append({
                        "init_method": init_method, "preprocessing": prep_name,
                        "window": window, "results": results, "error": None,
                    })
                except Exception as e:
                    combos.append({
                        "init_method": init_method, "preprocessing": prep_name,
                        "window": window, "results": None, "error": str(e),
                    })

    summary = _summarize_combos(combos, component_labels)
    return {"combos": combos, "summary": summary}


def _summarize_combos(combos, component_labels):
    summary = {}
    valid_combos = [c for c in combos if c["results"] is not None]
    for label in component_labels:
        labels_seen = []
        n_collapsed = 0
        for c in valid_combos:
            comp = next((r for r in c["results"] if r["true_label"] == label), None)
            if comp is None:
                continue
            labels_seen.append(comp["final_label"])
            if comp["component_collapsed"]:
                n_collapsed += 1
        n = len(labels_seen)
        n_good = sum(1 for l in labels_seen if l in GOOD_LABELS)
        stability_pct = 100.0 * n_good / n if n > 0 else float("nan")
        any_serious = any(l in SERIOUS_LABELS for l in labels_seen)
        summary[label] = {
            "stability_pct": stability_pct,
            "labels_seen": labels_seen,
            "any_serious_flag": any_serious,
            "n_combos_run": n,
            "n_combos_component_collapsed": n_collapsed,
            "n_combos_failed_to_run": len(combos) - len(valid_combos),
        }
    return summary


# ═══════════════════════════════════════════════════════════════════════
# 4. NOISE BOOTSTRAP: ulangi n_reps kali dengan noise tambahan kecil
# ═══════════════════════════════════════════════════════════════════════

def run_noise_bootstrap_robustness(D_raw, wavenumber, S_reference, component_labels,
                                    n_components=3, n_reps=15, noise_pct=1.0,
                                    init_method="simplisma",
                                    preprocessing_variant="baseline_norm",
                                    mcr_kwargs=None, base_seed=0):
    """
    Tambahkan noise Gaussian kecil (noise_pct% dari max(|D_raw|)) ke
    data MENTAH riil sebanyak n_reps kali dengan seed berbeda, jalankan
    pipeline penuh tiap kali, dan lihat apakah keputusan identitas
    tetap stabil terhadap gangguan noise realistis (bukan cuma
    perbedaan konfigurasi pipeline seperti run_robustness_grid).
    """
    prep_fn = PREPROCESSING_VARIANTS[preprocessing_variant]
    combo_mcr_kwargs = dict(mcr_kwargs or {})
    combo_mcr_kwargs.setdefault("s_nonneg", PREPROCESSING_S_NONNEG_DEFAULT[preprocessing_variant])
    combos = []
    for rep in range(n_reps):
        rng = np.random.default_rng(base_seed + rep)
        sigma = (noise_pct / 100.0) * float(np.max(np.abs(D_raw)))
        D_noisy = D_raw + rng.normal(0, sigma, size=D_raw.shape)
        try:
            D_proc = prep_fn(D_noisy, wavenumber)
            results = _run_and_evaluate(
                D_proc, wavenumber, n_components, S_reference, component_labels,
                init_method=init_method, mcr_kwargs=combo_mcr_kwargs, seed=base_seed + rep,
            )
            combos.append({"rep": rep, "results": results, "error": None})
        except Exception as e:
            combos.append({"rep": rep, "results": None, "error": str(e)})

    # _summarize_combos hanya butuh field "results", jadi bisa dipakai ulang
    summary = _summarize_combos(combos, component_labels)
    return {"combos": combos, "summary": summary}


# ═══════════════════════════════════════════════════════════════════════
# 5. GABUNGKAN JADI robustness_ok PER KOMPONEN (untuk
#    identity_decision_engine.evaluate_component_identity)
# ═══════════════════════════════════════════════════════════════════════

def summarize_robustness(grid_summary, bootstrap_summary, component_labels,
                          stability_threshold_pct=DEFAULT_STABILITY_THRESHOLD_PCT):
    """
    Gabungkan hasil run_robustness_grid() dan run_noise_bootstrap_robustness()
    jadi SATU verdict robustness_ok per komponen — nilai ini yang dipakai
    sebagai argumen robustness_ok di evaluate_component_identity() untuk
    keputusan identitas FINAL (menggantikan placeholder None).

    Aturan (konservatif, mengikuti prinsip "jangan klaim robust kecuali
    dua sumber bukti robustness sama-sama mendukung"):
      - robustness_ok = False kalau SALAH SATU dari grid/bootstrap pernah
        menunjukkan any_serious_flag (possible_false_resolution) —
        temuan serius tidak boleh "dirata-ratakan hilang".
      - robustness_ok = True kalau KEDUA stability_pct (grid & bootstrap)
        >= stability_threshold_pct DAN tidak ada serious flag.
      - robustness_ok = None (belum bisa disimpulkan) kalau salah satu
        dari grid/bootstrap gagal dijalankan sepenuhnya (n_combos_run=0)
        atau stability_pct di antara ambang (moderate, tidak jelas).
    """
    out = {}
    for label in component_labels:
        g = grid_summary.get(label, {})
        b = bootstrap_summary.get(label, {})
        g_stab = g.get("stability_pct", float("nan"))
        b_stab = b.get("stability_pct", float("nan"))
        serious = g.get("any_serious_flag", False) or b.get("any_serious_flag", False)

        if serious:
            out[label] = {
                "robustness_ok": False,
                "detail": (
                    f"'{label}': possible_false_resolution muncul pada setidaknya satu "
                    f"kombinasi robustness sweep (grid dan/atau bootstrap) — identitas "
                    f"TIDAK boleh diklaim robust, terlepas dari rata-rata stabilitas."
                ),
            }
            continue

        if np.isnan(g_stab) or np.isnan(b_stab):
            out[label] = {
                "robustness_ok": None,
                "detail": f"'{label}': robustness sweep belum lengkap dijalankan (grid atau bootstrap kosong/gagal).",
            }
            continue

        if g_stab >= stability_threshold_pct and b_stab >= stability_threshold_pct:
            out[label] = {
                "robustness_ok": True,
                "detail": (
                    f"'{label}': verdict identitas stabil di {g_stab:.0f}% kombinasi "
                    f"grid dan {b_stab:.0f}% replikat noise bootstrap (ambang "
                    f"{stability_threshold_pct:.0f}%)."
                ),
            }
        else:
            out[label] = {
                "robustness_ok": False,
                "detail": (
                    f"'{label}': stabilitas di bawah ambang ({stability_threshold_pct:.0f}%) "
                    f"— grid={g_stab:.0f}%, bootstrap={b_stab:.0f}%."
                ),
            }
    return out


def build_robustness_dataframe(grid_result, bootstrap_result, robustness_verdict, component_labels):
    """Tabel ringkas untuk fitur Laporan / tabel supplementary manuskrip,
    mengikuti pola build_*_dataframe yang sudah ada di
    mcr_replicate_extension.py dan identity_decision_engine.py."""
    import pandas as pd
    rows = []
    for label in component_labels:
        g = grid_result["summary"].get(label, {})
        b = bootstrap_result["summary"].get(label, {})
        v = robustness_verdict.get(label, {})
        rows.append({
            "Komponen": label,
            "Stabilitas grid (%)": g.get("stability_pct"),
            "Kombinasi grid dijalankan": g.get("n_combos_run"),
            "Kolaps saat grid": g.get("n_combos_component_collapsed"),
            "Stabilitas bootstrap noise (%)": b.get("stability_pct"),
            "Replikat bootstrap dijalankan": b.get("n_combos_run"),
            "possible_false_resolution pernah muncul": g.get("any_serious_flag", False) or b.get("any_serious_flag", False),
            "robustness_ok": v.get("robustness_ok"),
            "Keterangan": v.get("detail"),
        })
    return pd.DataFrame(rows)
