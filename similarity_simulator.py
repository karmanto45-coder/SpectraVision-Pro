"""
similarity_simulator.py — SpectraVision Pro
=============================================
Modul ADITIF untuk kebutuhan Section 13.1/30 brief riset JM/JE/MA:
studi simulasi dengan KORELASI SPEKTRAL antar dua komponen mirip (JM-JE)
yang BISA DIATUR secara eksak (bukan didekati/coba-coba), plus noise
terkendali, untuk membangun ground truth yang menentukan KAPAN
similarity-only identification mulai gagal dan apakah ambiguity-aware
framework (ambiguity_engine.py + identity_decision_engine.py) memberi
informasi tambahan pada titik itu.

BEDA dengan simulator_engine.py yang sudah ada di proyek ini (dari
histori pengembangan): simulator itu membangun mixture dari SPEKTRUM
MURNI RIIL (library) + rasio komposisi — cocok untuk validasi memakai
data nyata (santan, bumbu instan). Modul INI membangun spektrum murni
SINTETIS dengan korelasi Pearson yang DIJAMIN EKSAK terhadap target
(mis. 0.70/0.85/0.95/0.99, lihat tabel Section 13.1 brief), karena
studi simulasi khusus ini butuh sapuan (sweep) korelasi yang presisi
sebagai sumbu-x Figure 4 brief — sesuatu yang tidak bisa didapat dari
spektrum riil apa adanya. Dua modul ini melengkapi, bukan saling
menggantikan.

Konstruksi korelasi eksak: memakai dekomposisi ortogonal standar
(bukan trial-and-error/rejection sampling atas korelasi) —
    A_std, w_orth  : zero-mean, unit L2-norm, saling ORTOGONAL
    B_std = r * A_std + sqrt(1-r^2) * w_orth
menghasilkan corr(A_std, B_std) = r secara EKSAK (bukan pendekatan),
karena Pearson correlation dari dua vektor zero-mean sama dengan
dot-product-nya ketika keduanya di-unit-norm-kan (mengikuti definisi
pearson_corr() yang SUDAH ADA di mcr_engine.py — lihat verifikasi di
smoke test/docstring test_exact_correlation()). Pergeseran/skala aditif
untuk membuat spektrum non-negatif (absorbansi) TIDAK mengubah nilai
korelasi ini (Pearson invariant terhadap transformasi affine per
vektor).

PENTING — keterbatasan yang harus dinyatakan di manuskrip jika dipakai:
  - Spektrum sintetis di sini adalah kombinasi puncak Gaussian acak,
    BUKAN spektrum FTIR riil — dipakai untuk membangun ground truth
    KORELASI yang presisi (sumbu-x studi), bukan untuk klaim realisme
    kimia. Validasi akhir tetap harus memakai data eksperimen riil JM/
    JE/MA (Section 13.2/13.3 brief).
  - Noise model di sini adalah Gaussian additive homoskedastik/
    heteroskedastik sederhana (persentase dari intensitas maksimum),
    bukan model noise instrumen ATR-FTIR yang divalidasi.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from mcr_engine import run_mcr_als, pearson_corr, cosine_sim
from ambiguity_engine import compute_rotational_ambiguity
from identity_decision_engine import (
    compute_identity_robustness_index,
    evaluate_component_identity,
    SIRI_LOW_THRESHOLD_PCT,
    SIMILARITY_STRONG_THRESHOLD,
)

# Tabel kasus default — persis Section 13.1 brief (Easy/Moderate/Difficult/Extreme)
DEFAULT_CASES = [
    {"case_name": "Easy",      "target_correlation": 0.70, "noise_pct": 1.0},
    {"case_name": "Moderate",  "target_correlation": 0.85, "noise_pct": 1.0},
    {"case_name": "Difficult", "target_correlation": 0.95, "noise_pct": 2.0},
    {"case_name": "Extreme",   "target_correlation": 0.99, "noise_pct": 5.0},
]


# ═══════════════════════════════════════════════════════════════════════
# 1. GENERATOR SPEKTRUM SINTETIS
# ═══════════════════════════════════════════════════════════════════════

def _zero_mean_unit_norm(v):
    v = v - v.mean()
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


def _random_smooth_spectrum(wavenumber, n_peaks, peak_width_frac, rng,
                            wn_range=None):
    """Bangun 'spektrum' halus dari kombinasi beberapa puncak Gaussian
    acak — dipakai sebagai bahan baku sebelum konstruksi korelasi eksak.
    wn_range: (min,max) opsional untuk membatasi posisi puncak ke
    sub-region tertentu (mis. mensimulasikan region dominan maltodekstrin
    yang berbeda dari region dominan jahe)."""
    wn = np.asarray(wavenumber, dtype=float)
    lo, hi = wn_range if wn_range is not None else (wn.min(), wn.max())
    span = wn.max() - wn.min()
    spec = np.zeros_like(wn)
    for _ in range(n_peaks):
        center = rng.uniform(lo, hi)
        width = peak_width_frac * span * rng.uniform(0.6, 1.4)
        height = rng.uniform(0.3, 1.0)
        spec += height * np.exp(-((wn - center) ** 2) / (2 * width ** 2))
    return spec


def generate_correlated_pure_spectra(wavenumber, target_correlation,
                                      n_peaks_shared=5, n_peaks_unique=3,
                                      peak_width_frac=0.02, baseline_margin=1.05,
                                      seed=None):
    """
    Bangun dua spektrum murni sintetis (mis. 'JM-like' dan 'JE-like')
    dengan korelasi Pearson EKSAK sama dengan target_correlation.

    Parameters
    ----------
    wavenumber         : (n_points,) grid bilangan gelombang.
    target_correlation : float di [-1, 1] — korelasi Pearson yang
                         diinginkan antara dua spektrum keluaran.
    n_peaks_shared     : jumlah puncak 'backbone' bersama (mensimulasikan
                         kemiripan botani JM-JE).
    n_peaks_unique     : jumlah puncak unik tambahan pada spektrum A
                         SEBELUM dipakai sebagai basis dekomposisi
                         ortogonal (lihat catatan modul).
    peak_width_frac    : lebar puncak relatif terhadap rentang wavenumber.
    baseline_margin    : faktor pengali di atas |nilai ekstrem| untuk
                         offset non-negativity (>1.0 supaya tidak
                         menyentuh nol persis).
    seed               : RNG seed untuk reproducibility.

    Returns
    -------
    dict:
      S_a, S_b              : (n_points,) dua spektrum murni sintetis,
                               non-negatif.
      achieved_correlation   : korelasi Pearson AKTUAL (harus == target
                               correlation hingga presisi floating point
                               — lihat verifikasi di bawah).
    """
    rng = np.random.default_rng(seed)
    wn = np.asarray(wavenumber, dtype=float)
    r = float(np.clip(target_correlation, -0.999999, 0.999999))

    shared = _random_smooth_spectrum(wn, n_peaks_shared, peak_width_frac, rng)
    unique_a = _random_smooth_spectrum(wn, n_peaks_unique, peak_width_frac, rng)
    unique_b = _random_smooth_spectrum(wn, n_peaks_unique, peak_width_frac, rng)

    base_a = shared + unique_a
    A_std = _zero_mean_unit_norm(base_a)

    w_raw = _zero_mean_unit_norm(unique_b)
    w_orth = w_raw - float(np.dot(w_raw, A_std)) * A_std
    w_orth = _zero_mean_unit_norm(w_orth)

    B_std = r * A_std + np.sqrt(max(0.0, 1.0 - r ** 2)) * w_orth

    # ── Offset supaya non-negatif (absorbansi) — TIDAK mengubah korelasi
    # (Pearson invariant terhadap penambahan konstanta per vektor).
    offset = baseline_margin * float(np.max(np.abs(np.concatenate([A_std, B_std]))))
    S_a = A_std + offset
    S_b = B_std + offset

    achieved = pearson_corr(S_a, S_b)

    return {"S_a": S_a, "S_b": S_b, "achieved_correlation": achieved,
            "target_correlation": r}


def generate_independent_spectrum(wavenumber, references, max_correlation_vs_refs=0.30,
                                   n_peaks=5, peak_width_frac=0.025,
                                   baseline_margin=1.05, seed=None, max_attempts=100):
    """
    Bangun spektrum murni sintetis ketiga (mis. 'MA-like'/maltodekstrin)
    yang korelasinya terhadap SEMUA spektrum di `references` di bawah
    max_correlation_vs_refs — via rejection sampling (dicoba ulang
    sampai memenuhi syarat atau max_attempts habis).

    Berbeda dari generate_correlated_pure_spectra(): di sini korelasi
    TIDAK dikonstruksi eksak (karena syaratnya "cukup rendah", bukan
    nilai target presisi), jadi rejection sampling memang metode yang
    tepat dan cukup, konsisten dengan brief Section 4 bahwa JM-MA/JE-MA
    diharapkan JAUH lebih mudah dipisahkan daripada JM-JE.

    Returns
    -------
    dict: S_c, achieved_max_correlation, n_attempts, satisfied (bool)
    """
    rng = np.random.default_rng(seed)
    wn = np.asarray(wavenumber, dtype=float)
    best = None
    for attempt in range(1, max_attempts + 1):
        cand = _random_smooth_spectrum(wn, n_peaks, peak_width_frac, rng)
        offset = baseline_margin * float(np.max(np.abs(cand))) if np.max(np.abs(cand)) > 0 else 1.0
        cand_pos = cand + offset - cand.min()  # pastikan >=~0 dengan margin
        max_corr = max(abs(pearson_corr(cand_pos, ref)) for ref in references)
        if best is None or max_corr < best[1]:
            best = (cand_pos, max_corr)
        if max_corr <= max_correlation_vs_refs:
            return {"S_c": cand_pos, "achieved_max_correlation": max_corr,
                    "n_attempts": attempt, "satisfied": True}
    return {"S_c": best[0], "achieved_max_correlation": best[1],
            "n_attempts": max_attempts, "satisfied": False}


# ═══════════════════════════════════════════════════════════════════════
# 2. DESAIN KOMPOSISI & FORWARD MODEL (X = C @ S + noise)
# ═══════════════════════════════════════════════════════════════════════

def generate_dirichlet_design(n_samples, n_components, alpha=None, seed=None):
    """Rasio komposisi acak di simplex (closure otomatis: tiap baris
    berjumlah 1) via distribusi Dirichlet. alpha=None -> uniform
    (alpha=1 untuk semua komponen)."""
    rng = np.random.default_rng(seed)
    if alpha is None:
        alpha = np.ones(n_components)
    return rng.dirichlet(alpha, size=n_samples)


def simulate_linear_mixture(C, S, noise_pct=1.0, noise_mode="homoscedastic", seed=None):
    """
    Forward model Beer-Lambert linier: D = C @ S + noise.

    noise_mode : "none" | "homoscedastic" (std tetap = noise_pct% dari
                 max(|D_clean|)) | "heteroscedastic" (std per titik
                 proporsional terhadap sqrt(|D_clean per titik|), skala
                 disesuaikan agar rata-rata setara dengan mode
                 homoscedastic pada noise_pct yang sama — mensimulasikan
                 noise mirip-Poisson yang lebih besar di puncak).

    Returns
    -------
    D : (n_samples x n_points) — data campuran bersimulasi.
    D_clean : (n_samples x n_points) — tanpa noise (ground truth murni).
    """
    rng = np.random.default_rng(seed)
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    D_clean = C @ S

    if noise_mode == "none" or noise_pct <= 0:
        return D_clean.copy(), D_clean

    sigma_base = (noise_pct / 100.0) * float(np.max(np.abs(D_clean)))

    if noise_mode == "homoscedastic":
        noise = rng.normal(0, sigma_base, size=D_clean.shape)
    elif noise_mode == "heteroscedastic":
        local_scale = np.sqrt(np.abs(D_clean) + 1e-12)
        local_scale = local_scale / (np.mean(local_scale) + 1e-12)  # normalisasi rata-rata ~1
        noise = rng.normal(0, sigma_base, size=D_clean.shape) * local_scale
    else:
        raise ValueError(f"noise_mode tidak dikenal: {noise_mode!r}")

    return D_clean + noise, D_clean


# ═══════════════════════════════════════════════════════════════════════
# 3. PENCOCOKAN KOMPONEN HASIL MCR KE GROUND TRUTH (Hungarian assignment)
# ═══════════════════════════════════════════════════════════════════════

def match_components_to_truth(S_true, S_resolved):
    """
    Cocokkan tiap komponen ground truth ke komponen hasil MCR-ALS lewat
    optimal assignment (scipy.optimize.linear_sum_assignment) berbasis
    cosine similarity — pola yang sama dipakai batch_match_derivative()
    di mcr_engine.py untuk masalah assignment serupa (menghindari
    greedy nearest-neighbor yang bisa salah pasang kalau dua komponen
    saling mirip, PERSIS kasus JM-JE).

    Returns
    -------
    assignment    : list[int] — assignment[i] = indeks komponen hasil
                    MCR yang dipasangkan ke komponen ground truth ke-i.
    cosine_matrix : (k_true x k_resolved) — matriks cosine similarity
                    lengkap, untuk audit (mis. mengecek apakah ada
                    pasangan alternatif yang hampir sama baiknya, tanda
                    identifiability lemah).
    """
    S_true = np.asarray(S_true, dtype=float)
    S_resolved = np.asarray(S_resolved, dtype=float)
    k_true, k_res = S_true.shape[0], S_resolved.shape[0]
    cosine_matrix = np.zeros((k_true, k_res))
    for i in range(k_true):
        for j in range(k_res):
            cosine_matrix[i, j] = cosine_sim(S_true[i], S_resolved[j])

    row_idx, col_idx = linear_sum_assignment(-cosine_matrix)
    assignment = [None] * k_true
    for r_i, c_i in zip(row_idx, col_idx):
        assignment[r_i] = int(c_i)
    return assignment, cosine_matrix


# ═══════════════════════════════════════════════════════════════════════
# 4. SATU KASUS SIMULASI (korelasi + noise tertentu, satu replikat/seed)
# ═══════════════════════════════════════════════════════════════════════

def run_single_replicate(wavenumber, target_correlation, noise_pct,
                          n_mixture_samples=15, noise_mode="homoscedastic",
                          mcr_kwargs=None, seed=None,
                          similarity_reliable_threshold=SIMILARITY_STRONG_THRESHOLD,
                          n_directions_ambiguity=15):
    """
    Jalankan SATU replikat studi simulasi: bangun JM-like/JE-like (dengan
    korelasi target) + MA-like (independen), simulasikan mixture, jalankan
    MCR-ALS, lalu hitung similarity/ambiguity/SIRI dan bandingkan
    keputusan 'similarity-only' vs 'ambiguity-aware' terhadap ground
    truth yang kita ketahui persis (karena ini simulasi).

    Returns
    -------
    dict berisi metrik per komponen (list of dict, urutan mengikuti
    ground truth JM/JE/MA) plus info umum (achieved_correlation,
    converged, lof_final).
    """
    rng_seed = seed
    wn = np.asarray(wavenumber, dtype=float)
    mcr_kwargs = dict(mcr_kwargs or {})

    pair = generate_correlated_pure_spectra(wn, target_correlation, seed=rng_seed)
    S_jm, S_je = pair["S_a"], pair["S_b"]
    third = generate_independent_spectrum(wn, references=[S_jm, S_je],
                                           seed=None if rng_seed is None else rng_seed + 10_000)
    S_ma = third["S_c"]

    S_true = np.array([S_jm, S_je, S_ma])
    labels_true = ["JM", "JE", "MA"]

    C_true = generate_dirichlet_design(n_mixture_samples, 3,
                                        seed=None if rng_seed is None else rng_seed + 20_000)
    D, D_clean = simulate_linear_mixture(C_true, S_true, noise_pct=noise_pct,
                                          noise_mode=noise_mode,
                                          seed=None if rng_seed is None else rng_seed + 30_000)

    # closure=False (default project run_mcr_als) — closure=True bisa
    # menjebak ALS di local optimum bila skala intrinsik antar komponen
    # jauh berbeda (ditemukan lewat smoke test lintas modul; lihat
    # catatan di blind_validation_engine.py). User bisa override lewat
    # mcr_kwargs kalau desain closure memang dijamin oleh data riil.
    mcr_defaults = dict(max_iter=300, tol=1e-6, closure=False, init_method="simplisma")
    mcr_defaults.update(mcr_kwargs)
    C, S, lof_history, r2, converged, diagnostics = run_mcr_als(D, n_components=3, **mcr_defaults)

    assignment, cosine_matrix = match_components_to_truth(S_true, S)
    constraints_used = diagnostics.get("constraints_used", {
        "s_nonneg": mcr_defaults.get("s_nonneg", True),
        "closure": mcr_defaults.get("closure", False),
    })

    fit_ok = bool(converged) and float(diagnostics.get("lof_final", 100.0)) < 10.0

    ambiguity_result = None
    try:
        ambiguity_result = compute_rotational_ambiguity(
            C, S, constraints_used, method="auto",
            n_directions=n_directions_ambiguity, seed=rng_seed,
        )
    except Exception as e:
        ambiguity_result = {"error": str(e), "components": []}

    per_component = []
    for i, label in enumerate(labels_true):
        j = assignment[i]
        component_collapsed = bool(np.linalg.norm(S[j]) < 1e-8)
        pear = pearson_corr(S_true[i], S[j])
        cos = cosine_sim(S_true[i], S[j])

        amb_list = ambiguity_result.get("components", []) if ambiguity_result else []
        amb_comp = amb_list[j] if j < len(amb_list) else None

        siri_result = None
        if amb_comp is not None:
            siri_result = compute_identity_robustness_index(
                S[j], amb_comp["S_band_min"], amb_comp["S_band_max"], S_true[i],
            )

        decision = evaluate_component_identity(
            component_label=label,
            fit_ok=fit_ok,
            pearson=pear, cosine=cos,
            ambiguity_width_pct=amb_comp["ambiguity_width_pct"] if amb_comp else None,
            ambiguity_reliability_label=amb_comp["reliability"]["label"] if amb_comp else None,
            siri_result=siri_result,
        )

        similarity_only_reliable = min(pear, cos) >= similarity_reliable_threshold
        false_positive_similarity_only = bool(
            similarity_only_reliable and siri_result is not None
            and siri_result["siri_pct"] < SIRI_LOW_THRESHOLD_PCT
        )

        per_component.append({
            "true_label": label,
            "matched_resolved_idx": j,
            "component_collapsed": component_collapsed,
            "pearson": pear,
            "cosine": cos,
            "ambiguity_width_pct": amb_comp["ambiguity_width_pct"] if amb_comp else None,
            "siri_pct": siri_result["siri_pct"] if siri_result else None,
            "similarity_only_reliable": similarity_only_reliable,
            "ambiguity_aware_label": decision["final_label"],
            "false_positive_similarity_only": false_positive_similarity_only,
        })

    return {
        "target_correlation": target_correlation,
        "achieved_correlation": pair["achieved_correlation"],
        "noise_pct": noise_pct,
        "converged": bool(converged),
        "lof_final": float(diagnostics.get("lof_final", float("nan"))),
        "cosine_matrix_true_vs_resolved": cosine_matrix,
        "components": per_component,
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. STUDI PENUH: SAPUAN KORELASI x NOISE x REPLIKAT (Figure 4 brief)
# ═══════════════════════════════════════════════════════════════════════

def run_correlation_noise_study(wavenumber, cases=None, n_replicates=10,
                                 n_mixture_samples=15, base_seed=0,
                                 mcr_kwargs=None):
    """
    Jalankan studi lengkap Section 13.1/26 brief: untuk tiap kasus
    (korelasi JM-JE x level noise), ULANGI n_replicates kali dengan seed
    berbeda, lalu ringkas mean +/- SD per metrik (bukan cuma 1 angka
    tunggal) — sesuai Section 26 ("repeat multiple random seeds, report
    mean +/- SD").

    cases : list of dict {"case_name","target_correlation","noise_pct"}
            — default DEFAULT_CASES (tabel Easy/Moderate/Difficult/
            Extreme persis Section 13.1 brief).

    Returns
    -------
    list of dict, satu per kasus:
      case_name, target_correlation, noise_pct, n_replicates,
      raw_replicates (list hasil run_single_replicate mentah, untuk
                      audit/reproduksi),
      summary (dict per true_label JM/JE/MA berisi mean/SD pearson,
               cosine, ambiguity_width_pct, siri_pct, dan
               false_positive_rate_similarity_only [%] — statistik
               kunci Figure 4/Section 26).
    """
    cases = cases if cases is not None else DEFAULT_CASES
    results = []

    for case in cases:
        raw = []
        for rep in range(n_replicates):
            seed = base_seed + hash((case["case_name"], rep)) % 1_000_000
            rep_result = run_single_replicate(
                wavenumber, case["target_correlation"], case["noise_pct"],
                n_mixture_samples=n_mixture_samples, mcr_kwargs=mcr_kwargs,
                seed=seed,
            )
            raw.append(rep_result)

        summary = {}
        for label in ["JM", "JE", "MA"]:
            pearsons, cosines, ambs, siris, fps = [], [], [], [], []
            n_collapsed = 0
            for rep_result in raw:
                comp = next(c for c in rep_result["components"] if c["true_label"] == label)
                if comp["component_collapsed"]:
                    n_collapsed += 1
                pearsons.append(comp["pearson"])
                cosines.append(comp["cosine"])
                if comp["ambiguity_width_pct"] is not None:
                    ambs.append(comp["ambiguity_width_pct"])
                if comp["siri_pct"] is not None:
                    siris.append(comp["siri_pct"])
                fps.append(comp["false_positive_similarity_only"])

            def _mean_sd(vals):
                if not vals:
                    return float("nan"), float("nan")
                return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

            def _median(vals):
                return float(np.median(vals)) if vals else float("nan")

            pm, ps = _mean_sd(pearsons)
            cm, cs = _mean_sd(cosines)
            am, asd = _mean_sd(ambs)
            sm, ssd = _mean_sd(siris)

            summary[label] = {
                "pearson_mean": pm, "pearson_sd": ps, "pearson_median": _median(pearsons),
                "cosine_mean": cm, "cosine_sd": cs, "cosine_median": _median(cosines),
                "ambiguity_width_pct_mean": am, "ambiguity_width_pct_sd": asd,
                "ambiguity_width_pct_median": _median(ambs),
                "siri_pct_mean": sm, "siri_pct_sd": ssd, "siri_pct_median": _median(siris),
                "false_positive_rate_similarity_only_pct": 100.0 * float(np.mean(fps)) if fps else float("nan"),
                "n_replicates_component_collapsed": n_collapsed,
            }

        results.append({
            "case_name": case["case_name"],
            "target_correlation": case["target_correlation"],
            "noise_pct": case["noise_pct"],
            "n_replicates": n_replicates,
            "raw_replicates": raw,
            "summary": summary,
        })

    return results


def build_correlation_noise_study_dataframe(study_results):
    """
    Ratakan output run_correlation_noise_study() jadi satu pandas
    DataFrame (satu baris per kasus x komponen) — bahan langsung untuk
    Figure 4 brief (identity error/false-positive rate vs korelasi
    spektral & SNR) dan sheet laporan.
    """
    import pandas as pd
    rows = []
    for case_result in study_results:
        for label, s in case_result["summary"].items():
            rows.append({
                "Kasus": case_result["case_name"],
                "Target korelasi JM-JE": case_result["target_correlation"],
                "Noise (%)": case_result["noise_pct"],
                "Komponen": label,
                "Pearson (mean±SD / median)": f"{s['pearson_mean']:.4f} ± {s['pearson_sd']:.4f} / {s['pearson_median']:.4f}",
                "Cosine (mean±SD / median)": f"{s['cosine_mean']:.4f} ± {s['cosine_sd']:.4f} / {s['cosine_median']:.4f}",
                "Ambiguity width % (mean±SD / median)": f"{s['ambiguity_width_pct_mean']:.1f} ± {s['ambiguity_width_pct_sd']:.1f} / {s['ambiguity_width_pct_median']:.1f}",
                "SIRI % (mean±SD / median)": f"{s['siri_pct_mean']:.1f} ± {s['siri_pct_sd']:.1f} / {s['siri_pct_median']:.1f}",
                "False-positive rate similarity-only (%)": s["false_positive_rate_similarity_only_pct"],
                "Replikat komponen kolaps": f"{s['n_replicates_component_collapsed']}/{case_result['n_replicates']}",
                "n_replicates": case_result["n_replicates"],
            })
    return pd.DataFrame(rows)
