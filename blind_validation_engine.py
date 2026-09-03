"""
blind_validation_engine.py — SpectraVision Pro
=================================================
Modul ADITIF untuk Section 13.2/13.3/26 brief riset JM/JE/MA: validasi
"blind" gaya uji diagnostik — apakah pipeline (MCR-ALS -> ambiguity ->
SIRI -> hierarchical decision) BENAR menyatakan "identitas cocok dengan
reference target" ketika target itu MEMANG ADA (kelas positif), dan
BENAR menolak ketika yang ada justru adulterant/look-alike yang mirip
tapi BUKAN target (kelas negatif) — dengan komposisi disembunyikan dari
"analis" (pipeline dijalankan buta terhadap true_class, sama seperti
lab riil menerima sampel tanpa tahu isinya).

Kerangka: confusion matrix (TP/FP/TN/FN) -> sensitivity & specificity,
plus confidence interval via percentile bootstrap atas hasil trial
(bukan cuma satu titik estimasi) — sesuai Section 26 brief.

BEDA dengan run_correlation_noise_study() di similarity_simulator.py:
studi itu menjawab "seberapa akurat pipeline meng-assign SEMUA komponen
yang memang ada di campuran ke label ground truth-nya". Modul INI
menjawab pertanyaan yang lebih dekat ke penggunaan riil di lab: "kalau
saya punya SATU reference target (mis. JM), dan sampel yang datang bisa
saja BUKAN mengandung JM sama sekali (melainkan adulterant mirip), apakah
pipeline bisa membedakan keduanya?" — ini kerangka uji diagnostik biner
(present/absent terhadap SATU target), bukan multi-kelas assignment.

PENTING - keterbatasan yang harus dinyatakan di manuskrip jika dipakai:
  - "Impostor/adulterant" di sini juga spektrum SINTETIS (dibangun
    lewat generate_correlated_pure_spectra() dari similarity_simulator.py
    pada korelasi yang diatur terhadap target) — bukan sampel adulteran
    kimia riil. Berguna untuk memetakan KAPAN pipeline mulai gagal
    membedakan (fungsi dari korelasi look-alike), tapi validasi akhir
    tetap perlu adulterant riil.
  - Bootstrap CI di sini adalah percentile bootstrap SEDERHANA atas
    hasil trial biner (bukan bootstrap parametrik/BCa) — cukup untuk
    pelaporan awal, tapi nyatakan metodenya secara eksplisit di
    manuskrip kalau dipakai.
"""

import numpy as np

from mcr_engine import run_mcr_als, cosine_sim, pearson_corr
from ambiguity_engine import compute_rotational_ambiguity
from identity_decision_engine import (
    compute_identity_robustness_index,
    evaluate_component_identity,
)
from similarity_simulator import (
    generate_correlated_pure_spectra,
    generate_independent_spectrum,
    generate_dirichlet_design,
    simulate_linear_mixture,
)

GOOD_LABELS = {"reliably_identified", "provisional"}  # dianggap "predicted: target hadir"


# ═══════════════════════════════════════════════════════════════════════
# 1. SATU TRIAL BLIND (positif ATAU negatif)
# ═══════════════════════════════════════════════════════════════════════

def run_blind_trial(wavenumber, true_class, S_target, S_present_other,
                     S_impostor=None, n_mixture_samples=15, noise_pct=1.0,
                     mcr_kwargs=None, seed=None):
    """
    Jalankan SATU trial blind.

    true_class      : "positive" (target BENAR-BENAR ada di campuran)
                      atau "negative" (target TIDAK ada; kalau
                      S_impostor diberikan, look-alike itu yang dipakai
                      menggantikan slot target di campuran — kalau
                      None, campuran negatif hanya berisi
                      S_present_other saja tanpa pengganti).
    S_target        : (n_points,) spektrum reference target (mis. JM).
    S_present_other : list of (n_points,) — komponen LAIN yang SELALU
                      ada di campuran baik trial positif maupun negatif
                      (mis. JE, MA) — matrix/background yang sama,
                      supaya perbedaan hasil murni karena target
                      hadir/tidak, bukan karena komposisi background
                      berbeda.
    S_impostor      : (n_points,) opsional — spektrum look-alike yang
                      menempati slot target pada trial negatif.

    Returns
    -------
    dict: true_class, predicted_class, final_label, best_cosine_to_target,
          best_pearson_to_target, ambiguity_width_pct, siri_pct,
          converged, lof_final
    """
    rng_seed = seed
    mcr_kwargs = dict(mcr_kwargs or {})

    if true_class == "positive":
        components = [S_target] + list(S_present_other)
    elif true_class == "negative":
        components = (list(S_impostor if isinstance(S_impostor, list) else [S_impostor])
                      if S_impostor is not None else []) + list(S_present_other)
    else:
        raise ValueError("true_class harus 'positive' atau 'negative'")

    S_true = np.array(components)
    n_components = S_true.shape[0]

    C_true = generate_dirichlet_design(n_mixture_samples, n_components,
                                        seed=None if rng_seed is None else rng_seed + 1)
    D, D_clean = simulate_linear_mixture(C_true, S_true, noise_pct=noise_pct,
                                          seed=None if rng_seed is None else rng_seed + 2)

    # closure=False (default project run_mcr_als). PENTING - ditemukan
    # lewat smoke test modul ini: closure=True dapat menjebak ALS di
    # local optimum yang SAMA PERSIS terlepas dari metode init (diamati
    # LOF 43% vs 2.5% tanpa closure) ketika komponen punya skala
    # intensitas intrinsik jauh berbeda — kasus yang cukup umum untuk
    # spektrum sintetis/riil. Nyalakan closure=True lewat mcr_kwargs
    # HANYA kalau desain data memang menjamin closure fisik (total
    # fraksi = 1 dengan skala S yang saling sepadan) DAN sudah dicek
    # tidak menyebabkan LOF melonjak dibanding closure=False.
    mcr_defaults = dict(max_iter=300, tol=1e-6, closure=False, init_method="simplisma")
    mcr_defaults.update(mcr_kwargs)
    C, S, lof_history, r2, converged, diagnostics = run_mcr_als(
        D, n_components=n_components, **mcr_defaults
    )
    constraints_used = diagnostics.get("constraints_used", {
        "s_nonneg": mcr_defaults.get("s_nonneg", True),
        "closure": mcr_defaults.get("closure", False),
    })
    fit_ok = bool(converged) and float(diagnostics.get("lof_final", 100.0)) < 10.0

    # Cari kandidat resolved-component TERBAIK terhadap target (BUKAN
    # assignment penuh ke semua ground truth — ini murni pertanyaan
    # "apakah ADA slot resolved yang mirip target", sesuai skenario blind.
    cosines_to_target = [cosine_sim(S_target, S[k]) for k in range(S.shape[0])]
    best_j = int(np.argmax(cosines_to_target))
    best_cos = cosines_to_target[best_j]
    best_pear = pearson_corr(S_target, S[best_j])

    try:
        ambiguity_result = compute_rotational_ambiguity(
            C, S, constraints_used, method="auto", n_directions=15, seed=rng_seed,
        )
        amb_list = ambiguity_result.get("components", [])
    except Exception:
        amb_list = []
    amb_comp = amb_list[best_j] if best_j < len(amb_list) else None

    siri_result = None
    if amb_comp is not None:
        siri_result = compute_identity_robustness_index(
            S[best_j], amb_comp["S_band_min"], amb_comp["S_band_max"], S_target,
        )

    decision = evaluate_component_identity(
        component_label="target_candidate", fit_ok=fit_ok,
        pearson=best_pear, cosine=best_cos,
        ambiguity_width_pct=amb_comp["ambiguity_width_pct"] if amb_comp else None,
        ambiguity_reliability_label=amb_comp["reliability"]["label"] if amb_comp else None,
        siri_result=siri_result,
    )

    predicted_class = "positive" if decision["final_label"] in GOOD_LABELS else "negative"

    return {
        "true_class": true_class,
        "predicted_class": predicted_class,
        "final_label": decision["final_label"],
        "best_cosine_to_target": best_cos,
        "best_pearson_to_target": best_pear,
        "ambiguity_width_pct": amb_comp["ambiguity_width_pct"] if amb_comp else None,
        "siri_pct": siri_result["siri_pct"] if siri_result else None,
        "converged": bool(converged),
        "lof_final": float(diagnostics.get("lof_final", float("nan"))),
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. STUDI PENUH: banyak trial positif+negatif, per level korelasi impostor
# ═══════════════════════════════════════════════════════════════════════

def run_blind_validation_study(wavenumber, impostor_correlation_levels=(0.70, 0.85, 0.95, 0.99),
                                n_trials_per_class=20, n_mixture_samples=15, noise_pct=1.0,
                                n_other_components=1, mcr_kwargs=None, base_seed=0,
                                n_bootstrap=1000, ci_pct=95.0):
    """
    Jalankan studi validasi blind penuh: untuk tiap level korelasi
    impostor-vs-target, jalankan n_trials_per_class trial positif DAN
    n_trials_per_class trial negatif (target diganti impostor pada
    korelasi itu), lalu hitung confusion matrix, sensitivity,
    specificity, dan confidence interval via percentile bootstrap.

    Returns
    -------
    list of dict, satu per level korelasi:
      impostor_correlation, n_trials_per_class, confusion (dict TP/FP/TN/FN),
      sensitivity, specificity, sensitivity_ci, specificity_ci,
      raw_trials (list hasil run_blind_trial mentah, untuk audit).
    """
    results = []

    for corr in impostor_correlation_levels:
        seed_base_corr = base_seed + int(corr * 10_000)
        pair = generate_correlated_pure_spectra(wavenumber, corr, seed=seed_base_corr)
        S_target, S_impostor = pair["S_a"], pair["S_b"]

        S_other = []
        for k in range(n_other_components):
            other = generate_independent_spectrum(
                wavenumber, references=[S_target, S_impostor],
                seed=seed_base_corr + 100 + k,
            )
            S_other.append(other["S_c"])

        raw_trials = []
        for rep in range(n_trials_per_class):
            for true_class in ("positive", "negative"):
                trial_seed = seed_base_corr + 1000 + rep * 2 + (0 if true_class == "positive" else 1)
                trial = run_blind_trial(
                    wavenumber, true_class, S_target, S_other,
                    S_impostor=S_impostor, n_mixture_samples=n_mixture_samples,
                    noise_pct=noise_pct, mcr_kwargs=mcr_kwargs, seed=trial_seed,
                )
                raw_trials.append(trial)

        confusion = _build_confusion(raw_trials)
        sens, sens_ci = _rate_with_bootstrap_ci(
            raw_trials, true_class="positive", n_bootstrap=n_bootstrap, ci_pct=ci_pct,
        )
        spec, spec_ci = _rate_with_bootstrap_ci(
            raw_trials, true_class="negative", n_bootstrap=n_bootstrap, ci_pct=ci_pct,
        )

        results.append({
            "impostor_correlation": corr,
            "n_trials_per_class": n_trials_per_class,
            "confusion": confusion,
            "sensitivity_pct": sens, "sensitivity_ci_pct": sens_ci,
            "specificity_pct": spec, "specificity_ci_pct": spec_ci,
            "raw_trials": raw_trials,
        })

    return results


def _build_confusion(trials):
    tp = sum(1 for t in trials if t["true_class"] == "positive" and t["predicted_class"] == "positive")
    fn = sum(1 for t in trials if t["true_class"] == "positive" and t["predicted_class"] == "negative")
    tn = sum(1 for t in trials if t["true_class"] == "negative" and t["predicted_class"] == "negative")
    fp = sum(1 for t in trials if t["true_class"] == "negative" and t["predicted_class"] == "positive")
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp}


def _rate_with_bootstrap_ci(trials, true_class, n_bootstrap=1000, ci_pct=95.0, seed=0):
    """
    Hitung sensitivity (true_class='positive') atau specificity
    (true_class='negative') plus percentile bootstrap CI dari resampling
    hasil trial biner dengan penggantian (with replacement).
    """
    subset = [1 if t["predicted_class"] == true_class else 0
              for t in trials if t["true_class"] == true_class]
    if not subset:
        return float("nan"), (float("nan"), float("nan"))

    point = 100.0 * float(np.mean(subset))

    rng = np.random.default_rng(seed)
    arr = np.array(subset)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[b] = 100.0 * sample.mean()

    lo = (100 - ci_pct) / 2
    hi = 100 - lo
    ci = (float(np.percentile(boot_means, lo)), float(np.percentile(boot_means, hi)))
    return point, ci


def build_blind_validation_dataframe(study_results):
    """Tabel ringkas untuk fitur Laporan / Figure 4-style plot dan tabel
    supplementary manuskrip — mengikuti pola build_*_dataframe yang
    sudah dipakai modul-modul lain di proyek ini."""
    import pandas as pd
    rows = []
    for r in study_results:
        c = r["confusion"]
        rows.append({
            "Korelasi impostor vs target": r["impostor_correlation"],
            "n trial/kelas": r["n_trials_per_class"],
            "TP": c["TP"], "FN": c["FN"], "TN": c["TN"], "FP": c["FP"],
            "Sensitivity (%)": round(r["sensitivity_pct"], 1),
            "Sensitivity 95% CI": f"[{r['sensitivity_ci_pct'][0]:.1f}, {r['sensitivity_ci_pct'][1]:.1f}]",
            "Specificity (%)": round(r["specificity_pct"], 1),
            "Specificity 95% CI": f"[{r['specificity_ci_pct'][0]:.1f}, {r['specificity_ci_pct'][1]:.1f}]",
        })
    return pd.DataFrame(rows)
