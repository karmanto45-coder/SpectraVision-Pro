"""
reproducibility_engine.py — SpectraVision Pro
================================================
Modul ADITIF untuk Evidence 1 (Reproducibility, brief Section 5) dan
Identity Stability P_ID (brief Section 16) dari
MCR_Spurious_Component_Existence_and_Identity_Framework.md.

Dua pertanyaan yang dijawab modul ini, KEDUANYA butuh MCR-ALS dijalankan
BERULANG KALI pada data yang SAMA dengan variasi acak tiap run (restart
acak + gangguan noise kecil):

  1. compute_existence_reproducibility() — pertanyaan EKSISTENSI: apakah
     faktor dengan bentuk tertentu MUNCUL KONSISTEN di across banyak run,
     atau cuma muncul sesekali (indikasi faktor itu artefak numerik dari
     kondisi awal/noise tertentu, bukan sumber variasi kimia yang nyata)?
     TIDAK butuh reference library — murni soal konsistensi internal.

  2. compute_identity_stability() — pertanyaan IDENTITAS (P_ID, brief
     Section 16): dari sekian run, berapa persen yang secara konsisten
     meng-assign kandidat resolved factor ke TARGET (bukan ke kompetitor)?
     Ini BEDA dari robustness_sweep_engine.py yang sudah ada: modul itu
     menguji sensitivitas KEPUTUSAN AKHIR (identity_decision_engine's
     final_label) terhadap variasi KONFIGURASI (init/preprocessing/
     window/noise) — sedikit kombinasi bermakna (mis. pca vs simplisma
     vs nmf, masing² SATU kali). Modul INI menguji frekuensi ASSIGNMENT
     mentah (menang vs kompetitor) lewat BANYAK (puluhan-ratusan) restart
     ACAK murni — dua sudut pandang berbeda pada axis "seberapa stabil".

PENTING - keterbatasan yang harus dinyatakan di manuskrip jika dipakai:
  - Sumber variasi antar-run di sini adalah kombinasi (a) restart acak
    (S_init non-negatif acak, dicampur dengan simplisma/pca/nmf yang
    deterministik) dan (b) gangguan Gaussian kecil pada data (residual-
    bootstrap-like). Ini BUKAN daftar lengkap semua sumber variasi yang
    brief sebut (mis. tidak termasuk variasi preprocessing/model-order —
    itu domain robustness_sweep_engine.py, bukan modul ini).
  - Run yang GAGAL konvergen TETAP dihitung sebagai bukti (bukan dibuang
    diam-diam) — brief Section 5 eksplisit: faktor yang sering gagal
    muncul/konvergen justru itu sinyal kecurigaannya, bukan noise yang
    boleh diabaikan.
"""

import numpy as np

from mcr_engine import run_mcr_als, cosine_sim
from similarity_simulator import match_components_to_truth

DEFAULT_REPRO_COSINE_THRESHOLD = 0.90


# ═══════════════════════════════════════════════════════════════════════
# 1. DRIVER: JALANKAN MCR-ALS BERULANG DENGAN VARIASI ACAK
# ═══════════════════════════════════════════════════════════════════════

def run_repeated_mcr(D, n_components, n_runs=50,
                      deterministic_init_methods=("simplisma", "pca", "nmf"),
                      use_random_restart=True,
                      add_noise_perturbation=True, noise_pct=0.5,
                      mcr_kwargs=None, base_seed=0):
    """
    Jalankan MCR-ALS n_runs kali pada data D yang SAMA, dengan tiap run
    memakai kombinasi acak: metode inisialisasi (salah satu dari
    deterministic_init_methods, ATAU S_init acak non-negatif kalau
    use_random_restart=True) DAN (opsional) gangguan Gaussian kecil pada
    D. Restart acak adalah cara paling standar di literatur MCR untuk
    menguji stabilitas ALS (bukan cuma mencoba beberapa metode init
    yang masing² deterministik — itu domain robustness_sweep_engine.py).

    Returns
    -------
    list of dict, satu per run: run_index, init_method_used, noise_added,
    C, S (None kalau run gagal), converged, lof_final, error (None kalau
    sukses). Run gagal TETAP disertakan (lihat catatan modul di atas).
    """
    mcr_kwargs = dict(mcr_kwargs or {})
    init_options = list(deterministic_init_methods) + (["random_restart"] if use_random_restart else [])
    if not init_options:
        raise ValueError("Setidaknya satu opsi inisialisasi harus tersedia.")

    D = np.asarray(D, dtype=float)
    runs = []
    for run_idx in range(n_runs):
        rng = np.random.default_rng(base_seed + run_idx)
        init_choice = init_options[int(rng.integers(len(init_options)))]

        D_run = D
        noise_added = False
        if add_noise_perturbation:
            sigma = (noise_pct / 100.0) * float(np.max(np.abs(D)))
            D_run = D + rng.normal(0, sigma, size=D.shape)
            noise_added = True

        run_kwargs = dict(mcr_kwargs)
        run_kwargs.setdefault("max_iter", 300)
        run_kwargs.setdefault("tol", 1e-6)
        run_kwargs.setdefault("closure", False)
        if init_choice == "random_restart":
            n_points = D.shape[1]
            run_kwargs["S_init"] = rng.uniform(0.01, 1.0, size=(n_components, n_points))
        else:
            run_kwargs["init_method"] = init_choice

        try:
            C, S, lof_hist, r2, converged, diag = run_mcr_als(D_run, n_components=n_components, **run_kwargs)
            runs.append({
                "run_index": run_idx, "init_method_used": init_choice, "noise_added": noise_added,
                "C": C, "S": S, "converged": bool(converged),
                "lof_final": float(diag.get("lof_final", float("nan"))), "error": None,
            })
        except Exception as e:
            runs.append({
                "run_index": run_idx, "init_method_used": init_choice, "noise_added": noise_added,
                "C": None, "S": None, "converged": False, "lof_final": float("nan"), "error": str(e),
            })
    return runs


# ═══════════════════════════════════════════════════════════════════════
# 2. EXISTENCE REPRODUCIBILITY (Evidence 1, Section 5 brief)
# ═══════════════════════════════════════════════════════════════════════

def compute_existence_reproducibility(runs, canonical_S=None,
                                       cosine_threshold=DEFAULT_REPRO_COSINE_THRESHOLD):
    """
    Untuk tiap komponen "kanonis" (canonical_S — kalau None, dipakai S
    dari run PERTAMA yang sukses konvergen), hitung berapa persen dari
    seluruh run yang punya faktor hasil MCR yang cukup mirip (Hungarian-
    matched via match_components_to_truth, cosine >= cosine_threshold)
    — persis skema brief Section 5 (mis. "C1: 98/100, C4: 8/100").

    Run yang gagal konvergen dihitung SEBAGAI TIDAK REPRODUKSI untuk
    SEMUA komponen kanonis (bukan diabaikan) — konsisten dengan catatan
    modul di atas.

    Returns
    -------
    list of dict per komponen kanonis: component_index, n_reproducible,
    n_total_runs, p_repro_pct, classification.
    """
    valid_runs = [r for r in runs if r["S"] is not None]
    if canonical_S is None:
        if not valid_runs:
            raise ValueError("Tidak ada run yang sukses — tidak bisa menentukan canonical_S.")
        canonical_S = valid_runs[0]["S"]

    n_components = canonical_S.shape[0]
    n_total = len(runs)
    counts = np.zeros(n_components)

    for r in runs:
        if r["S"] is None:
            continue  # gagal konvergen -> dihitung TIDAK reproduksi (tidak menambah counts)
        assignment, cosmat = match_components_to_truth(canonical_S, r["S"])
        for i in range(n_components):
            j = assignment[i]
            if cosmat[i, j] >= cosine_threshold:
                counts[i] += 1

    results = []
    for i in range(n_components):
        p_repro_pct = 100.0 * counts[i] / n_total if n_total > 0 else float("nan")
        if p_repro_pct < 50.0:
            classification = {
                "label": "mencurigakan",
                "detail": (
                    f"Hanya muncul konsisten di {p_repro_pct:.0f}% run — PATUT "
                    f"DICURIGAI sebagai faktor yang tidak reproducible (artefak "
                    f"kondisi awal/noise tertentu), TAPI ini bukan vonis akhir; "
                    f"gabungkan dengan bukti contribution ratio/null-test."
                ),
            }
        else:
            classification = {
                "label": "wajar",
                "detail": f"Muncul konsisten di {p_repro_pct:.0f}% run — reproducible.",
            }
        results.append({
            "component_index": i, "n_reproducible": int(counts[i]),
            "n_total_runs": n_total, "p_repro_pct": p_repro_pct,
            "classification": classification,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════
# 3. IDENTITY STABILITY P_ID (Section 16 brief)
# ═══════════════════════════════════════════════════════════════════════

def compute_identity_stability(runs, reference_set, target_label, metric="cosine"):
    """
    Untuk tiap run, cari komponen resolved yang PALING mirip ke
    target_label (kandidat slot identitas itu), lalu lihat referensi
    MANA (termasuk target sendiri) yang paling mirip dengan kandidat itu
    — kalau target menang, run itu dihitung sebagai "assign ke target".
    P_ID(target) = proporsi run yang menang untuk target.

    Parameters
    ----------
    runs          : output run_repeated_mcr().
    reference_set : dict {label: S_ref} — HARUS memuat target_label dan
                    minimal satu kompetitor lain.
    target_label  : label identitas yang diuji stabilitasnya.

    Returns
    -------
    dict: p_id_target_pct, winners (list label pemenang per run, None
    untuk run yang gagal konvergen), n_total_runs, classification.
    """
    if target_label not in reference_set:
        raise ValueError(f"target_label '{target_label}' tidak ada di reference_set.")
    sim_fn = cosine_sim if metric == "cosine" else None
    if sim_fn is None:
        from mcr_engine import pearson_corr
        sim_fn = pearson_corr

    S_target = reference_set[target_label]
    winners = []
    for r in runs:
        if r["S"] is None:
            winners.append(None)
            continue
        S = r["S"]
        sims_to_target = [sim_fn(S[k], S_target) for k in range(S.shape[0])]
        best_k = int(np.argmax(sims_to_target))
        all_sims = {label: sim_fn(S[best_k], ref) for label, ref in reference_set.items()}
        winners.append(max(all_sims, key=all_sims.get))

    n_total = len(runs)
    n_valid = sum(1 for w in winners if w is not None)
    n_target_wins = sum(1 for w in winners if w == target_label)
    p_id_pct = 100.0 * n_target_wins / n_total if n_total > 0 else float("nan")

    if p_id_pct >= 90.0:
        classification = {"label": "stabil", "detail": f"P_ID({target_label}) = {p_id_pct:.0f}% — identitas sangat stabil lintas run acak."}
    elif p_id_pct >= 50.0:
        classification = {"label": "sedang", "detail": f"P_ID({target_label}) = {p_id_pct:.0f}% — identitas menang mayoritas tapi tidak dominan; perlu bukti pendukung lain."}
    else:
        classification = {"label": "tidak_stabil", "detail": f"P_ID({target_label}) = {p_id_pct:.0f}% — identitas TIDAK stabil, kompetitor lebih sering menang dari run acak berulang."}

    return {
        "p_id_target_pct": p_id_pct, "winners": winners,
        "n_total_runs": n_total, "n_valid_runs": n_valid,
        "n_target_wins": n_target_wins, "classification": classification,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. TABEL RINGKAS UNTUK LAPORAN
# ═══════════════════════════════════════════════════════════════════════

def build_reproducibility_dataframe(reproducibility_results):
    import pandas as pd
    return pd.DataFrame([{
        "Komponen": f"C{r['component_index'] + 1}",
        "Reproducible (run)": f"{r['n_reproducible']}/{r['n_total_runs']}",
        "P_repro (%)": round(r["p_repro_pct"], 1),
        "Status": r["classification"]["label"],
        "Keterangan": r["classification"]["detail"],
    } for r in reproducibility_results])


def build_identity_stability_dataframe(stability_results_by_label):
    """stability_results_by_label: dict {target_label: hasil compute_identity_stability()}"""
    import pandas as pd
    rows = []
    for label, r in stability_results_by_label.items():
        rows.append({
            "Target": label,
            "P_ID (%)": round(r["p_id_target_pct"], 1),
            "Menang / total run": f"{r['n_target_wins']}/{r['n_total_runs']}",
            "Run gagal konvergen": r["n_total_runs"] - r["n_valid_runs"],
            "Status": r["classification"]["label"],
            "Keterangan": r["classification"]["detail"],
        })
    return pd.DataFrame(rows)
