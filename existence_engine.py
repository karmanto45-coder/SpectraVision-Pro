"""
existence_engine.py — SpectraVision Pro
=========================================
Modul ADITIF yang membuka dimensi BARU dari framework JM/JE/MA: bukan
"apa identitas komponen ini" (sudah dijawab identity_decision_engine.py),
melainkan pertanyaan yang lebih mendasar dan sebelumnya belum tersentuh
sama sekali — "apakah komponen hasil MCR ini benar-benar merepresentasikan
sumber variasi spektral yang independen di sampel, atau cuma faktor yang
matematis diizinkan tapi kimiawi tidak nyata (spurious/over-factoring)?"
Lihat MCR_Spurious_Component_Existence_and_Identity_Framework.md,
terutama Section 2-4 (pemisahan existence vs identity) dan Section 24
(kerangka akhir: Existence -> Identity -> Robustness -> Decision).

Dua alat bukti EKSISTENSI yang dibangun di modul ini (Evidence 2 & bagian
dari Section 10 pada brief):

  1. compute_contribution_ratio() — Rk = ||Ck·Sk^T||_F^2 / ||X||_F^2,
     proporsi energi Frobenius yang disumbang tiap komponen terhadap
     data ASLI (bukan data rekonstruksi C@S, karena brief eksplisit
     memakai ||X||_F^2 dari data asli sebagai penyebut).

  2. detect_spectral_redundancy() — cek semua pasangan komponen hasil
     MCR: kalau spektrum (S) DAN profil konsentrasi (C) SAMA-SAMA
     berkorelasi sangat tinggi, itu indikasi "mathematical splitting"
     (Section 10 brief) — satu kontribusi kimia asli yang pecah jadi
     dua faktor matematis, bukan dua komponen independen.

PENTING - keterbatasan yang harus dinyatakan di manuskrip jika dipakai
(brief sendiri eksplisit memperingatkan ini, lihat Section 6 & 10):
  - Kontribusi yang sangat kecil TIDAK OTOMATIS berarti spurious —
    komponen minor asli juga bisa berkontribusi kecil. compute_contribution_ratio()
    HANYA melaporkan angka dan flag interpretatif, BUKAN vonis akhir.
    Vonis akhir harus menunggu bukti lain (reproducibility, null test,
    identity) — lihat modul-modul berikutnya di backlog ini.
  - Redundansi tinggi (S dan C sama-sama berkorelasi) adalah indikasi
    KUAT tapi bukan BUKTI MUTLAK mathematical splitting — dua komponen
    kimia yang secara kebetulan berperilaku sangat mirip (co-eluting/
    co-varying secara fisik) juga bisa menghasilkan pola serupa.
"""

import numpy as np

from mcr_engine import cosine_sim, pearson_corr, run_mcr_als

# Ambang default — KONVENSI INTERPRETATIF proyek ini (sama seperti
# threshold-threshold lain di ambiguity_engine.py/identity_decision_engine.py),
# BUKAN standar baku dari literatur.
CONTRIBUTION_SUSPICIOUS_THRESHOLD_PCT = 1.0    # brief dummy example: C4=0.04% jelas mencurigakan
REDUNDANCY_COSINE_S_THRESHOLD = 0.98
REDUNDANCY_PEARSON_C_THRESHOLD = 0.95


# ═══════════════════════════════════════════════════════════════════════
# 1. CONTRIBUTION RATIO (Rk)
# ═══════════════════════════════════════════════════════════════════════

def compute_contribution_ratio(C, S, D, suspicious_threshold_pct=CONTRIBUTION_SUSPICIOUS_THRESHOLD_PCT):
    """
    Hitung Rk = ||Ck·Sk^T||_F^2 / ||X||_F^2 untuk tiap komponen —
    proporsi energi (variansi spektral) di DATA ASLI X yang dijelaskan
    oleh kontribusi rank-1 komponen k saja (Ck outer Sk), persis formula
    Section 6 brief.

    Parameters
    ----------
    C : (n_sampel x n_komponen) — profil konsentrasi hasil MCR-ALS.
    S : (n_komponen x n_wavenumber) — spektrum murni hasil MCR-ALS.
    D : (n_sampel x n_wavenumber) — DATA ASLI (sebelum MCR), dipakai
        sebagai penyebut ||X||_F^2. WAJIB data asli, bukan rekonstruksi
        C@S — brief eksplisit membandingkan terhadap data asli supaya
        Rk mencerminkan proporsi VARIANSI NYATA yang dijelaskan, bukan
        proporsi terhadap model sendiri (yang akan selalu berjumlah
        100% secara sirkular kalau dibagi dengan C@S).

    Returns
    -------
    list of dict, satu per komponen:
      component_index, contribution_pct, frobenius_norm_sq_component,
      classification {"label","detail"}
    """
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    D = np.asarray(D, dtype=float)
    n_components = S.shape[0]

    total_energy = float(np.sum(D ** 2))
    if total_energy <= 0:
        raise ValueError("||X||_F^2 == 0 — data asli (D) kosong atau semuanya nol.")

    results = []
    for k in range(n_components):
        Ck = C[:, k:k + 1]           # (n_sampel, 1)
        Sk = S[k:k + 1, :]           # (1, n_wavenumber)
        Xk = Ck @ Sk                 # kontribusi rank-1 komponen k
        energy_k = float(np.sum(Xk ** 2))
        contribution_pct = 100.0 * energy_k / total_energy

        if contribution_pct < suspicious_threshold_pct:
            classification = {
                "label": "mencurigakan",
                "detail": (
                    f"Kontribusi sangat kecil ({contribution_pct:.3f}% < "
                    f"{suspicious_threshold_pct:.2f}%) — PATUT DICURIGAI sebagai "
                    f"faktor spurious/overfitting, TAPI ini bukan vonis akhir: "
                    f"komponen minor asli juga bisa berkontribusi kecil. Perlu "
                    f"dikonfirmasi dengan bukti reproducibility/null-test."
                ),
            }
        else:
            classification = {
                "label": "wajar",
                "detail": f"Kontribusi {contribution_pct:.3f}% berada di atas ambang kecurigaan.",
            }

        results.append({
            "component_index": k,
            "contribution_pct": contribution_pct,
            "frobenius_norm_sq_component": energy_k,
            "classification": classification,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════
# 2. REDUNDANSI SPEKTRAL / MATHEMATICAL SPLITTING
# ═══════════════════════════════════════════════════════════════════════

def detect_spectral_redundancy(C, S,
                                cosine_s_threshold=REDUNDANCY_COSINE_S_THRESHOLD,
                                pearson_c_threshold=REDUNDANCY_PEARSON_C_THRESHOLD):
    """
    Cek SEMUA pasangan komponen hasil MCR untuk indikasi "mathematical
    splitting" (Section 10 brief): satu kontribusi kimia asli yang
    matematis terpecah jadi dua faktor MCR, ditandai dengan spektrum (S)
    DAN profil konsentrasi (C) yang SAMA-SAMA berkorelasi sangat tinggi
    antar pasangan itu.

    Kenapa BUTUH DUA bukti sekaligus (bukan cukup salah satu): dua
    komponen bisa punya S mirip tapi C independen (mis. dua isomer
    dengan spektrum nyaris sama tapi konsentrasi berbeda-beda antar
    sampel — itu dua komponen NYATA, bukan splitting), atau C mirip
    tapi S beda (dua komponen yang kebetulan co-varying secara fisik
    tapi kimiawi berbeda). Splitting matematis butuh KEDUANYA tinggi.

    Returns
    -------
    list of dict, satu per pasangan (i,j) dengan i<j:
      component_i, component_j, cosine_S, pearson_C, suspicious_redundancy
    """
    S = np.asarray(S, dtype=float)
    C = np.asarray(C, dtype=float)
    n_components = S.shape[0]

    pairs = []
    for i in range(n_components):
        for j in range(i + 1, n_components):
            cos_S = cosine_sim(S[i], S[j])
            corr_C = pearson_corr(C[:, i], C[:, j])
            suspicious = (cos_S >= cosine_s_threshold) and (abs(corr_C) >= pearson_c_threshold)
            pairs.append({
                "component_i": i,
                "component_j": j,
                "cosine_S": cos_S,
                "pearson_C": corr_C,
                "suspicious_redundancy": suspicious,
                "detail": (
                    f"C{i+1} vs C{j+1}: cosine(S)={cos_S:.4f}, pearson(C)={corr_C:.4f} — "
                    + ("INDIKASI mathematical splitting (kedua metrik di atas ambang); "
                       "pertimbangkan menjalankan ulang dengan n_komponen lebih kecil."
                       if suspicious else
                       "tidak ada indikasi splitting matematis pada pasangan ini.")
                ),
            })
    return pairs


# ═══════════════════════════════════════════════════════════════════════
# 3. TABEL RINGKAS UNTUK LAPORAN
# ═══════════════════════════════════════════════════════════════════════

def build_existence_evidence_dataframe(contribution_results, redundancy_results):
    """
    Gabungkan hasil compute_contribution_ratio() dan
    detect_spectral_redundancy() jadi tabel bukti eksistensi — mengikuti
    pola build_*_dataframe yang sudah dipakai modul-modul lain di
    proyek ini. Dua tabel terpisah (kontribusi per-komponen, redundansi
    per-pasangan) karena bentuknya memang beda (satu per komponen vs
    satu per pasangan) — digabung di sini hanya untuk kenyamanan return
    tunggal, bukan dipaksa jadi satu tabel.
    """
    import pandas as pd

    df_contrib = pd.DataFrame([{
        "Komponen": f"C{r['component_index'] + 1}",
        "Kontribusi (%)": round(r["contribution_pct"], 4),
        "Status": r["classification"]["label"],
        "Keterangan": r["classification"]["detail"],
    } for r in contribution_results])

    df_redundancy = pd.DataFrame([{
        "Pasangan": f"C{r['component_i'] + 1} vs C{r['component_j'] + 1}",
        "Cosine(S)": round(r["cosine_S"], 4),
        "Pearson(C)": round(r["pearson_C"], 4),
        "Diduga splitting?": "YA" if r["suspicious_redundancy"] else "Tidak",
        "Keterangan": r["detail"],
    } for r in redundancy_results]) if redundancy_results else pd.DataFrame(
        columns=["Pasangan", "Cosine(S)", "Pearson(C)", "Diduga splitting?", "Keterangan"]
    )

    return {"contribution": df_contrib, "redundancy": df_redundancy}


# ═══════════════════════════════════════════════════════════════════════
# 3. NESTED-MODEL ΔLOF TEST (Evidence 3, Section 7 brief)
# ═══════════════════════════════════════════════════════════════════════

def compute_delta_lof_nested(D, n_components_lower, n_components_higher,
                              mcr_kwargs=None):
    """
    Bandingkan M_{k-1} vs M_k: jalankan MCR-ALS pada DATA ASLI di kedua
    model-order, hitung ΔLOF = LOF_{k-1} - LOF_k (brief Section 7).

    PENTING: ΔLOF di sini SENDIRIAN belum menjawab "apakah perbaikannya
    signifikan" — brief eksplisit bilang "a lower LOF alone is not
    enough". Fungsi ini hanya menyediakan angka observed; signifikansinya
    baru bisa dinilai lewat run_null_delta_lof_test() (Evidence 4) di
    bawah, yang membandingkan angka ini terhadap distribusi null.

    Returns
    -------
    dict: lof_lower, lof_higher, delta_lof, C_lower, S_lower,
          C_higher, S_higher (dikembalikan supaya bisa dipakai ulang,
          mis. residual_lower untuk null test, tanpa fit ulang).
    """
    mcr_kwargs = dict(mcr_kwargs or {})
    mcr_defaults = dict(max_iter=300, tol=1e-6, closure=False, init_method="simplisma")
    mcr_defaults.update(mcr_kwargs)

    D = np.asarray(D, dtype=float)
    C_lower, S_lower, _, _, conv_lower, diag_lower = run_mcr_als(
        D, n_components=n_components_lower, **mcr_defaults)
    C_higher, S_higher, _, _, conv_higher, diag_higher = run_mcr_als(
        D, n_components=n_components_higher, **mcr_defaults)

    lof_lower = float(diag_lower["lof_final"])
    lof_higher = float(diag_higher["lof_final"])

    return {
        "n_components_lower": n_components_lower, "n_components_higher": n_components_higher,
        "lof_lower": lof_lower, "lof_higher": lof_higher, "delta_lof": lof_lower - lof_higher,
        "converged_lower": bool(conv_lower), "converged_higher": bool(conv_higher),
        "C_lower": C_lower, "S_lower": S_lower, "C_higher": C_higher, "S_higher": S_higher,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. NULL / PERMUTATION / RESIDUAL-BOOTSTRAP TEST (Evidence 4, Section 8)
# ═══════════════════════════════════════════════════════════════════════
#
# H0 (brief Section 8): "the extra factor explains residual/noise
# structure rather than an independent chemical component". Null
# dataset dibangun dengan menganggap model (k-1) komponen SUDAH benar
# (M_{k-1}), lalu residualnya di-resample (bootstrap/permutation) dan
# ditambahkan kembali ke rekonstruksi (k-1)-komponen — persis metode
# "residual bootstrap" yang brief sebut sebagai opsi pertama.

def _generate_null_dataset(D_hat_lower, residual_lower, method, rng):
    flat = residual_lower.ravel()
    if method == "bootstrap":
        resampled = rng.choice(flat, size=flat.shape[0], replace=True)
    elif method == "permutation":
        resampled = rng.permutation(flat)
    else:
        raise ValueError(f"method harus 'bootstrap' atau 'permutation', dapat {method!r}")
    return D_hat_lower + resampled.reshape(residual_lower.shape)


def run_null_delta_lof_test(D, n_components_lower, n_components_higher,
                             n_null_datasets=200, method="bootstrap",
                             mcr_kwargs=None, base_seed=0):
    """
    Uji signifikansi ΔLOF (Evidence 3+4 brief digabung jadi satu alur):

      1. Fit data ASLI di k-1 dan k komponen -> delta_lof_observed.
      2. Bangun n_null_datasets dataset null di bawah H0 (residual model
         k-1 di-resample dan ditambahkan kembali ke rekonstruksi
         k-1-komponen — data yang KONSISTEN dengan "cuma k-1 komponen
         nyata, sisanya noise").
      3. Fit SETIAP dataset null di k-1 dan k komponen -> distribusi
         delta_lof_null.
      4. Bandingkan delta_lof_observed terhadap persentil-95 distribusi
         null (delta_lof_critical_95) dan hitung p-value satu-arah.

    PERINGATAN BIAYA KOMPUTASI: setiap null dataset butuh 2 kali fit
    MCR-ALS (k-1 dan k) — n_null_datasets=200 berarti ~400 fit MCR
    tambahan. Untuk eksplorasi cepat pakai n_null_datasets kecil
    (20-30); untuk angka final manuskrip, brief menyarankan orde
    ratusan (konsisten dengan p-value yang presisinya wajar).

    Returns
    -------
    dict: n_components_lower/higher, lof_lower_observed, lof_higher_observed,
          delta_lof_observed, delta_lof_null_distribution (np.array),
          n_null_datasets_used, n_null_datasets_failed, critical_95,
          p_value, method, classification.
    """
    nested_observed = compute_delta_lof_nested(D, n_components_lower, n_components_higher, mcr_kwargs)
    delta_lof_observed = nested_observed["delta_lof"]

    D = np.asarray(D, dtype=float)
    D_hat_lower = nested_observed["C_lower"] @ nested_observed["S_lower"]
    residual_lower = D - D_hat_lower

    mcr_kwargs = dict(mcr_kwargs or {})
    mcr_defaults = dict(max_iter=300, tol=1e-6, closure=False, init_method="simplisma")
    mcr_defaults.update(mcr_kwargs)

    delta_lof_null = []
    n_failed = 0
    for i in range(n_null_datasets):
        rng = np.random.default_rng(base_seed + i)
        D_null = _generate_null_dataset(D_hat_lower, residual_lower, method, rng)
        try:
            _, _, _, _, _, diag_null_lower = run_mcr_als(
                D_null, n_components=n_components_lower, **mcr_defaults)
            _, _, _, _, _, diag_null_higher = run_mcr_als(
                D_null, n_components=n_components_higher, **mcr_defaults)
            delta_null = float(diag_null_lower["lof_final"]) - float(diag_null_higher["lof_final"])
            delta_lof_null.append(delta_null)
        except Exception:
            n_failed += 1
            continue

    if not delta_lof_null:
        raise RuntimeError(
            "Semua null dataset gagal di-fit — coba kurangi n_components_higher, "
            "periksa data, atau kurangi n_null_datasets untuk debug lebih cepat."
        )
    delta_lof_null = np.array(delta_lof_null)
    critical_95 = float(np.percentile(delta_lof_null, 95))
    p_value = float(np.mean(delta_lof_null >= delta_lof_observed))

    if delta_lof_observed > critical_95:
        classification = {
            "label": "signifikan",
            "detail": (
                f"ΔLOF observed ({delta_lof_observed:.4f}%) MELEBIHI ambang kritis "
                f"95% distribusi null ({critical_95:.4f}%, p≈{p_value:.3f}) — "
                f"perbaikan LOF pada model {n_components_higher}-komponen kemungkinan "
                f"besar BUKAN sekadar kapasitas fleksibilitas tambahan. Ini bukti yang "
                f"jauh lebih kuat daripada sekadar cutoff LOF sembarang (brief Section 8)."
            ),
        }
    else:
        classification = {
            "label": "tidak_signifikan",
            "detail": (
                f"ΔLOF observed ({delta_lof_observed:.4f}%) TIDAK melebihi ambang kritis "
                f"95% distribusi null ({critical_95:.4f}%, p≈{p_value:.3f}) — perbaikan "
                f"LOF ini KONSISTEN dengan yang bisa dihasilkan noise/fleksibilitas "
                f"tambahan saja. Faktor ke-{n_components_higher} PATUT DICURIGAI spurious "
                f"berdasarkan uji ini (meski tetap perlu digabung dengan bukti lain — "
                f"contribution ratio, reproducibility — sebelum vonis akhir)."
            ),
        }

    return {
        "n_components_lower": n_components_lower, "n_components_higher": n_components_higher,
        "lof_lower_observed": nested_observed["lof_lower"], "lof_higher_observed": nested_observed["lof_higher"],
        "delta_lof_observed": delta_lof_observed,
        "delta_lof_null_distribution": delta_lof_null,
        "n_null_datasets_used": int(len(delta_lof_null)), "n_null_datasets_failed": n_failed,
        "critical_95": critical_95, "p_value": p_value,
        "method": method, "classification": classification,
    }
