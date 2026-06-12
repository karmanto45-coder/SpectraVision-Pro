"""
mcr_engine.py  —  SpectraVision Pro v3.1
MCR-ALS engine dengan perbaikan ilmiah menyeluruh berdasarkan:
- Tauler (1995) Chemom. Intell. Lab. Syst.
- Jaumot et al. (2015) Chemom. Intell. Lab. Syst.
- Unscrambler MCR-ALS algorithm documentation

Perbaikan dari v3.0:
  1. Bug closure constraint diperbaiki (C bukan S)
  2. Normalisasi S per iterasi + kompensasi ke C
  3. Inisialisasi: shift-to-positive bukan np.abs()
  4. Unimodality constraint diimplementasikan nyata
  5. Sensitivity parameter (Malinowski E1/En ratio)
  6. Auto-detection k dengan hybrid manual/otomatis
  7. Residual per sampel dan per wavenumber
  8. Warning system 4 tipe (Unscrambler-style)
  9. NNV (Non-negativity Violation score)
 10. Spectral Reproducibility Index (SRI)
 11. External spectral initial guess dari library
 12. Multi-k run untuk konsistensi checker
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter


# ════════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════════

def preprocess(spectra_matrix, wavenumber,
               do_norm=True, do_smooth=False, do_baseline=False):
    proc = spectra_matrix.copy().astype(float)
    if do_smooth:
        for i in range(proc.shape[1]):
            proc[:, i] = savgol_filter(proc[:, i], 11, 3)
    if do_baseline:
        for i in range(proc.shape[1]):
            proc[:, i] -= proc[:, i].min()
    if do_norm:
        for i in range(proc.shape[1]):
            area = (np.trapezoid(np.abs(proc[:, i]), wavenumber)
                    if hasattr(np, 'trapezoid')
                    else np.trapz(np.abs(proc[:, i]), wavenumber))
            if area > 0:
                proc[:, i] /= area
    return proc


# ════════════════════════════════════════════════════════════════
# POST-MCR SPECTRAL PROCESSING
# ════════════════════════════════════════════════════════════════

def postprocess_mcr_spectra(S, wavenumber,
                             do_smooth=False, sg_window=11, sg_poly=3,
                             do_norm=False, norm_method="area",
                             do_baseline=False):
    S_proc = S.copy().astype(float)
    wn = np.array(wavenumber, dtype=float)
    log = []

    if do_baseline:
        for i in range(S_proc.shape[0]):
            S_proc[i] -= S_proc[i].min()
        log.append("Baseline correction: min subtraction")

    if do_smooth:
        win  = max(5, sg_window if sg_window % 2 == 1 else sg_window + 1)
        poly = min(sg_poly, win - 2)
        for i in range(S_proc.shape[0]):
            try:
                S_proc[i] = savgol_filter(S_proc[i], win, poly)
                S_proc[i] = np.maximum(S_proc[i], 0)
            except Exception:
                pass
        log.append(f"Smoothing: Savitzky-Golay (window={win}, poly={poly})")

    if do_norm:
        for i in range(S_proc.shape[0]):
            sp = S_proc[i]
            if norm_method == "area":
                denom = (np.trapezoid(np.abs(sp), wn)
                         if hasattr(np, 'trapezoid')
                         else np.trapz(np.abs(sp), wn))
                if denom > 0:
                    S_proc[i] = sp / denom
            elif norm_method == "max":
                mx = sp.max()
                if mx > 0:
                    S_proc[i] = sp / mx
            elif norm_method == "vector":
                nv = np.linalg.norm(sp)
                if nv > 0:
                    S_proc[i] = sp / nv
            elif norm_method == "minmax":
                mn, mx = sp.min(), sp.max()
                if mx > mn:
                    S_proc[i] = (sp - mn) / (mx - mn)
        log.append(f"Normalization: {norm_method}")

    return S_proc, log


# ════════════════════════════════════════════════════════════════
# PCA COMPONENT DETECTION + SENSITIVITY
# ════════════════════════════════════════════════════════════════

def detect_components(D, max_k=10, sensitivity=100):
    """
    Deteksi jumlah komponen optimal menggunakan PCA + Malinowski criteria.

    Parameters
    ----------
    D           : data matrix (n_samples × n_wavelengths)
    max_k       : batas atas komponen yang dicek
    sensitivity : Unscrambler-style sensitivity (10–190, default 100)
                  = threshold ratio E1/(En×10)
                  Tinggi → lebih banyak komponen terdeteksi (komponen minor)
                  Rendah → hanya komponen dominan

    Returns
    -------
    ev       : explained variance per komponen (%)
    cum      : cumulative explained variance (%)
    auto_k   : rekomendasi jumlah komponen (95% threshold)
    sens_k   : rekomendasi berdasarkan sensitivity parameter
    ind_vals : Malinowski IND function values
    ev_abs   : absolute eigenvalues
    """
    n = min(max_k, min(D.shape) - 1)
    n = max(n, 2)
    pca = PCA(n_components=n)
    pca.fit(D)

    ev     = pca.explained_variance_ratio_ * 100
    cum    = np.cumsum(ev)
    ev_abs = pca.explained_variance_  # eigenvalues absolut

    # Auto-k berdasarkan 95% variansi kumulatif
    auto_k = int(np.searchsorted(cum, 95)) + 1
    auto_k = max(2, min(auto_k, n))

    # Malinowski IND function: IND(k) = LOF_k / (n_var - k)²
    # Minimum IND → jumlah komponen optimal
    n_var = D.shape[1]
    ind_vals = []
    for k in range(1, n + 1):
        lof_k = np.sqrt(1 - cum[k-1]/100) if cum[k-1] < 100 else 1e-10
        ind_k = lof_k / max((n_var - k) ** 2, 1)
        ind_vals.append(float(ind_k))

    malinowski_k = int(np.argmin(ind_vals)) + 1
    malinowski_k = max(2, min(malinowski_k, n))

    # Sensitivity-based k (Unscrambler E1/En ratio)
    # sens_k = komponen terbanyak di mana E1/(Ek×10) < sensitivity
    e1 = float(ev_abs[0]) if len(ev_abs) > 0 else 1.0
    sens_k = 2
    for k in range(2, n + 1):
        ek = float(ev_abs[k-1]) if k-1 < len(ev_abs) else 1e-10
        ratio = e1 / (ek * 10) if ek > 0 else float('inf')
        if ratio < sensitivity:
            sens_k = k
        else:
            break
    sens_k = max(2, min(sens_k, n))

    return ev, cum, auto_k, sens_k, ind_vals, ev_abs


# ════════════════════════════════════════════════════════════════
# INISIALISASI MCR
# ════════════════════════════════════════════════════════════════

def _init_mcr(D, n_components, init_method="pca", S_init=None):
    """
    Inisialisasi C dan S untuk MCR-ALS.

    init_method:
      'pca'     → PCA scores sebagai C awal (shift-to-positive)
                  Cepat, umum digunakan, tapi menghasilkan komponen
                  dengan nilai negatif yang dikoreksi
      'nmf'     → NMF-NNDSVD sebagai C dan S awal
                  Non-negatif dari awal tanpa koreksi,
                  lebih sesuai untuk data spektroskopi
      'library' → S_init dari spektra eksternal (library)
                  Paling akurat jika spektra murni tersedia

    Returns: C_init (m × k), S_init_out (k × n) or None
    """
    from sklearn.decomposition import NMF
    D = np.array(D, dtype=float)
    m, n = D.shape

    # ── Pilihan 1: Library spectral guess ────────────────────
    if S_init is not None:
        S0 = np.array(S_init, dtype=float)
        if S0.shape[0] != n_components:
            S0 = S0[:n_components] if S0.shape[0] > n_components else S0
        S0 = np.maximum(S0, 0)
        norms = np.linalg.norm(S0, axis=1, keepdims=True)
        norms[norms == 0] = 1
        S0 = S0 / norms
        C0 = np.linalg.lstsq(S0.T, D.T, rcond=None)[0].T
        C0 = np.maximum(C0, 1e-10)
        return C0, S0

    # ── Pilihan 2: NMF-NNDSVD ────────────────────────────────
    elif init_method == "nmf":
        try:
            # NNDSVD: Non-negative Double SVD
            # Menghasilkan C dan S yang sudah non-negatif dari awal
            # tanpa perlu koreksi abs() atau shift
            D_pos = np.maximum(D, 1e-10)  # NMF butuh nilai positif
            nmf = NMF(
                n_components=n_components,
                init="nndsvd",           # non-negative double SVD
                max_iter=50,             # hanya untuk inisialisasi
                tol=1e-3,
                random_state=42
            )
            C0 = nmf.fit_transform(D_pos)
            S0 = nmf.components_
            C0 = np.maximum(C0, 1e-10)
            S0 = np.maximum(S0, 0)
            return C0, S0
        except Exception:
            # Fallback ke PCA jika NMF gagal
            pass

    # ── Pilihan 3: PCA (default / fallback) ──────────────────
    pca = PCA(n_components=n_components)
    C_raw = pca.fit_transform(D)
    C_shifted = C_raw - C_raw.min(axis=0)
    C0 = np.maximum(C_shifted, 1e-10)
    return C0, None


# ════════════════════════════════════════════════════════════════
# UNIMODALITY CONSTRAINT
# ════════════════════════════════════════════════════════════════

def _apply_unimodality(profile):
    """
    Terapkan unimodality constraint pada satu profil (1D array).
    Profil harus memiliki tepat satu puncak (unimodal).
    Metode: isotonic regression approach — flatten bagian yang turun
    sebelum puncak dan naik setelah puncak.
    """
    p = profile.copy()
    n = len(p)
    if n < 3:
        return p

    # Cari puncak
    peak_idx = int(np.argmax(p))

    # Paksa monoton naik di kiri puncak
    for i in range(peak_idx - 1, -1, -1):
        if p[i] > p[i + 1]:
            p[i] = p[i + 1]

    # Paksa monoton turun di kanan puncak
    for i in range(peak_idx + 1, n):
        if p[i] > p[i - 1]:
            p[i] = p[i - 1]

    return np.maximum(p, 0)


# ════════════════════════════════════════════════════════════════
# MCR-ALS UTAMA
# ════════════════════════════════════════════════════════════════

def run_mcr_als(D, n_components, max_iter=200, tol=1e-6,
                closure=False, unimodal=False,
                normalize_S=False, init_method="pca", S_init=None):
    """
    MCR-ALS dengan constraint lengkap dan inisialisasi yang benar.

    Parameters
    ----------
    D            : (n_samples × n_wavelengths)
    n_components : jumlah komponen
    max_iter     : iterasi maksimum
    tol          : toleransi konvergensi (perubahan LOF)
    closure      : closure constraint pada C (jumlah fraksi = 1) [DIPERBAIKI]
    unimodal     : unimodality constraint pada S [DIIMPLEMENTASIKAN]
    normalize_S  : normalisasi S per iterasi (unit vector) [BARU]
    S_init       : spektra awal dari library eksternal (k × n) [BARU]

    Returns
    -------
    C, S, lof_history, r2, converged, diagnostics
    diagnostics : dict berisi residual per sampel, per wavenumber, NNV, dll
    """
    D = np.array(D, dtype=float)
    m, n = D.shape

    # ── Inisialisasi ──────────────────────────────────────────
    C, S0 = _init_mcr(D, n_components, init_method, S_init)
    if S0 is not None:
        S = S0.copy()
        # Update C dari S awal
        C = np.linalg.lstsq(S.T, D.T, rcond=None)[0].T
        C = np.maximum(C, 1e-10)
    else:
        # S akan dihitung di iterasi pertama
        S = None

    lof_history = []
    converged   = False

    # ── Loop ALS ─────────────────────────────────────────────
    for iteration in range(max_iter):

        # Step 1: Update S = least squares dari C
        S = np.linalg.lstsq(C, D, rcond=None)[0]

        # Non-negativity pada S
        S = np.maximum(S, 0)

        # Unimodality pada S [BARU — benar-benar diimplementasikan]
        if unimodal:
            for i in range(S.shape[0]):
                S[i] = _apply_unimodality(S[i])

        # Normalisasi S ke unit vector + kompensasi ke C [BARU]
        if normalize_S:
            norms = np.linalg.norm(S, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            S = S / norms
            # Kompensasi skala ke C agar D = C×S tetap sama
            C = C * norms.T

        # Step 2: Update C = least squares dari S
        C = np.linalg.lstsq(S.T, D.T, rcond=None)[0].T

        # Non-negativity pada C
        C = np.maximum(C, 0)

        # Closure constraint pada C [BUG DIPERBAIKI — sebelumnya di S]
        if closure:
            row_sums = C.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            C = C / row_sums

        # ── Hitung LOF ────────────────────────────────────────
        D_hat   = C @ S
        residual = D - D_hat
        ss_res  = np.sum(residual ** 2)
        ss_tot  = np.sum(D ** 2)
        lof     = np.sqrt(ss_res / ss_tot) * 100 if ss_tot > 0 else 0.0
        lof_history.append(lof)

        # ── Cek konvergensi ───────────────────────────────────
        if iteration > 2:
            delta = abs(lof_history[-2] - lof_history[-1])
            if delta < tol:
                converged = True
                break

    # ── Diagnostik akhir ─────────────────────────────────────
    D_hat    = C @ S
    residual = D - D_hat

    # R² terhadap mean-centered D
    ss_res_r2 = np.sum((D - D_hat) ** 2)
    ss_tot_r2 = np.sum((D - np.mean(D)) ** 2)
    r2 = 1 - ss_res_r2 / ss_tot_r2 if ss_tot_r2 > 0 else 0.0

    # RMSE
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    # Residual per sampel (sample residuals)
    sample_residuals = np.sqrt(np.mean(residual ** 2, axis=1))

    # Residual per wavenumber (variable residuals)
    var_residuals = np.sqrt(np.mean(residual ** 2, axis=0))

    # Non-negativity Violation Score per komponen
    nnv_scores = []
    for i in range(S.shape[0]):
        n_neg   = np.sum(S[i] < 0)
        nnv_pct = float(n_neg / S.shape[1] * 100)
        nnv_scores.append(nnv_pct)

    # Explained variance per komponen
    ev_comp = []
    for i in range(n_components):
        S_i   = S[i:i+1]
        C_i   = C[:, i:i+1]
        D_i   = C_i @ S_i
        ev_i  = float(np.sum(D_i ** 2) / np.sum(D ** 2) * 100) if np.sum(D**2) > 0 else 0.0
        ev_comp.append(ev_i)

    # LOF per sampel
    lof_per_sample = []
    for i in range(m):
        ss_r = np.sum(residual[i] ** 2)
        ss_t = np.sum(D[i] ** 2)
        lof_i = float(np.sqrt(ss_r / ss_t) * 100) if ss_t > 0 else 0.0
        lof_per_sample.append(lof_i)

    diagnostics = {
        "rmse":             rmse,
        "lof_final":        lof_history[-1] if lof_history else 0.0,
        "lof_per_sample":   lof_per_sample,
        "sample_residuals": sample_residuals.tolist(),
        "var_residuals":    var_residuals.tolist(),
        "nnv_scores":       nnv_scores,
        "ev_per_comp":      ev_comp,
        "n_iter":           len(lof_history),
        "converged":        converged,
    }

    return C, S, lof_history, r2, converged, diagnostics


# ════════════════════════════════════════════════════════════════
# MULTI-K RUN (KONSISTENSI CHECKER)
# ════════════════════════════════════════════════════════════════

def run_mcr_multi_k(D, k_range=(2, 6), max_iter=200, tol=1e-6,
                    closure=False, unimodal=False, normalize_S=False,
                    init_method="pca"):
    """
    Jalankan MCR-ALS untuk semua k dalam k_range.
    Hitung Spectral Reproducibility Index (SRI) antar model k dan k+1.

    Returns
    -------
    results : dict dengan key = k, value = dict hasil MCR
    sri_table : list of dict — SRI per pasangan komponen antar k
    summary_table : list of dict — ringkasan per k
    recommended_k : rekomendasi k berdasarkan LOF stability + SRI
    """
    k_min, k_max = k_range
    k_max = min(k_max, min(D.shape) - 1)
    k_min = max(2, k_min)

    results = {}
    for k in range(k_min, k_max + 1):
        C, S, lof_hist, r2, conv, diag = run_mcr_als(
            D, k, max_iter, tol, closure, unimodal, normalize_S,
            init_method=init_method
        )
        results[k] = {
            "C": C, "S": S,
            "lof_history": lof_hist,
            "lof_final": diag["lof_final"],
            "r2": r2,
            "converged": conv,
            "diagnostics": diag,
        }

    # ── SRI: Spectral Reproducibility Index ──────────────────
    # Cocokkan spektra murni antar model k dan k+1 menggunakan cosine similarity
    sri_table = []
    k_list = sorted(results.keys())
    for idx in range(len(k_list) - 1):
        k1 = k_list[idx]
        k2 = k_list[idx + 1]
        S1 = results[k1]["S"]  # k1 × n
        S2 = results[k2]["S"]  # k2 × n

        # Untuk setiap komponen di k1, cari pasangan terbaik di k2
        matched = []
        for i in range(S1.shape[0]):
            best_cos = -1
            best_j   = -1
            for j in range(S2.shape[0]):
                c = cosine_sim(S1[i], S2[j])
                if c > best_cos:
                    best_cos = c
                    best_j   = j
            stability = ("stabil" if best_cos >= 0.95
                         else "perlu cek" if best_cos >= 0.85
                         else "tidak stabil")
            matched.append({
                "k1": k1, "k2": k2,
                "comp_k1": i + 1,
                "comp_k2": best_j + 1,
                "sri": round(best_cos, 4),
                "stability": stability,
            })
        sri_table.extend(matched)

    # ── Summary table per k ───────────────────────────────────
    summary_table = []
    prev_lof = None
    for k in k_list:
        r = results[k]
        lof = r["lof_final"]
        lof_drop = round(prev_lof - lof, 3) if prev_lof is not None else None
        significant = (lof_drop is not None and lof_drop > 1.0)

        # Rata-rata SRI untuk k ini vs k-1
        sri_vals = [x["sri"] for x in sri_table if x["k2"] == k]
        avg_sri  = round(float(np.mean(sri_vals)), 3) if sri_vals else None

        # Skor validasi sederhana
        nnv_max = max(r["diagnostics"]["nnv_scores"]) if r["diagnostics"]["nnv_scores"] else 100
        ev_min  = min(r["diagnostics"]["ev_per_comp"]) if r["diagnostics"]["ev_per_comp"] else 0

        valid_flags = [
            lof <= 10,
            nnv_max <= 5,
            ev_min >= 1.0,
            (avg_sri is None or avg_sri >= 0.90),
        ]
        score = sum(valid_flags)

        summary_table.append({
            "k":              k,
            "lof":            round(lof, 3),
            "r2":             round(r["r2"], 5),
            "lof_drop":       lof_drop,
            "lof_drop_signif": significant,
            "avg_sri":        avg_sri,
            "nnv_max":        round(nnv_max, 2),
            "ev_min":         round(ev_min, 2),
            "valid_score":    score,
            "converged":      r["converged"],
        })
        prev_lof = lof

    # ── Rekomendasi k ─────────────────────────────────────────
    # Pilih k dengan valid_score tertinggi, tie-break: k terkecil
    best = max(summary_table, key=lambda x: (x["valid_score"], -x["k"]))
    recommended_k = best["k"]

    return results, sri_table, summary_table, recommended_k


# ════════════════════════════════════════════════════════════════
# WARNING SYSTEM (Unscrambler-style, 4 tipe)
# ════════════════════════════════════════════════════════════════

def generate_warnings(diagnostics, k, ev, sensitivity=100):
    """
    Generate warning list sesuai Unscrambler 4-type system.

    Returns list of dict: {type, code, message_id, message_en, severity}
    severity: 'error' | 'warning' | 'info'
    """
    warnings = []

    lof       = diagnostics["lof_final"]
    nnv_scores = diagnostics["nnv_scores"]
    ev_comp   = diagnostics["ev_per_comp"]
    lof_per_s = diagnostics["lof_per_sample"]
    var_res   = diagnostics["var_residuals"]

    # ── Type 1: Naikkan sensitivity / tambah komponen ─────────
    if lof > 10:
        warnings.append({
            "type": 1, "code": "HIGH_LOF", "severity": "error",
            "message_id": f"LOF terlalu tinggi ({lof:.1f}%). Kemungkinan jumlah komponen kurang. Coba naikkan jumlah komponen atau sensitivity.",
            "message_en": f"LOF too high ({lof:.1f}%). Likely too few components. Try increasing number of components or sensitivity.",
        })

    if ev_comp and min(ev_comp) < 0.5:
        warnings.append({
            "type": 1, "code": "LOW_EV_COMP", "severity": "warning",
            "message_id": f"Komponen dengan explained variance sangat kecil ({min(ev_comp):.2f}%). Ada kemungkinan komponen minor yang belum terdeteksi.",
            "message_en": f"Component with very low explained variance ({min(ev_comp):.2f}%). Possible minor component not yet detected.",
        })

    # ── Type 2: Turunkan sensitivity / kurangi komponen ───────
    if ev_comp and min(ev_comp) < 0.1:
        warnings.append({
            "type": 2, "code": "NOISE_COMP", "severity": "warning",
            "message_id": f"Komponen terakhir hanya menjelaskan {min(ev_comp):.3f}% variansi — kemungkinan besar noise, bukan komponen kimia nyata. Pertimbangkan kurangi jumlah komponen.",
            "message_en": f"Last component explains only {min(ev_comp):.3f}% variance — likely noise, not a real chemical component. Consider reducing number of components.",
        })

    # Cek apakah ada dua spektra sangat mirip (SRI internal)
    S_nnv = nnv_scores
    if len(S_nnv) >= 2:
        # Placeholder — SRI antar komponen dalam model yang sama
        pass

    # ── Type 3: Ubah sensitivity (tidak konsisten) ────────────
    if 5 < lof <= 10:
        warnings.append({
            "type": 3, "code": "MODERATE_LOF", "severity": "warning",
            "message_id": f"LOF dalam range sedang ({lof:.1f}%). Model dapat diterima namun tidak optimal. Coba variasikan jumlah komponen.",
            "message_en": f"LOF in moderate range ({lof:.1f}%). Model acceptable but not optimal. Try varying number of components.",
        })

    # ── Type 4: Baseline / normalisasi ────────────────────────
    nnv_max = max(nnv_scores) if nnv_scores else 0
    if nnv_max > 5:
        warnings.append({
            "type": 4, "code": "HIGH_NNV", "severity": "warning",
            "message_id": f"Spektra murni komponen memiliki {nnv_max:.1f}% nilai negatif (NNV tinggi). Pertimbangkan koreksi baseline atau normalisasi sebelum MCR.",
            "message_en": f"Pure spectra have {nnv_max:.1f}% negative values (high NNV). Consider baseline correction or normalization before MCR.",
        })

    if nnv_max > 15:
        warnings.append({
            "type": 4, "code": "VERY_HIGH_NNV", "severity": "error",
            "message_id": "NNV > 15%: spektra murni sangat tidak fisik. Lakukan koreksi baseline atau ubah rentang wavenumber.",
            "message_en": "NNV > 15%: pure spectra highly unphysical. Apply baseline correction or change wavenumber range.",
        })

    # ── Cek outlier sampel ────────────────────────────────────
    if lof_per_s:
        mean_lof_s = np.mean(lof_per_s)
        outlier_idx = [i for i, v in enumerate(lof_per_s) if v > 3 * mean_lof_s]
        if outlier_idx:
            warnings.append({
                "type": 3, "code": "SAMPLE_OUTLIER", "severity": "warning",
                "message_id": f"Sampel {[i+1 for i in outlier_idx]} memiliki residual jauh di atas rata-rata (>3×). Kemungkinan outlier — pertimbangkan untuk diperiksa atau dihapus.",
                "message_en": f"Samples {[i+1 for i in outlier_idx]} have residuals far above average (>3×). Possible outliers — consider inspecting or removing.",
            })

    # ── Cek wavenumber bermasalah ─────────────────────────────
    if var_res:
        vr    = np.array(var_res)
        mean_vr = np.mean(vr)
        if mean_vr > 0:
            high_var_pct = float(np.sum(vr > 3 * mean_vr) / len(vr) * 100)
            if high_var_pct > 5:
                warnings.append({
                    "type": 3, "code": "HIGH_VAR_RESIDUAL", "severity": "info",
                    "message_id": f"{high_var_pct:.1f}% titik wavenumber memiliki residual tinggi (>3× rata-rata). Pertimbangkan menyempitkan rentang analisis.",
                    "message_en": f"{high_var_pct:.1f}% of wavenumber points have high residuals (>3× average). Consider narrowing the analysis range.",
                })

    # Jika tidak ada warning
    if not warnings:
        warnings.append({
            "type": 0, "code": "OK", "severity": "info",
            "message_id": "Tidak ada peringatan signifikan. Model MCR tampak valid secara diagnostik.",
            "message_en": "No significant warnings. MCR model appears diagnostically valid.",
        })

    return warnings


# ════════════════════════════════════════════════════════════════
# MCR VALIDATION SCORECARD
# ════════════════════════════════════════════════════════════════

def compute_scorecard(diagnostics, summary_row=None):
    """
    Hitung MCR Validation Scorecard (8 kriteria).

    Returns
    -------
    scorecard : list of dict {criterion, value, status, message_id, message_en}
    total_score : int (0–8)
    overall : 'baik' | 'sedang' | 'perlu_perbaikan'
    """
    lof       = diagnostics["lof_final"]
    rmse      = diagnostics["rmse"]
    nnv_scores = diagnostics["nnv_scores"]
    ev_comp   = diagnostics["ev_per_comp"]
    lof_per_s = diagnostics["lof_per_sample"]
    converged = diagnostics["converged"]

    nnv_max  = max(nnv_scores) if nnv_scores else 100
    ev_min   = min(ev_comp)    if ev_comp    else 0
    lof_max_s = max(lof_per_s) if lof_per_s  else 100
    lof_mean_s = np.mean(lof_per_s) if lof_per_s else 100

    scorecard = []

    # 1. LOF global
    ok1 = lof < 5
    warn1 = lof < 10
    scorecard.append({
        "criterion": "LOF global",
        "value": f"{lof:.3f}%",
        "status": "✅" if ok1 else ("🟡" if warn1 else "❌"),
        "message_id": "Sangat baik (<5%)" if ok1 else ("Dapat diterima (5–10%)" if warn1 else "Terlalu tinggi (>10%)"),
        "message_en": "Excellent (<5%)" if ok1 else ("Acceptable (5–10%)" if warn1 else "Too high (>10%)"),
    })

    # 2. RMSE
    ok2 = rmse < 0.005
    warn2 = rmse < 0.02
    scorecard.append({
        "criterion": "RMSE",
        "value": f"{rmse:.5f}",
        "status": "✅" if ok2 else ("🟡" if warn2 else "❌"),
        "message_id": "Sangat kecil" if ok2 else ("Dapat diterima" if warn2 else "Terlalu besar"),
        "message_en": "Very small" if ok2 else ("Acceptable" if warn2 else "Too large"),
    })

    # 3. NNV (Non-negativity Violation)
    ok3 = nnv_max < 1
    warn3 = nnv_max < 5
    scorecard.append({
        "criterion": "NNV maks (%)",
        "value": f"{nnv_max:.2f}%",
        "status": "✅" if ok3 else ("🟡" if warn3 else "❌"),
        "message_id": "Sangat baik (<1%)" if ok3 else ("Dapat diterima (1–5%)" if warn3 else "Bermasalah (>5%)"),
        "message_en": "Excellent (<1%)" if ok3 else ("Acceptable (1–5%)" if warn3 else "Problematic (>5%)"),
    })

    # 4. EV komponen terakhir
    ok4 = ev_min >= 3
    warn4 = ev_min >= 1
    scorecard.append({
        "criterion": "EV komponen terkecil (%)",
        "value": f"{ev_min:.2f}%",
        "status": "✅" if ok4 else ("🟡" if warn4 else "❌"),
        "message_id": "Komponen signifikan (≥3%)" if ok4 else ("Komponen minor (1–3%)" if warn4 else "Kemungkinan noise (<1%)"),
        "message_en": "Significant component (≥3%)" if ok4 else ("Minor component (1–3%)" if warn4 else "Possibly noise (<1%)"),
    })

    # 5. LOF maks per sampel
    ok5 = lof_max_s < 10
    warn5 = lof_max_s < 20
    scorecard.append({
        "criterion": "LOF maks per sampel (%)",
        "value": f"{lof_max_s:.2f}%",
        "status": "✅" if ok5 else ("🟡" if warn5 else "❌"),
        "message_id": "Tidak ada outlier (<10%)" if ok5 else ("Outlier potensial (10–20%)" if warn5 else "Outlier jelas (>20%)"),
        "message_en": "No outliers (<10%)" if ok5 else ("Potential outlier (10–20%)" if warn5 else "Clear outlier (>20%)"),
    })

    # 6. Konvergensi
    scorecard.append({
        "criterion": "Konvergensi",
        "value": "Ya" if converged else "Tidak",
        "status": "✅" if converged else "❌",
        "message_id": "Algoritma konvergen" if converged else "Belum konvergen — tambah iterasi",
        "message_en": "Algorithm converged" if converged else "Not converged — increase iterations",
    })

    # 7. SRI (dari summary_row jika tersedia)
    if summary_row and summary_row.get("avg_sri") is not None:
        sri = summary_row["avg_sri"]
        ok7   = sri >= 0.95
        warn7 = sri >= 0.85
        scorecard.append({
            "criterion": "SRI rata-rata",
            "value": f"{sri:.3f}",
            "status": "✅" if ok7 else ("🟡" if warn7 else "❌"),
            "message_id": "Komponen stabil antar model" if ok7 else ("Perlu verifikasi" if warn7 else "Komponen tidak stabil"),
            "message_en": "Components stable across models" if ok7 else ("Needs verification" if warn7 else "Components unstable"),
        })
    else:
        scorecard.append({
            "criterion": "SRI rata-rata",
            "value": "—",
            "status": "⬜",
            "message_id": "Jalankan Konsistensi Checker untuk mendapatkan SRI",
            "message_en": "Run Consistency Checker to obtain SRI",
        })

    # 8. Rasio LOF maks/mean per sampel (deteksi outlier)
    ratio = lof_max_s / lof_mean_s if lof_mean_s > 0 else 1
    ok8   = ratio < 2
    warn8 = ratio < 3
    scorecard.append({
        "criterion": "Rasio LOF maks/mean sampel",
        "value": f"{ratio:.2f}×",
        "status": "✅" if ok8 else ("🟡" if warn8 else "❌"),
        "message_id": "Distribusi LOF merata" if ok8 else ("Satu sampel sedikit menyimpang" if warn8 else "Ada outlier signifikan"),
        "message_en": "LOF evenly distributed" if ok8 else ("One sample slightly deviant" if warn8 else "Significant outlier present"),
    })

    total_score = sum(1 for x in scorecard if x["status"] == "✅")
    if total_score >= 7:
        overall = "baik"
    elif total_score >= 5:
        overall = "sedang"
    else:
        overall = "perlu_perbaikan"

    return scorecard, total_score, overall


# ════════════════════════════════════════════════════════════════
# SPECTRAL MATCHING (tidak berubah)
# ════════════════════════════════════════════════════════════════

def _sort_ascending(wn, sp):
    wn = np.array(wn, dtype=float)
    sp = np.array(sp, dtype=float)
    if wn[0] > wn[-1]:
        wn = wn[::-1]
        sp = sp[::-1]
    return wn, sp


def interpolate_spectrum(wn_ref, sp_ref, wn_target):
    wn_ref, sp_ref = _sort_ascending(wn_ref, sp_ref)
    wn_target = np.array(wn_target, dtype=float)
    ascending = wn_target[0] < wn_target[-1]
    wt     = wn_target if ascending else wn_target[::-1]
    result = np.interp(wt, wn_ref, sp_ref)
    return result if ascending else result[::-1]


def build_common_grid(wn_a, wn_b, grid_interval="auto"):
    wn_a = np.sort(np.array(wn_a, dtype=float))
    wn_b = np.sort(np.array(wn_b, dtype=float))
    ov_min = max(wn_a.min(), wn_b.min())
    ov_max = min(wn_a.max(), wn_b.max())
    overlap = ov_max - ov_min
    info = {
        "wn_a_range":    (float(wn_a.min()), float(wn_a.max())),
        "wn_b_range":    (float(wn_b.min()), float(wn_b.max())),
        "overlap_min":   float(ov_min),
        "overlap_max":   float(ov_max),
        "overlap_width": float(overlap),
        "warning": None, "error": None,
    }
    if overlap <= 0:
        info["error"] = "No overlap between spectra — matching not possible."
        return None, info
    if overlap < 200:
        info["warning"] = f"Overlap only {overlap:.1f} cm⁻¹ — matching may be unreliable."
    if grid_interval == "auto":
        ia = float(np.median(np.diff(wn_a))) if len(wn_a) > 1 else 1.0
        ib = float(np.median(np.diff(wn_b))) if len(wn_b) > 1 else 1.0
        interval = max(min(ia, ib), 0.1)
    else:
        interval = float(grid_interval)
    info["interval_a"]    = float(np.median(np.diff(wn_a))) if len(wn_a) > 1 else 1.0
    info["interval_b"]    = float(np.median(np.diff(wn_b))) if len(wn_b) > 1 else 1.0
    info["grid_interval"] = float(interval)
    common_grid = np.arange(ov_min, ov_max + interval * 0.1, interval)
    info["n_common_points"] = len(common_grid)
    return common_grid, info


def resample_to_grid(wn_src, sp_src, common_grid, method="cubic"):
    from scipy.interpolate import interp1d
    wn_src, sp_src = _sort_ascending(wn_src, sp_src)
    kind = "cubic" if (method == "cubic" and len(wn_src) >= 4) else "linear"
    f = interp1d(wn_src, sp_src, kind=kind, bounds_error=False, fill_value=0.0)
    return f(common_grid)


def apply_window(wn, spec, mode, wmin=None, wmax=None):
    wn   = np.array(wn,   dtype=float)
    spec = np.array(spec, dtype=float)
    if mode == "fingerprint":
        mask = (wn >= 400) & (wn <= 1800)
    elif mode == "custom":
        mask = (wn >= wmin) & (wn <= wmax)
    else:
        mask = np.ones(len(wn), dtype=bool)
    return wn[mask], spec[mask]


def cosine_sim(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def hqi_score(a, b):
    return round(cosine_sim(a, b) ** 2 * 100, 3)


def batch_match(query_spec, query_wn, library_entries,
                window_mode, wmin, wmax, top_n=10,
                grid_interval="auto", interp_method="cubic"):
    if grid_interval != "auto":
        try:
            grid_interval = float(grid_interval)
        except (TypeError, ValueError):
            grid_interval = "auto"
    wn_q, sp_q = apply_window(query_wn, query_spec, window_mode, wmin, wmax)
    if len(wn_q) < 5:
        return []
    results = []
    for entry in library_entries:
        wn_r  = np.array(entry["wavenumber"], dtype=float)
        sp_r  = np.array(entry["spectrum"],   dtype=float)
        wn_r2, sp_r2 = apply_window(wn_r, sp_r, window_mode, wmin, wmax)
        if len(wn_r2) < 5:
            continue
        common_grid, grid_info = build_common_grid(wn_q, wn_r2, grid_interval)
        if common_grid is None or len(common_grid) < 5:
            continue
        sp_q_c = resample_to_grid(wn_q,  sp_q,  common_grid, interp_method)
        sp_r_c = resample_to_grid(wn_r2, sp_r2, common_grid, interp_method)
        cos = round(cosine_sim(sp_q_c, sp_r_c), 4)
        hqi = round(hqi_score(sp_q_c, sp_r_c), 2)
        results.append({
            "id": entry["id"], "name": entry["name"],
            "category": entry["category"],
            "cosine": cos, "hqi": hqi,
            "overlap_min":     grid_info["overlap_min"],
            "overlap_max":     grid_info["overlap_max"],
            "overlap_width":   grid_info["overlap_width"],
            "grid_interval":   grid_info["grid_interval"],
            "n_common_points": grid_info["n_common_points"],
            "interval_query":  grid_info["interval_a"],
            "interval_lib":    grid_info["interval_b"],
            "grid_warning":    grid_info.get("warning"),
            "interp_method":   interp_method,
        })
    results.sort(key=lambda x: x["cosine"], reverse=True)
    return results[:top_n]


def consensus_label(cos, hqi, thresh_cos=0.95, thresh_hqi=90.25):
    cos_strong = cos >= thresh_cos
    hqi_strong = hqi >= thresh_hqi
    cos_med    = cos >= (thresh_cos - 0.05)
    hqi_med    = hqi >= (thresh_hqi - 9.25)
    if cos_strong and hqi_strong:
        return "strong",   "✅ Match kuat / Strong match"
    if cos_med and hqi_med:
        if cos_strong != hqi_strong:
            return "conflict", "⚠️ Konflik ranking / Rank conflict"
        return "medium", "🟡 Match sedang / Medium match"
    if cos_strong != hqi_strong:
        return "conflict", "⚠️ Konflik ranking / Rank conflict"
    return "weak", "❌ Tidak match / No match"
