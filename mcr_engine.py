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
from scipy.optimize import linear_sum_assignment


# ════════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════════

def validate_derivative_params(order, poly, window):
    """
    Validasi parameter Savitzky-Golay untuk domain turunan, sebelum
    dipakai. Mengembalikan (ok: bool, messages: list[str]) — messages
    berisi peringatan/error dalam Bahasa Indonesia untuk ditampilkan
    di UI. Tidak melempar exception — biar UI yang memutuskan apakah
    tetap melanjutkan atau tidak.

    Rambu-rambu (lihat diskusi):
      - poly harus >= order, wajib, atau turunan tidak bisa dihitung sama
        sekali (error, bukan cuma warning).
      - poly == 1 untuk order >= 1 valid secara matematis hanya untuk
        1st derivative, tapi mendistorsi puncak tajam (bias di apex) —
        untuk 2nd derivative poly=1 sama sekali tidak bisa menghasilkan
        turunan kedua (error).
      - window harus ganjil dan > poly (bukan >=, karena SG butuh titik
        lebih banyak dari jumlah koefisien polinomial).
    """
    messages = []
    ok = True

    if poly < order:
        messages.append(
            f"❌ Polynomial order ({poly}) harus ≥ derivative order ({order}) — "
            f"turunan tidak dapat dihitung dengan setelan ini."
        )
        ok = False

    if poly == 1 and order >= 2:
        messages.append(
            "❌ Polynomial order 1 tidak valid untuk 2nd derivative — "
            "naikkan ke minimal 2 (rekomendasi: 3)."
        )
        ok = False
    elif poly == 1:
        messages.append(
            "⚠️ Polynomial order 1 (linear fit per jendela) berisiko mendistorsi "
            "amplitudo di puncak tajam (bias kurvatura). Untuk spektra FTIR/UV-Vis "
            "dengan pita sempit, disarankan poly order ≥ 2."
        )

    if window % 2 == 0:
        messages.append(
            f"❌ Smoothing window ({window}) harus ganjil — akan dibulatkan otomatis."
        )
        ok = False

    if window <= poly:
        messages.append(
            f"❌ Smoothing window ({window}) harus lebih besar dari polynomial order ({poly})."
        )
        ok = False

    if ok and poly != order + 1:
        messages.append(
            f"ℹ️ Rekomendasi umum: polynomial order = derivative order + 1 "
            f"({order + 1}) untuk mengurangi distorsi bentuk puncak. "
            f"Setelan Anda saat ini: poly={poly}."
        )

    return ok, messages


def apply_derivative(spectra_matrix, order=1, poly=None, window=15,
                     symmetric=True):
    """
    Terapkan turunan Savitzky-Golay (1st/2nd derivative, dst.) ke setiap
    baris (spektrum) dalam spectra_matrix.

    Parameters
    ----------
    spectra_matrix : (n_samples x n_wavelengths) — spektrum domain mentah
    order     : orde turunan (1 = 1st derivative, 2 = 2nd derivative)
    poly      : polynomial order SG. Default None → order + 1 (rekomendasi
                standar untuk mengurangi distorsi puncak — lihat
                validate_derivative_params).
    window    : lebar jendela smoothing (titik data), harus ganjil dan
                lebih besar dari poly. Dibulatkan otomatis jika genap.
    symmetric : SG standar (scipy.signal.savgol_filter) selalu memakai
                kernel simetris di titik interior — parameter ini
                dipertahankan untuk keterlacakan/audit trail dan untuk
                konsistensi terminologi dengan software seperti
                Unscrambler. Titik-titik di ujung spektrum otomatis
                ditangani scipy dengan mode 'interp' (polynomial
                fit-extrapolation), bukan kernel asimetris manual.

    Returns
    -------
    deriv_matrix : (n_samples x n_wavelengths) — spektrum turunan
    info         : dict — parameter aktual yang dipakai + validasi
    """
    spectra_matrix = np.asarray(spectra_matrix, dtype=float)
    if poly is None:
        poly = order + 1

    win = window if window % 2 == 1 else window + 1
    win = max(win, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)

    ok, messages = validate_derivative_params(order, poly, win)

    deriv_matrix = np.zeros_like(spectra_matrix)
    for i in range(spectra_matrix.shape[0]):
        deriv_matrix[i] = savgol_filter(
            spectra_matrix[i], window_length=win, polyorder=poly,
            deriv=order, mode="interp"
        )

    info = {
        "order": order, "poly": poly, "window": win,
        "symmetric": bool(symmetric),
        "valid": ok, "messages": messages,
    }
    return deriv_matrix, info


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

def simplisma(D, n_components, f=0.05):
    """
    SIMPLISMA — SIMPle-to-use Interactive Self-modeling Mixture Analysis
    (Windig & Guilment, Anal. Chem. 1991, 63, 1425-1432).

    Mencari "purest variables" - kolom (wavenumber) yang sinyalnya paling
    didominasi oleh satu komponen tunggal di seluruh sampel. Ini adalah
    metode inisialisasi standar yang dipakai mayoritas software MCR-ALS
    komersial (termasuk Unscrambler) karena langsung memberi estimasi awal
    C dari data nyata (bukan proyeksi abstrak PCA), sehingga jauh lebih
    tahan terhadap rotational ambiguity dibanding inisialisasi PCA biasa.

    Parameters
    ----------
    D  : (n_samples x n_variables) matriks data (baris = spektrum campuran)
    n_components : jumlah variabel murni yang dicari
    f  : fraksi offset noise (0-1) relatif terhadap mean kolom maksimum;
         mencegah kolom dengan mean mendekati nol mendominasi purity
         spectrum akibat pembagian oleh angka kecil.

    Returns
    -------
    pure_idx : list index kolom (variabel) yang dipilih sebagai "purest"
    purity   : purity spectrum (n_variables,) - untuk keperluan diagnostik
    """
    D = np.asarray(D, dtype=float)
    n_samples, n_vars = D.shape
    n_components = min(n_components, n_vars)

    mu = D.mean(axis=0)
    sigma = D.std(axis=0, ddof=0)
    alpha = f * mu.max() if mu.max() > 0 else 1e-10

    purity = sigma / (mu + alpha)

    length = np.sqrt(mu ** 2 + alpha ** 2)
    length[length == 0] = 1e-12
    Dw = D / length
    COV = (Dw.T @ Dw) / n_samples

    pure_idx = [int(np.argmax(purity))]
    for _ in range(1, n_components):
        best_j, best_det = -1, -np.inf
        for j in range(n_vars):
            if j in pure_idx:
                continue
            idx = pure_idx + [j]
            M = COV[np.ix_(idx, idx)]
            det = np.linalg.det(M)
            if det > best_det:
                best_det = det
                best_j = j
        if best_j == -1:
            break
        pure_idx.append(best_j)

    return pure_idx, purity


def _init_mcr(D, n_components, init_method="pca", S_init=None, s_nonneg=True):
    """
    Inisialisasi C dan S untuk MCR-ALS.

    init_method:
      'simplisma' → Purest-variable selection (Windig & Guilment 1991)
                  C awal diambil langsung dari kolom data nyata (bukan
                  PCA), non-negatif dari awal. Pendekatan ini paling
                  mendekati bagaimana software MCR-ALS komersial
                  (termasuk Unscrambler) melakukan inisialisasi default,
                  dan paling efektif menekan rotational ambiguity tanpa
                  perlu library eksternal. Direkomendasikan sebagai default.
      'pca'     → PCA scores sebagai C awal (shift-to-positive)
                  Cepat, umum digunakan, tapi menghasilkan komponen
                  dengan nilai negatif yang dikoreksi
      'nmf'     → NMF-NNDSVD sebagai C dan S awal
                  Non-negatif dari awal tanpa koreksi,
                  lebih sesuai untuk data spektroskopi
      'library' → S_init dari spektra eksternal (library)
                  Paling akurat jika spektra murni tersedia

    Returns: C_init (m x k), S_init_out (k x n) or None
    """
    from sklearn.decomposition import NMF
    D = np.array(D, dtype=float)
    m, n = D.shape

    # ── Pilihan 1: Library spectral guess ────────────────────
    if S_init is not None:
        S0 = np.array(S_init, dtype=float)
        if S0.shape[0] != n_components:
            S0 = S0[:n_components] if S0.shape[0] > n_components else S0
        if s_nonneg:
            S0 = np.maximum(S0, 0)
        norms = np.linalg.norm(S0, axis=1, keepdims=True)
        norms[norms == 0] = 1
        S0 = S0 / norms
        C0 = np.linalg.lstsq(S0.T, D.T, rcond=None)[0].T
        C0 = np.maximum(C0, 1e-10)
        return C0, S0

    # ── Pilihan 2: SIMPLISMA (purest-variable, default direkomendasikan) ──
    if init_method == "simplisma":
        pure_idx, _purity = simplisma(D, n_components)
        C0 = D[:, pure_idx].copy()
        C0 = np.maximum(C0, 1e-10)
        return C0, None

    # ── Pilihan 3: NMF-NNDSVD ────────────────────────────────
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
                normalize_S=False, init_method="pca", S_init=None,
                fixed_spectra=None, fixed_conc_zero=None,
                unimodal_C=False, smooth_S=False,
                smooth_window=15, smooth_poly=3,
                s_nonneg=True):
    """
    MCR-ALS dengan constraint lengkap dan inisialisasi yang benar.

    Parameters
    ----------
    D            : (n_samples x n_wavelengths)
    n_components : jumlah komponen
    max_iter     : iterasi maksimum
    tol          : toleransi konvergensi (perubahan LOF)
    closure      : closure constraint pada C (jumlah fraksi = 1) [DIPERBAIKI]
    unimodal     : unimodality constraint pada S (spektra). PERINGATAN:
                   spektra IR/UV-Vis murni pada umumnya memiliki BANYAK
                   puncak (multi-band), sehingga memaksa unimodalitas di
                   sini akan menghancurkan struktur multi-puncak yang nyata.
                   Constraint ini hanya masuk akal jika S memang mewakili
                   profil dengan satu puncak tunggal secara fisik. Untuk
                   data spektroskopi FTIR/UV-Vis biasa, gunakan
                   unimodal_C (di bawah), BUKAN parameter ini.
    s_nonneg     : non-negativity constraint pada S (spektra). [BARU]
                   Default True (perilaku lama, benar untuk domain data
                   mentah/absorbansi). WAJIB di-set False untuk domain
                   spektra turunan (1st/2nd derivative Savitzky-Golay),
                   karena spektrum turunan secara matematis harus memiliki
                   nilai negatif (lereng turun kurva) — memaksa
                   non-negativity di domain ini akan meng-clip dan merusak
                   bentuk turunan yang valid, memaksa ALS ke solusi yang
                   tidak konsisten dengan model data. Non-negativity pada
                   C TIDAK dipengaruhi parameter ini — C tetap selalu
                   non-negatif karena konsentrasi fisik tidak pernah
                   negatif di domain apapun.
    normalize_S  : normalisasi S per iterasi (unit vector) [BARU]
    S_init       : spektra awal dari library eksternal (k x n) [BARU]
    fixed_spectra: dict {komponen_index: spektra_array} — selectivity constraint [BARU]
                   Spektra komponen yang dikunci tidak akan berubah selama iterasi.
                   Contoh: {0: spektra_air} → komponen 0 (air) dikunci
                   Default None = semua komponen bebas dioptimasi (perilaku normal)
    fixed_conc_zero: list of tuple (sample_idx, comp_idx) — windowing /
                   equality constraint pada matriks konsentrasi (C). [BARU]
                   Nilai C di baris sample_idx, kolom comp_idx dipaksa nol
                   pada setiap iterasi. Berguna jika diketahui secara independen
                   bahwa suatu sampel tidak mengandung komponen tertentu
                   (mis. titik blank/background tanpa deposit analit).
                   Contoh: [(3, 0), (3, 1)] → sampel baris ke-3 (0-indexed)
                   dipaksa konsentrasi komponen 0 dan 1 = 0.
                   Indeks di luar batas matriks diabaikan secara aman (tidak crash).
                   Default None = tidak ada constraint tambahan (perilaku normal).
    unimodal_C   : unimodality constraint pada PROFIL KONSENTRASI (kolom C),
                   bukan pada spektra. Ini adalah penerapan unimodality yang
                   benar secara kimia untuk seri sampel yang berurutan
                   (mis. seri dilusi/rasio konsentrasi meningkat monoton),
                   dan tidak merusak struktur multi-puncak spektra IR/UV-Vis.
    smooth_S     : terapkan smoothing Savitzky-Golay ringan pada S setiap
                   iterasi (dengan non-negativity dipertahankan) untuk
                   menekan noise pada komponen minor yang under-determined —
                   membantu ketika satu komponen menyerap noise residual
                   dan tampak jauh lebih "kasar" dibanding software lain.
    smooth_window, smooth_poly : parameter Savitzky-Golay untuk smooth_S.

    Returns
    -------
    C, S, lof_history, r2, converged, diagnostics
    diagnostics : dict berisi residual per sampel, per wavenumber, NNV, dll,
                  termasuk "constraints_used" untuk keterlacakan (audit trail).
    """
    D = np.array(D, dtype=float)
    m, n = D.shape

    # ── Inisialisasi ──────────────────────────────────────────
    C, S0 = _init_mcr(D, n_components, init_method, S_init, s_nonneg=s_nonneg)
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

        # Non-negativity pada S — HANYA untuk domain data mentah.
        # Untuk domain spektra turunan (s_nonneg=False), nilai negatif
        # dipertahankan karena itu bagian valid dari bentuk turunan.
        if s_nonneg:
            S = np.maximum(S, 0)

        # Unimodality pada S [BARU — benar-benar diimplementasikan]
        # PERINGATAN: hanya masuk akal untuk profil S yang secara fisik
        # memang unimodal (bukan spektra IR/UV-Vis multi-puncak biasa).
        if unimodal:
            for i in range(S.shape[0]):
                S[i] = _apply_unimodality(S[i])

        # Smoothing ringan pada S per iterasi [BARU]
        # Menekan noise pada komponen yang under-determined tanpa
        # mengubah bentuk puncak secara signifikan (window kecil,
        # non-negativity dipertahankan setelahnya).
        if smooth_S:
            win  = max(5, smooth_window if smooth_window % 2 == 1 else smooth_window + 1)
            win  = min(win, S.shape[1] - (1 - S.shape[1] % 2))
            poly = min(smooth_poly, win - 2)
            if win >= 5 and poly >= 1:
                for i in range(S.shape[0]):
                    try:
                        S[i] = savgol_filter(S[i], win, poly)
                    except Exception:
                        pass
                if s_nonneg:
                    S = np.maximum(S, 0)

        # Normalisasi S ke unit vector + kompensasi ke C [BARU]
        if normalize_S:
            norms = np.linalg.norm(S, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            S = S / norms
            # Kompensasi skala ke C agar D = C×S tetap sama
            C = C * norms.T

        # Selectivity constraint — kunci spektra komponen yang diketahui [BARU]
        # Setelah semua constraint lain diterapkan, kembalikan spektra yang dikunci
        # ke nilai referensi. Ini memastikan komponen diketahui tidak berubah
        # selama iterasi (equality/selectivity constraint).
        if fixed_spectra is not None:
            for comp_idx, sp_fixed in fixed_spectra.items():
                if 0 <= comp_idx < S.shape[0]:
                    S[comp_idx] = np.array(sp_fixed, dtype=float)

        # Step 2: Update C = least squares dari S
        C = np.linalg.lstsq(S.T, D.T, rcond=None)[0].T

        # Non-negativity pada C
        C = np.maximum(C, 0)

        # Unimodality pada profil konsentrasi (per kolom komponen) [BARU]
        # Ini adalah penerapan unimodality yang benar secara kimia untuk
        # seri sampel berurutan (mis. seri rasio/dilusi monoton), berbeda
        # dari parameter 'unimodal' di atas yang bekerja pada spektra.
        if unimodal_C:
            for j in range(C.shape[1]):
                C[:, j] = _apply_unimodality(C[:, j])

        # Windowing / equality constraint pada konsentrasi [BARU]
        # Paksa C[sample_idx, comp_idx] = 0 untuk pasangan yang diketahui
        # tidak mengandung komponen tersebut (mis. titik blank/background).
        # Diterapkan sebelum closure agar renormalisasi closure (jika aktif)
        # menggunakan nilai C yang sudah benar.
        if fixed_conc_zero is not None:
            for (samp_idx, comp_idx) in fixed_conc_zero:
                if 0 <= samp_idx < C.shape[0] and 0 <= comp_idx < C.shape[1]:
                    C[samp_idx, comp_idx] = 0.0

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

    # Non-negativity Violation Score per komponen.
    # CATATAN: metrik ini hanya bermakna sebagai "pelanggaran" ketika
    # s_nonneg=True (domain data mentah). Di domain turunan (s_nonneg=
    # False), nilai negatif adalah bagian valid dari bentuk turunan,
    # bukan pelanggaran — lihat diagnostics["nnv_meaningful"].
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
        "nnv_meaningful":   bool(s_nonneg),
        "ev_per_comp":      ev_comp,
        "n_iter":           len(lof_history),
        "converged":        converged,
        "constraints_used": {
            "init_method":   init_method,
            "closure":       bool(closure),
            "unimodal_S":    bool(unimodal),
            "unimodal_C":    bool(unimodal_C),
            "normalize_S":   bool(normalize_S),
            "smooth_S":      bool(smooth_S),
            "s_nonneg":      bool(s_nonneg),
            "selectivity":   fixed_spectra is not None,
            "windowing":     fixed_conc_zero is not None,
            "library_init":  S_init is not None,
        },
    }

    return C, S, lof_history, r2, converged, diagnostics


# ════════════════════════════════════════════════════════════════
# MULTI-K RUN (KONSISTENSI CHECKER)
# ════════════════════════════════════════════════════════════════

def run_mcr_multi_k(D, k_range=(2, 6), max_iter=200, tol=1e-6,
                    closure=False, unimodal=False, normalize_S=False,
                    init_method="pca", fixed_spectra=None, fixed_conc_zero=None,
                    unimodal_C=False, smooth_S=False, s_nonneg=True):
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
            init_method=init_method, fixed_spectra=fixed_spectra,
            fixed_conc_zero=fixed_conc_zero,
            unimodal_C=unimodal_C, smooth_S=smooth_S, s_nonneg=s_nonneg
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
    """Cosine similarity mentah (TIDAK mean-centered).
    CATATAN: sensitif terhadap offset/baseline/envelope yang sama-sama
    dimiliki dua spektrum, sehingga bisa memberi skor tinggi meski pola
    puncak kimianya berbeda. Gunakan pearson_corr() untuk perbandingan
    bentuk puncak murni (lihat dokumentasi di bawah)."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def pearson_corr(a, b):
    """Korelasi Pearson antara dua spektrum (mean-centered cosine).

    Berbeda dari cosine_sim(), di sini rata-rata masing-masing vektor
    dikurangkan terlebih dahulu sebelum dot product. Ini menghilangkan
    kontribusi offset/baseline/envelope bersama yang sering membuat
    cosine similarity mentah "membesar-besarkan" kemiripan dua spektrum
    ATR-FTIR yang sebenarnya berbeda kimia — masalah yang sering muncul
    saat spektrum hasil MCR belum sepenuhnya murni (rotational
    ambiguity) atau saat baseline dua sumber (query vs library) tidak
    identik. Pearson lebih menekankan pada KESESUAIAN POLA PUNCAK
    (posisi & rasio intensitas relatif), sehingga umumnya lebih
    diskriminatif untuk identifikasi spektra dibanding cosine mentah.

    KETERBATASAN Pearson (dan cosine): keduanya TIDAK tahan terhadap
    (1) rasio intensitas relatif antar puncak yang berubah akibat efek
    matriks/path length, (2) pergeseran posisi puncak (peak shift) akibat
    drift kalibrasi instrumen, dan (3) memperlakukan semua titik data
    setara meski sebagian tidak informatif secara kimia. Lihat
    derivative_corr() dan shift_tolerant_corr() sebagai pelengkap yang
    menutupi keterbatasan #1 dan #2.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    return float(np.dot(a_c, b_c) / denom) if denom > 0 else 0.0


def derivative_corr(a, b, order=1, window=11, poly=3):
    """Korelasi Pearson pada spektrum turunan (1st/2nd derivative).

    Menerapkan turunan Savitzky-Golay pada kedua spektrum sebelum
    menghitung Pearson. Turunan menghilangkan baseline/envelope secara
    TOTAL (bukan cuma dikurangi rata-rata seperti pearson_corr biasa)
    dan menonjolkan posisi & bentuk puncak tajam, sehingga jauh lebih
    tahan terhadap perbedaan rasio intensitas relatif antar puncak
    akibat efek matriks/konsentrasi/path length ATR yang tidak seragam.
    Ini adalah pendekatan standar pada identifikasi spektra forensik &
    otentikasi pangan (mis. AOAC/ASTM) untuk kasus seperti ini.

    order : 1 = first derivative (paling umum dipakai), 2 = second
            derivative (lebih menekankan puncak tajam, lebih sensitif
            noise).
    Jika salah satu spektrum terlalu pendek untuk window Savitzky-Golay
    yang diminta, window otomatis diperkecil.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    n = min(len(a), len(b))
    win = min(window, n - (1 - n % 2))
    win = max(5, win if win % 2 == 1 else win - 1)
    p = min(poly, win - 2)
    if win < 5 or p < 1 or n < win:
        # Data terlalu pendek untuk differensiasi Savitzky-Golay yang
        # andal — fallback ke pearson biasa daripada memberi angka
        # yang tidak bermakna.
        return pearson_corr(a, b)
    try:
        da = savgol_filter(a, win, p, deriv=order)
        db = savgol_filter(b, win, p, deriv=order)
    except Exception:
        return pearson_corr(a, b)
    return pearson_corr(da, db)


def shift_tolerant_corr(a, b, max_shift=5):
    """Pearson correlation terbaik di antara beberapa pergeseran kecil.

    Mencari nilai pearson_corr(a, b digeser sejauh s) tertinggi untuk
    s dalam rentang [-max_shift, +max_shift] (dalam satuan titik grid,
    BUKAN cm⁻¹ — sesuaikan max_shift dengan grid_interval yang dipakai
    kalau ingin toleransi dalam cm⁻¹ tertentu). Ini mengakomodasi
    pergeseran posisi puncak kecil akibat drift kalibrasi instrumen
    atau efek ikatan hidrogen/suhu, yang membuat pearson_corr() biasa
    (dan cosine_sim()) menghukum dua spektrum yang sebenarnya senyawa
    sama hanya karena puncaknya bergeser 1-2 titik grid.

    Mengembalikan (skor_terbaik, pergeseran_optimal).
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    best_score = -1.0
    best_shift = 0
    for s in range(-max_shift, max_shift + 1):
        if s > 0:
            a_s, b_s = a[s:], b[:-s] if s > 0 else b
        elif s < 0:
            a_s, b_s = a[:s], b[-s:]
        else:
            a_s, b_s = a, b
        if len(a_s) < 5:
            continue
        score = pearson_corr(a_s, b_s)
        if score > best_score:
            best_score = score
            best_shift = s
    return float(best_score), best_shift


def find_significant_extrema(spectrum, wavenumber, order=1, prominence_frac=0.1):
    """
    Deteksi posisi ekstrem signifikan (band center) pada spektrum turunan,
    untuk dipakai sebagai "sidik jari" posisi pita yang jauh lebih tahan
    noise/skala dibanding perbandingan bentuk kurva penuh.

    Untuk 1st derivative: titik zero-crossing (naik→turun) menandai posisi
    puncak asli. Untuk 2nd derivative (dan orde genap lainnya): lembah
    (minimum lokal, karena puncak asli berubah jadi lembah tajam) adalah
    penanda posisi pita yang dipakai.

    Parameters
    ----------
    spectrum   : array turunan (1D)
    wavenumber : array wavenumber yang berpadanan (1D)
    order      : orde turunan (1 atau 2) — menentukan kriteria ekstrem
    prominence_frac : ambang prominence relatif terhadap rentang
                 amplitudo spektrum (0-1). Menyaring noise minor supaya
                 tidak dianggap "band" — naikkan nilai ini jika data
                 masih terlalu berisik meski sudah di-smoothing.

    Returns
    -------
    positions : list posisi wavenumber dari ekstrem signifikan yang
                terdeteksi (band centers), diurutkan menaik.
    """
    from scipy.signal import find_peaks

    spectrum = np.asarray(spectrum, dtype=float)
    wavenumber = np.asarray(wavenumber, dtype=float)
    amp_range = spectrum.max() - spectrum.min()
    if amp_range <= 0:
        return []
    prominence = prominence_frac * amp_range

    if order % 2 == 1:
        # Order ganjil (1st, 3rd, ...): posisi pita = zero-crossing
        # dari + ke - (turunan berubah tanda melewati puncak asli).
        # Amplitudo pita didekati dengan kecuraman lokal (local_slope) di
        # titik zero-crossing tsb — makin tajam transisinya, makin
        # "kuat" pita aslinya secara relatif terhadap pita lain di
        # spektrum yang sama.
        sign = np.sign(spectrum)
        crossings = np.where((sign[:-1] > 0) & (sign[1:] <= 0))[0]
        # Saring crossing yang terlalu datar (bukan transisi tajam)
        peaks = []
        for idx in crossings:
            local_slope = abs(spectrum[max(0, idx - 2)] - spectrum[min(len(spectrum) - 1, idx + 2)])
            if local_slope >= prominence:
                # interpolasi linear posisi zero-crossing yang presisi
                x0, x1 = wavenumber[idx], wavenumber[idx + 1]
                y0, y1 = spectrum[idx], spectrum[idx + 1]
                frac = y0 / (y0 - y1) if (y0 - y1) != 0 else 0.5
                pos = float(x0 + frac * (x1 - x0))
                peaks.append((pos, float(local_slope)))
        return sorted(peaks, key=lambda p: p[0])
    else:
        # Order genap (2nd, 4th, ...): posisi pita = lembah (minimum
        # lokal) — puncak asli menjadi lembah tajam pada 2nd derivative.
        # Amplitudo pita = kedalaman lembah tsb.
        valley_idx, _ = find_peaks(-spectrum, prominence=prominence)
        peaks = [(float(wavenumber[i]), float(abs(spectrum[i]))) for i in valley_idx]
        return sorted(peaks, key=lambda p: p[0])


def peak_position_match(query_spec, query_wn, ref_spec, ref_wn,
                        order=1, tolerance_cm=6.0, prominence_frac=0.1,
                        intensity_weight=1.0):
    """
    Cocokkan posisi PITA dan intensitas relatifnya antara spektrum
    turunan query (mis. komponen hasil MCR) dan spektrum turunan
    referensi murni.

    Pemasangan pita menggunakan penugasan optimal global (algoritma
    Hungarian / linear_sum_assignment) berdasarkan matriks biaya yang
    menggabungkan (1) jarak posisi wavenumber dan (2) selisih
    intensitas relatif — BUKAN pencocokan greedy per-urutan seperti
    versi sebelumnya, yang bisa terjebak pada pasangan lokal yang
    sub-optimal ketika beberapa pita saling berdekatan.

    Catatan desain: intensitas dinormalisasi terhadap pita TERKUAT di
    masing-masing sisi (query vs referensi) sebelum dibandingkan.
    Ini disengaja — tujuannya menilai apakah POLA RELATIF tinggi-rendah
    pita sudah konsisten (mis. pita A dua kali lebih kuat dari pita B
    di kedua spektrum), bukan menuntut kesamaan skala turunan absolut,
    yang secara fisik tidak bermakna (skala turunan bergantung pada
    parameter Savitzky-Golay dan tidak perlu identik antar entri).

    Kenapa bukan algoritma genetika (GA): ini adalah assignment
    problem berskala kecil (biasanya <20 pita) dengan struktur berurut
    (posisi query & referensi sama-sama monoton terhadap wavenumber).
    Hungarian algorithm memberi solusi OPTIMAL GLOBAL secara eksak,
    deterministik, dan jauh lebih murah secara komputasi — GA hanya
    relevan untuk ruang pencarian besar/non-konveks yang tidak dimiliki
    problem ini.

    Returns
    -------
    dict berisi:
      n_query, n_ref        : jumlah pita signifikan terdeteksi tiap sisi
      n_matched              : jumlah pasangan pita yang posisinya cocok
                              dalam toleransi
      match_fraction         : n_matched / max(n_query, n_ref) — proporsi
                              pita yang berhasil dicocokkan (0-1)
      mean_abs_shift         : rata-rata pergeseran (cm⁻¹) dari pasangan
                              yang cocok — makin kecil, makin presisi
      intensity_agreement    : 1 - rata-rata selisih intensitas relatif
                              (0-1) pada pasangan yang cocok — makin
                              tinggi, makin konsisten pola tinggi-rendah
                              pitanya. None jika tidak ada pasangan.
      matched_pairs          : list (posisi_query, posisi_ref, shift,
                              intensitas_relatif_query, intensitas_relatif_ref)
      unmatched_query        : posisi pita query yang tidak ada
                              pasangannya di referensi (indikasi identitas
                              kimia berbeda atau komponen belum murni)
    """
    q_peaks = find_significant_extrema(query_spec, query_wn, order, prominence_frac)
    r_peaks = find_significant_extrema(ref_spec, ref_wn, order, prominence_frac)
    q_pos = [p[0] for p in q_peaks]
    r_pos = [p[0] for p in r_peaks]

    # Intensitas relatif (0-1) terhadap pita terkuat pada sisi masing-masing
    q_max = max((p[1] for p in q_peaks), default=0.0) or 1.0
    r_max = max((p[1] for p in r_peaks), default=0.0) or 1.0
    q_rel = [p[1] / q_max for p in q_peaks]
    r_rel = [p[1] / r_max for p in r_peaks]

    matched_pairs = []
    unmatched_query = list(q_pos)

    if q_peaks and r_peaks:
        nq, nr = len(q_peaks), len(r_peaks)
        # Biaya = jarak posisi ternormalisasi (terhadap tolerance_cm) +
        # selisih intensitas relatif berbobot. Pasangan yang jaraknya
        # jauh melampaui toleransi diberi biaya sangat tinggi supaya
        # Hungarian tidak "terpaksa" memasangkan pita yang jelas beda.
        cost = np.zeros((nq, nr))
        for i in range(nq):
            for j in range(nr):
                d = abs(q_pos[i] - r_pos[j])
                pos_cost = d / tolerance_cm
                if d > tolerance_cm:
                    pos_cost += 100.0  # penalti besar, efektif melarang pasangan ini
                int_cost = intensity_weight * abs(q_rel[i] - r_rel[j])
                cost[i, j] = pos_cost + int_cost

        row_ind, col_ind = linear_sum_assignment(cost)
        used_query = set()
        for i, j in zip(row_ind, col_ind):
            d = abs(q_pos[i] - r_pos[j])
            if d <= tolerance_cm:
                matched_pairs.append((q_pos[i], r_pos[j], q_pos[i] - r_pos[j],
                                       q_rel[i], r_rel[j]))
                used_query.add(q_pos[i])
        unmatched_query = [p for p in q_pos if p not in used_query]

    n_matched = len(matched_pairs)
    denom = max(len(q_pos), len(r_pos), 1)
    mean_abs_shift = (float(np.mean([abs(p[2]) for p in matched_pairs]))
                       if matched_pairs else None)
    intensity_agreement = (float(1.0 - np.mean([abs(p[3] - p[4]) for p in matched_pairs]))
                            if matched_pairs else None)

    return {
        "n_query": len(q_pos),
        "n_ref": len(r_pos),
        "n_matched": n_matched,
        "match_fraction": round(n_matched / denom, 4),
        "mean_abs_shift": (round(mean_abs_shift, 3) if mean_abs_shift is not None else None),
        "intensity_agreement": (round(intensity_agreement, 4) if intensity_agreement is not None else None),
        "matched_pairs": matched_pairs,
        "unmatched_query": unmatched_query,
    }


def hqi_score(a, b):
    """HQI klasik = cosine² × 100. PERINGATAN: metrik ini SECARA
    MATEMATIS diturunkan langsung dari cosine_sim() (bukan pengukuran
    independen). Jangan diperlakukan sebagai "validasi kedua" yang
    berbeda dari cosine — nilainya akan selalu bergerak searah dan
    tidak pernah benar-benar "berkonflik" dengan cosine. Untuk validasi
    silang yang sungguh independen, bandingkan cosine_sim() dengan
    pearson_corr()."""
    return round(cosine_sim(a, b) ** 2 * 100, 3)


def compare_two_spectra(wn_a, sp_a, wn_b, sp_b, window_mode="full",
                        wmin=None, wmax=None, grid_interval="auto",
                        interp_method="cubic", deriv_order=1,
                        shift_tolerance=5):
    """
    Bandingkan sepasang spektrum generik apa saja - dipakai fitur
    "Perbandingan Manual (Admin)" untuk kombinasi bebas: komponen MCR,
    spektra acuan library, atau spektra eksternal input manual. Ini
    BUKAN pencarian satu query vs banyak entri library (itu tugas
    batch_match()) - hanya menghitung satu pasang saja.

    Alur alignment (windowing -> common grid -> resampling) dan metrik
    yang dihasilkan identik dengan batch_match() (lihat docstring-nya
    untuk arti tiap metrik: cosine, pearson, hqi, derivative, shift,
    composite), supaya angka yang tampil di kedua fitur bisa saling
    dibandingkan/dipercaya secara konsisten.

    Returns
    -------
    dict berisi semua metrik + info grid alignment, atau None jika
    kedua spektrum (setelah windowing) tidak overlap sama sekali -
    pemanggil wajib menangani kasus None ini sendiri.
    """
    wn_a2, sp_a2 = apply_window(wn_a, sp_a, window_mode, wmin, wmax)
    wn_b2, sp_b2 = apply_window(wn_b, sp_b, window_mode, wmin, wmax)
    if len(wn_a2) < 5 or len(wn_b2) < 5:
        return None
    common_grid, grid_info = build_common_grid(wn_a2, wn_b2, grid_interval)
    if common_grid is None or len(common_grid) < 5:
        return None
    sp_a_c = resample_to_grid(wn_a2, sp_a2, common_grid, interp_method)
    sp_b_c = resample_to_grid(wn_b2, sp_b2, common_grid, interp_method)

    pear = round(pearson_corr(sp_a_c, sp_b_c), 4)
    deriv = round(derivative_corr(sp_a_c, sp_b_c, order=deriv_order), 4)
    shift_score, shift_lag = shift_tolerant_corr(sp_a_c, sp_b_c, max_shift=shift_tolerance)
    shift_score = round(shift_score, 4)

    return {
        "cosine":     round(cosine_sim(sp_a_c, sp_b_c), 4),
        "pearson":    pear,
        "hqi":        round(hqi_score(sp_a_c, sp_b_c), 2),
        "derivative": deriv,
        "shift":      shift_score,
        "shift_lag":  shift_lag,
        "composite":  round((pear + deriv + shift_score) / 3, 4),
        "overlap_min":     grid_info["overlap_min"],
        "overlap_max":     grid_info["overlap_max"],
        "overlap_width":   grid_info["overlap_width"],
        "grid_interval":   grid_info["grid_interval"],
        "n_common_points": grid_info["n_common_points"],
        "grid_warning":    grid_info.get("warning"),
    }


def batch_match(query_spec, query_wn, library_entries,
                window_mode, wmin, wmax, top_n=10,
                grid_interval="auto", interp_method="cubic",
                rank_by="composite", ambiguous_margin=0.03,
                deriv_order=1, shift_tolerance=5):
    """
    Parameters tambahan
    --------------------
    rank_by : "composite" (default, direkomendasikan), "pearson",
              "cosine", "derivative", atau "shift".
              - "pearson"   : lihat pearson_corr() — tahan baseline/
                              envelope bersama, TIDAK tahan peak-shift
                              atau perubahan rasio intensitas antar puncak.
              - "derivative": lihat derivative_corr() — tahan perubahan
                              rasio intensitas antar puncak (efek
                              matriks/path length), TIDAK tahan peak-shift.
              - "shift"     : lihat shift_tolerant_corr() — tahan
                              pergeseran posisi puncak kecil (drift
                              kalibrasi), dihitung di atas pearson.
              - "composite" : rata-rata pearson + derivative + shift,
                              paling robust karena tiga kelemahan di
                              atas saling menutupi; kandidat yang benar
                              secara kimia umumnya tinggi di ketiganya,
                              sedangkan kandidat yang cuma mirip
                              baseline/envelope biasanya jatuh di
                              minimal satu metrik.
    ambiguous_margin : jika selisih skor (rank_by) antara kandidat #1
              dan #2 lebih kecil dari nilai ini, kandidat #1 ditandai
              "ambiguous": True — artinya sistem TIDAK cukup yakin
              untuk membedakan kandidat #1 dari #2, walau skor mutlaknya
              tinggi. Ini mencegah kasus seperti dua kandidat dengan
              cosine 0.8391 vs 0.8389 dilaporkan seolah #1 pasti benar.
    deriv_order : orde turunan untuk derivative_corr() (1 atau 2).
    shift_tolerance : rentang pergeseran (titik grid) untuk
              shift_tolerant_corr().
    """
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
        cos  = round(cosine_sim(sp_q_c, sp_r_c), 4)
        pear = round(pearson_corr(sp_q_c, sp_r_c), 4)
        hqi  = round(hqi_score(sp_q_c, sp_r_c), 2)
        deriv = round(derivative_corr(sp_q_c, sp_r_c, order=deriv_order), 4)
        shift_score, shift_lag = shift_tolerant_corr(sp_q_c, sp_r_c, max_shift=shift_tolerance)
        shift_score = round(shift_score, 4)
        composite = round((pear + deriv + shift_score) / 3, 4)
        results.append({
            "id": entry["id"], "name": entry["name"],
            "category": entry["category"],
            "cosine": cos, "pearson": pear, "hqi": hqi,
            "derivative": deriv,
            "shift": shift_score, "shift_lag": shift_lag,
            "composite": composite,
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

    rank_key = rank_by if rank_by in ("pearson", "cosine", "derivative", "shift", "composite") else "composite"
    results.sort(key=lambda x: x[rank_key], reverse=True)

    # Tandai kandidat #1 sebagai "ambiguous" jika terlalu dekat dengan
    # kandidat #2 pada metrik ranking — mencegah laporan identifikasi
    # yang terkesan pasti padahal skornya nyaris seri (mis. 0.8391 vs
    # 0.8389). Semua entri diberi field ini agar konsisten.
    for r in results:
        r["ambiguous"] = False
    if len(results) >= 2:
        margin = results[0][rank_key] - results[1][rank_key]
        results[0]["margin_to_next"] = round(margin, 4)
        if margin < ambiguous_margin:
            results[0]["ambiguous"] = True

    return results[:top_n]


def batch_match_derivative(query_deriv_spec, query_wn, deriv_library_entries,
                           deriv_order, window_mode="fingerprint",
                           wmin=None, wmax=None, top_n=10,
                           grid_interval="auto", interp_method="cubic",
                           tolerance_cm=6.0, prominence_frac=0.1,
                           shift_tolerance_cm=10.0, intensity_weight=1.0,
                           ambiguous_margin=0.03):
    """
    Matching khusus domain turunan — TIDAK boleh dipakai untuk
    membandingkan spektrum turunan dengan library domain mentah
    (lihat diskusi: dua domain berbeda, hasil tidak bermakna).

    Perbedaan dari batch_match() biasa:
      1. Hanya membandingkan terhadap entri library yang deriv_order-nya
         SAMA PERSIS dengan query (entry["deriv_order"] must match) —
         entri lain di-skip otomatis, bukan cuma diberi skor rendah.
      2. Default window_mode = "fingerprint" (bukan "full") karena di
         domain turunan, region non-pita bernilai ~0 di semua komponen
         dan akan menggembungkan cosine similarity secara artifisial
         jika dihitung full-range.
      3. Menambahkan peak_position_match() sebagai metrik utama —
         mencocokkan POSISI dan INTENSITAS RELATIF band (lembah untuk
         2nd derivative, zero-crossing untuk 1st derivative) via
         penugasan optimal (Hungarian algorithm), bukan cuma bentuk
         kurva, karena interpretasi visual bentuk turunan kedua sangat
         sulit dan gampang salah tafsir.

    CATATAN PERBAIKAN (v3.2): versi sebelumnya memanggil
    derivative_corr(sp_q_c, sp_r_c, order=1) di sini — padahal sp_q_c/
    sp_r_c SUDAH berupa spektrum domain turunan, sehingga itu secara
    tidak sengaja menurunkan sekali lagi (turunan-dari-turunan) dan
    parameter deriv_order tidak pernah benar-benar dipakai untuk
    menentukan order-nya. Metrik itu dihapus dari sini; posisi dan
    intensitas pita (peak_position_match) sudah menutupi perannya
    dengan cara yang benar secara domain.

    deriv_library_entries : list of dict, tiap entri wajib punya key
        "deriv_order" selain "id","name","category","wavenumber","spectrum".
    shift_tolerance_cm : toleransi pergeseran puncak untuk
        shift_tolerant_corr(), dalam cm⁻¹ — dikonversi ke jumlah titik
        grid berdasarkan grid_interval aktual tiap pasangan (bukan
        angka titik grid tetap, yang artinya berubah-ubah tergantung
        kerapatan grid).
    intensity_weight : bobot selisih intensitas relatif pita di dalam
        matriks biaya Hungarian pada peak_position_match(). Naikkan
        jika ingin kesesuaian intensitas lebih menentukan dibanding
        kesesuaian posisi murni.

    Returns : list of dict, sudah diurutkan menurun berdasarkan
              "composite" — rata-rata dari 3 KELUARGA metrik yang
              saling independen: (1) kemiripan bentuk kurva (pearson +
              shift-tolerant, dirata-rata jadi satu skor karena
              keduanya sama-sama berbasis korelasi bentuk), (2)
              kesesuaian POSISI pita, (3) kesesuaian INTENSITAS RELATIF
              pita — top_n entri, dengan flag "ambiguous" seperti
              batch_match() biasa.
    """
    if grid_interval != "auto":
        try:
            grid_interval = float(grid_interval)
        except (TypeError, ValueError):
            grid_interval = "auto"

    wn_q, sp_q = apply_window(query_wn, query_deriv_spec, window_mode, wmin, wmax)
    if len(wn_q) < 5:
        return []

    results = []
    for entry in deriv_library_entries:
        if int(entry.get("deriv_order", -1)) != int(deriv_order):
            continue  # domain berbeda — jangan pernah dibandingkan
        wn_r = np.array(entry["wavenumber"], dtype=float)
        sp_r = np.array(entry["spectrum"],   dtype=float)
        wn_r2, sp_r2 = apply_window(wn_r, sp_r, window_mode, wmin, wmax)
        if len(wn_r2) < 5:
            continue
        common_grid, grid_info = build_common_grid(wn_q, wn_r2, grid_interval)
        if common_grid is None or len(common_grid) < 5:
            continue
        sp_q_c = resample_to_grid(wn_q,  sp_q,  common_grid, interp_method)
        sp_r_c = resample_to_grid(wn_r2, sp_r2, common_grid, interp_method)

        pear = round(pearson_corr(sp_q_c, sp_r_c), 4)
        # shift_tolerance_cm -> jumlah titik grid, berdasarkan grid_interval
        # AKTUAL pasangan ini (bukan angka titik tetap)
        shift_pts = max(1, round(shift_tolerance_cm / grid_info["grid_interval"]))
        shift_score, shift_lag = shift_tolerant_corr(sp_q_c, sp_r_c, max_shift=shift_pts)
        shift_score = round(shift_score, 4)

        peak_info = peak_position_match(
            sp_q_c, common_grid, sp_r_c, common_grid,
            order=deriv_order, tolerance_cm=tolerance_cm,
            prominence_frac=prominence_frac, intensity_weight=intensity_weight
        )

        shape_score = round((pear + shift_score) / 2, 4)
        position_score = peak_info["match_fraction"]
        # intensity_agreement bisa None jika tidak ada pasangan pita sama
        # sekali (identitas kimia jelas berbeda) — dalam kasus itu jangan
        # diam-diam diabaikan dari rata-rata, tapi diberi nilai 0 supaya
        # composite tetap mencerminkan kegagalan pencocokan.
        intensity_score = (peak_info["intensity_agreement"]
                            if peak_info["intensity_agreement"] is not None else 0.0)

        composite = round((shape_score + position_score + intensity_score) / 3, 4)

        results.append({
            "id": entry["id"], "name": entry["name"],
            "category": entry.get("category", ""),
            "deriv_order": int(entry.get("deriv_order", deriv_order)),
            "pearson": pear,
            "shift": shift_score, "shift_lag": shift_lag,
            "shape_score": shape_score,
            "peak_match_fraction": peak_info["match_fraction"],
            "peak_mean_abs_shift": peak_info["mean_abs_shift"],
            "peak_intensity_agreement": peak_info["intensity_agreement"],
            "n_peaks_query": peak_info["n_query"],
            "n_peaks_ref": peak_info["n_ref"],
            "composite": composite,
            "overlap_width":   grid_info["overlap_width"],
            "grid_warning":    grid_info.get("warning"),
        })

    results.sort(key=lambda x: x["composite"], reverse=True)
    for r in results:
        r["ambiguous"] = False
    if len(results) >= 2:
        margin = results[0]["composite"] - results[1]["composite"]
        results[0]["margin_to_next"] = round(margin, 4)
        if margin < ambiguous_margin:
            results[0]["ambiguous"] = True

    return results[:top_n]


def consensus_label(cos, hqi, pearson=None, thresh_cos=0.95, thresh_hqi=90.25,
                     thresh_pearson=0.90, ambiguous=False):
    """
    CATATAN PERBAIKAN: versi lama membandingkan cosine vs HQI seolah dua
    ukuran independen — padahal hqi = cosine² × 100 (turunan langsung
    dari cosine), sehingga "konflik" yang dilaporkan sebelumnya nyaris
    tidak pernah mencerminkan konflik nyata. Versi ini memakai
    pearson_corr() (mean-centered, lihat dokumentasinya) sebagai
    cross-check yang SUNGGUH independen dari cosine: pearson bisa turun
    jauh dari cosine ketika kemiripan cosine sebagian besar berasal dari
    baseline/envelope yang kebetulan sama, bukan pola puncak kimia yang
    sama. Jika pearson tidak diberikan, fallback ke perilaku lama
    (dengan HQI) untuk kompatibilitas.
    """
    if ambiguous:
        return ("ambiguous",
                "🟠 Ambigu — skor nyaris sama dengan kandidat lain / "
                "Ambiguous — score nearly tied with next candidate")

    if pearson is not None:
        cos_strong = cos >= thresh_cos
        pear_strong = pearson >= thresh_pearson
        cos_med  = cos >= (thresh_cos - 0.05)
        pear_med = pearson >= (thresh_pearson - 0.10)
        if cos_strong and pear_strong:
            return "strong", "✅ Match kuat / Strong match"
        if cos_strong != pear_strong:
            # cosine tinggi tapi pearson jauh lebih rendah (atau
            # sebaliknya) → indikasi kemiripan cosine didorong oleh
            # baseline/envelope bersama, bukan pola puncak yang sama.
            return "conflict", ("⚠️ Cosine & Pearson berbeda jauh — cek baseline/pola puncak / "
                                 "Cosine & Pearson diverge — check baseline/peak pattern")
        if cos_med and pear_med:
            return "medium", "🟡 Match sedang / Medium match"
        return "weak", "❌ Tidak match / No match"

    # ── Fallback lama (tanpa pearson) ──────────────────────────
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
