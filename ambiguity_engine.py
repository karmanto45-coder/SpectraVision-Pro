"""
ambiguity_engine.py
====================
Modul kuantifikasi ambiguitas rotasi (rotational ambiguity) untuk hasil
MCR-ALS SpectraVision Pro — pendekatan MCR-BANDS.

Prinsip dasar: untuk matriks transformasi T (k x k) apa pun, C' = C @ T dan
S' = inv(T) @ S menghasilkan C' @ S' = C @ S — identik dengan solusi asli.
Jadi pencarian ambiguitas BUKAN mencocokkan ulang data D, melainkan mencari
rentang T yang membuat C' dan S' tetap memenuhi constraint yang sama
(non-negativity, closure) dengan solusi MCR-ALS aslinya.

Dua tingkat metode:
  - k <= K_EXACT_LIMIT : pencarian EKSAK via optimasi terkendala (SLSQP),
    memaksimalkan/meminimalkan luas area profil konsentrasi tiap komponen.
  - k >  K_EXACT_LIMIT : estimasi via sampling terarah-gradien (SLSQP
    dengan arah objektif acak) — dilabeli jelas sebagai perkiraan, bukan
    batas eksak, karena hanya menjelajahi sejumlah arah terbatas.

Penanganan closure: closure (jumlah tiap baris C = 1) dijaga sebagai
CONSTRAINT STRUKTURAL langsung pada T, yaitu setiap baris T harus berjumlah
1 (T @ 1_k = 1_k). Ini menjamin C' = C@T otomatis tetap closure-consistent
tanpa perlu renormalisasi pasca-hoc yang bisa merusak closure dari solusi
nominal itu sendiri (lihat catatan di _normalize_rows).

Referensi konsep: Tauler (1995) Chemom. Intell. Lab. Syst.; metode
MCR-BANDS (Tauler & kawan-kawan) untuk estimasi rentang solusi feasible
akibat ambiguitas rotasi. RMSERA mengikuti formalisme Chiappini, Alcaraz,
et al., Anal. Chem. 2020, 92(10), 7255-7263 (δRA/√12 untuk batas bawah,
δRA/√3 untuk batas atas) — DIADAPTASI ke skala konsentrasi relatif MCR-ALS
(bukan satuan absolut hasil kalibrasi second-order eksternal), karena
tidak ada nilai referensi/kalibrasi silang pada alur kerja SpectraVision
Pro. δRA di sini didekati sebagai rata-rata sebaran konsentrasi per
sampel (selisih area maks-min dibagi jumlah sampel), BUKAN dari satu
sampel uji tunggal seperti pada definisi kalibrasi second-order asli —
adaptasi ini WAJIB dinyatakan secara eksplisit di metodologi jika dipakai
untuk publikasi.
"""

# Ambang klasifikasi (dapat diubah pemanggil; nilai default adalah
# kriteria INTERPRETATIF praktis, bukan standar baku tunggal)
AMBIGUITY_LOW_THRESHOLD_PCT = 30.0     # < ini -> ambiguitas rendah
AMBIGUITY_HIGH_THRESHOLD_PCT = 100.0   # > ini -> ambiguitas tinggi
LOCAL_AMBIGUITY_THRESHOLD_PCT = 15.0   # ambang lebar pita LOKAL (per titik)

import numpy as np
from scipy.optimize import minimize

K_EXACT_LIMIT = 4  # batas jumlah komponen untuk pencarian eksak (SLSQP)


# ---------------------------------------------------------------------------
# Utilitas transformasi & normalisasi
# ---------------------------------------------------------------------------

def _apply_transform(T, C, S):
    """C' = C @ T ,  S' = inv(T) @ S"""
    T_inv = np.linalg.inv(T)
    C_new = C @ T
    S_new = T_inv @ S
    return C_new, S_new


def _normalize_rows(C_new, S_new, closure=False):
    """
    Menghilangkan ambiguitas SKALA trivial (bukan ambiguitas rotasi/bentuk
    yang sebenarnya ingin diukur):

    - closure=False : normalisasi S per baris ke unit-norm + kompensasi
      skala ke C (konvensi sama dengan constraint normalize_S di
      run_mcr_als). Tanpa closure, skala tiap komponen bebas sepenuhnya
      kalau tidak dipatok di sini.
    - closure=True  : TIDAK melakukan rescaling S sama sekali. Closure
      sudah memfiksasi skala melalui constraint T@1=1 pada pencarian T
      (lihat _closure_constraint) — merescale S di sini justru merusak
      closure dari solusi nominal sekalipun (T=identitas).
    """
    if closure:
        return C_new, S_new
    norms = np.linalg.norm(S_new, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    S_norm = S_new / norms
    C_norm = C_new * norms.T
    return C_norm, S_norm


def _area(profile):
    return float(np.sum(np.abs(profile)))


def _closure_constraint(k):
    """
    Constraint kesetaraan: setiap baris T harus berjumlah 1, supaya
    C' = C@T otomatis mempertahankan closure baris C (jumlah = 1) tanpa
    renormalisasi tambahan. Dikembalikan sebagai dict siap-pakai untuk
    scipy.optimize.minimize(constraints=[...]).
    """
    def fun(t_flat):
        T = t_flat.reshape(k, k)
        return T.sum(axis=1) - 1.0
    return {"type": "eq", "fun": fun}


def _feasibility_check(C_norm, S_norm, s_nonneg, tol=1e-5):
    """True jika semua elemen relevan >= -tol (non-negativity feasible)."""
    parts = [C_norm.flatten()]
    if s_nonneg:
        parts.append(S_norm.flatten())
    return float(np.concatenate(parts).min()) >= -tol


# ---------------------------------------------------------------------------
# Klasifikasi keandalan (kuantitatif / semi-kuantitatif / kualitatif)
# ---------------------------------------------------------------------------

def classify_reliability(ambiguity_width_pct, fit_ok=True,
                          low_threshold=AMBIGUITY_LOW_THRESHOLD_PCT,
                          high_threshold=AMBIGUITY_HIGH_THRESHOLD_PCT):
    """
    Menggabungkan status fit (dari scorecard yang SUDAH ADA di mcr_engine,
    mis. overall scorecard 'baik'/tidak, ATAU LOF/R2 per komponen) dengan
    lebar ambiguitas rotasi komponen ini, menjadi satu label keandalan.

    fit_ok : bool — hasil fit dianggap baik (dari scorecard existing).
             Kalau False, klasifikasi langsung "tidak dapat diandalkan"
             terlepas dari ambiguitas, karena masalah fit lebih mendasar.

    Returns
    -------
    dict {"label", "detail"}
    """
    if not fit_ok:
        return {
            "label": "tidak_dapat_diandalkan",
            "detail": "Fit MCR-ALS tidak memenuhi kriteria scorecard — "
                      "ambiguitas rotasi belum relevan dibahas sebelum "
                      "masalah fit ini diatasi.",
        }
    if ambiguity_width_pct < low_threshold:
        return {
            "label": "kuantitatif_kualitatif",
            "detail": f"Ambiguitas rendah (<{low_threshold:.0f}%) dan fit baik — "
                      f"angka konsentrasi & identitas komponen dapat diklaim langsung.",
        }
    if ambiguity_width_pct <= high_threshold:
        return {
            "label": "semi_kuantitatif_kualitatif",
            "detail": f"Ambiguitas sedang ({low_threshold:.0f}-{high_threshold:.0f}%) — "
                      f"pola/tren relatif dapat dipercaya, tapi angka konsentrasi "
                      f"absolut perlu dilaporkan dengan kehati-hatian (bukan angka tunggal presisi).",
        }
    return {
        "label": "kualitatif_saja",
        "detail": f"Ambiguitas tinggi (>{high_threshold:.0f}%) — hanya klaim "
                  f"identifikasi/keberadaan komponen yang disarankan; hindari "
                  f"klaim angka konsentrasi untuk komponen ini.",
    }


# ---------------------------------------------------------------------------
# Profil ambiguitas LOKAL (per titik bilangan gelombang)
# ---------------------------------------------------------------------------

def compute_local_ambiguity_profile(S_nominal_row, S_band_min_row, S_band_max_row,
                                     local_threshold_pct=LOCAL_AMBIGUITY_THRESHOLD_PCT):
    """
    Menghitung lebar pita AMBIGUITAS LOKAL di tiap titik bilangan gelombang,
    dinormalisasi terhadap intensitas MAKSIMUM spektrum komponen ini (bukan
    per-titik) — supaya titik baseline yang nyaris nol tidak menghasilkan
    persentase yang meledak/tidak bermakna.

    KETERBATASAN PENTING: S_band_min/max di sini (untuk metode EKSAK)
    berasal dari HANYA 2 solusi anchor (area-maks dan area-min), diambil
    elementwise min/max keduanya — BUKAN hasil optimasi maks/min di SETIAP
    titik bilangan gelombang secara individual. Profil ini kemungkinan
    UNDER-ESTIMATE ambiguitas lokal sesungguhnya, karena titik ekstrem
    sejati di suatu panjang gelombang tertentu bisa saja dicapai oleh T
    lain yang tidak termasuk di antara 2 solusi anchor tersebut. Untuk
    metode heuristik (k>K_EXACT_LIMIT), profil ini relatif lebih
    representatif karena diambil dari envelope banyak solusi (n_directions),
    tapi tetap bukan pencarian eksak per titik.

    Returns
    -------
    dict {
      "local_width_pct": array (n_lambda,) — lebar pita lokal (%)
      "pct_region_low_ambiguity": float — % titik dengan lebar <= threshold
      "low_ambiguity_mask": array bool (n_lambda,) — titik mana yang "aman"
    }
    """
    scale = np.max(np.abs(S_nominal_row))
    if scale < 1e-12:
        scale = 1.0
    local_width_pct = 100.0 * (S_band_max_row - S_band_min_row) / scale
    low_mask = local_width_pct <= local_threshold_pct
    pct_low = 100.0 * float(np.mean(low_mask))
    return {
        "local_width_pct": local_width_pct,
        "pct_region_low_ambiguity": pct_low,
        "low_ambiguity_mask": low_mask,
    }


# ---------------------------------------------------------------------------
# Metode EKSAK (k <= K_EXACT_LIMIT) — optimasi SLSQP
# ---------------------------------------------------------------------------

def _optimize_component_exact(C, S, comp_idx, direction, s_nonneg, closure,
                               n_restarts=4, maxiter=150, seed=None):
    """
    direction: 'max' atau 'min' — arah optimasi luas area profil
    konsentrasi komponen comp_idx, atas ruang transformasi T (k x k).

    Returns dict {area, C_profile, S_profile, success}
    """
    k = C.shape[1]
    rng = np.random.default_rng(seed)
    sign = -1.0 if direction == "max" else 1.0

    def objective(t_flat):
        T = t_flat.reshape(k, k)
        try:
            C_new, S_new = _apply_transform(T, C, S)
        except np.linalg.LinAlgError:
            return 1e6
        C_norm, _ = _normalize_rows(C_new, S_new, closure)
        return sign * _area(C_norm[:, comp_idx])

    def constraint_nonneg(t_flat):
        T = t_flat.reshape(k, k)
        try:
            C_new, S_new = _apply_transform(T, C, S)
        except np.linalg.LinAlgError:
            return np.array([-1e6])
        C_norm, S_norm = _normalize_rows(C_new, S_new, closure)
        parts = [C_norm.flatten()]
        if s_nonneg:
            parts.append(S_norm.flatten())
        return np.concatenate(parts)  # semua elemen >= 0

    constraints = [{"type": "ineq", "fun": constraint_nonneg}]
    if closure:
        constraints.append(_closure_constraint(k))

    best = None
    T0_base = np.eye(k).flatten()
    for attempt in range(n_restarts):
        t0 = T0_base.copy() if attempt == 0 else T0_base + rng.normal(0, 0.15, size=k * k)
        res = minimize(objective, t0, method="SLSQP", constraints=constraints,
                        options={"maxiter": maxiter, "ftol": 1e-8})
        if res.success:
            candidate_area = sign * res.fun
            better = (best is None
                      or (direction == "max" and candidate_area > best["area"])
                      or (direction == "min" and candidate_area < best["area"]))
            if better:
                T_opt = res.x.reshape(k, k)
                C_new, S_new = _apply_transform(T_opt, C, S)
                C_norm, S_norm = _normalize_rows(C_new, S_new, closure)
                best = {
                    "area": candidate_area,
                    "C_profile": C_norm[:, comp_idx],
                    "S_profile": S_norm[comp_idx, :],
                    "success": True,
                }

    if best is None:
        # Fallback: semua restart gagal konvergen -> pakai solusi nominal
        C_norm, S_norm = _normalize_rows(C, S, closure)
        best = {
            "area": _area(C_norm[:, comp_idx]),
            "C_profile": C_norm[:, comp_idx],
            "S_profile": S_norm[comp_idx, :],
            "success": False,
        }
    return best


# ---------------------------------------------------------------------------
# Metode ESTIMASI (k > K_EXACT_LIMIT) — sampling terarah-gradien
# ---------------------------------------------------------------------------
#
# CATATAN DESAIN: percobaan awal memakai sampling T = I + perturbasi ACAK
# MURNI di sekitar identitas, difilter dengan cek non-negativity. Ternyata
# ini nyaris SELALU gagal menemukan sampel feasible pada data spektrum
# nyata: banyak titik spektrum yang nilainya sangat dekat nol (baseline),
# dan perturbasi acak tanpa arah membuat titik-titik ini punya peluang
# ~50% jatuh negatif secara independen — mensyaratkan semuanya tetap
# non-negatif sekaligus (ratusan/ribuan titik) menjadi hampir mustahil
# lewat tebakan buta. Karena itu, metode ini memakai pendekatan yang sama
# prinsipnya dengan metode eksak (optimasi SLSQP terkendala non-negativity),
# hanya arah objektifnya berupa kombinasi bobot ACAK antar komponen (bukan
# satu-per-satu maks/min per komponen) — supaya jumlah optimasi tidak
# meningkat proporsional dengan k, dan tetap terarah menuju batas wilayah
# feasible, bukan menebak buta.

def _sample_ambiguity_heuristic(C, S, s_nonneg, closure, n_directions=20,
                                 n_restarts_per_direction=1, maxiter=80, seed=None):
    """
    Jalankan optimasi SLSQP berulang dengan arah objektif (bobot antar
    komponen) yang diacak, mengumpulkan solusi feasible di dekat batas
    wilayah ambiguitas. Dari kumpulan solusi ini diambil envelope
    (min/max per titik) untuk C dan S.

    PENTING: ini ESTIMASI, bukan batas eksak — hanya menjelajahi sejumlah
    arah terbatas (n_directions), sehingga berpotensi UNDER-ESTIMATE lebar
    ambiguitas sebenarnya dibanding pencarian eksak penuh per komponen.
    """
    k = C.shape[1]
    rng = np.random.default_rng(seed)

    C_min = np.full_like(C, np.inf)
    C_max = np.full_like(C, -np.inf)
    S_min = np.full_like(S, np.inf)
    S_max = np.full_like(S, -np.inf)
    n_valid = 0
    T0_base = np.eye(k).flatten()

    def constraint_nonneg(t_flat):
        T = t_flat.reshape(k, k)
        try:
            C_new, S_new = _apply_transform(T, C, S)
        except np.linalg.LinAlgError:
            return np.array([-1e6])
        C_norm, S_norm = _normalize_rows(C_new, S_new, closure)
        parts = [C_norm.flatten()]
        if s_nonneg:
            parts.append(S_norm.flatten())
        return np.concatenate(parts)

    constraints = [{"type": "ineq", "fun": constraint_nonneg}]
    if closure:
        constraints.append(_closure_constraint(k))

    for _ in range(n_directions):
        w = rng.normal(0, 1, size=k)  # arah objektif acak antar komponen

        def objective(t_flat, w=w):
            T = t_flat.reshape(k, k)
            try:
                C_new, S_new = _apply_transform(T, C, S)
            except np.linalg.LinAlgError:
                return 1e6
            C_norm, _ = _normalize_rows(C_new, S_new, closure)
            return float(np.dot(w, [_area(C_norm[:, i]) for i in range(k)]))

        for attempt in range(n_restarts_per_direction):
            t0 = T0_base + rng.normal(0, 0.1, size=k * k)
            res = minimize(objective, t0, method="SLSQP", constraints=constraints,
                            options={"maxiter": maxiter, "ftol": 1e-8})
            if not res.success:
                continue
            T_opt = res.x.reshape(k, k)
            C_new, S_new = _apply_transform(T_opt, C, S)
            C_norm, S_norm = _normalize_rows(C_new, S_new, closure)
            if not _feasibility_check(C_norm, S_norm, s_nonneg):
                continue
            n_valid += 1
            C_min = np.minimum(C_min, C_norm)
            C_max = np.maximum(C_max, C_norm)
            S_min = np.minimum(S_min, S_norm)
            S_max = np.maximum(S_max, S_norm)

    if n_valid == 0:
        C_norm, S_norm = _normalize_rows(C, S, closure)
        return {"n_valid": 0, "n_total": n_directions * n_restarts_per_direction,
                "C_min": C_norm, "C_max": C_norm,
                "S_min": S_norm, "S_max": S_norm}

    return {"n_valid": n_valid, "n_total": n_directions * n_restarts_per_direction,
            "C_min": C_min, "C_max": C_max,
            "S_min": S_min, "S_max": S_max}


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------

def compute_rotational_ambiguity(C, S, constraints_used, method="auto",
                                  n_directions=20, k_exact_limit=K_EXACT_LIMIT,
                                  seed=None, fit_ok_per_component=None,
                                  local_ambiguity_threshold_pct=LOCAL_AMBIGUITY_THRESHOLD_PCT,
                                  reliability_low_threshold=AMBIGUITY_LOW_THRESHOLD_PCT,
                                  reliability_high_threshold=AMBIGUITY_HIGH_THRESHOLD_PCT):
    """
    Hitung ambiguitas rotasi (pendekatan MCR-BANDS) untuk hasil MCR-ALS,
    dilengkapi RMSERA (adaptasi relatif), klasifikasi keandalan
    (kuantitatif/semi-kuantitatif/kualitatif), dan profil ambiguitas lokal
    per titik bilangan gelombang.

    Parameters tambahan
    --------------------
    fit_ok_per_component : list[bool] atau None
        Status fit per komponen dari scorecard yang SUDAH ADA di
        mcr_engine.py (mis. berdasarkan LOF/R2 threshold). Jika None,
        diasumsikan True untuk semua komponen (fit dianggap baik).
    local_ambiguity_threshold_pct, reliability_low_threshold,
    reliability_high_threshold : ambang yang bisa disesuaikan pemanggil;
        default mengikuti konvensi interpretatif di kepala modul ini.

    Returns
    -------
    dict {
      "method_used", "k", "diagnostics",
      "components": [
         {
           ... field lama (area_nominal, area_min, area_max,
               ambiguity_width_pct, C_band_min/max, S_band_min/max,
               optimizer_success),
           "rmsera_lower", "rmsera_upper", "rmsera_upper_relative_pct",
           "reliability": {"label", "detail"},
           "local_ambiguity": {"local_width_pct", "pct_region_low_ambiguity",
                                "low_ambiguity_mask"},
         }, ...
      ],
    }
    """
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    k = C.shape[1]
    n_samples = C.shape[0]

    if k < 2:
        raise ValueError("Ambiguitas rotasi hanya bermakna untuk k >= 2 komponen.")

    if fit_ok_per_component is None:
        fit_ok_per_component = [True] * k

    s_nonneg = bool(constraints_used.get("s_nonneg", True))
    closure = bool(constraints_used.get("closure", False))

    if method == "auto":
        method = "exact" if k <= k_exact_limit else "heuristic"

    C_nom, S_nom = _normalize_rows(C, S, closure)
    components = []

    if method == "exact":
        for i in range(k):
            res_max = _optimize_component_exact(C, S, i, "max", s_nonneg, closure, seed=seed)
            res_min = _optimize_component_exact(C, S, i, "min", s_nonneg, closure, seed=seed)
            area_nom = _area(C_nom[:, i])
            width_pct = (100.0 * (res_max["area"] - res_min["area"]) / area_nom
                         if area_nom > 0 else 0.0)
            # PENTING: profil dari solusi "area-maks" dan "area-min" bisa
            # SALING SILANG di titik-titik tertentu (bukan satu selalu lebih
            # tinggi dari yang lain di SELURUH rentang) — band envelope yang
            # benar harus diambil elementwise min/max dari keduanya, bukan
            # diasumsikan res_max selalu di atas res_min di semua titik.
            C_lo = np.minimum(res_min["C_profile"], res_max["C_profile"])
            C_hi = np.maximum(res_min["C_profile"], res_max["C_profile"])
            S_lo = np.minimum(res_min["S_profile"], res_max["S_profile"])
            S_hi = np.maximum(res_min["S_profile"], res_max["S_profile"])
            components.append({
                "component_idx": i,
                "area_nominal": area_nom,
                "area_min": res_min["area"], "area_max": res_max["area"],
                "ambiguity_width_pct": width_pct,
                "C_band_min": C_lo, "C_band_max": C_hi,
                "S_band_min": S_lo, "S_band_max": S_hi,
                "optimizer_success": bool(res_min["success"] and res_max["success"]),
            })
        diag = {"n_valid_samples": None, "n_total_samples": None}

    else:  # heuristic
        mc = _sample_ambiguity_heuristic(C, S, s_nonneg, closure,
                                          n_directions=n_directions, seed=seed)
        for i in range(k):
            area_nom = _area(C_nom[:, i])
            area_min = _area(mc["C_min"][:, i])
            area_max = _area(mc["C_max"][:, i])
            width_pct = (100.0 * (area_max - area_min) / area_nom
                         if area_nom > 0 else 0.0)
            components.append({
                "component_idx": i,
                "area_nominal": area_nom,
                "area_min": area_min, "area_max": area_max,
                "ambiguity_width_pct": width_pct,
                "C_band_min": mc["C_min"][:, i], "C_band_max": mc["C_max"][:, i],
                "S_band_min": mc["S_min"][i, :], "S_band_max": mc["S_max"][i, :],
                "optimizer_success": mc["n_valid"] > 0,
            })
        diag = {"n_valid_samples": mc["n_valid"], "n_total_samples": mc["n_total"]}

    # ── Lengkapi tiap komponen dengan RMSERA, klasifikasi, & profil lokal ──
    for comp in components:
        i = comp["component_idx"]

        # RMSERA (adaptasi relatif, lihat catatan di kepala modul)
        delta_ra_avg = (comp["area_max"] - comp["area_min"]) / n_samples
        rmsera_lower = delta_ra_avg / np.sqrt(12)
        rmsera_upper = delta_ra_avg / np.sqrt(3)
        mean_c_nominal = comp["area_nominal"] / n_samples
        rmsera_upper_pct = (100.0 * rmsera_upper / mean_c_nominal
                             if mean_c_nominal > 0 else 0.0)
        comp["rmsera_lower"] = float(rmsera_lower)
        comp["rmsera_upper"] = float(rmsera_upper)
        comp["rmsera_upper_relative_pct"] = float(rmsera_upper_pct)

        # Klasifikasi keandalan
        comp["reliability"] = classify_reliability(
            comp["ambiguity_width_pct"],
            fit_ok=fit_ok_per_component[i] if i < len(fit_ok_per_component) else True,
            low_threshold=reliability_low_threshold,
            high_threshold=reliability_high_threshold,
        )

        # Profil ambiguitas lokal per titik bilangan gelombang
        comp["local_ambiguity"] = compute_local_ambiguity_profile(
            S_nom[i, :], comp["S_band_min"], comp["S_band_max"],
            local_threshold_pct=local_ambiguity_threshold_pct,
        )

    return {
        "method_used": method,
        "k": k,
        "components": components,
        "diagnostics": diag,
    }
