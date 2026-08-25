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
akibat ambiguitas rotasi.
"""

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
                                  seed=None):
    """
    Hitung ambiguitas rotasi (pendekatan MCR-BANDS) untuk hasil MCR-ALS.

    Parameters
    ----------
    C : (n_sampel x k)  -- profil konsentrasi hasil MCR-ALS (mcr_C)
    S : (k x n_lambda)  -- profil spektra hasil MCR-ALS (mcr_S)
    constraints_used : dict dari diagnostics["constraints_used"] (hasil
                       run_mcr_als) — dipakai supaya pencarian ambiguitas
                       menghormati closure & s_nonneg yang SAMA dengan run
                       MCR-ALS aslinya, bukan menebak ulang dari nol.
    method : 'auto' | 'exact' | 'heuristic'
             'auto' -> exact jika k <= k_exact_limit, else heuristic
    n_directions : jumlah arah yang dijelajahi untuk metode heuristik
                   (tidak dipakai untuk metode exact). Lebih besar = lebih
                   teliti tapi lebih lambat.

    Returns
    -------
    dict {
      "method_used": "exact" | "heuristic",
      "k": int,
      "components": [
         {
           "component_idx", "area_nominal", "area_min", "area_max",
           "ambiguity_width_pct",
           "C_band_min", "C_band_max",   # untuk overlay plot profil C
           "S_band_min", "S_band_max",   # untuk overlay plot profil S
           "optimizer_success",
         }, ...
      ],
      "diagnostics": {"n_valid_samples", "n_total_samples"}  # khusus heuristic
    }
    """
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    k = C.shape[1]

    if k < 2:
        raise ValueError("Ambiguitas rotasi hanya bermakna untuk k >= 2 komponen.")

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
            components.append({
                "component_idx": i,
                "area_nominal": area_nom,
                "area_min": res_min["area"], "area_max": res_max["area"],
                "ambiguity_width_pct": width_pct,
                "C_band_min": res_min["C_profile"], "C_band_max": res_max["C_profile"],
                "S_band_min": res_min["S_profile"], "S_band_max": res_max["S_profile"],
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

    return {
        "method_used": method,
        "k": k,
        "components": components,
        "diagnostics": diag,
    }
