"""
2D Correlation Spectroscopy (2D-COS) Module
Based on: Noda, I. (1993). Generalized two-dimensional correlation method
          applicable to infrared, Raman, and other types of spectroscopy.
          Applied Spectroscopy, 47(9), 1329-1336.
"""

import numpy as np


# ── Core 2D-COS computation ───────────────────────────────────────────────────

def compute_dynamic_spectra(D):
    """
    Compute dynamic spectra by subtracting the mean spectrum.
    D : (m × n) — m perturbation steps, n wavenumber points
    Returns Ã : (m × n)
    """
    D = np.array(D, dtype=float)
    mean_spec = D.mean(axis=0)
    return D - mean_spec


def compute_synchronous(D_dyn):
    """
    Synchronous 2D correlation map.
    Φ(ν₁,ν₂) = 1/(m-1) · Ã^T · Ã
    Returns Phi : (n × n)
    """
    m = D_dyn.shape[0]
    Phi = (D_dyn.T @ D_dyn) / (m - 1)
    return Phi


def compute_asynchronous(D_dyn):
    """
    Asynchronous 2D correlation map using Hilbert-Noda matrix N.
    Ψ(ν₁,ν₂) = 1/(m-1) · Ã^T · N · Ã
    N[i,j] = 0 if i==j else 1/π(j-i)
    Returns Psi : (n × n)
    """
    m = D_dyn.shape[0]
    # Build Hilbert-Noda matrix
    N = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                N[i, j] = 1.0 / (np.pi * (j - i))
    Psi = (D_dyn.T @ N @ D_dyn) / (m - 1)
    return Psi


def compute_autopower(Phi):
    """
    Autopower spectrum = diagonal of synchronous map.
    Identifies wavenumber bands that change significantly.
    """
    return np.diag(Phi)


def compute_2dcos(D, wavenumber, wmin=None, wmax=None):
    """
    Full 2D-COS pipeline.

    Parameters
    ----------
    D          : (m × n) spectral matrix — rows=perturbation, cols=wavenumber
    wavenumber : (n,) wavenumber array
    wmin, wmax : optional window limits

    Returns
    -------
    dict with Phi, Psi, autopower, wn_used, D_dyn
    """
    D = np.array(D, dtype=float)
    wn = np.array(wavenumber, dtype=float)

    # Ensure wavenumber ascending for consistent indexing
    if wn[0] > wn[-1]:
        wn = wn[::-1]
        D = D[:, ::-1]

    # Apply window
    if wmin is not None and wmax is not None:
        mask = (wn >= wmin) & (wn <= wmax)
        wn_used = wn[mask]
        D_win   = D[:, mask]
    else:
        wn_used = wn
        D_win   = D

    if D_win.shape[1] < 4 or D_win.shape[0] < 3:
        return None

    D_dyn  = compute_dynamic_spectra(D_win)
    Phi    = compute_synchronous(D_dyn)
    Psi    = compute_asynchronous(D_dyn)
    Auto   = compute_autopower(Phi)

    return {
        "Phi":       Phi,
        "Psi":       Psi,
        "autopower": Auto,
        "wn":        wn_used,
        "D_dyn":     D_dyn,
        "D_win":     D_win,
        "n_steps":   D_win.shape[0],
        "n_points":  D_win.shape[1],
    }


# ── Cross-peak analysis ───────────────────────────────────────────────────────

def find_crosspeaks(Phi, Psi, wn, threshold_phi=0.0, threshold_psi=0.0,
                    top_n=20):
    """
    Find significant cross-peaks and apply Noda's rules.

    Returns list of dicts with wn1, wn2, phi, psi, sign_phi, sign_psi,
    noda_rule, sequential_order.
    """
    n = len(wn)
    peaks = []

    for i in range(n):
        for j in range(i + 1, n):
            phi_val = Phi[i, j]
            psi_val = Psi[i, j]

            if abs(phi_val) < threshold_phi and abs(psi_val) < threshold_psi:
                continue

            sign_phi = "+" if phi_val >= 0 else "-"
            sign_psi = "+" if psi_val >= 0 else "-"

            # Noda's rules
            noda, order = apply_nodas_rules(
                phi_val, psi_val, wn[i], wn[j]
            )

            peaks.append({
                "wn1":       round(float(wn[i]), 2),
                "wn2":       round(float(wn[j]), 2),
                "phi":       round(float(phi_val), 6),
                "psi":       round(float(psi_val), 6),
                "sign_phi":  sign_phi,
                "sign_psi":  sign_psi,
                "abs_phi":   abs(phi_val),
                "noda_rule": noda,
                "order":     order,
            })

    # Sort by |phi| descending
    peaks.sort(key=lambda x: x["abs_phi"], reverse=True)
    return peaks[:top_n]


def apply_nodas_rules(phi, psi, wn1, wn2):
    """
    Apply Noda's rules to determine sequential order.

    Returns (rule_description, order_description)
    """
    if phi == 0:
        return "Tidak dapat ditentukan / Undetermined", "—"

    if phi > 0 and psi > 0:
        rule = "Φ>0, Ψ>0"
        order = f"{wn1:.1f} cm⁻¹ lebih dulu / first"
    elif phi > 0 and psi < 0:
        rule = "Φ>0, Ψ<0"
        order = f"{wn2:.1f} cm⁻¹ lebih dulu / first"
    elif phi < 0 and psi > 0:
        rule = "Φ<0, Ψ>0"
        order = f"{wn2:.1f} cm⁻¹ lebih dulu / first"
    elif phi < 0 and psi < 0:
        rule = "Φ<0, Ψ<0"
        order = f"{wn1:.1f} cm⁻¹ lebih dulu / first"
    else:
        rule = "Ψ=0"
        order = "Bersamaan / Simultaneous"

    return rule, order


# ── Perturbation axis ─────────────────────────────────────────────────────────

PERTURBATION_PRESETS = {
    "Konsentrasi / Concentration (%)": {
        "unit": "%", "symbol": "C",
        "description": "Serial pengenceran / dilution series"
    },
    "Waktu / Time (menit)": {
        "unit": "menit", "symbol": "t",
        "description": "Time-resolved spectroscopy"
    },
    "Suhu / Temperature (°C)": {
        "unit": "°C", "symbol": "T",
        "description": "Temperature-dependent spectroscopy"
    },
    "pH": {
        "unit": "pH", "symbol": "pH",
        "description": "pH-dependent spectroscopy"
    },
    "Tekanan / Pressure (bar)": {
        "unit": "bar", "symbol": "P",
        "description": "Pressure-dependent spectroscopy"
    },
    "Lainnya / Other (isi manual)": {
        "unit": "a.u.", "symbol": "x",
        "description": "Custom perturbation"
    },
}
