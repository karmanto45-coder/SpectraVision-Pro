"""
existence_identity_decision_tree.py — SpectraVision Pro
==========================================================
Modul ADITIF penggabung AKHIR — mengimplementasikan decision tree
Section 17 brief MCR_Spurious_Component_Existence_and_Identity_Framework.md,
menggabungkan SEMUA modul yang sudah dibangun di backlog ini jadi SATU
verdict akhir per komponen:

  reproducibility_engine.py  -> P_repro (reproducibility)
  existence_engine.py        -> contribution ratio (Rk) + null/bootstrap
                                 ΔLOF significance test
  identity_decision_engine.py -> similarity/SIRI/ambiguity (final_label)
                                 + Similarity Margin (SM_min)

Pohon keputusan (persis Section 17 brief, path lengkap):

    Apakah faktor reproducible? (P_repro >= ambang)
      TIDAK -> Apakah ada bukti kuat lain (null-test signifikan)?
                 YA  -> UNCERTAIN
                 TIDAK -> SPURIOUS
      YA    -> Apakah kontribusinya bermakna? (Rk >= ambang)
                 TIDAK -> SPURIOUS
                 YA    -> Apakah reference-compatible? (similarity >= ambang)
                            TIDAK -> REAL_UNASSIGNED
                            YA    -> Apakah identitasnya robust?
                                       (reliably_identified DAN SM_min > 0)
                                       TIDAK -> NON_IDENTIFIABLE
                                       YA    -> VALID

PENTING - hal yang WAJIB dinyatakan kalau dipakai untuk publikasi (brief
sendiri eksplisit di baris terakhir Section 17): "This is a conceptual
framework; numerical decision thresholds should be empirically
validated." Semua ambang di bawah adalah KONVENSI INTERPRETATIF proyek
ini (konsisten dengan ambang-ambang lain di seluruh backlog ini),
BUKAN standar baku dari literatur — nyatakan ini secara eksplisit di
metodologi manuskrip.

Modul ini SENGAJA tidak menghitung ulang apa pun — murni penggabung
(pure combiner) atas hasil yang SUDAH dihitung modul-modul lain,
mengikuti pola evaluate_component_identity() di identity_decision_engine.py.
Kalau salah satu bukti belum tersedia (None), verdict yang dikembalikan
adalah "EVIDENCE_INCOMPLETE" dengan flag yang jelas menyebutkan bukti
apa yang masih kurang — BUKAN menebak dengan asumsi default, supaya
tidak diam-diam menurunkan/menaikkan vonis tanpa bukti.
"""

DEFAULT_REPRODUCIBLE_THRESHOLD_PCT = 50.0
DEFAULT_CONTRIBUTION_THRESHOLD_PCT = 1.0
DEFAULT_REFERENCE_COMPATIBLE_THRESHOLD = 0.90


def classify_component_existence_identity(
    component_label,
    p_repro_pct=None,
    null_test_result=None,
    contribution_pct=None,
    pearson=None, cosine=None,
    identity_decision=None,
    sm_min_result=None,
    reproducible_threshold_pct=DEFAULT_REPRODUCIBLE_THRESHOLD_PCT,
    contribution_threshold_pct=DEFAULT_CONTRIBUTION_THRESHOLD_PCT,
    reference_compatible_threshold=DEFAULT_REFERENCE_COMPATIBLE_THRESHOLD,
):
    """
    Implementasi pohon keputusan Section 17 brief. Lihat docstring modul
    untuk struktur pohon lengkap.

    Parameters
    ----------
    p_repro_pct       : dari compute_existence_reproducibility()["p_repro_pct"]
                        untuk komponen ini, atau None kalau belum dihitung.
    null_test_result  : dict dari run_null_delta_lof_test(), atau None.
                        Dipakai sebagai "bukti kuat lain" di cabang TIDAK
                        reproducible.
    contribution_pct  : dari compute_contribution_ratio()["contribution_pct"]
                        untuk komponen ini, atau None.
    pearson, cosine   : similarity solusi nominal komponen ini terhadap
                        reference yang diklaim (dipakai untuk cek
                        reference-compatible).
    identity_decision : dict dari evaluate_component_identity(), atau None.
    sm_min_result     : dict dari compute_similarity_margin_min_under_ambiguity(),
                        atau None (kalau None, cek "identity robust" HANYA
                        bergantung pada identity_decision, dengan flag
                        bahwa SM_min belum dinilai).

    Returns
    -------
    dict: component_label, final_label (salah satu dari VALID/
    REAL_UNASSIGNED/NON_IDENTIFIABLE/UNCERTAIN/SPURIOUS/EVIDENCE_INCOMPLETE),
    tier_evidence (audit trail lengkap), flags, narrative.
    """
    tier_evidence = {
        "p_repro_pct": p_repro_pct,
        "null_test_label": null_test_result["classification"]["label"] if null_test_result else None,
        "contribution_pct": contribution_pct,
        "pearson": pearson, "cosine": cosine,
        "identity_final_label": identity_decision["final_label"] if identity_decision else None,
        "sm_min": sm_min_result["sm_min"] if sm_min_result else None,
    }

    # ── Node 1: apakah faktor reproducible? ─────────────────────────
    if p_repro_pct is None:
        return _incomplete(component_label, tier_evidence,
                           "Reproducibility belum dihitung — jalankan "
                           "reproducibility_engine.run_repeated_mcr() + "
                           "compute_existence_reproducibility() dulu.")

    is_reproducible = p_repro_pct >= reproducible_threshold_pct

    if not is_reproducible:
        # ── Cabang TIDAK reproducible: apakah ada bukti kuat lain? ──
        if null_test_result is None:
            return _incomplete(component_label, tier_evidence,
                               f"Reproducibility rendah ({p_repro_pct:.0f}%) — butuh "
                               f"null/bootstrap ΔLOF test (existence_engine."
                               f"run_null_delta_lof_test()) untuk menentukan "
                               f"UNCERTAIN vs SPURIOUS.")
        strong_evidence = null_test_result["classification"]["label"] == "signifikan"
        if strong_evidence:
            return _finalize(component_label, tier_evidence, "UNCERTAIN", [],
                             f"Reproducibility rendah ({p_repro_pct:.0f}% < "
                             f"{reproducible_threshold_pct:.0f}%) TAPI null-test "
                             f"menunjukkan perbaikan LOF signifikan — bukti saling "
                             f"bertentangan, verdict TIDAK bisa disimpulkan tegas "
                             f"ke arah mana pun. Perlu data/replikat tambahan.")
        return _finalize(component_label, tier_evidence, "SPURIOUS", [],
                         f"Reproducibility rendah ({p_repro_pct:.0f}%) DAN null-test "
                         f"tidak signifikan — dua bukti eksistensi independen "
                         f"sama-sama menolak faktor ini sebagai komponen kimia nyata.")

    # ── Node 2 (reproducible=True): apakah kontribusi bermakna? ─────
    if contribution_pct is None:
        return _incomplete(component_label, tier_evidence,
                           "Contribution ratio belum dihitung — jalankan "
                           "existence_engine.compute_contribution_ratio() dulu.")
    if contribution_pct < contribution_threshold_pct:
        return _finalize(component_label, tier_evidence, "SPURIOUS", [],
                         f"Reproducible ({p_repro_pct:.0f}%) TAPI kontribusi energi "
                         f"sangat kecil ({contribution_pct:.3f}% < "
                         f"{contribution_threshold_pct:.2f}%) — faktor ini stabil "
                         f"secara numerik tapi nyaris tidak menjelaskan variansi "
                         f"data nyata.")

    # ── Node 3: apakah reference-compatible? ─────────────────────────
    if pearson is None or cosine is None:
        return _incomplete(component_label, tier_evidence,
                           "Similarity terhadap reference belum dihitung — "
                           "perlu pearson & cosine terhadap kandidat identitas.")
    reference_compatible = min(pearson, cosine) >= reference_compatible_threshold
    if not reference_compatible:
        return _finalize(component_label, tier_evidence, "REAL_UNASSIGNED", [],
                         f"Reproducible ({p_repro_pct:.0f}%) DAN kontribusi bermakna "
                         f"({contribution_pct:.2f}%) — faktor ini KEMUNGKINAN BESAR "
                         f"komponen kimia nyata — TAPI similarity terhadap kandidat "
                         f"reference yang tersedia rendah (min(pearson,cosine)="
                         f"{min(pearson, cosine):.3f}). Komponen nyata tapi belum "
                         f"teridentifikasi; mungkin butuh reference lain di library.")

    # ── Node 4: apakah identitasnya robust? ──────────────────────────
    if identity_decision is None:
        return _incomplete(component_label, tier_evidence,
                           "Existence sudah mendukung (reproducible, kontribusi "
                           "bermakna, reference-compatible) — tinggal jalankan "
                           "identity_decision_engine.evaluate_component_identity() "
                           "untuk menentukan VALID vs NON_IDENTIFIABLE.")

    flags = []
    identity_ok = identity_decision["final_label"] == "reliably_identified"
    if not identity_ok:
        flags.append(f"identity_final_label={identity_decision['final_label']}")

    sm_min_ok = True
    if sm_min_result is not None:
        sm_min_ok = sm_min_result["classification"]["label"] == "robust"
        if not sm_min_ok:
            flags.append("sm_min_ambigu")
    else:
        flags.append("sm_min_belum_dinilai")

    identity_robust = identity_ok and sm_min_ok

    if not identity_robust:
        return _finalize(component_label, tier_evidence, "NON_IDENTIFIABLE", flags,
                         f"Existence lengkap mendukung (reproducible "
                         f"{p_repro_pct:.0f}%, kontribusi {contribution_pct:.2f}%, "
                         f"reference-compatible) — komponen ini KEMUNGKINAN BESAR "
                         f"nyata, TAPI identitas spesifiknya (JM vs JE vs kandidat "
                         f"lain) TIDAK cukup robust untuk dipastikan "
                         f"(non-identifiable != spurious — lihat brief Section 1).")

    return _finalize(component_label, tier_evidence, "VALID", flags,
                     f"SELURUH evidence tier mendukung: reproducible "
                     f"({p_repro_pct:.0f}%), kontribusi bermakna "
                     f"({contribution_pct:.2f}%), reference-compatible, DAN "
                     f"identitas robust (termasuk terhadap kompetitor di bawah "
                     f"ambiguitas rotasi). Komponen ini valid identified.")


def _incomplete(component_label, tier_evidence, message):
    return {
        "component_label": component_label, "final_label": "EVIDENCE_INCOMPLETE",
        "tier_evidence": tier_evidence, "flags": ["bukti_belum_lengkap"],
        "narrative": message,
    }


def _finalize(component_label, tier_evidence, final_label, flags, message):
    return {
        "component_label": component_label, "final_label": final_label,
        "tier_evidence": tier_evidence, "flags": flags, "narrative": message,
    }


def build_decision_tree_dataframe(results):
    """Tabel ringkas untuk fitur Laporan / tabel utama manuskrip —
    mengikuti pola build_*_dataframe modul-modul lain di proyek ini."""
    import pandas as pd
    rows = []
    for r in results:
        ev = r["tier_evidence"]
        rows.append({
            "Komponen": r["component_label"],
            "Verdict": r["final_label"],
            "P_repro (%)": ev["p_repro_pct"],
            "Null-test": ev["null_test_label"],
            "Kontribusi (%)": ev["contribution_pct"],
            "Pearson": ev["pearson"], "Cosine": ev["cosine"],
            "Identity label": ev["identity_final_label"],
            "SM_min": ev["sm_min"],
            "Flags": ", ".join(r["flags"]) if r["flags"] else "-",
        })
    return pd.DataFrame(rows)
