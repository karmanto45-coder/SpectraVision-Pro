"""
identity_decision_engine.py — SpectraVision Pro
=================================================
Modul ADITIF (tidak mengubah mcr_engine.py, ambiguity_engine.py, atau
mcr_replicate_extension.py yang sudah ada) untuk dua kebutuhan riset
"ambiguity-aware chemical identity" (JM/JE/MA, brief Q1):

  1. compute_identity_robustness_index()
     — "Spectral Identity Robustness Index" (SIRI), lihat brief
       Section 7. Mengukur apakah REFERENCE SPECTRUM (bukan spektrum
       itu sendiri) tetap konsisten dengan feasible solution band hasil
       ambiguity_engine.compute_rotational_ambiguity(). Ini BEDA dari
       compute_local_ambiguity_profile() yang sudah ada di
       ambiguity_engine.py — fungsi itu menormalisasi lebar band
       TERHADAP DIRINYA SENDIRI (intensitas maksimum komponen), tidak
       membandingkan ke reference eksternal. SIRI di sini persis
       mengikuti formulasi konseptual brief:

           IR = 1 - (A_mismatch / A_total)

       di mana A_mismatch = luas area reference yang JATUH DI LUAR
       band feasible [S_min, S_max], dan A_total = luas area reference
       itu sendiri.

  2. evaluate_component_identity()
     — Hierarchical identity decision (brief Section 8, 17, 21): bukan
       weighted arbitrary score, melainkan kombinasi bertingkat dari
       similarity (Tier 1) -> ambiguity+SIRI (Tier 3) -> diagnostic
       structure (Tier 2, opsional) -> robustness (Tier 4, opsional,
       BELUM diimplementasi sebagai modul terpisah — lihat catatan di
       docstring fungsi) -> fit/reconstruction (gate awal, Tier 5).

       Secara eksplisit dirancang untuk bisa MENANGKAP kasus sentral
       brief ("false resolved spectrum", Section 9): similarity TINGGI
       tetapi ambiguitas rotasi tinggi / SIRI rendah -> diberi label
       khusus, BUKAN otomatis "identified", supaya paper bisa
       menunjukkan bahwa good fit/similarity saja tidak cukup.

PENTING - keterbatasan yang harus dinyatakan di manuskrip jika dipakai:
  - Ambang batas (threshold) similarity/ambiguity/SIRI di sini adalah
    KONVENSI INTERPRETATIF proyek ini, sama seperti threshold yang
    sudah ada di ambiguity_engine.py (AMBIGUITY_LOW/HIGH_THRESHOLD_PCT)
    — bukan standar baku tunggal dari literatur. Harus dinyatakan
    eksplisit di metodologi bila dipakai untuk klaim publikasi.
  - Parameter robustness_ok di evaluate_component_identity() masih
    berupa placeholder input (None = "belum dinilai") sampai modul
    robustness-sweep (item backlog berikutnya) selesai dibangun. Verdict
    yang dihasilkan SEBELUM robustness-sweep tersedia harus dibaca
    sebagai "provisional", bukan final, sesuai brief Section 14/24
    (Tier 4 robustness dianggap bagian evidence hierarchy, bukan opsional
    tambahan).
"""

import numpy as np

from mcr_engine import cosine_sim, pearson_corr

# ── Ambang default (konvensi interpretatif proyek ini) ──────────────────
SIRI_LOW_THRESHOLD_PCT = 70.0     # < ini -> SIRI lemah (band tidak konsisten dgn reference)
SIRI_HIGH_THRESHOLD_PCT = 90.0    # > ini -> SIRI kuat

SIMILARITY_MODERATE_THRESHOLD = 0.90   # min(pearson, cosine) di bawah ini -> "tidak didukung"
SIMILARITY_STRONG_THRESHOLD = 0.95     # min(pearson, cosine) di atas ini -> "similarity kuat"


# ═══════════════════════════════════════════════════════════════════════
# 1. SPECTRAL IDENTITY ROBUSTNESS INDEX (SIRI)
# ═══════════════════════════════════════════════════════════════════════

def compute_identity_robustness_index(S_nominal, S_band_min, S_band_max, S_reference,
                                       low_threshold_pct=SIRI_LOW_THRESHOLD_PCT,
                                       high_threshold_pct=SIRI_HIGH_THRESHOLD_PCT,
                                       min_reference_energy=1e-8):
    """
    Hitung SIRI: proporsi wilayah spektral (berbasis area, bukan hitung
    titik biner) di mana reference spectrum tetap berada di dalam
    feasible band [S_band_min, S_band_max] hasil
    ambiguity_engine.compute_rotational_ambiguity().

    Parameters
    ----------
    S_nominal   : (n_points,) — spektrum komponen hasil MCR-ALS nominal
                  (T=identitas), dipakai HANYA untuk best-fit scaling
                  reference (bukan untuk perbandingan langsung).
    S_band_min, S_band_max : (n_points,) — dari
                  compute_rotational_ambiguity()["components"][i]
                  ("S_band_min"/"S_band_max"). WAJIB grid wavenumber
                  yang SAMA dengan S_nominal dan S_reference (resample
                  dulu pakai build_common_grid/resample_to_grid dari
                  mcr_engine.py kalau grid berbeda, sama seperti pola
                  yang sudah dipakai di batch_match()).
    S_reference : (n_points,) — spektrum referensi/library murni untuk
                  komponen yang diklaim (bukan hasil MCR).
    min_reference_energy : ambang ||S_reference||^2 minimum untuk
                  menghindari pembagian oleh nol pada best-fit scaling.

    Returns
    -------
    dict:
      siri_pct              : 100 * (1 - mismatch_area/total_area)
      mismatch_area_pct      : 100 * A_mismatch / A_total
      scale_factor           : faktor skala best-fit (least-squares)
                               yang diterapkan ke S_reference sebelum
                               dibandingkan ke band (menghilangkan
                               ambiguitas skala trivial, konvensi yang
                               sama dengan fix_intensity_scale di
                               mcr_engine.compute_mcr_bands)
      S_reference_scaled     : S_reference setelah discale
      excess                 : (n_points,) selisih di luar band per titik
                               (0 jika di dalam band; tanda + berarti di
                               atas band, - berarti di bawah band)
      pct_points_consistent  : % titik (hitung sederhana, bukan berbasis
                               area) di mana reference berada di dalam
                               band — statistik pelengkap untuk audit,
                               BUKAN dasar SIRI (SIRI berbasis area).
      classification         : {"label", "detail"}
    """
    S_nominal = np.asarray(S_nominal, dtype=float)
    S_min = np.asarray(S_band_min, dtype=float)
    S_max = np.asarray(S_band_max, dtype=float)
    S_ref = np.asarray(S_reference, dtype=float)

    if not (len(S_nominal) == len(S_min) == len(S_max) == len(S_ref)):
        raise ValueError(
            "S_nominal, S_band_min, S_band_max, dan S_reference harus "
            "berada pada grid wavenumber yang sama panjang — resample "
            "dulu (build_common_grid/resample_to_grid) sebelum memanggil "
            "fungsi ini."
        )

    # ── Best-fit scale (least-squares) — menyamakan skala reference ke
    # skala S_nominal, supaya perbandingan ke band murni soal BENTUK,
    # bukan skala absolut (yang memang ambigu tanpa penjangkar eksternal,
    # lihat catatan fix_intensity_scale di mcr_engine.compute_mcr_bands).
    ref_energy = float(np.dot(S_ref, S_ref))
    if ref_energy < min_reference_energy:
        scale = 1.0
    else:
        scale = float(np.dot(S_ref, S_nominal) / ref_energy)
        if scale <= 0:
            # Best-fit scale negatif/nol berarti reference dan nominal
            # tidak searah sama sekali — sinyal kuat bahwa ini BUKAN
            # komponen yang sama; jangan paksa scale negatif (akan
            # membalik tanda spektrum dan menyesatkan perbandingan band).
            scale = 0.0

    S_ref_scaled = S_ref * scale

    excess = np.where(
        S_ref_scaled < S_min, S_ref_scaled - S_min,
        np.where(S_ref_scaled > S_max, S_ref_scaled - S_max, 0.0)
    )

    mismatch_area = float(np.sum(np.abs(excess)))
    total_area = float(np.sum(np.abs(S_ref_scaled)))
    mismatch_area_pct = (100.0 * mismatch_area / total_area) if total_area > 0 else 100.0
    siri_pct = max(0.0, 100.0 - mismatch_area_pct)

    consistent_mask = np.abs(excess) < 1e-12
    pct_points_consistent = 100.0 * float(np.mean(consistent_mask))

    classification = _classify_siri(siri_pct, low_threshold_pct, high_threshold_pct)

    return {
        "siri_pct": siri_pct,
        "mismatch_area_pct": mismatch_area_pct,
        "scale_factor": scale,
        "S_reference_scaled": S_ref_scaled,
        "excess": excess,
        "pct_points_consistent": pct_points_consistent,
        "classification": classification,
    }


def _classify_siri(siri_pct, low_threshold_pct, high_threshold_pct):
    if siri_pct < low_threshold_pct:
        return {
            "label": "lemah",
            "detail": (
                f"SIRI rendah (<{low_threshold_pct:.0f}%) — reference spectrum "
                f"berada di luar feasible solution band pada sebagian besar "
                f"wilayah spektral (berbasis area). Identitas komponen TIDAK "
                f"cukup didukung oleh feasible solution space, terlepas dari "
                f"skor similarity satu solusi nominal."
            ),
        }
    if siri_pct <= high_threshold_pct:
        return {
            "label": "sedang",
            "detail": (
                f"SIRI sedang ({low_threshold_pct:.0f}-{high_threshold_pct:.0f}%) — "
                f"reference konsisten dengan band pada sebagian besar tapi "
                f"tidak seluruh wilayah spektral; identitas perlu bukti "
                f"pendukung tambahan (diagnostic bands/robustness) sebelum "
                f"diklaim kuat."
            ),
        }
    return {
        "label": "kuat",
        "detail": (
            f"SIRI tinggi (>{high_threshold_pct:.0f}%) — reference tetap "
            f"konsisten dengan feasible solution band di hampir seluruh "
            f"wilayah spektral; identitas komponen robust terhadap "
            f"rotational ambiguity untuk data ini."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. HIERARCHICAL IDENTITY DECISION (Tier 1-5, brief Section 8/17/21)
# ═══════════════════════════════════════════════════════════════════════

def evaluate_component_identity(component_label,
                                 fit_ok,
                                 pearson, cosine,
                                 ambiguity_width_pct=None,
                                 ambiguity_reliability_label=None,
                                 siri_result=None,
                                 diagnostic_ok=None,
                                 robustness_ok=None,
                                 similarity_moderate_threshold=SIMILARITY_MODERATE_THRESHOLD,
                                 similarity_strong_threshold=SIMILARITY_STRONG_THRESHOLD):
    """
    Gabungkan bukti Tier 1-5 menjadi SATU verdict identitas kimia per
    komponen, sebagai KEPUTUSAN BERTINGKAT (hierarchical), BUKAN skor
    tertimbang (brief eksplisit minta ini di Section 17: "Lebih baik
    dibuat sebagai hierarchical decision framework daripada weighted
    score arbitrer").

    Parameters
    ----------
    component_label : str/int — nama/identitas komponen untuk laporan.
    fit_ok          : bool — status fit MCR-ALS (dari compute_scorecard
                      yang SUDAH ADA), gate paling awal. Kalau False,
                      verdict langsung "tidak_dapat_diandalkan" —
                      konsisten dengan gate yang sudah dipakai
                      ambiguity_engine.classify_reliability().
    pearson, cosine : float — similarity komponen MCR vs reference
                      (dari mcr_engine.pearson_corr/cosine_sim atau
                      batch_match). Brief Section 16 eksplisit minta
                      Pearson & cosine JANGAN diperlakukan sebagai dua
                      bukti independen (karena berkorelasi) — di sini
                      keduanya digabung lewat min(pearson, cosine),
                      bukan dijumlah/dirata-rata sebagai "2 bukti".
    ambiguity_width_pct, ambiguity_reliability_label :
                      dari compute_rotational_ambiguity()["components"][i]
                      (field "ambiguity_width_pct" dan
                      reliability["label"]). Boleh None jika ambiguity
                      analysis belum dijalankan untuk komponen ini
                      (verdict akan menandai tier ini "belum dinilai").
    siri_result     : dict dari compute_identity_robustness_index(), atau
                      None jika belum dihitung.
    diagnostic_ok   : True/False/None — hasil pengecekan manual/otomatis
                      apakah komponen ini konsisten dengan diagnostic
                      band JM-vs-JE dari mcr_replicate_extension.
                      assess_group_separability() (mis. tanda/perilaku
                      relatif di region kuat sesuai arah yang
                      diharapkan). None = belum dinilai.
    robustness_ok   : True/False/None — PLACEHOLDER untuk modul
                      robustness-sweep (belum dibangun terpisah — lihat
                      catatan modul di atas). None = belum dinilai;
                      verdict akan ditandai "provisional".

    Returns
    -------
    dict:
      component_label, final_label, is_provisional, flags (list[str]),
      tier_evidence (dict berisi semua bukti per tier untuk audit trail
      / tabel supplementary manuskrip), narrative (str, ID+EN ringkas).

    final_label salah satu dari:
      "unreliable"                — fit gagal.
      "not_supported"             — similarity sendiri sudah rendah.
      "possible_false_resolution" — similarity TINGGI tapi ambiguitas
                                     tinggi / SIRI rendah: kandidat
                                     "false resolved spectrum" (brief
                                     Section 9) — kasus paling penting
                                     untuk didiskusikan di manuskrip.
      "provisional"               — similarity kuat/sedang, ambiguitas
                                     sedang, ATAU bukti robustness/
                                     diagnostic belum lengkap.
      "reliably_identified"       — similarity kuat, ambiguitas rendah,
                                     SIRI tinggi, DAN diagnostic_ok/
                                     robustness_ok tidak False (True
                                     atau memang sudah dinilai True;
                                     kalau salah satunya masih None,
                                     tetap diturunkan ke "provisional"
                                     sesuai catatan modul di atas).
    """
    tier_evidence = {
        "fit_ok": fit_ok,
        "pearson": pearson,
        "cosine": cosine,
        "ambiguity_width_pct": ambiguity_width_pct,
        "ambiguity_reliability_label": ambiguity_reliability_label,
        "siri_pct": siri_result["siri_pct"] if siri_result else None,
        "siri_label": siri_result["classification"]["label"] if siri_result else None,
        "diagnostic_ok": diagnostic_ok,
        "robustness_ok": robustness_ok,
    }
    flags = []

    # ── Gate 0: fit / reconstruction quality ────────────────────────
    if not fit_ok:
        return {
            "component_label": component_label,
            "final_label": "unreliable",
            "is_provisional": False,
            "flags": ["fit_gagal"],
            "tier_evidence": tier_evidence,
            "narrative": (
                "Fit MCR-ALS tidak memenuhi kriteria scorecard — identitas "
                "kimia tidak dapat dinilai sebelum masalah fit ini "
                "diatasi. / MCR-ALS fit does not meet scorecard criteria — "
                "chemical identity cannot be assessed until the fit issue "
                "is resolved."
            ),
        }

    # ── Tier 1: similarity (Pearson & cosine digabung, BUKAN dijumlah) ──
    sim_min = min(float(pearson), float(cosine))
    if sim_min < similarity_moderate_threshold:
        return {
            "component_label": component_label,
            "final_label": "not_supported",
            "is_provisional": False,
            "flags": ["similarity_rendah"],
            "tier_evidence": tier_evidence,
            "narrative": (
                f"Similarity terhadap reference rendah (min Pearson/cosine "
                f"= {sim_min:.3f} < {similarity_moderate_threshold:.2f}) — "
                f"identitas komponen ini tidak didukung, terlepas dari "
                f"hasil ambiguity/SIRI. / Similarity to reference is low — "
                f"component identity is not supported regardless of "
                f"ambiguity/SIRI results."
            ),
        }
    sim_level = "kuat" if sim_min >= similarity_strong_threshold else "sedang"

    # ── Tier 3: ambiguity + SIRI (evaluasi bersama, bukan terpisah) ──
    amb_assessed = ambiguity_width_pct is not None
    siri_assessed = siri_result is not None

    amb_high = amb_assessed and ambiguity_reliability_label == "kualitatif_saja"
    siri_weak = siri_assessed and siri_result["classification"]["label"] == "lemah"
    amb_low = amb_assessed and ambiguity_reliability_label == "kuantitatif_kualitatif"
    siri_strong = siri_assessed and siri_result["classification"]["label"] == "kuat"

    if sim_level == "kuat" and (amb_high or siri_weak):
        # Ini kasus sentral brief Section 9: fit/similarity bagus TAPI
        # ambiguitas tinggi atau SIRI rendah -> kandidat false resolved
        # spectrum. JANGAN dilabeli "reliably_identified" walau
        # similarity tinggi.
        flags.append("kandidat_false_resolved_spectrum")
        return {
            "component_label": component_label,
            "final_label": "possible_false_resolution",
            "is_provisional": False,
            "flags": flags,
            "tier_evidence": tier_evidence,
            "narrative": (
                "Similarity terhadap reference tinggi, TETAPI ambiguitas "
                "rotasi tinggi dan/atau SIRI rendah — resolved spectrum ini "
                "berpotensi menjadi 'false resolved spectrum': rekonstruksi "
                "matematis baik, namun identitas kimia belum dapat "
                "dipastikan karena feasible solution space tidak konsisten "
                "dengan reference. / High similarity but high rotational "
                "ambiguity and/or low SIRI — this resolved spectrum is a "
                "candidate 'false resolved spectrum': good mathematical "
                "reconstruction does not by itself establish chemical "
                "identity here."
            ),
        }

    if not amb_assessed:
        flags.append("ambiguity_belum_dinilai")
    if not siri_assessed:
        flags.append("siri_belum_dihitung")
    if diagnostic_ok is None:
        flags.append("diagnostic_region_belum_dinilai")
    elif diagnostic_ok is False:
        flags.append("diagnostic_region_tidak_konsisten")
    if robustness_ok is None:
        flags.append("robustness_sweep_belum_dijalankan")
    elif robustness_ok is False:
        flags.append("tidak_stabil_pada_robustness_sweep")

    # ── Final tier: hanya "reliably_identified" kalau SEMUA tier yang
    # sudah dinilai konsisten DAN tidak ada tier penting yang masih None.
    all_assessed_and_good = (
        sim_level == "kuat"
        and amb_low and siri_strong
        and diagnostic_ok is not False
        and robustness_ok is not False
    )
    nothing_missing = not any(f.endswith("belum_dinilai") or f.endswith("belum_dihitung")
                              or f.endswith("belum_dijalankan") for f in flags)

    if all_assessed_and_good and nothing_missing:
        return {
            "component_label": component_label,
            "final_label": "reliably_identified",
            "is_provisional": False,
            "flags": flags,
            "tier_evidence": tier_evidence,
            "narrative": (
                "Similarity kuat, ambiguitas rotasi rendah, SIRI tinggi, "
                "dan bukti diagnostic/robustness tidak bertentangan — "
                "identitas komponen didukung secara konsisten di seluruh "
                "tier evidence. / Strong similarity, low rotational "
                "ambiguity, high SIRI, and diagnostic/robustness evidence "
                "not contradicting — component identity is consistently "
                "supported across evidence tiers."
            ),
        }

    return {
        "component_label": component_label,
        "final_label": "provisional",
        "is_provisional": True,
        "flags": flags,
        "tier_evidence": tier_evidence,
        "narrative": (
            "Similarity mendukung identitas komponen, dan tidak ditemukan "
            "bukti kuat yang menolaknya, TETAPI belum seluruh tier evidence "
            "(ambiguity/SIRI/diagnostic/robustness) selesai dinilai atau "
            "hasilnya baru sedang (bukan kuat) — verdict ini bersifat "
            "sementara sampai tier yang tersisa dilengkapi. / Similarity "
            "supports the identity and no strong contradicting evidence "
            "found, but not all evidence tiers are yet complete or results "
            "are moderate rather than strong — this verdict is provisional "
            "pending the remaining tiers."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. TABEL UNTUK LAPORAN (pola sama dengan build_*_dataframe di
#    mcr_replicate_extension.py)
# ═══════════════════════════════════════════════════════════════════════

def build_identity_decision_dataframe(results):
    """
    Susun list hasil evaluate_component_identity() jadi satu pandas
    DataFrame — untuk sheet "Identity Decision" di fitur Laporan, dan
    sebagai bahan tabel supplementary manuskrip (audit trail lengkap
    per komponen, bukan cuma kesimpulan akhir).

    Parameters
    ----------
    results : list[dict] — tiap elemen adalah output evaluate_component_identity()

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd
    rows = []
    for r in results:
        ev = r["tier_evidence"]
        rows.append({
            "Komponen": r["component_label"],
            "Verdict": r["final_label"],
            "Provisional": r["is_provisional"],
            "Pearson": ev["pearson"],
            "Cosine": ev["cosine"],
            "Ambiguity width (%)": ev["ambiguity_width_pct"],
            "SIRI (%)": ev["siri_pct"],
            "Diagnostic OK": ev["diagnostic_ok"],
            "Robustness OK": ev["robustness_ok"],
            "Flags": ", ".join(r["flags"]) if r["flags"] else "-",
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 4. SIMILARITY MARGIN (SM) — Section 14/15 riset eksistensi-identitas
# ═══════════════════════════════════════════════════════════════════════
#
# Beda dari evaluate_component_identity() di atas (yang pakai AMBANG
# ABSOLUT terhadap satu reference target): SM mengukur seberapa
# DISKRIMINATIF identitas itu dibanding kandidat pesaing TERKUAT.
# Motivasi (brief Section 14): dua kandidat bisa sama-sama "similarity
# tinggi ke target" (mis. 0.9999 dan 0.9994) tapi salah satunya jauh
# LEBIH mirip ke kompetitor juga (margin kecil) — similarity tinggi
# SENDIRIAN bukan bukti identitas kuat kalau kompetitor sama kuatnya.

def compute_similarity_margin(S, S_target, S_competitors, metric="cosine"):
    """
    SM = Sim(S, target) - max_j Sim(S, competitor_j)  — brief Section 14.

    Parameters
    ----------
    S             : (n_points,) — spektrum yang dinilai (biasanya solusi
                    nominal hasil MCR, TAPI bisa juga anchor band — lihat
                    compute_similarity_margin_min_under_ambiguity()).
    S_target      : (n_points,) — reference untuk identitas yang diklaim.
    S_competitors : list of (label, S_j) — kandidat identitas PESAING,
                    mis. [("JE", S_je), ("MA", S_ma)] kalau target-nya JM.
                    Kalau kosong, margin tidak bisa dihitung (dikembalikan
                    None dengan classification yang menjelaskan kenapa).
    metric        : "cosine" (default, sesuai gaya angka contoh brief
                    Section 14, mis. 0.99990/0.99833) atau "pearson".

    Returns
    -------
    dict: sim_target, sim_competitor_best, competitor_label, margin,
          classification {"label","detail"}
    """
    sim_fn = cosine_sim if metric == "cosine" else pearson_corr
    sim_target = sim_fn(S, S_target)

    if not S_competitors:
        return {
            "sim_target": sim_target, "sim_competitor_best": None,
            "competitor_label": None, "margin": None,
            "classification": {
                "label": "tidak_ada_kompetitor",
                "detail": ("Tidak ada kandidat pesaing diberikan — SM tidak bisa "
                           "dihitung. Similarity tinggi terhadap target saja TIDAK "
                           "cukup untuk klaim identitas diskriminatif (brief Section 14); "
                           "sertakan minimal satu kandidat pesaing yang relevan."),
            },
        }

    competitor_sims = [(label, sim_fn(S, Sj)) for label, Sj in S_competitors]
    competitor_label, sim_competitor_best = max(competitor_sims, key=lambda x: x[1])
    margin = sim_target - sim_competitor_best

    return {
        "sim_target": sim_target, "sim_competitor_best": sim_competitor_best,
        "competitor_label": competitor_label, "margin": margin,
        "classification": _classify_margin(margin, competitor_label),
    }


def _classify_margin(margin, competitor_label):
    if margin > 0:
        return {
            "label": "diskriminatif",
            "detail": (
                f"Margin positif ({margin:.5f}) terhadap kompetitor terkuat "
                f"('{competitor_label}') — target tetap lebih mirip daripada "
                f"kandidat pesaing manapun."
            ),
        }
    return {
        "label": "ambigu",
        "detail": (
            f"Margin <= 0 ({margin:.5f}) terhadap kompetitor terkuat "
            f"('{competitor_label}') — kompetitor SAMA KUAT atau LEBIH KUAT; "
            f"identitas TIDAK diskriminatif meski similarity ke target sendiri "
            f"mungkin tinggi (persis peringatan brief Section 14: "
            f"'high similarity != strong identity evidence')."
        ),
    }


def compute_similarity_margin_min_under_ambiguity(S_nominal, S_band_min, S_band_max,
                                                   S_target, S_competitors, metric="cosine"):
    """
    Aproksimasi SM_min (worst-case Similarity Margin di bawah rotational
    ambiguity, brief Section 15) memakai TIGA anchor spectra yang sudah
    tersedia dari ambiguity_engine.compute_rotational_ambiguity(): solusi
    nominal, S_band_min, dan S_band_max.

    PENTING - tingkat aproksimasi ini SENGAJA disamakan dengan yang sudah
    dipakai ambiguity_engine.compute_local_ambiguity_profile() (lihat
    catatan di modul itu): S_band_min/S_band_max adalah ENVELOPE
    ELEMENTWISE (min/max per titik wavenumber dari solusi-solusi yang
    dieksplorasi), BUKAN otomatis satu spektrum yang benar-benar
    realizable dari satu T tunggal. Brief Section 15 sendiri eksplisit
    mengizinkan pendekatan ini: "F adalah feasible solution set ATAU
    aproksimasi yang berdasar (justified approximation)". Hasil sm_min
    di sini adalah ESTIMASI KONSERVATIF dari 3 titik anchor yang
    tersedia, BUKAN hasil optimisasi eksak atas seluruh F.

    Returns
    -------
    dict: sm_min, worst_case_anchor, per_anchor (dict nama_anchor ->
          hasil compute_similarity_margin di anchor itu), classification
    """
    anchors = {"nominal": S_nominal, "band_min": S_band_min, "band_max": S_band_max}
    per_anchor = {
        name: compute_similarity_margin(S, S_target, S_competitors, metric=metric)
        for name, S in anchors.items()
    }
    valid = {k: v for k, v in per_anchor.items() if v["margin"] is not None}
    if not valid:
        return {
            "sm_min": None, "worst_case_anchor": None, "per_anchor": per_anchor,
            "classification": {
                "label": "tidak_ada_kompetitor",
                "detail": "Tidak ada kandidat pesaing diberikan — SM_min tidak bisa dihitung.",
            },
        }
    worst_case_anchor = min(valid, key=lambda k: valid[k]["margin"])
    sm_min = valid[worst_case_anchor]["margin"]

    label = "robust" if sm_min > 0 else "ambigu"
    detail = (
        (f"SM_min > 0 ({sm_min:.5f}) di anchor '{worst_case_anchor}' — target tetap "
         f"lebih diskriminatif daripada kompetitor terkuat di SELURUH tiga anchor "
         f"feasible band yang diperiksa (nominal/band_min/band_max).")
        if sm_min > 0 else
        (f"SM_min <= 0 ({sm_min:.5f}) di anchor '{worst_case_anchor}' — di titik itu, "
         f"kompetitor menjadi SAMA KUAT atau LEBIH KUAT daripada target; identitas "
         f"TIDAK robust terhadap rotational ambiguity (brief Section 15: "
         f"'SM_min <= 0 -> ambiguous identity').")
    )
    return {
        "sm_min": sm_min, "worst_case_anchor": worst_case_anchor,
        "per_anchor": per_anchor, "classification": {"label": label, "detail": detail},
    }
