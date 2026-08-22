"""
mcr_replicate_extension.py — SpectraVision Pro
Ekstensi ADITIF untuk mcr_engine.py — menangani replikat multi-sampel
per varietas/komponen (mis. >2 spektrum jahe merah, >2 spektrum jahe
emprit) untuk membangun referensi rata-rata yang lebih robust dan
menguji apakah perbedaan antar-kelompok signifikan dibanding variasi
alami di dalam kelompok itu sendiri.

PENTING — INI TIDAK MENGUBAH APA PUN DI mcr_engine.py YANG SUDAH ADA:
  - run_mcr_als(), _init_mcr(), simplisma(), detect_components() TIDAK
    disentuh sama sekali dan tetap menerima array biasa seperti
    sebelumnya (S_init, fixed_spectra) — algoritma inti tidak perlu
    tahu apakah array itu berasal dari 1 spektrum atau rata-rata
    beberapa replikat.
  - Dua fungsi baru di file ini murni (pure function): tidak ada efek
    samping, tidak memanggil balik ke mcr_engine.py atau app.py.

Cara pakai (di app.py, HANYA sebagai penambahan opsional, tidak
menggantikan alur multiselect 1-spektrum-per-komponen yang sudah ada):

    from mcr_replicate_extension import (
        average_replicate_spectra, assess_group_separability
    )

    # setelah user pilih >1 entri library untuk komponen yang sama:
    mean_spec, std_spec, cv_spec, info = average_replicate_spectra(
        [entry1["spectrum"], entry2["spectrum"], entry3["spectrum"]],
        [entry1["wavenumber"], entry2["wavenumber"], entry3["wavenumber"]],
    )
    # mean_spec ini yang dipakai sebagai S_init / fixed_spectra,
    # persis seperti sebelumnya user pakai 1 spektrum tunggal.
"""

import numpy as np


# ════════════════════════════════════════════════════════════════
# 1. RATA-RATA REPLIKAT (dengan resampling otomatis jika grid beda)
# ════════════════════════════════════════════════════════════════

def average_replicate_spectra(spectra_list, wavenumber_list=None,
                               interp_method="cubic"):
    """
    Menggabungkan >=2 spektrum replikat (mis. beberapa rimpang jahe
    merah berbeda) menjadi satu spektrum representatif, sekaligus
    mengembalikan ukuran variasi alami di dalam kelompok tersebut.

    Parameters
    ----------
    spectra_list     : list of array — tiap elemen satu spektrum replikat
    wavenumber_list  : list of array atau None.
                       - Jika None: semua spektrum diasumsikan SUDAH
                         berada di grid wavenumber yang sama persis
                         (sama seperti asumsi yang sudah dipakai di
                         seluruh mcr_engine.py untuk data satu file
                         upload — lihat tab_input, semua kolom share
                         satu kolom wavenumber).
                       - Jika diberikan: tiap spektrum boleh punya grid
                         wavenumber berbeda (mis. diambil dari entri
                         library terpisah); akan di-resample ke grid
                         yang paling sempit overlap-nya, memakai
                         build_common_grid()/resample_to_grid() gaya
                         yang sama dengan modul matching yang sudah ada.
    interp_method    : metode interpolasi untuk resampling ("cubic"/"linear")

    Returns
    -------
    mean_spectrum : (n_points,) — rata-rata, dipakai sebagai referensi
                    (S_init / fixed_spectra), pengganti langsung dari
                    "1 spektrum tunggal" yang selama ini dipakai.
    std_spectrum  : (n_points,) — simpangan baku per titik wavenumber,
                    ukuran variasi ALAMI di dalam kelompok (bukan noise
                    alat, tapi variasi biologis antar-rimpang/replikat).
    cv_spectrum   : (n_points,) — coefficient of variation (%) per titik,
                    berguna untuk melihat region mana yang paling
                    stabil vs paling bervariasi antar replikat.
    info          : dict — n_replicates, common_wavenumber (jika ada
                    resampling), warning jika replikat terlalu sedikit
                    atau overlap grid terlalu sempit.
    """
    n = len(spectra_list)
    info = {"n_replicates": n, "warnings": []}

    if n < 2:
        info["warnings"].append(
            "⚠️ Hanya 1 spektrum — tidak ada variasi replikat yang bisa "
            "dihitung. std_spectrum akan bernilai nol di semua titik; "
            "hasil ini setara dengan memakai 1 spektrum tunggal seperti "
            "alur yang sudah ada sebelumnya."
        )
        single = np.asarray(spectra_list[0], dtype=float)
        return single.copy(), np.zeros_like(single), np.zeros_like(single), info

    if n < 3:
        info["warnings"].append(
            "⚠️ Hanya 2 replikat — std masih bisa dihitung tapi estimasi "
            "variasi alami akan kurang stabil. Disarankan >=3 replikat "
            "per kelompok (sesuai rancangan eksperimen: >=5-10 idealnya)."
        )

    # ── Samakan grid wavenumber jika diperlukan ───────────────
    if wavenumber_list is None:
        matrix = np.array([np.asarray(s, dtype=float) for s in spectra_list])
        common_wn = None
    else:
        # Import lokal supaya file ini tetap bisa dipakai berdiri
        # sendiri (mis. untuk unit test) tanpa wajib ada mcr_engine.py
        # di path yang sama; kalau dipasang di dalam proyek yang sama,
        # baris ini otomatis memakai fungsi yang sudah ada dan teruji.
        try:
            from mcr_engine import build_common_grid, resample_to_grid
        except ImportError as e:
            raise ImportError(
                "wavenumber_list diberikan (grid berbeda-beda) tapi "
                "build_common_grid/resample_to_grid dari mcr_engine.py "
                "tidak ditemukan. Pastikan file ini berada di folder yang "
                "sama dengan mcr_engine.py, atau berikan spektrum yang "
                "sudah berada di grid yang sama (wavenumber_list=None)."
            ) from e

        wn_arrays = [np.asarray(w, dtype=float) for w in wavenumber_list]
        common = wn_arrays[0]
        for wn_i in wn_arrays[1:]:
            common, grid_info = build_common_grid(common, wn_i)
            if common is None:
                raise ValueError(
                    "Tidak ada overlap wavenumber yang cukup antar salah "
                    "satu pasangan replikat — periksa range wavenumber "
                    "tiap entri library."
                )
        resampled = []
        for wn_i, sp_i in zip(wn_arrays, spectra_list):
            resampled.append(resample_to_grid(wn_i, np.asarray(sp_i, dtype=float),
                                               common, interp_method))
        matrix = np.array(resampled)
        common_wn = common
        info["common_wavenumber_n_points"] = len(common)

    mean_spectrum = matrix.mean(axis=0)
    std_spectrum = matrix.std(axis=0, ddof=1)  # ddof=1: sample std, replikat = sampel

    # CV dihitung relatif terhadap |mean| + offset kecil agar tidak
    # meledak di titik mean~0 (pola alpha-offset yang sama dipakai di
    # simplisma() pada mcr_engine.py untuk alasan yang sama)
    alpha = 0.01 * np.max(np.abs(mean_spectrum)) if np.max(np.abs(mean_spectrum)) > 0 else 1e-10
    cv_spectrum = (std_spectrum / (np.abs(mean_spectrum) + alpha)) * 100

    if common_wn is not None:
        info["common_wavenumber"] = common_wn

    return mean_spectrum, std_spectrum, cv_spectrum, info


# ════════════════════════════════════════════════════════════════
# 2. UJI KETERPISAHAN ANTAR-KELOMPOK (signal vs within-group noise)
# ════════════════════════════════════════════════════════════════

def assess_group_separability(group_a_spectra, group_b_spectra, wavenumber,
                               name_a="Grup A", name_b="Grup B",
                               snr_threshold_candidate=1.0,
                               snr_threshold_strong=3.0,
                               min_region_width_pts=3):
    """
    Menilai apakah dua kelompok replikat (mis. jahe merah vs jahe
    emprit) memiliki perbedaan spektral yang MELEBIHI variasi alami
    di dalam masing-masing kelompok — jawaban langsung atas
    keterbatasan yang tercatat sebelumnya (analisis 1-lawan-1
    sebelumnya hanya menunjukkan hipotesis kandidat, belum diuji
    terhadap variasi biologis replikat).

    Asumsi: group_a_spectra dan group_b_spectra SUDAH berada pada
    grid wavenumber yang sama (satu file upload dengan kolom-kolom
    replikat, sama seperti asumsi seluruh tab_mcr yang sudah ada).
    Kalau grid berbeda-beda antar entri, resample dulu memakai
    build_common_grid()/resample_to_grid() dari mcr_engine.py sebelum
    memanggil fungsi ini.

    Parameters
    ----------
    group_a_spectra, group_b_spectra : list of array — replikat tiap kelompok
    wavenumber        : array — grid wavenumber bersama (n_points,)
    name_a, name_b     : label kelompok untuk pelaporan
    snr_threshold_candidate : ambang signal/noise minimum untuk disebut
                       "kandidat region pembeda" (default 1.0 — selisih
                       rata-rata setara 1x gabungan simpangan baku)
    snr_threshold_strong    : ambang untuk "pembeda kuat" (default 3.0,
                       longgar setara aturan 3-sigma sederhana)
    min_region_width_pts    : lebar minimum (titik grid berurutan) agar
                       suatu region dilaporkan sebagai satu kandidat,
                       bukan titik tunggal yang mungkin cuma noise satu
                       piksel

    Returns
    -------
    dict berisi:
      snr_per_point      : (n_points,) rasio |selisih mean| / (std_a+std_b)
      mean_a, mean_b      : rata-rata tiap kelompok (dari average_replicate_spectra)
      std_a, std_b        : simpangan baku tiap kelompok
      candidate_regions   : list of dict {wn_start, wn_end, max_snr, peak_wn}
                            — region berurutan dengan snr >= threshold_candidate
      strong_regions      : subset candidate_regions dengan max_snr >= threshold_strong
      overall_verdict     : ringkasan teks (Indonesia) — apakah kedua kelompok
                            terpisahkan secara statistik atau tidak, dan
                            berapa banyak region kandidat ditemukan
      n_replicates_a, n_replicates_b, warnings : info replikasi (dari
                            average_replicate_spectra masing-masing kelompok)
    """
    wn = np.asarray(wavenumber, dtype=float)

    mean_a, std_a, _, info_a = average_replicate_spectra(group_a_spectra)
    mean_b, std_b, _, info_b = average_replicate_spectra(group_b_spectra)

    if mean_a.shape[0] != wn.shape[0]:
        raise ValueError(
            f"Panjang wavenumber ({wn.shape[0]}) tidak cocok dengan panjang "
            f"spektrum kelompok ({mean_a.shape[0]}). Pastikan semua replikat "
            f"sudah berada di grid wavenumber yang sama sebelum memanggil "
            f"fungsi ini."
        )

    diff = np.abs(mean_a - mean_b)
    noise = std_a + std_b
    noise_floor = 1e-10 * np.max(np.abs(np.concatenate([mean_a, mean_b])))
    noise_safe = np.maximum(noise, noise_floor if noise_floor > 0 else 1e-10)

    snr = diff / noise_safe

    # ── Kelompokkan titik-titik candidate jadi region berurutan ──
    def _find_regions(mask, snr_arr, wn_arr):
        regions = []
        i = 0
        n_pts = len(mask)
        while i < n_pts:
            if mask[i]:
                j = i
                while j < n_pts and mask[j]:
                    j += 1
                if (j - i) >= min_region_width_pts:
                    seg_snr = snr_arr[i:j]
                    seg_wn = wn_arr[i:j]
                    peak_local = int(np.argmax(seg_snr))
                    regions.append({
                        "wn_start": float(min(seg_wn[0], seg_wn[-1])),
                        "wn_end":   float(max(seg_wn[0], seg_wn[-1])),
                        "max_snr":  float(seg_snr[peak_local]),
                        "peak_wn":  float(seg_wn[peak_local]),
                        "n_points": int(j - i),
                    })
                i = j
            else:
                i += 1
        regions.sort(key=lambda r: -r["max_snr"])
        return regions

    candidate_mask = snr >= snr_threshold_candidate
    strong_mask = snr >= snr_threshold_strong

    candidate_regions = _find_regions(candidate_mask, snr, wn)
    strong_regions = _find_regions(strong_mask, snr, wn)

    n_candidate = len(candidate_regions)
    n_strong = len(strong_regions)

    if n_strong >= 1:
        verdict = (
            f"✅ {name_a} vs {name_b}: ditemukan {n_strong} region dengan "
            f"perbedaan KUAT (SNR≥{snr_threshold_strong}) — perbedaan spektral "
            f"melebihi variasi alami di dalam masing-masing kelompok replikat. "
            f"Region terkuat di sekitar {strong_regions[0]['peak_wn']:.1f} cm⁻¹ "
            f"(SNR={strong_regions[0]['max_snr']:.2f})."
        )
    elif n_candidate >= 1:
        verdict = (
            f"🟡 {name_a} vs {name_b}: ditemukan {n_candidate} region kandidat "
            f"(SNR≥{snr_threshold_candidate}) tapi belum ada yang mencapai "
            f"ambang kuat — perbedaan ada tapi tergolong marginal dibanding "
            f"variasi alami. Region terbaik di sekitar "
            f"{candidate_regions[0]['peak_wn']:.1f} cm⁻¹ "
            f"(SNR={candidate_regions[0]['max_snr']:.2f}). Pertimbangkan "
            f"menambah jumlah replikat untuk memperkuat estimasi."
        )
    else:
        verdict = (
            f"❌ {name_a} vs {name_b}: tidak ditemukan region dengan perbedaan "
            f"yang melebihi variasi alami replikat (SNR maksimum "
            f"{float(np.max(snr)):.2f} < ambang kandidat {snr_threshold_candidate}). "
            f"Berdasarkan data replikat ini, kedua kelompok TIDAK terpisahkan "
            f"secara statistik pada level noise saat ini — pemisahan MCR-ALS "
            f"tanpa constraint kuat (selectivity/library init) berisiko sangat "
            f"tinggi menghasilkan solusi rotational-ambiguous, bukan solusi "
            f"kimia yang benar."
        )

    return {
        "snr_per_point": snr,
        "mean_a": mean_a, "mean_b": mean_b,
        "std_a": std_a, "std_b": std_b,
        "candidate_regions": candidate_regions,
        "strong_regions": strong_regions,
        "overall_verdict": verdict,
        "n_replicates_a": info_a["n_replicates"],
        "n_replicates_b": info_b["n_replicates"],
        "warnings": info_a["warnings"] + info_b["warnings"],
    }


# ════════════════════════════════════════════════════════════════
# 3. VISUALISASI & DATA UNTUK LAPORAN (Plotly — sudah jadi dependency
#    aplikasi ini, TIDAK menambah library baru)
# ════════════════════════════════════════════════════════════════

def build_separability_dataframe(result, wavenumber, name_a="Kelompok A", name_b="Kelompok B"):
    """
    Susun hasil assess_group_separability() jadi tabel rapi (pandas
    DataFrame) — dipakai untuk sheet tambahan di fitur Laporan (tab_rep),
    supaya semua angka perhitungan (bukan cuma kesimpulan) ikut
    terekam untuk keperluan audit/reproduksi (mis. ISO 17025).

    Parameters
    ----------
    result     : dict — output dari assess_group_separability()
    wavenumber : array — grid wavenumber yang dipakai (harus sama
                 panjang dengan array di dalam result)
    name_a, name_b : label kolom kelompok

    Returns
    -------
    pandas.DataFrame dengan kolom:
        Wavenumber, Mean_<name_a>, Mean_<name_b>, Std_<name_a>,
        Std_<name_b>, SNR
    """
    import pandas as pd
    return pd.DataFrame({
        "Wavenumber (cm-1)":              np.asarray(wavenumber, dtype=float),
        f"Mean_{name_a}":                 result["mean_a"],
        f"Mean_{name_b}":                 result["mean_b"],
        f"Std_{name_a}":                  result["std_a"],
        f"Std_{name_b}":                  result["std_b"],
        "SNR (|selisih| / gabungan std)": result["snr_per_point"],
    })


def build_separability_regions_dataframe(result):
    """
    Ringkasan region kandidat & kuat dalam bentuk tabel — untuk sheet
    "Ringkasan Region" di laporan, memuat rentang bilangan gelombang,
    puncak SNR, dan kategori (kuat/kandidat) untuk tiap region.

    Returns
    -------
    pandas.DataFrame dengan kolom:
        Kategori, Rentang (cm-1), Puncak SNR di (cm-1), Nilai SNR Puncak
    """
    import pandas as pd
    rows = []
    strong_wn = {r["peak_wn"] for r in result["strong_regions"]}
    for r in result["strong_regions"]:
        rows.append({
            "Kategori": "Kuat (strong)",
            "Rentang (cm-1)": f"{r['wn_start']:.0f}–{r['wn_end']:.0f}",
            "Puncak SNR di (cm-1)": round(r["peak_wn"], 1),
            "Nilai SNR Puncak": round(r["max_snr"], 3),
        })
    for r in result["candidate_regions"]:
        if r["peak_wn"] in strong_wn:
            continue  # sudah masuk sebagai "strong", jangan duplikat
        rows.append({
            "Kategori": "Kandidat (belum kuat)",
            "Rentang (cm-1)": f"{r['wn_start']:.0f}–{r['wn_end']:.0f}",
            "Puncak SNR di (cm-1)": round(r["peak_wn"], 1),
            "Nilai SNR Puncak": round(r["max_snr"], 3),
        })
    if not rows:
        rows.append({
            "Kategori": "Tidak ada region signifikan",
            "Rentang (cm-1)": "-", "Puncak SNR di (cm-1)": "-", "Nilai SNR Puncak": "-",
        })
    return pd.DataFrame(rows)


def build_separability_figure(result, wavenumber, name_a="Kelompok A", name_b="Kelompok B"):
    """
    Chart 3-panel (Plotly) yang sama konsepnya dengan analisis awal
    jahe merah vs emprit: (1) rata-rata tiap kelompok, (2) pita
    ±1 std di sekitar rata-rata (menunjukkan sebaran replikat), dan
    (3) kurva SNR per titik dengan region kuat/kandidat ditandai.

    Memakai plotly.graph_objects — SUDAH jadi dependency aplikasi ini
    (dipakai di banyak tab lain), jadi TIDAK menambah library baru
    atau baris apa pun di requirements.txt.

    Returns
    -------
    plotly.graph_objects.Figure — siap ditampilkan dengan
    st.plotly_chart(fig, use_container_width=True)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    wn = np.asarray(wavenumber, dtype=float)
    mean_a, mean_b = result["mean_a"], result["mean_b"]
    std_a, std_b = result["std_a"], result["std_b"]
    snr = result["snr_per_point"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=(
            f"Rata-rata {name_a} vs {name_b} (± 1 std antar replikat)",
            f"Sebaran replikat (pita ±1 std)",
            "Kurva SNR per titik — area kuning/hijau = region signifikan",
        ),
        vertical_spacing=0.08,
    )

    # Panel 1 — mean overlay
    fig.add_trace(go.Scatter(x=wn, y=mean_a, name=name_a, mode="lines",
                             line=dict(color="#b3282d", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=wn, y=mean_b, name=name_b, mode="lines",
                             line=dict(color="#2d7a3a", width=1.5)), row=1, col=1)

    # Panel 2 — mean ± std band (pakai fill 'toself' per kelompok)
    fig.add_trace(go.Scatter(
        x=np.concatenate([wn, wn[::-1]]),
        y=np.concatenate([mean_a + std_a, (mean_a - std_a)[::-1]]),
        fill="toself", fillcolor="rgba(179,40,45,0.2)",
        line=dict(color="rgba(0,0,0,0)"), name=f"{name_a} ±1 std",
        showlegend=True
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=np.concatenate([wn, wn[::-1]]),
        y=np.concatenate([mean_b + std_b, (mean_b - std_b)[::-1]]),
        fill="toself", fillcolor="rgba(45,122,58,0.2)",
        line=dict(color="rgba(0,0,0,0)"), name=f"{name_b} ±1 std",
        showlegend=True
    ), row=2, col=1)
    fig.add_trace(go.Scatter(x=wn, y=mean_a, mode="lines",
                             line=dict(color="#b3282d", width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=wn, y=mean_b, mode="lines",
                             line=dict(color="#2d7a3a", width=1), showlegend=False), row=2, col=1)

    # Panel 3 — SNR curve + shaded regions
    fig.add_trace(go.Scatter(x=wn, y=snr, mode="lines",
                             line=dict(color="#94a3b8", width=1), name="SNR"), row=3, col=1)
    for r in result["strong_regions"]:
        fig.add_vrect(x0=r["wn_start"], x1=r["wn_end"],
                      fillcolor="gold", opacity=0.35, line_width=0, row=3, col=1)
    for r in result["candidate_regions"]:
        if r not in result["strong_regions"]:
            fig.add_vrect(x0=r["wn_start"], x1=r["wn_end"],
                          fillcolor="lightgreen", opacity=0.2, line_width=0, row=3, col=1)

    fig.update_xaxes(autorange="reversed", title_text="Bilangan gelombang (cm⁻¹)", row=3, col=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        height=700, margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig
