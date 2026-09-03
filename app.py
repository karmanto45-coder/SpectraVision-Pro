import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime

from auth import render_login, is_logged_in, is_admin, logout
from database import (init_db, add_spectrum, delete_spectrum,
                      update_spectrum_meta, get_all_meta,
                      get_spectrum_by_id, get_all_spectra_for_matching,
                      count_spectra, get_categories, import_from_json,
                      add_derivative_spectrum, delete_derivative_spectrum,
                      get_all_derivative_meta, get_derivative_spectrum_by_id,
                      get_all_derivative_spectra_for_matching,
                      count_derivative_spectra)
from mcr_engine import (preprocess, detect_components, run_mcr_als,
                        run_mcr_multi_k, generate_warnings, compute_scorecard,
                        postprocess_mcr_spectra,
                        batch_match, consensus_label, interpolate_spectrum,
                        apply_window,
                        apply_derivative, validate_derivative_params,
                        batch_match_derivative, peak_position_match,
                        compare_two_spectra, pearson_corr, cosine_sim)
from ambiguity_engine import compute_rotational_ambiguity, K_EXACT_LIMIT
from identity_decision_engine import (compute_identity_robustness_index,
                                      evaluate_component_identity,
                                      build_identity_decision_dataframe)
from similarity_simulator import (run_correlation_noise_study,
                                  build_correlation_noise_study_dataframe,
                                  generate_correlated_pure_spectra,
                                  generate_independent_spectrum,
                                  generate_dirichlet_design,
                                  simulate_linear_mixture,
                                  DEFAULT_CASES)
from robustness_sweep_engine import (run_robustness_grid,
                                     run_noise_bootstrap_robustness,
                                     summarize_robustness,
                                     build_robustness_dataframe,
                                     PREPROCESSING_VARIANTS)
from blind_validation_engine import (run_blind_validation_study,
                                     build_blind_validation_dataframe)
from cos2d import (compute_2dcos, find_crosspeaks, apply_nodas_rules,
                   PERTURBATION_PRESETS)


# ── External spectrum parsing helpers (fitur Perbandingan Manual Admin) ─
# Dipakai untuk membaca spektra yang diinput admin secara manual (bukan
# dari MCR maupun library), baik lewat paste teks maupun upload file.

def _parse_pasted_spectrum(raw_text: str):
    """
    Parse teks yang di-paste admin menjadi (wavenumber, intensitas).
    Menerima pemisah koma, titik-koma, tab, atau spasi — satu pasang
    angka per baris. Baris yang tidak bisa dikonversi ke dua angka
    (mis. header kolom) dilewati saja, tidak menggagalkan seluruh parsing.
    """
    import re
    wn_list, sp_list = [], []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[,;\t]+|\s+", line) if p != ""]
        if len(parts) < 2:
            continue
        try:
            w, s = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        wn_list.append(w)
        sp_list.append(s)
    if len(wn_list) < 5:
        raise ValueError(
            "Kurang dari 5 titik data valid ditemukan / "
            "Fewer than 5 valid data points found."
        )
    return np.array(wn_list, dtype=float), np.array(sp_list, dtype=float)


def _parse_uploaded_spectrum(uploaded_file):
    """
    Parse file yang diupload admin (CSV/TXT/Excel) menjadi
    (wavenumber, intensitas). Mengambil dua kolom numerik pertama;
    baris header non-numerik otomatis dilewati.
    """
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, header=None)
    else:
        raw_bytes = uploaded_file.read()
        text = raw_bytes.decode("utf-8", errors="ignore")
        sep = None
        try:
            import csv as _csv
            sep = _csv.Sniffer().sniff(text[:2048], delimiters=",;\t ").delimiter
        except Exception:
            sep = None
        try:
            df = pd.read_csv(io.StringIO(text), header=None, sep=sep, engine="python")
        except Exception:
            df = pd.read_csv(io.StringIO(text), header=None, delim_whitespace=True)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError(
            "File harus memiliki minimal 2 kolom numerik / "
            "File must contain at least 2 numeric columns."
        )
    df2 = df.iloc[:, :2].dropna()
    if len(df2) < 5:
        raise ValueError(
            "Kurang dari 5 baris numerik valid ditemukan / "
            "Fewer than 5 valid numeric rows found."
        )
    wn = df2.iloc[:, 0].to_numpy(dtype=float)
    sp = df2.iloc[:, 1].to_numpy(dtype=float)
    return wn, sp


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SpectraVision Pro",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
[data-testid="stToolbar"] {visibility:hidden;}
a[href*="github"] {display:none !important;}
.stDeployButton {display:none !important;}

.app-header{
  background:linear-gradient(135deg,#0d1117,#131c2e);
  border:1px solid #1e3a5f;border-radius:14px;
  padding:1.2rem 1.8rem;margin-bottom:1.2rem;
}
.app-title{
  font-family:'DM Mono',monospace;font-size:1.5rem;font-weight:500;
  color:#e2e8f0;margin:0;letter-spacing:-0.5px;
}
.app-sub{color:#64748b;font-size:0.82rem;margin:3px 0 0;}
.badge{display:inline-block;font-size:0.68rem;padding:2px 8px;
  border-radius:4px;font-family:'DM Mono',monospace;margin-left:8px;}
.badge-admin{background:#1e0a3c;color:#c084fc;}
.badge-user{background:#0a1e2a;color:#7dd3fc;}
.badge-version{background:#0f2a1a;color:#4ade80;}
.metric-card{background:#161b27;border:1px solid #2a3142;
  border-radius:10px;padding:0.9rem 1.1rem;text-align:center;}
.metric-value{font-family:'DM Mono',monospace;font-size:1.5rem;
  font-weight:500;color:#7dd3fc;}
.metric-label{font-size:0.72rem;color:#64748b;margin-top:2px;
  text-transform:uppercase;letter-spacing:0.05em;}
.sec-hdr{font-family:'DM Mono',monospace;font-size:0.68rem;color:#475569;
  text-transform:uppercase;letter-spacing:0.1em;margin:1.2rem 0 0.6rem;
  padding-bottom:5px;border-bottom:1px solid #1e293b;}
.match-card{border-radius:10px;padding:0.8rem 1rem;
  margin-bottom:0.45rem;border-left:3px solid;}
.m-strong{background:#0d2018;border-color:#22c55e;}
.m-medium{background:#1a1a08;border-color:#eab308;}
.m-conflict{background:#12100d;border-color:#f97316;}
.m-weak{background:#1a0a08;border-color:#ef4444;}
.m-name{font-weight:500;color:#e2e8f0;font-size:0.92rem;}
.m-scores{font-family:'DM Mono',monospace;font-size:0.78rem;color:#94a3b8;margin-top:3px;}
.window-chip{display:inline-block;background:#1e293b;border:1px solid #334155;
  border-radius:6px;padding:3px 10px;font-family:'DM Mono',monospace;
  font-size:0.76rem;color:#7dd3fc;margin-right:6px;}
.proc-box{background:#0d1829;border:1px solid #1e3a5f;border-radius:10px;
  padding:0.9rem 1.2rem;margin-bottom:0.8rem;}
.proc-box-title{font-family:'DM Mono',monospace;font-size:0.72rem;
  color:#7dd3fc;text-transform:uppercase;letter-spacing:0.08em;
  margin-bottom:0.5rem;}
.step-badge{display:inline-block;background:#1e3a5f;color:#7dd3fc;
  border-radius:50%;width:20px;height:20px;text-align:center;
  line-height:20px;font-size:0.7rem;font-weight:600;margin-right:6px;}
</style>
""", unsafe_allow_html=True)

# ── Init ──────────────────────────────────────────────────────
init_db()

# ── Auth gate ─────────────────────────────────────────────────
if not is_logged_in():
    render_login()
    st.stop()

# ── Language helper ───────────────────────────────────────────
lang = st.session_state.get("lang", "id")
def t(id_text, en_text):
    return en_text if lang == "en" else id_text

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    role  = st.session_state.get("role", "user")
    uname = st.session_state.get("display_name", "User")
    badge = "badge-admin" if role == "admin" else "badge-user"
    blabel = "Admin" if role == "admin" else "User"
    st.markdown(f"""
    <div style="padding:0.5rem 0 1rem;">
      <p style="font-family:'DM Mono',monospace;font-size:1rem;
         color:#e2e8f0;margin:0;">{uname}
        <span class="badge {badge}">{blabel}</span>
      </p>
      <p style="font-size:0.75rem;color:#475569;margin:2px 0 0;">
        {st.session_state.get('username','')}
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<p class="sec-hdr">{t("Statistik","Stats")}</p>', unsafe_allow_html=True)
    n_lib = count_spectra()
    st.markdown(f'<div class="metric-card"><div class="metric-value">{n_lib:,}</div>'
                f'<div class="metric-label">{t("Spektra library","Library spectra")}</div></div>',
                unsafe_allow_html=True)

    st.markdown("")
    lang_choice = st.selectbox("🌐 Language",
        ["🇮🇩 Bahasa Indonesia", "🇬🇧 English"],
        index=0 if lang == "id" else 1)
    st.session_state["lang"] = "en" if "English" in lang_choice else "id"

    st.markdown("---")
    if st.button(t("Keluar","Logout"), use_container_width=True):
        logout()
        st.rerun()

# ── Header ────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <p class="app-title">SpectraVision Pro
    <span class="badge badge-version">v3.0</span>
    <span class="badge badge-admin" style="font-size:0.62rem;">New</span>
  </p>
  <p class="app-sub">
    {t("Multivariate Curve Resolution · Post-MCR Processing · Identifikasi Spektra ATR-FTIR",
       "Multivariate Curve Resolution · Post-MCR Processing · ATR-FTIR Spectral Identification")}
  </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab_labels = (
    [t("📂 Input Data","📂 Input Data"),
     t("🔬 Analisis MCR","🔬 MCR Analysis"),
     t("✨ Proses Spektra MCR","✨ Process MCR Spectra"),
     t("🔍 Identifikasi","🔍 Identification"),
     t("📈 2D-COS","📈 2D-COS"),
     t("🧬 Analisis Turunan","🧬 Derivative Analysis"),
     t("📚 Library","📚 Library"),
     t("⚙️ Admin","⚙️ Admin"),
     t("🧪 Studi Riset","🧪 Research Studies"),
     t("📊 Laporan","📊 Report")]
    if is_admin() else
    [t("📂 Input Data","📂 Input Data"),
     t("🔬 Analisis MCR","🔬 MCR Analysis"),
     t("✨ Proses Spektra MCR","✨ Process MCR Spectra"),
     t("🔍 Identifikasi","🔍 Identification"),
     t("📈 2D-COS","📈 2D-COS"),
     t("🧬 Analisis Turunan","🧬 Derivative Analysis"),
     t("📊 Laporan","📊 Report")]
)

tabs = st.tabs(tab_labels)
tab_input   = tabs[0]
tab_mcr     = tabs[1]
tab_postmcr = tabs[2]
tab_match   = tabs[3]
tab_cos     = tabs[4]
tab_deriv   = tabs[5]
tab_lib      = tabs[6] if is_admin() else None
tab_admin    = tabs[7] if is_admin() else None
tab_research = tabs[8] if is_admin() else None
tab_rep     = tabs[9] if is_admin() else tabs[6]

# ════════════════════════════════════════════════════════════════
# TAB 1 — INPUT DATA
# ════════════════════════════════════════════════════════════════
with tab_input:
    st.markdown(f'<p class="sec-hdr">{t("Upload data spektra","Upload spectral data")}</p>',
                unsafe_allow_html=True)

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            t("Upload file (Excel / CSV / TXT)","Upload file (Excel / CSV / TXT)"),
            type=["xlsx","xls","csv","txt","jdx","dx"]
        )
    with col_info:
        st.info(t(
            "**Format kolom:**\nKolom 1 = wavenumber (cm⁻¹)\nKolom 2+ = spektra sampel\nMinimum 4 spektra",
            "**Column format:**\nCol 1 = wavenumber (cm⁻¹)\nCol 2+ = sample spectra\nMinimum 4 spectra"
        ))

    if not uploaded:
        if st.session_state.get("_uploaded_filename"):
            for _k in ["wavenumber","spectra","spec_names","_uploaded_filename",
                       "mcr_C","mcr_S","mcr_lof","mcr_r2","mcr_ncomp",
                       "mcr_converged","mcr_S_proc","mcr_proc_log",
                       "match_results","cos2d_result","cos2d_perturb",
                       "cos2d_unit","cos2d_name"]:
                st.session_state.pop(_k, None)
            st.info(t("File dihapus — semua hasil analisis direset.",
                      "File removed — all analysis results have been reset."))

    if uploaded:
        try:
            name = uploaded.name.lower()
            if name.endswith((".xlsx",".xls")):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded, sep=None, engine="python", comment="#")

            wn_col    = df.columns[0]
            spec_cols = df.columns[1:]
            wavenumber = df[wn_col].values.astype(float)
            raw_matrix = df[spec_cols].values.astype(float)

            n_spec = len(spec_cols)
            n_pts  = len(wavenumber)

            c1,c2,c3,c4 = st.columns(4)
            for col, val, lbl in zip(
                [c1,c2,c3,c4],
                [n_spec, n_pts, wavenumber.min(), wavenumber.max()],
                [t("Jumlah spektra","Spectra count"),
                 t("Titik data","Data points"),
                 t("Wavenum. min","Wavenum. min"),
                 t("Wavenum. max","Wavenum. max")]
            ):
                col.markdown(
                    f'<div class="metric-card"><div class="metric-value">{val:.0f}</div>'
                    f'<div class="metric-label">{lbl}</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown(f'<p class="sec-hdr">{t("Pra-pemrosesan input","Input preprocessing")}</p>',
                        unsafe_allow_html=True)
            p1,p2,p3 = st.columns(3)
            do_norm     = p1.checkbox(t("Normalisasi","Normalize"), value=True)
            do_smooth   = p2.checkbox(t("Smoothing (SG)","Smoothing (SG)"), value=False)
            do_baseline = p3.checkbox(t("Koreksi baseline","Baseline correction"), value=False)

            proc = preprocess(raw_matrix, wavenumber, do_norm, do_smooth, do_baseline)

            prev_file = st.session_state.get("_uploaded_filename", None)
            curr_file = uploaded.name + str(uploaded.size)
            is_new_file = (prev_file != curr_file)

            st.session_state["wavenumber"]  = wavenumber
            st.session_state["spectra"]     = proc
            st.session_state["spec_names"]  = list(spec_cols.astype(str))
            st.session_state["_uploaded_filename"] = curr_file

            if is_new_file:
                for _k in ["mcr_C","mcr_S","mcr_lof","mcr_r2","mcr_ncomp",
                           "mcr_converged","mcr_S_proc","mcr_proc_log",
                           "match_results","cos2d_result","cos2d_perturb"]:
                    st.session_state.pop(_k, None)

            st.markdown(f'<p class="sec-hdr">{t("Visualisasi spektra","Spectral visualization")}</p>',
                        unsafe_allow_html=True)
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, col in enumerate(spec_cols):
                fig.add_trace(go.Scatter(
                    x=wavenumber, y=proc[:,i],
                    name=str(col), mode="lines",
                    line=dict(width=1.2, color=colors[i % len(colors)])
                ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(autorange="reversed", gridcolor="#1e293b",
                           title=t("Wavenumber (cm⁻¹)","Wavenumber (cm⁻¹)")),
                yaxis=dict(gridcolor="#1e293b", title="Absorbance"),
                legend=dict(
                    title=dict(text=t("Spektra Sampel","Sample Spectra"),
                               font=dict(size=11, color="#7dd3fc")),
                    bgcolor="#161b27", bordercolor="#2a3142", borderwidth=1,
                    font=dict(size=11, color="#e2e8f0"),
                    orientation="v", x=1.02, xanchor="left", y=1
                ),
                height=370, margin=dict(l=20,r=160,t=20,b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            if n_spec < 4:
                st.warning(t(f"⚠️ Hanya {n_spec} spektra. Minimum rekomendasi: 4 spektra.",
                             f"⚠️ Only {n_spec} spectra. Recommended minimum: 4."))
            elif n_spec < 10:
                st.warning(t("⚠️ Data cukup untuk analisis, tapi disarankan 10+ spektra.",
                             "⚠️ Sufficient for analysis, but 10+ spectra recommended."))
            else:
                st.success(t(f"✅ {n_spec} spektra siap dianalisis.",
                             f"✅ {n_spec} spectra ready for analysis."))
        except Exception as e:
            st.error(f"Error: {e}")

# ════════════════════════════════════════════════════════════════
# TAB 2 — MCR ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab_mcr:
    if "spectra" not in st.session_state:
        st.info(t("Upload data spektra di tab Input Data terlebih dahulu.",
                  "Please upload spectral data in the Input Data tab first."))
    else:
        wn = st.session_state["wavenumber"]
        D  = st.session_state["spectra"].T

        st.markdown(f'<p class="sec-hdr">{t("Deteksi komponen (PCA)","Component detection (PCA)")}</p>',
                    unsafe_allow_html=True)

        ev, cum, auto_k, sens_k, ind_vals, ev_abs = detect_components(D)
        fig_pca = make_subplots(rows=1, cols=2,
            subplot_titles=(
                t("Variansi tiap komponen (%)","Variance per component (%)"),
                t("Variansi kumulatif (%)","Cumulative variance (%)")
            ))
        fig_pca.add_trace(go.Bar(x=list(range(1,len(ev)+1)), y=ev,
            marker_color="#7dd3fc", name="Var%"), row=1, col=1)
        fig_pca.add_trace(go.Scatter(x=list(range(1,len(cum)+1)), y=cum,
            mode="lines+markers", line=dict(color="#f97316"), name="Cum%"), row=1, col=2)
        fig_pca.add_hline(y=95, line_dash="dash", line_color="#475569",
            annotation_text="95%", row=1, col=2)
        fig_pca.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117", height=260, showlegend=False,
            margin=dict(l=20,r=20,t=40,b=20))
        fig_pca.update_xaxes(gridcolor="#1e293b")
        fig_pca.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption(t(f"Saran otomatis PCA: **{auto_k} komponen** (≥95% variansi) | Sensitivity: **{sens_k} komponen**",
                     f"PCA suggestion: **{auto_k} components** (≥95% variance) | Sensitivity: **{sens_k} components**"))

        st.markdown(f'<p class="sec-hdr">{t("Parameter MCR-ALS","MCR-ALS parameters")}</p>',
                    unsafe_allow_html=True)
        a1,a2,a3,a4 = st.columns(4)
        n_comp   = a1.number_input(t("Jumlah komponen","Components"), 2, 10, auto_k)
        max_iter = a2.number_input(t("Iterasi max","Max iterations"), 50, 1000, 200, step=50)
        tol      = a3.selectbox(t("Toleransi","Tolerance"),
                       [1e-4,1e-5,1e-6,1e-7], index=2,
                       format_func=lambda x: f"{x:.0e}")
        closure  = a4.checkbox(t("Closure constraint","Closure constraint"), value=False,
                       help=t("Paksa jumlah fraksi konsentrasi per sampel = 1. "
                              "Hanya aktifkan jika total komposisi sampel memang "
                              "diketahui/didesain sama dengan 1 (mis. rasio campuran). "
                              "JANGAN dikombinasikan dengan 'Normalisasi S per iterasi' "
                              "— keduanya mengunci skala C dan S secara bertentangan "
                              "dan terbukti bisa menjebak ALS pada solusi yang salah.",
                              "Force each sample's concentration fractions to sum to 1. "
                              "Enable only if the sample's total composition is known/"
                              "designed to equal 1 (e.g. mixture ratio design). "
                              "Do NOT combine with 'Normalize S per iteration' — both "
                              "lock the C/S scale in conflicting ways and have been "
                              "shown to trap the ALS in a wrong solution."))

        b1, b2, b3 = st.columns(3)

        # ── Metode inisialisasi ───────────────────────────────
        init_method = b1.radio(
            t("Metode inisialisasi","Initialization method"),
            ["simplisma", "pca", "nmf"],
            format_func=lambda x: {
                "simplisma": t("SIMPLISMA — direkomendasikan (default Unscrambler-style)",
                               "SIMPLISMA — recommended (Unscrambler-style default)"),
                "pca": t("PCA — umum, cepat","PCA — general, fast"),
                "nmf": t("NMF-NNDSVD — lebih baik untuk spektroskopi",
                         "NMF-NNDSVD — better for spectroscopy"),
            }[x],
            key="mcr_init_method",
            help=t(
                "SIMPLISMA (Windig & Guilment 1991): memilih 'purest variables' "
                "langsung dari data nyata sebagai estimasi awal C. Ini pendekatan "
                "yang paling mendekati inisialisasi default software MCR-ALS "
                "komersial (termasuk Unscrambler), dan pada pengujian internal "
                "konsisten konvergen lebih cepat serta lebih dekat ke spektra murni "
                "asli dibanding PCA. PCA: inisialisasi standar, cepat, tapi rawan "
                "rotational ambiguity pada dataset dengan sedikit sampel. "
                "NMF-NNDSVD: non-negatif dari awal tanpa koreksi.",
                "SIMPLISMA (Windig & Guilment 1991): selects 'purest variables' "
                "directly from the real data as the initial C estimate. This is "
                "the closest match to how commercial MCR-ALS software (including "
                "Unscrambler) initializes by default, and in internal testing "
                "converges faster and closer to the true pure spectra than PCA. "
                "PCA: standard, fast, but prone to rotational ambiguity on "
                "small-sample datasets. NMF-NNDSVD: non-negative from the start."
            )
        )

        # ── Constraint tambahan ───────────────────────────────
        unimodal    = b1.checkbox(
            t("Unimodality pada spektra (S)","Unimodality on spectra (S)"), value=False,
            help=t("⚠️ Paksa spektra murni memiliki SATU puncak saja. Spektra "
                   "FTIR/UV-Vis pada umumnya multi-puncak (banyak gugus fungsi) — "
                   "mengaktifkan ini akan MENGHANCURKAN puncak-puncak lain yang "
                   "sah. Gunakan hanya jika komponen memang secara fisik "
                   "unimodal. Untuk kebanyakan kasus spektroskopi, gunakan "
                   "'Unimodality pada konsentrasi (C)' di bawah, bukan ini.",
                   "⚠️ Forces the pure spectrum to have a SINGLE peak. FTIR/UV-Vis "
                   "spectra normally have multiple bands (multiple functional "
                   "groups) — enabling this will DESTROY other genuine peaks. "
                   "Use only if the component is physically unimodal. For most "
                   "spectroscopy cases, use 'Unimodality on concentration (C)' "
                   "below instead."))
        unimodal_C  = b1.checkbox(
            t("Unimodality pada konsentrasi (C)","Unimodality on concentration (C)"),
            value=False,
            help=t("Cocok untuk seri sampel berurutan (mis. seri rasio/dilusi "
                   "yang meningkat monoton). Tidak merusak struktur multi-puncak "
                   "spektra, karena constraint ini bekerja pada profil "
                   "konsentrasi tiap komponen, bukan pada spektranya.",
                   "Appropriate for ordered sample series (e.g. a monotonically "
                   "increasing ratio/dilution series). Does not damage the "
                   "spectra's multi-peak structure, since this constraint acts "
                   "on each component's concentration profile, not its spectrum."))
        normalize_S = b1.checkbox(
            t("Normalisasi S per iterasi","Normalize S per iteration"), value=False,
            help=t("Normalisasi unit vector S setiap iterasi. "
                   "Aktifkan hanya jika skala komponen sangat berbeda. "
                   "JANGAN dikombinasikan dengan Closure constraint (lihat "
                   "peringatan di atas).",
                   "Unit-vector normalize S each iteration. "
                   "Enable only if component scales differ greatly. "
                   "Do NOT combine with Closure constraint (see warning above)."))
        smooth_S    = b1.checkbox(
            t("Smoothing ringan pada S tiap iterasi","Light smoothing on S per iteration"),
            value=False,
            help=t("Savitzky-Golay ringan pada spektra tiap iterasi (non-negativity "
                   "tetap dijaga). Membantu bila satu komponen menyerap noise "
                   "residual dan tampak jauh lebih kasar/bergerigi dibanding "
                   "komponen lain.",
                   "Light Savitzky-Golay smoothing on the spectra each iteration "
                   "(non-negativity preserved). Helps when one component absorbs "
                   "residual noise and looks much rougher/noisier than the others."))

        if closure and normalize_S:
            st.warning(t(
                "⚠️ Closure constraint dan Normalisasi S per iterasi aktif "
                "bersamaan. Kombinasi ini terbukti (pada pengujian internal) "
                "dapat menjebak ALS pada solusi lokal yang salah (LOF melonjak "
                "tinggi meski status 'konvergen'). Nonaktifkan salah satu.",
                "⚠️ Closure constraint and Normalize S per iteration are both "
                "enabled. This combination has been shown (in internal testing) "
                "to trap the ALS in a wrong local solution (LOF spikes high even "
                "though it reports 'converged'). Disable one of them."))

        sensitivity = b2.slider(
            t("Sensitivity komponen minor","Minor component sensitivity"),
            10, 190, 100, step=10,
            help=t("Tinggi = lebih banyak komponen terdeteksi (termasuk minor). "
                   "Rendah = hanya komponen dominan.",
                   "High = more components detected (including minor). "
                   "Low = dominant components only."))
        _, _, _, sens_k, _, _ = detect_components(D, sensitivity=sensitivity)

        # Info box ringkas
        init_desc = {
            "simplisma": t("SIMPLISMA → purest variables → C awal",
                           "SIMPLISMA → purest variables → initial C"),
            "pca": t("PCA → shift-to-positive → C awal",
                     "PCA → shift-to-positive → initial C"),
            "nmf": t("NMF-NNDSVD → C & S non-negatif dari awal",
                     "NMF-NNDSVD → C & S non-negative from start"),
        }
        b3.info(
            t(f"Sensitivity → **{sens_k} komponen**",
              f"Sensitivity → **{sens_k} components**")
            + "  \n" + init_desc[init_method]
        )

        # ── Initial Guess dari Library ────────────────────────
        st.markdown(f'<p class="sec-hdr">{t("Initial guess spektra (opsional)","Spectral initial guess (optional)")}</p>',
                    unsafe_allow_html=True)

        use_init_guess = st.checkbox(
            t("Gunakan spektra library sebagai initial guess S",
              "Use library spectra as initial guess for S"),
            value=False, key="mcr_use_init_guess",
            help=t(
                "Pilih spektra murni dari library sebagai titik awal MCR-ALS. "
                "Mengurangi rotational ambiguity dan meningkatkan konsistensi hasil. "
                "Jumlah spektra yang dipilih harus sama dengan jumlah komponen.",
                "Select pure spectra from library as MCR-ALS starting point. "
                "Reduces rotational ambiguity and improves result consistency. "
                "Number of selected spectra must equal number of components."
            )
        )

        from mcr_replicate_extension import average_replicate_spectra, assess_group_separability

        use_multi_replicate = st.checkbox(
            "Gabungkan beberapa entri sebagai satu referensi (rata-rata replikat)",
            value=False, key="mcr_use_replicate_avg"
        )
        # Kalau tidak dicentang -> perilaku 100% sama seperti sebelumnya (default OFF)
        if use_multi_replicate:
            from mcr_replicate_extension import assess_group_separability

            spec_names_all = st.session_state.get("spec_names", [])
            spectra_all = st.session_state["spectra"]  # shape: (n_wavenumber x n_sampel)

            rcol1, rcol2 = st.columns(2)
            group_a_names = rcol1.multiselect(
                t("Pilih kolom Kelompok A (mis. semua replikat jahe merah)",
                  "Select Group A columns (e.g. all red ginger replicates)"),
                spec_names_all, key="mcr_group_a"
            )
            group_b_names = rcol2.multiselect(
                t("Pilih kolom Kelompok B (mis. semua replikat jahe emprit)",
                  "Select Group B columns (e.g. all white ginger replicates)"),
                spec_names_all, key="mcr_group_b"
            )

            if len(group_a_names) >= 2 and len(group_b_names) >= 2:
                idx_a = [spec_names_all.index(n) for n in group_a_names]
                idx_b = [spec_names_all.index(n) for n in group_b_names]
                group_a_spectra = [spectra_all[:, i] for i in idx_a]
                group_b_spectra = [spectra_all[:, i] for i in idx_b]

                result = assess_group_separability(
                    group_a_spectra, group_b_spectra, wn,
                    name_a="Kelompok A", name_b="Kelompok B"
                )

                st.markdown(result["overall_verdict"])
                from mcr_replicate_extension import build_separability_figure
                fig_sep = build_separability_figure(result, wn, "Kelompok A", "Kelompok B")
                st.plotly_chart(fig_sep, use_container_width=True)

                # Simpan untuk dipakai di fitur Laporan (tab Export data)
                st.session_state["replicate_sep_result"] = result
                st.session_state["replicate_sep_wn"] = wn
                st.session_state["replicate_sep_names"] = ("Kelompok A", "Kelompok B")            
                if result["strong_regions"]:
                    st.markdown(t("**Region pembeda kuat ditemukan:**",
                                  "**Strong distinguishing regions found:**"))
                    for reg in result["strong_regions"][:8]:
                        st.write(f"- {reg['wn_start']:.0f}–{reg['wn_end']:.0f} cm⁻¹ "
                                 f"(puncak SNR di {reg['peak_wn']:.0f} cm⁻¹, "
                                 f"SNR={reg['max_snr']:.2f})")
                elif result["candidate_regions"]:
                    st.markdown(t("**Region kandidat (belum kuat):**",
                                  "**Candidate regions (not yet strong):**"))
                    for reg in result["candidate_regions"][:8]:
                        st.write(f"- {reg['wn_start']:.0f}–{reg['wn_end']:.0f} cm⁻¹ "
                                 f"(SNR={reg['max_snr']:.2f})")
            else:
                st.info(t(
                    "Pilih minimal 2 kolom di tiap kelompok untuk menjalankan analisis.",
                    "Select at least 2 columns in each group to run the analysis."
                ))
        S_init_guess = None
        if use_init_guess:
            from database import get_all_spectra_for_matching, get_spectrum_by_id
            all_lib = get_all_spectra_for_matching()
            if not all_lib:
                st.warning(t("Library masih kosong. Tambahkan spektra referensi terlebih dahulu.",
                             "Library is empty. Please add reference spectra first."))
            else:
                lib_options = {f"{e['name']} [{e['category']}] (ID:{e['id']})": e['id']
                               for e in all_lib}
                selected_names = st.multiselect(
                    t(f"Pilih {n_comp} spektra dari library (harus = jumlah komponen)",
                      f"Select {n_comp} spectra from library (must equal components)"),
                    list(lib_options.keys()),
                    key="mcr_init_guess_select"
                )
                if selected_names:
                    n_selected = len(selected_names)
                    if n_selected != int(n_comp):
                        st.warning(t(
                            f"⚠️ Dipilih {n_selected} spektra, dibutuhkan {n_comp} "
                            f"(sama dengan jumlah komponen). Sesuaikan pilihan.",
                            f"⚠️ Selected {n_selected} spectra, need {n_comp} "
                            f"(equal to number of components). Adjust selection."
                        ))
                    else:
                        # Kumpulkan spektra dari library dan resample ke grid data
                        from mcr_engine import resample_to_grid, build_common_grid
                        S_init_list = []
                        wn_data = np.array(wn)
                        all_ok = True
                        for sname in selected_names:
                            sid  = lib_options[sname]
                            entry = get_spectrum_by_id(sid)
                            if entry is None:
                                st.error(f"Spektra ID {sid} tidak ditemukan.")
                                all_ok = False
                                break
                            wn_lib = np.array(entry["wavenumber"], dtype=float)
                            sp_lib = np.array(entry["spectrum"],   dtype=float)
                            # Resample ke grid wavenumber data
                            common_grid, grid_info = build_common_grid(wn_data, wn_lib)
                            if common_grid is None:
                                st.error(t(
                                    f"Tidak ada overlap wavenumber antara data dan '{entry['name']}'. "
                                    f"Periksa range wavenumber library.",
                                    f"No wavenumber overlap between data and '{entry['name']}'. "
                                    f"Check library wavenumber range."
                                ))
                                all_ok = False
                                break
                            sp_resampled = resample_to_grid(wn_lib, sp_lib, wn_data)
                            S_init_list.append(sp_resampled)

                        if all_ok and len(S_init_list) == int(n_comp):
                            S_init_guess = np.array(S_init_list)
                            st.success(t(
                                f"✅ {n_comp} spektra siap sebagai initial guess: "
                                f"{', '.join([n.split(' [')[0] for n in selected_names])}",
                                f"✅ {n_comp} spectra ready as initial guess: "
                                f"{', '.join([n.split(' [')[0] for n in selected_names])}"
                            ))
                            # Preview overlay
                            with st.expander(t("Preview initial guess vs data","Preview initial guess vs data"),
                                             expanded=False):
                                fig_ig = go.Figure()
                                colors_ig = px.colors.qualitative.Pastel
                                for i, sp in enumerate(S_init_list):
                                    fig_ig.add_trace(go.Scatter(
                                        x=wn, y=sp,
                                        name=selected_names[i].split(" [")[0],
                                        mode="lines",
                                        line=dict(width=1.8, color=colors_ig[i % len(colors_ig)])
                                    ))
                                fig_ig.update_layout(
                                    template="plotly_dark",
                                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                                    xaxis=dict(autorange="reversed", gridcolor="#1e293b",
                                               title="Wavenumber (cm⁻¹)"),
                                    yaxis=dict(gridcolor="#1e293b",
                                               title=t("Intensitas","Intensity")),
                                    legend=dict(
                                        bgcolor="#161b27", bordercolor="#2a3142",
                                        borderwidth=1, font=dict(size=11, color="#e2e8f0"),
                                        orientation="v", x=1.02, xanchor="left", y=1
                                    ),
                                    height=260, margin=dict(l=20,r=160,t=10,b=40)
                                )
                                st.plotly_chart(fig_ig, use_container_width=True)

        # ── Selectivity Constraint (opsional) ───────────────
        st.markdown(f'<p class="sec-hdr">{t("Selectivity constraint (opsional)","Selectivity constraint (optional)")}</p>',
                    unsafe_allow_html=True)

        use_selectivity = st.checkbox(
            t("Kunci spektra komponen yang diketahui (selectivity constraint)",
              "Lock known component spectra (selectivity constraint)"),
            value=False, key="mcr_use_selectivity",
            help=t(
                "Pilih komponen mana yang spektranya sudah diketahui dan dikunci "
                "agar tidak berubah selama iterasi MCR-ALS. "
                "Contoh: air sebagai perturbasi yang diketahui. "
                "Ini mengurangi rotational ambiguity secara signifikan.",
                "Select which components have known spectra to be locked "
                "during MCR-ALS iterations. "
                "Example: water as a known perturbation component. "
                "This significantly reduces rotational ambiguity."
            )
        )

        fixed_spectra_dict = None
        if use_selectivity:
            from database import get_all_spectra_for_matching, get_spectrum_by_id
            all_lib_sel = get_all_spectra_for_matching()
            if not all_lib_sel:
                st.warning(t("Library masih kosong. Tambahkan spektra referensi terlebih dahulu.",
                             "Library is empty. Please add reference spectra first."))
            else:
                st.markdown(f"""
                <div style="background:#0d1829;border:1px solid #1e3a5f;
                  border-radius:8px;padding:8px 14px;margin-bottom:8px;
                  font-size:0.82rem;color:#7dd3fc;">
                  ℹ️ {t(
                    "Untuk setiap komponen yang ingin dikunci, pilih indeks komponen "
                    "(0 = komponen pertama) dan spektra dari library yang akan dikunci.",
                    "For each component to lock, select the component index "
                    "(0 = first component) and the library spectrum to lock it to."
                  )}
                </div>
                """, unsafe_allow_html=True)

                lib_sel_options = {
                    f"{e['name']} [{e['category']}] (ID:{e['id']})": e['id']
                    for e in all_lib_sel
                }
                fixed_spectra_dict = {}

                # Maksimum bisa kunci semua komponen, tapi minimal 1 bebas
                max_fixed = max(1, int(n_comp) - 1)
                n_fixed = st.number_input(
                    t(f"Jumlah komponen yang dikunci (maks {max_fixed})",
                      f"Number of components to lock (max {max_fixed})"),
                    min_value=1, max_value=max_fixed, value=1,
                    key="mcr_n_fixed"
                )

                all_fixed_ok = True
                for fi in range(int(n_fixed)):
                    fc1, fc2 = st.columns([1, 3])
                    comp_idx_sel = fc1.number_input(
                        t(f"Indeks komponen #{fi+1}","Component index #{fi+1}"),
                        min_value=0, max_value=int(n_comp)-1,
                        value=fi, key=f"mcr_fixed_idx_{fi}",
                        help=t("0 = komponen pertama, 1 = kedua, dst.",
                               "0 = first component, 1 = second, etc.")
                    )
                    sel_lib_name = fc2.selectbox(
                        t(f"Spektra library untuk komponen {comp_idx_sel}",
                          f"Library spectrum for component {comp_idx_sel}"),
                        list(lib_sel_options.keys()),
                        key=f"mcr_fixed_lib_{fi}"
                    )

                    # Ambil dan resample spektra dari library
                    sid_sel = lib_sel_options[sel_lib_name]
                    entry_sel = get_spectrum_by_id(sid_sel)
                    if entry_sel:
                        from mcr_engine import resample_to_grid
                        wn_lib_sel = np.array(entry_sel["wavenumber"], dtype=float)
                        sp_lib_sel = np.array(entry_sel["spectrum"],   dtype=float)
                        sp_resampled_sel = resample_to_grid(wn_lib_sel, sp_lib_sel,
                                                            np.array(wn))
                        # Non-negativity
                        sp_resampled_sel = np.maximum(sp_resampled_sel, 0)
                        fixed_spectra_dict[int(comp_idx_sel)] = sp_resampled_sel
                    else:
                        st.error(f"Spektra ID {sid_sel} tidak ditemukan.")
                        all_fixed_ok = False

                if fixed_spectra_dict and all_fixed_ok:
                    # Cek tidak ada indeks duplikat
                    if len(fixed_spectra_dict) != int(n_fixed):
                        st.warning(t("⚠️ Ada indeks komponen yang sama. Gunakan indeks berbeda.",
                                     "⚠️ Duplicate component indices. Use different indices."))
                        fixed_spectra_dict = None
                    else:
                        st.success(t(
                            f"✅ {len(fixed_spectra_dict)} komponen dikunci: "
                            f"indeks {list(fixed_spectra_dict.keys())}",
                            f"✅ {len(fixed_spectra_dict)} component(s) locked: "
                            f"indices {list(fixed_spectra_dict.keys())}"
                        ))
        # ── Windowing / Equality Constraint (opsional) ───────
        st.markdown(f'<p class="sec-hdr">{t("Windowing constraint (opsional)","Windowing constraint (optional)")}</p>',
                    unsafe_allow_html=True)

        use_window = st.checkbox(
            t("Paksa konsentrasi nol pada sampel tertentu (windowing constraint)",
              "Force zero concentration on specific samples (windowing constraint)"),
            value=False, key="mcr_use_window",
            help=t(
                "Pilih sampel yang diketahui secara independen TIDAK mengandung "
                "komponen tertentu (mis. titik blank/background tanpa deposit analit). "
                "Konsentrasi komponen tersebut dipaksa nol di sampel itu selama iterasi. "
                "Ini mengurangi rotational ambiguity dengan cara yang berbeda dari "
                "selectivity constraint (yang mengunci bentuk spektra, bukan nilai konsentrasi).",
                "Select samples that are independently known NOT to contain a given "
                "component (e.g. blank/background points without analyte deposit). "
                "The concentration of that component is forced to zero in those samples "
                "during iteration. This reduces rotational ambiguity differently from "
                "the selectivity constraint (which locks spectral shape, not concentration)."
            )
        )

        fixed_conc_zero_list = None
        if use_window:
            spec_names_all = st.session_state.get("spec_names", [])
            if not spec_names_all:
                st.warning(t("Nama sampel tidak ditemukan. Upload ulang data di tab Input Data.",
                             "Sample names not found. Re-upload data in the Input Data tab."))
            else:
                w1, w2 = st.columns([2, 1])
                blank_samples = w1.multiselect(
                    t("Sampel blank/background (konsentrasi dipaksa nol)",
                      "Blank/background samples (concentration forced to zero)"),
                    spec_names_all,
                    key="mcr_window_samples",
                    help=t("Pilih satu atau lebih titik yang diketahui tidak mengandung analit.",
                           "Select one or more points known to contain no analyte.")
                )
                zero_comp_idx = w2.multiselect(
                    t("Komponen yang dipaksa nol","Components forced to zero"),
                    list(range(int(n_comp))),
                    key="mcr_window_comps",
                    help=t("0 = komponen pertama, 1 = kedua, dst.",
                           "0 = first component, 1 = second, etc.")
                )

                if blank_samples and zero_comp_idx:
                    name_to_idx = {nm: i for i, nm in enumerate(spec_names_all)}
                    fixed_conc_zero_list = [
                        (name_to_idx[nm], int(ci))
                        for nm in blank_samples
                        for ci in zero_comp_idx
                        if nm in name_to_idx
                    ]
                    st.success(t(
                        f"✅ {len(blank_samples)} sampel × {len(zero_comp_idx)} komponen "
                        f"= {len(fixed_conc_zero_list)} pasangan (sampel, komponen) dipaksa nol.",
                        f"✅ {len(blank_samples)} sample(s) × {len(zero_comp_idx)} component(s) "
                        f"= {len(fixed_conc_zero_list)} (sample, component) pairs forced to zero."
                    ))
                elif blank_samples or zero_comp_idx:
                    st.info(t("Pilih minimal satu sampel DAN minimal satu komponen untuk mengaktifkan constraint.",
                              "Select at least one sample AND one component to activate the constraint."))

        if st.button(f"▶  {t('Jalankan MCR-ALS','Run MCR-ALS')}",
                     use_container_width=True):
            with st.spinner(t("Menjalankan MCR-ALS...","Running MCR-ALS...")):
                C, S, lof_hist, r2, conv, diag = run_mcr_als(
                    D, int(n_comp), int(max_iter), float(tol),
                    closure, unimodal,
                    normalize_S=normalize_S,
                    init_method=init_method,
                    S_init=S_init_guess,
                    fixed_spectra=fixed_spectra_dict if use_selectivity else None,
                    fixed_conc_zero=fixed_conc_zero_list if use_window else None,
                    unimodal_C=unimodal_C,
                    smooth_S=smooth_S
                )
                warnings_list = generate_warnings(diag, int(n_comp), ev)
                st.session_state.update({
                    "mcr_C": C, "mcr_S": S, "mcr_lof": lof_hist,
                    "mcr_r2": r2, "mcr_ncomp": int(n_comp),
                    "mcr_converged": conv,
                    "mcr_diag": diag,
                    "mcr_warnings": warnings_list,
                })
                # Reset post-MCR processing on new MCR run
                for _key in ["mcr_S_proc","mcr_proc_log","match_results",
                             "cos2d_result","cos2d_perturb"]:
                    st.session_state.pop(_key, None)
            conv_msg = t("Konvergen","Converged") if conv else t("Belum konvergen","Not converged")
            st.success(f"✅ {conv_msg} — {len(lof_hist)} {t('iterasi','iterations')} "
                       f"| LOF: {lof_hist[-1]:.4f}% | R²: {r2:.5f} "
                       f"| RMSE: {diag['rmse']:.5f}")
            # Tampilkan warning system
            for w in warnings_list:
                sev = w["severity"]
                msg = w["message_en"] if lang == "en" else w["message_id"]
                if sev == "error":
                    st.error(f"[Tipe {w['type']}] {msg}")
                elif sev == "warning":
                    st.warning(f"[Tipe {w['type']}] {msg}")
                else:
                    if w["code"] != "OK":
                        st.info(f"[Tipe {w['type']}] {msg}")

        if "mcr_S" in st.session_state:
            S_res = st.session_state["mcr_S"]
            C_res = st.session_state["mcr_C"]
            lof_h = st.session_state["mcr_lof"]
            r2    = st.session_state["mcr_r2"]
            nc    = st.session_state["mcr_ncomp"]

            rmse_val = st.session_state.get("mcr_diag", {}).get("rmse")
            rmse_str = f"{rmse_val:.5f}" if rmse_val is not None else "—"

            m1,m2,m3,m4,m5 = st.columns(5)
            for col, val, lbl in zip(
                [m1,m2,m3,m4,m5],
                [f"{lof_h[-1]:.3f}%", f"{r2:.4f}", rmse_str, len(lof_h), nc],
                ["LOF", "R²", "RMSE", t("Iterasi","Iterations"), t("Komponen","Components")]
            ):
                col.markdown(
                    f'<div class="metric-card"><div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{lbl}</div></div>',
                    unsafe_allow_html=True
                )

            # ── Validation Scorecard ──────────────────────────
            if "mcr_diag" in st.session_state:
                diag_stored = st.session_state["mcr_diag"]
                scorecard, total_sc, overall = compute_scorecard(diag_stored)
                sc_color = {"baik":"#0d2018","sedang":"#1a1a08","perlu_perbaikan":"#1a0a08"}[overall]
                sc_border = {"baik":"#22c55e","sedang":"#eab308","perlu_perbaikan":"#ef4444"}[overall]
                sc_text   = {"baik":"#4ade80","sedang":"#fde047","perlu_perbaikan":"#f87171"}[overall]
                sc_label  = {"baik": t("BAIK","GOOD"),
                             "sedang": t("SEDANG","MODERATE"),
                             "perlu_perbaikan": t("PERLU PERBAIKAN","NEEDS IMPROVEMENT")}[overall]

                with st.expander(
                    t(f"📋 Validation Scorecard — {total_sc}/8 kriteria terpenuhi ({sc_label})",
                      f"📋 Validation Scorecard — {total_sc}/8 criteria met ({sc_label})"),
                    expanded=(overall != "baik")
                ):
                    for sc in scorecard:
                        msg = sc["message_en"] if lang == "en" else sc["message_id"]
                        st.markdown(
                            f'<div style="display:flex;gap:12px;padding:5px 0;'
                            f'border-bottom:0.5px solid #1e293b;font-size:0.83rem;">'
                            f'<span style="width:24px;text-align:center;">{sc["status"]}</span>'
                            f'<span style="width:220px;color:#94a3b8;">{sc["criterion"]}</span>'
                            f'<span style="width:100px;font-family:monospace;color:#7dd3fc;">{sc["value"]}</span>'
                            f'<span style="color:#e2e8f0;">{msg}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    # Residual per sampel
                    st.markdown(f'<p class="sec-hdr" style="margin-top:1rem;">'
                                f'{t("Residual per sampel","Residual per sample")}</p>',
                                unsafe_allow_html=True)
                    lof_ps = diag_stored["lof_per_sample"]
                    snames_sc = st.session_state.get("spec_names",
                                [f"S{i+1}" for i in range(len(lof_ps))])
                    if len(snames_sc) != len(lof_ps):
                        snames_sc = [f"S{i+1}" for i in range(len(lof_ps))]
                    fig_sr = go.Figure(go.Bar(
                        x=snames_sc, y=lof_ps,
                        marker_color=["#ef4444" if v > 3*np.mean(lof_ps) else "#378ADD"
                                      for v in lof_ps]
                    ))
                    mean_lof_s = float(np.mean(lof_ps))
                    fig_sr.add_hline(y=mean_lof_s*3, line_dash="dash",
                                     line_color="#f97316",
                                     annotation_text=t("3× rata-rata","3× mean"))
                    fig_sr.update_layout(
                        template="plotly_dark", paper_bgcolor="#0f1117",
                        plot_bgcolor="#0f1117", height=200,
                        xaxis=dict(gridcolor="#1e293b"),
                        yaxis=dict(gridcolor="#1e293b",
                                   title=t("LOF per sampel (%)","LOF per sample (%)")),
                        margin=dict(l=20,r=20,t=10,b=40)
                    )
                    st.plotly_chart(fig_sr, use_container_width=True)

                # ── Ambiguitas Rotasi (MCR-BANDS) ──────────────────────────
                with st.expander(
                    t("📐 Ambiguitas Rotasi (MCR-BANDS)", "📐 Rotational Ambiguity (MCR-BANDS)"),
                    expanded=False
                ):
                    st.caption(t(
                        "Mengukur seberapa lebar rentang solusi C/S alternatif yang "
                        "masih memenuhi constraint yang sama (non-negativity, closure) "
                        "tanpa mengubah kecocokan terhadap data. Lebar besar = hasil "
                        "kurang unik, perlu constraint tambahan atau data selektif.",
                        "Measures how wide the range of alternative C/S solutions is "
                        "that still satisfy the same constraints (non-negativity, "
                        "closure) without changing the fit to data. Large width = "
                        "less unique result, needs more constraints or selective data."
                    ))

                    method_note = (
                        t(f"Metode EKSAK akan dipakai (≤{K_EXACT_LIMIT} komponen).",
                          f"EXACT method will be used (≤{K_EXACT_LIMIT} components).")
                        if nc <= K_EXACT_LIMIT else
                        t(f"Metode ESTIMASI (heuristik) akan dipakai (>{K_EXACT_LIMIT} komponen) "
                          f"— hasil berupa perkiraan, bukan batas eksak.",
                          f"HEURISTIC (estimated) method will be used (>{K_EXACT_LIMIT} components) "
                          f"— result is an approximation, not an exact bound.")
                    )
                    st.info(method_note)

                    n_dir = 20
                    if nc > K_EXACT_LIMIT:
                        n_dir = st.slider(
                            t("Jumlah arah eksplorasi (lebih besar = lebih teliti, lebih lambat)",
                              "Number of exploration directions (higher = more thorough, slower)"),
                            min_value=10, max_value=60, value=20, step=5,
                        )

                    if st.button(t("🔍 Hitung Ambiguitas", "🔍 Compute Ambiguity"), key="btn_ambiguity"):
                        constraints_used = diag_stored.get("constraints_used", {})
                        with st.spinner(t("Menghitung rentang ambiguitas...", "Computing ambiguity range...")):
                            try:
                                amb_result = compute_rotational_ambiguity(
                                    C_res, S_res, constraints_used,
                                    method="auto", n_directions=n_dir,
                                )
                                st.session_state["mcr_ambiguity"] = amb_result
                            except Exception as e:
                                st.error(t(f"Gagal menghitung ambiguitas: {e}",
                                           f"Failed to compute ambiguity: {e}"))

                    if "mcr_ambiguity" in st.session_state:
                        amb = st.session_state["mcr_ambiguity"]

                        badge = (t("✅ Eksak (SLSQP)", "✅ Exact (SLSQP)")
                                 if amb["method_used"] == "exact"
                                 else t("⚠️ Estimasi (heuristik)", "⚠️ Estimated (heuristic)"))
                        st.markdown(f"**{badge}**")

                        if amb["method_used"] == "heuristic":
                            nv = amb["diagnostics"]["n_valid_samples"]
                            nt = amb["diagnostics"]["n_total_samples"]
                            st.caption(t(
                                f"{nv}/{nt} arah eksplorasi berhasil menemukan solusi feasible. "
                                f"Rentang di bawah kemungkinan UNDER-ESTIMATE lebar ambiguitas sebenarnya.",
                                f"{nv}/{nt} exploration directions found a feasible solution. "
                                f"The ranges below may UNDER-ESTIMATE the true ambiguity width."
                            ))
                            if nv == 0:
                                st.warning(t(
                                    "Tidak ada solusi feasible ditemukan — coba naikkan jumlah arah "
                                    "eksplorasi, atau hasil MCR-ALS mungkin terlalu dekat batas constraint.",
                                    "No feasible solution found — try increasing the number of "
                                    "exploration directions, or the MCR-ALS result may be too "
                                    "close to the constraint boundary."
                                ))

                        # Tabel ringkasan lebar ambiguitas, RMSERA, dan klasifikasi keandalan
                        reliability_badge = {
                            "kuantitatif_kualitatif": "🟢 " + t("Kuantitatif+Kualitatif", "Quantitative+Qualitative"),
                            "semi_kuantitatif_kualitatif": "🟡 " + t("Semi-kuantitatif+Kualitatif", "Semi-quant.+Qualitative"),
                            "kualitatif_saja": "🟠 " + t("Kualitatif saja", "Qualitative only"),
                            "tidak_dapat_diandalkan": "🔴 " + t("Tidak dapat diandalkan", "Not reliable"),
                        }
                        rows = []
                        for comp in amb["components"]:
                            rows.append({
                                t("Komponen", "Component"): f"C{comp['component_idx']+1}",
                                t("Lebar Ambiguitas (%)", "Ambiguity Width (%)"): round(comp["ambiguity_width_pct"], 2),
                                "RMSERA (δRA/√12)": round(comp["rmsera_lower"], 5),
                                "RMSERA (δRA/√3)": round(comp["rmsera_upper"], 5),
                                t("RMSERA relatif (%)", "RMSERA relative (%)"): round(comp["rmsera_upper_relative_pct"], 2),
                                t("% Wilayah Spektrum Aman", "% Safe Spectral Region"): round(
                                    comp["local_ambiguity"]["pct_region_low_ambiguity"], 1),
                                t("Klasifikasi Keandalan", "Reliability Classification"):
                                    reliability_badge.get(comp["reliability"]["label"], "-"),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        st.caption(t(
                            "RMSERA di sini adalah ADAPTASI relatif (tanpa kalibrasi eksternal) dari "
                            "formalisme Chiappini, Alcaraz, et al., Anal. Chem. 2020, 92(10), 7255-7263 "
                            "— nyatakan sebagai adaptasi, bukan RMSERA baku, jika dipakai di publikasi. "
                            "'% Wilayah Spektrum Aman' = persentase titik bilangan gelombang dengan lebar "
                            "pita lokal ≤15% dari intensitas maksimum komponen.",
                            "RMSERA here is a relative ADAPTATION (no external calibration) of the "
                            "formalism by Chiappini, Alcaraz, et al., Anal. Chem. 2020, 92(10), 7255-7263 "
                            "— state it as an adaptation, not standard RMSERA, if used in publication. "
                            "'% Safe Spectral Region' = percentage of wavenumber points with local band "
                            "width ≤15% of the component's maximum intensity."
                        ))

                        for comp in amb["components"]:
                            with st.expander(
                                t(f"ℹ️ Detail interpretasi C{comp['component_idx']+1}",
                                  f"ℹ️ Interpretation detail C{comp['component_idx']+1}")
                            ):
                                st.write(comp["reliability"]["detail"])

                        # Grafik pita feasible per komponen (overlay pada profil nominal),
                        # dengan shading hijau tipis di wilayah bilangan gelombang yang
                        # ambiguitas lokalnya rendah (<=15%, "aman" untuk interpretasi bentuk).
                        wn_axis = wn  # variabel 'wn' sudah tersedia di scope tab_mcr (baris 397 app.py)
                        comp_to_plot = st.selectbox(
                            t("Pilih komponen untuk visualisasi pita", "Select component to visualize band"),
                            options=list(range(nc)),
                            format_func=lambda i: f"C{i+1}",
                            key="ambig_comp_select",
                        )
                        comp_data = amb["components"][comp_to_plot]
                        safe_mask = comp_data["local_ambiguity"]["low_ambiguity_mask"]

                        fig_amb = go.Figure()

                        # Shading wilayah "aman" (ambiguitas lokal rendah) sebagai vrect
                        # per segmen kontinu, supaya tidak satu-satu per titik (mahal & ramai)
                        if np.any(safe_mask):
                            edges = np.diff(safe_mask.astype(int))
                            starts = list(np.where(edges == 1)[0] + 1)
                            ends = list(np.where(edges == -1)[0] + 1)
                            if safe_mask[0]:
                                starts = [0] + starts
                            if safe_mask[-1]:
                                ends = ends + [len(safe_mask)]
                            for s_idx, e_idx in zip(starts, ends):
                                fig_amb.add_vrect(
                                    x0=wn_axis[s_idx], x1=wn_axis[min(e_idx, len(wn_axis) - 1)],
                                    fillcolor="rgba(74,222,128,0.10)", line_width=0, layer="below",
                                )

                        fig_amb.add_trace(go.Scatter(
                            x=wn_axis, y=comp_data["S_band_max"], mode="lines",
                            line=dict(width=0), showlegend=False, hoverinfo="skip",
                        ))
                        fig_amb.add_trace(go.Scatter(
                            x=wn_axis, y=comp_data["S_band_min"], mode="lines",
                            line=dict(width=0), fill="tonexty", fillcolor="rgba(125,211,252,0.25)",
                            name=t("Pita feasible", "Feasible band"),
                        ))
                        fig_amb.add_trace(go.Scatter(
                            x=wn_axis, y=S_res[comp_to_plot], mode="lines",
                            line=dict(color="#7dd3fc", width=2),
                            name=t("Profil terpilih (nominal)", "Selected profile (nominal)"),
                        ))
                        fig_amb.update_layout(
                            title=t(f"Pita Ambiguitas Spektrum — Komponen C{comp_to_plot+1}",
                                    f"Spectral Ambiguity Band — Component C{comp_to_plot+1}"),
                            xaxis_title=t("Bilangan gelombang (cm⁻¹)", "Wavenumber (cm⁻¹)"),
                            yaxis_title=t("Absorbansi (ternormalisasi)", "Absorbance (normalized)"),
                            height=380,
                        )
                        st.plotly_chart(fig_amb, use_container_width=True)
                        st.caption(t(
                            "🟩 Area hijau tipis = wilayah bilangan gelombang dengan ambiguitas bentuk "
                            "lokal rendah (aman untuk interpretasi puncak/identifikasi di wilayah itu).",
                            "🟩 Light green area = wavenumber region with low local shape ambiguity "
                            "(safe for peak interpretation/identification in that region)."
                        ))

                # ── Keputusan Identitas Kimia (Hierarchical Decision) ──────
                with st.expander(
                    t("🎯 Keputusan Identitas Kimia (Hierarchical)",
                      "🎯 Chemical Identity Decision (Hierarchical)"),
                    expanded=False
                ):
                    st.caption(t(
                        "Menggabungkan status fit, similarity terhadap reference, dan ambiguitas "
                        "rotasi (lewat Spectral Identity Robustness Index/SIRI) menjadi SATU "
                        "verdict identitas per komponen — BUKAN skor tertimbang, melainkan "
                        "keputusan bertingkat. Fit MCR yang baik dan similarity tinggi SAJA "
                        "TIDAK otomatis berarti identitas kimia benar; kombinasi ini bisa "
                        "menandai secara eksplisit kasus 'similarity tinggi tapi ambigu' "
                        "(kandidat false resolved spectrum).",
                        "Combines fit status, similarity to a reference, and rotational "
                        "ambiguity (via the Spectral Identity Robustness Index/SIRI) into ONE "
                        "identity verdict per component — NOT a weighted score, but a "
                        "hierarchical decision. Good MCR fit and high similarity ALONE do NOT "
                        "automatically mean correct chemical identity; this combination can "
                        "explicitly flag the 'high similarity but ambiguous' case (a candidate "
                        "false resolved spectrum)."
                    ))

                    if "mcr_ambiguity" not in st.session_state:
                        st.info(t(
                            "Jalankan '📐 Ambiguitas Rotasi' di atas dulu — keputusan identitas "
                            "membutuhkan hasil ambiguitas (untuk SIRI) sebagai salah satu bukti.",
                            "Run '📐 Rotational Ambiguity' above first — the identity decision "
                            "needs the ambiguity result (for SIRI) as one piece of evidence."
                        ))
                    else:
                        amb_id = st.session_state["mcr_ambiguity"]
                        from database import get_all_spectra_for_matching as _get_lib_id
                        library_entries_id = _get_lib_id()

                        if not library_entries_id:
                            st.warning(t(
                                "Belum ada spektra referensi di library — tambahkan dulu di tab "
                                "Kelola Library (admin) sebelum menjalankan keputusan identitas.",
                                "No reference spectra in the library yet — add some via the "
                                "Manage Library tab (admin) before running the identity decision."
                            ))
                        else:
                            fit_ok_scorecard = (overall != "perlu_perbaikan")
                            ref_options = [(None, t("— pilih reference —", "— select reference —"))] + \
                                          [(e["id"], e.get("name", f"#{e['id']}")) for e in library_entries_id]

                            decisions_id = []
                            for i in range(nc):
                                st.markdown(f"**C{i+1}**")
                                default_idx = 0
                                if "match_results" in st.session_state and i < len(st.session_state["match_results"]):
                                    top_res_id = st.session_state["match_results"][i]
                                    if top_res_id:
                                        ids_only = [o[0] for o in ref_options]
                                        if top_res_id[0]["id"] in ids_only:
                                            default_idx = ids_only.index(top_res_id[0]["id"])

                                sel_ref = st.selectbox(
                                    t(f"Reference identitas yang diklaim untuk C{i+1}",
                                      f"Claimed identity reference for C{i+1}"),
                                    options=ref_options, format_func=lambda o: o[1],
                                    index=default_idx, key=f"identity_ref_select_{i}",
                                )
                                ref_id = sel_ref[0]
                                if ref_id is None:
                                    st.caption(t("Lewati komponen ini — belum ada reference dipilih.",
                                                 "Skip this component — no reference selected yet."))
                                    continue

                                lib_entry_id = get_spectrum_by_id(ref_id)
                                S_ref_resampled = interpolate_spectrum(
                                    lib_entry_id["wavenumber"], lib_entry_id["spectrum"], wn
                                )

                                pear_id = pearson_corr(S_res[i], S_ref_resampled)
                                cos_id = cosine_sim(S_res[i], S_ref_resampled)

                                amb_comp_id = (amb_id["components"][i]
                                              if i < len(amb_id.get("components", [])) else None)
                                siri_result_id = None
                                if amb_comp_id is not None:
                                    siri_result_id = compute_identity_robustness_index(
                                        S_res[i], amb_comp_id["S_band_min"], amb_comp_id["S_band_max"],
                                        S_ref_resampled,
                                    )

                                decision_id = evaluate_component_identity(
                                    component_label=f"C{i+1}", fit_ok=fit_ok_scorecard,
                                    pearson=pear_id, cosine=cos_id,
                                    ambiguity_width_pct=amb_comp_id["ambiguity_width_pct"] if amb_comp_id else None,
                                    ambiguity_reliability_label=amb_comp_id["reliability"]["label"] if amb_comp_id else None,
                                    siri_result=siri_result_id,
                                )
                                decisions_id.append(decision_id)

                                badge_map_id = {
                                    "reliably_identified": "🟢 " + t("Teridentifikasi kuat", "Reliably identified"),
                                    "provisional": "🟡 " + t("Sementara (provisional)", "Provisional"),
                                    "possible_false_resolution": "🟠 " + t(
                                        "Kandidat False Resolved Spectrum", "Possible false resolved spectrum"),
                                    "not_supported": "🔴 " + t("Tidak didukung", "Not supported"),
                                    "unreliable": "⚫ " + t("Tidak dapat diandalkan (fit gagal)", "Unreliable (fit failed)"),
                                }
                                st.markdown(f"{badge_map_id.get(decision_id['final_label'], '-')}")
                                st.caption(decision_id["narrative"])
                                if decision_id["flags"]:
                                    st.caption(t("Catatan: ", "Notes: ") + ", ".join(decision_id["flags"]))

                            if decisions_id:
                                st.markdown(t("**Ringkasan seluruh komponen:**", "**Summary across components:**"))
                                st.dataframe(build_identity_decision_dataframe(decisions_id),
                                            use_container_width=True, hide_index=True)
                                st.caption(t(
                                    "robustness_ok belum tersedia otomatis di sini (butuh modul "
                                    "robustness sweep terpisah dijalankan lebih dulu) — verdict "
                                    "'reliably_identified' di atas karenanya bersifat provisional "
                                    "sampai robustness dicek.",
                                    "robustness_ok is not yet available automatically here (needs "
                                    "the separate robustness sweep module run first) — "
                                    "'reliably_identified' verdicts above are therefore "
                                    "provisional until robustness is checked."
                                ))
                                st.session_state["identity_decisions"] = decisions_id


            colors = px.colors.qualitative.Pastel

            st.markdown(f'<p class="sec-hdr">{t("Spektra murni hasil MCR (raw)","MCR pure spectra (raw)")}</p>',
                        unsafe_allow_html=True)
            fig_s = go.Figure()
            for i in range(nc):
                fig_s.add_trace(go.Scatter(
                    x=wn, y=S_res[i], name=f"{t('Komponen','Component')} {i+1}",
                    mode="lines", line=dict(width=1.8, color=colors[i%len(colors)])
                ))
            fig_s.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(autorange="reversed", gridcolor="#1e293b",
                           title="Wavenumber (cm⁻¹)"),
                yaxis=dict(gridcolor="#1e293b", title=t("Intensitas","Intensity")),
                legend=dict(
                    title=dict(text=t("Komponen MCR","MCR Components"),
                               font=dict(size=11, color="#7dd3fc")),
                    bgcolor="#161b27", bordercolor="#2a3142", borderwidth=1,
                    font=dict(size=11, color="#e2e8f0"),
                    orientation="v", x=1.02, xanchor="left", y=1
                ),
                height=340, margin=dict(l=20,r=160,t=20,b=40)
            )
            st.plotly_chart(fig_s, use_container_width=True)

            st.markdown(f'<p class="sec-hdr">{t("Profil konsentrasi","Concentration profiles")}</p>',
                        unsafe_allow_html=True)
            snames = st.session_state.get("spec_names", [f"S{i+1}" for i in range(C_res.shape[0])])
            if len(snames) != C_res.shape[0]:
                snames = [f"S{i+1}" for i in range(C_res.shape[0])]
            fig_c = go.Figure()
            for i in range(nc):
                fig_c.add_trace(go.Bar(
                    name=f"{t('Komponen','Component')} {i+1}",
                    x=snames, y=C_res[:,i],
                    marker_color=colors[i%len(colors)]
                ))
            fig_c.update_layout(
                barmode="stack", template="plotly_dark",
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(gridcolor="#1e293b", title=t("Sampel","Sample")),
                yaxis=dict(gridcolor="#1e293b", title=t("Kontribusi relatif","Relative contribution")),
                legend=dict(
                    title=dict(text=t("Komponen","Components"),
                               font=dict(size=11, color="#7dd3fc")),
                    bgcolor="#161b27", bordercolor="#2a3142", borderwidth=1,
                    font=dict(size=11, color="#e2e8f0"),
                    orientation="v", x=1.02, xanchor="left", y=1
                ),
                height=300, margin=dict(l=20,r=160,t=20,b=40)
            )
            st.plotly_chart(fig_c, use_container_width=True)

            st.markdown(f'<p class="sec-hdr">{t("Konvergensi LOF","LOF convergence")}</p>',
                        unsafe_allow_html=True)
            fig_lof = go.Figure(go.Scatter(y=lof_h, mode="lines",
                line=dict(color="#f97316", width=1.5)))
            fig_lof.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(gridcolor="#1e293b", title=t("Iterasi","Iteration")),
                yaxis=dict(gridcolor="#1e293b", title="LOF (%)"),
                height=200, margin=dict(l=20,r=20,t=10,b=40)
            )
            st.plotly_chart(fig_lof, use_container_width=True)

            st.info(t(
                "💡 Lanjutkan ke tab **✨ Proses Spektra MCR** untuk smoothing & normalisasi "
                "spektra komponen sebelum identifikasi.",
                "💡 Proceed to **✨ Process MCR Spectra** tab for smoothing & normalization "
                "of component spectra before identification."
            ))

# ════════════════════════════════════════════════════════════════
# TAB 3 — POST-MCR SPECTRAL PROCESSING  ← NEW
# ════════════════════════════════════════════════════════════════
with tab_postmcr:
    if "mcr_S" not in st.session_state:
        st.info(t("Jalankan MCR-ALS terlebih dahulu di tab Analisis MCR.",
                  "Please run MCR-ALS first in the MCR Analysis tab."))
    else:
        wn  = st.session_state["wavenumber"]
        S   = st.session_state["mcr_S"]
        nc  = st.session_state["mcr_ncomp"]

        st.markdown(f"""
        <div class="proc-box">
          <div class="proc-box-title">
            {t("Pemrosesan Pasca-MCR","Post-MCR Processing")}
          </div>
          <span style="font-size:0.82rem;color:#94a3b8;">
            {t(
              "Terapkan smoothing dan/atau normalisasi pada spektra murni hasil MCR "
              "secara terpisah. Spektra yang diproses inilah yang akan digunakan untuk "
              "pencocokan dengan library.",
              "Apply smoothing and/or normalization to MCR pure-component spectra "
              "independently. The processed spectra will be used for library matching."
            )}
          </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Step 1: Baseline ──────────────────────────────────
        st.markdown(f'<p class="sec-hdr"><span class="step-badge">1</span>'
                    f'{t("Koreksi Baseline","Baseline Correction")}</p>',
                    unsafe_allow_html=True)
        do_bl = st.checkbox(
            t("Aktifkan koreksi baseline (min subtraction)",
              "Enable baseline correction (min subtraction)"),
            value=False, key="postmcr_baseline"
        )

        # ── Step 2: Smoothing ─────────────────────────────────
        st.markdown(f'<p class="sec-hdr"><span class="step-badge">2</span>'
                    f'{t("Smoothing Savitzky-Golay","Savitzky-Golay Smoothing")}</p>',
                    unsafe_allow_html=True)
        do_sg = st.checkbox(
            t("Aktifkan smoothing Savitzky-Golay",
              "Enable Savitzky-Golay smoothing"),
            value=False, key="postmcr_smooth"
        )
        sg_cols = st.columns(2)
        sg_window = sg_cols[0].number_input(
            t("Window length (ganjil)","Window length (odd)"),
            min_value=5, max_value=51, value=11, step=2,
            disabled=not do_sg, key="postmcr_sg_window",
            help=t("Harus ganjil dan > orde polinomial + 2",
                   "Must be odd and > polynomial order + 2")
        )
        sg_poly = sg_cols[1].number_input(
            t("Orde polinomial","Polynomial order"),
            min_value=2, max_value=5, value=3,
            disabled=not do_sg, key="postmcr_sg_poly",
            help=t("Orde 3 direkomendasikan untuk FTIR",
                   "Order 3 recommended for FTIR")
        )
        if do_sg:
            win_ok = (sg_window % 2 == 1) and (sg_window > sg_poly + 2)
            if not win_ok:
                st.warning(t(
                    f"⚠️ Window {sg_window} tidak valid untuk poly={sg_poly}. "
                    f"Minimum window = {sg_poly + 3} (ganjil).",
                    f"⚠️ Window {sg_window} invalid for poly={sg_poly}. "
                    f"Minimum window = {sg_poly + 3} (odd)."
                ))

        # ── Step 3: Normalization ─────────────────────────────
        st.markdown(f'<p class="sec-hdr"><span class="step-badge">3</span>'
                    f'{t("Normalisasi Spektra","Spectral Normalization")}</p>',
                    unsafe_allow_html=True)
        do_norm_post = st.checkbox(
            t("Aktifkan normalisasi","Enable normalization"),
            value=False, key="postmcr_norm"
        )

        norm_method_labels = {
            "area":   t("Area (trapezoid) — direkomendasikan untuk FTIR",
                        "Area (trapezoid) — recommended for FTIR"),
            "max":    t("Max intensity = 1","Max intensity = 1"),
            "vector": t("Vector (L2 norm)","Vector (L2 norm)"),
            "minmax": t("Min-Max → [0, 1]","Min-Max → [0, 1]"),
        }
        norm_keys = list(norm_method_labels.keys())
        norm_selected = st.selectbox(
            t("Metode normalisasi","Normalization method"),
            norm_keys,
            format_func=lambda k: norm_method_labels[k],
            disabled=not do_norm_post,
            key="postmcr_norm_method"
        )

        # Descriptions
        norm_desc = {
            "area":   t("Membagi spektra dengan luas di bawah kurva → sebanding dengan konsentrasi.",
                        "Divides by area under curve → proportional to concentration."),
            "max":    t("Menjadikan puncak tertinggi = 1 → mudah dibandingkan bentuk.",
                        "Sets highest peak = 1 → easy shape comparison."),
            "vector": t("L2 normalization → vektor unit, cocok untuk cosine similarity.",
                        "L2 normalization → unit vector, ideal for cosine similarity."),
            "minmax": t("Rentang [0,1] → menghilangkan offset baseline.",
                        "Range [0,1] → removes baseline offset."),
        }
        if do_norm_post:
            st.caption(f"ℹ️ {norm_desc[norm_selected]}")

        # ── Apply processing ───────────────────────────────────
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)

        if col_btn1.button(
            f"✨ {t('Terapkan pemrosesan','Apply processing')}",
            use_container_width=True
        ):
            if not do_bl and not do_sg and not do_norm_post:
                st.warning(t("Pilih minimal satu jenis pemrosesan.",
                             "Select at least one processing type."))
            else:
                with st.spinner(t("Memproses spektra MCR...","Processing MCR spectra...")):
                    S_proc, proc_log = postprocess_mcr_spectra(
                        S, wn,
                        do_smooth   = do_sg,
                        sg_window   = int(sg_window),
                        sg_poly     = int(sg_poly),
                        do_norm     = do_norm_post,
                        norm_method = norm_selected,
                        do_baseline = do_bl
                    )
                    st.session_state["mcr_S_proc"] = S_proc
                    st.session_state["mcr_proc_log"] = proc_log
                    st.session_state.pop("match_results", None)
                st.success(t(
                    f"✅ Spektra diproses: {' → '.join(proc_log)}",
                    f"✅ Spectra processed: {' → '.join(proc_log)}"
                ))

        if col_btn2.button(
            t("↩ Gunakan spektra MCR original","↩ Use original MCR spectra"),
            use_container_width=True
        ):
            st.session_state.pop("mcr_S_proc", None)
            st.session_state.pop("mcr_proc_log", None)
            st.session_state.pop("match_results", None)
            st.success(t("✅ Kembali ke spektra MCR original.",
                         "✅ Reverted to original MCR spectra."))

        # ── Comparison plot ────────────────────────────────────
        S_display = st.session_state.get("mcr_S_proc", S)
        proc_log  = st.session_state.get("mcr_proc_log", [])
        is_processed = "mcr_S_proc" in st.session_state

        st.markdown(f'<p class="sec-hdr">'
                    f'{t("Perbandingan: Spektra MCR Original vs Diproses","Comparison: Original vs Processed MCR Spectra")}'
                    f'</p>', unsafe_allow_html=True)

        colors_pastel = px.colors.qualitative.Pastel
        comp_select = st.selectbox(
            t("Pilih komponen untuk ditampilkan","Select component to display"),
            [f"{t('Komponen','Component')} {i+1}" for i in range(nc)],
            key="postmcr_comp_select"
        )
        comp_idx = int(comp_select.split()[-1]) - 1

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=wn, y=S[comp_idx],
            name=t("Original (MCR)", "Original (MCR)"),
            mode="lines",
            line=dict(width=1.5, color="#475569", dash="dot"),
            opacity=0.7
        ))
        fig_cmp.add_trace(go.Scatter(
            x=wn, y=S_display[comp_idx],
            name=t("Diproses","Processed") if is_processed else t("Original","Original"),
            mode="lines",
            line=dict(width=2.0, color=colors_pastel[comp_idx % len(colors_pastel)])
        ))
        proc_title = (" · ".join(proc_log)) if proc_log else t("Tidak ada pemrosesan","No processing")
        fig_cmp.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            title=dict(text=proc_title, font=dict(size=11, color="#7dd3fc"), x=0),
            xaxis=dict(autorange="reversed", gridcolor="#1e293b",
                       title="Wavenumber (cm⁻¹)"),
            yaxis=dict(gridcolor="#1e293b", title=t("Intensitas","Intensity")),
            legend=dict(
                bgcolor="#161b27", bordercolor="#2a3142", borderwidth=1,
                font=dict(size=11, color="#e2e8f0"),
                orientation="v", x=1.02, xanchor="left", y=1
            ),
            height=360, margin=dict(l=20,r=160,t=40,b=40)
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── All components overview ────────────────────────────
        with st.expander(t("📊 Tampilkan semua komponen (diproses)",
                           "📊 Show all components (processed)"), expanded=False):
            fig_all = go.Figure()
            for i in range(nc):
                fig_all.add_trace(go.Scatter(
                    x=wn, y=S_display[i],
                    name=f"{t('Komponen','Component')} {i+1}",
                    mode="lines",
                    line=dict(width=1.8, color=colors_pastel[i % len(colors_pastel)])
                ))
            fig_all.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(autorange="reversed", gridcolor="#1e293b",
                           title="Wavenumber (cm⁻¹)"),
                yaxis=dict(gridcolor="#1e293b", title=t("Intensitas","Intensity")),
                legend=dict(
                    title=dict(text=t("Komponen","Components"),
                               font=dict(size=11, color="#7dd3fc")),
                    bgcolor="#161b27", bordercolor="#2a3142", borderwidth=1,
                    font=dict(size=11, color="#e2e8f0"),
                    orientation="v", x=1.02, xanchor="left", y=1
                ),
                height=340, margin=dict(l=20,r=160,t=10,b=40)
            )
            st.plotly_chart(fig_all, use_container_width=True)

        # ── Download processed spectra ─────────────────────────
        if is_processed:
            df_proc_exp = pd.DataFrame(
                S_display.T, index=wn,
                columns=[f"Component_{i+1}_processed" for i in range(nc)]
            )
            df_proc_exp.index.name = "Wavenumber (cm-1)"
            st.download_button(
                t("⬇ Download spektra diproses (CSV)","⬇ Download processed spectra (CSV)"),
                df_proc_exp.to_csv(),
                f"MCR_processed_spectra_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )

        # Status for identification tab
        if is_processed:
            st.success(t(
                "✅ Spektra sudah diproses dan siap untuk identifikasi di tab Identifikasi.",
                "✅ Spectra processed and ready for identification in the Identification tab."
            ))
        else:
            st.info(t(
                "ℹ️ Spektra MCR original akan digunakan untuk identifikasi jika pemrosesan "
                "tidak diterapkan.",
                "ℹ️ Original MCR spectra will be used for identification if no processing "
                "is applied."
            ))

# ════════════════════════════════════════════════════════════════
# TAB 4 — SPECTRAL IDENTIFICATION
# ════════════════════════════════════════════════════════════════
with tab_match:
    if "mcr_S" not in st.session_state:
        st.info(t("Jalankan MCR-ALS terlebih dahulu.",
                  "Please run MCR-ALS first."))
    else:
        n_lib = count_spectra()
        if n_lib == 0:
            st.warning(t("Library masih kosong. Admin perlu menambahkan spektra acuan.",
                         "Library is empty. Admin needs to add reference spectra."))
        else:
            wn  = st.session_state["wavenumber"]
            nc  = st.session_state["mcr_ncomp"]

            # Determine which spectra to use: processed or original
            if "mcr_S_proc" in st.session_state:
                S = st.session_state["mcr_S_proc"]
                proc_log = st.session_state.get("mcr_proc_log", [])
                proc_label = " · ".join(proc_log) if proc_log else "processed"
                st.markdown(f"""
                <div style="background:#0d2018;border:1px solid #22c55e;border-radius:8px;
                  padding:8px 14px;margin-bottom:10px;font-size:0.82rem;">
                  <span style="color:#4ade80;font-weight:600;">✨ 
                    {t("Menggunakan spektra yang sudah diproses:","Using processed spectra:")}
                  </span>
                  <span style="color:#86efac;font-family:monospace;"> {proc_label}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                S = st.session_state["mcr_S"]
                st.markdown(f"""
                <div style="background:#1a1205;border:1px solid #f97316;border-radius:8px;
                  padding:8px 14px;margin-bottom:10px;font-size:0.82rem;">
                  <span style="color:#fb923c;">
                    ℹ️ {t("Menggunakan spektra MCR original (belum diproses). "
                          "Kunjungi tab ✨ Proses Spektra MCR untuk smoothing/normalisasi.",
                          "Using original MCR spectra (not processed). "
                          "Visit ✨ Process MCR Spectra tab for smoothing/normalization.")}
                  </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f'<p class="sec-hdr">{t("Pengaturan window & threshold","Window & threshold settings")}</p>',
                        unsafe_allow_html=True)
            w1,w2,w3 = st.columns([2,1,1])
            window_mode = w1.selectbox(
                t("Rentang analisis","Analysis range"),
                [t("Fingerprint (400–1800 cm⁻¹)","Fingerprint (400–1800 cm⁻¹)"),
                 t("Full range","Full range"),
                 t("Custom range","Custom range")]
            )
            wmin_input = w2.number_input("Min (cm⁻¹)", value=400, step=50,
                disabled="Custom" not in window_mode)
            wmax_input = w3.number_input("Max (cm⁻¹)", value=4000, step=50,
                disabled="Custom" not in window_mode)

            t1,t2,t3 = st.columns(3)
            top_n      = t1.number_input(t("Top-N kandidat","Top-N candidates"), 3, 20, 10)
            thresh_cos = t2.slider(t("Threshold cosine","Threshold cosine"), 0.70, 1.00, 0.95, 0.01)
            thresh_hqi = t3.slider(t("Threshold HQI (%)","Threshold HQI (%)"), 50.0, 100.0, 90.25, 0.25)

            m1, m2, m3 = st.columns(3)
            rank_metric = m1.selectbox(
                t("Metode ranking kandidat","Candidate ranking method"),
                [t("Komposit (rekomendasi)","Composite (recommended)"),
                 "Pearson", "Cosine",
                 t("Turunan (derivative)","Derivative"),
                 t("Toleran-geser (shift)","Shift-tolerant")],
                help=t("Komposit = rata-rata Pearson + Turunan + Toleran-geser — "
                       "paling robust karena tiap metrik punya kelemahan berbeda "
                       "(lihat penjelasan di bawah hasil) dan saling menutupi. "
                       "Cosine mentah sebaiknya hanya untuk pembanding, bukan "
                       "acuan utama.",
                       "Composite = average of Pearson + Derivative + Shift-tolerant "
                       "— most robust since each metric has different weaknesses "
                       "(see explanation below the results) that offset each other. "
                       "Raw cosine should only be used for comparison, not as the "
                       "primary reference.")
            )
            rank_map = {
                t("Komposit (rekomendasi)","Composite (recommended)"): "composite",
                "Pearson": "pearson", "Cosine": "cosine",
                t("Turunan (derivative)","Derivative"): "derivative",
                t("Toleran-geser (shift)","Shift-tolerant"): "shift",
            }
            rank_by = rank_map.get(rank_metric, "composite")
            thresh_pearson = m2.slider(t("Threshold Pearson","Pearson threshold"),
                                        0.50, 1.00, 0.90, 0.01)
            ambiguous_margin = m3.slider(
                t("Margin ambigu","Ambiguous margin"), 0.0, 0.10, 0.03, 0.005,
                help=t("Jika selisih skor kandidat #1 dan #2 lebih kecil dari nilai ini, "
                       "kandidat #1 ditandai ambigu (tidak cukup bisa dibedakan).",
                       "If the score gap between candidate #1 and #2 is smaller than "
                       "this, candidate #1 is flagged ambiguous (not clearly distinguishable).")
            )

            st.markdown(f'<p class="sec-hdr">{t("Pengaturan metrik Turunan & Geser","Derivative & Shift metric settings")}</p>',
                        unsafe_allow_html=True)
            st.caption(t(
                "Parameter ini mengatur perhitungan kolom Turunan (Savitzky-Golay) "
                "dan Geser (rentang toleransi pergeseran) pada hasil identifikasi "
                "di bawah.",
                "These parameters control the Derivative (Savitzky-Golay) and "
                "Shift (shift-tolerance range) columns in the identification "
                "results below."
            ))
            dv1, dv2, dv3, dv4 = st.columns(4)
            deriv_order_id = dv1.selectbox(
                t("Orde turunan","Derivative order"), [1, 2], index=0,
                key="ident_deriv_order",
                help=t("1 = paling umum. 2 = lebih menonjolkan puncak tajam, lebih sensitif noise.",
                       "1 = most common. 2 = emphasizes sharp peaks more, more noise-sensitive.")
            )
            sg_window_id = dv2.number_input(
                t("Window Savitzky-Golay","Savitzky-Golay window"),
                min_value=5, max_value=51, value=11, step=2,
                key="ident_sg_window",
                help=t("Harus ganjil dan > poly + 1.","Must be odd and > poly + 1.")
            )
            sg_poly_id = dv3.number_input(
                t("Polynomial order","Polynomial order"),
                min_value=1, max_value=6, value=3, step=1,
                key="ident_sg_poly"
            )
            shift_tolerance_id = dv4.number_input(
                t("Toleransi geser (titik grid)","Shift tolerance (grid points)"),
                min_value=1, max_value=30, value=5, step=1,
                key="ident_shift_tolerance",
                help=t("Rentang pencarian pergeseran ±N titik grid untuk skor Geser. "
                       "Perbesar kalau skor Geser sering mentok di lag = ±N (mungkin "
                       "pergeseran sebenarnya lebih besar dari rentang saat ini).",
                       "Search range ±N grid points for the Shift score. Increase this "
                       "if the Shift score often hits lag = ±N at the boundary (the "
                       "true shift may be larger than the current range).")
            )
            ok_deriv_id, deriv_msgs_id = validate_derivative_params(
                deriv_order_id, int(sg_poly_id), int(sg_window_id)
            )
            for _msg in deriv_msgs_id:
                if _msg.startswith("❌"):
                    st.error(_msg)
                elif _msg.startswith("⚠️"):
                    st.warning(_msg)
                else:
                    st.caption(_msg)

            st.markdown(f'<p class="sec-hdr">{t("Pengaturan penyesuaian grid","Grid alignment settings")}</p>',
                        unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            interp_method = g1.selectbox(
                t("Metode interpolasi","Interpolation method"),
                ["cubic", "linear"],
                help=t("Cubic: lebih akurat untuk puncak tajam. Linear: lebih cepat.",
                       "Cubic: more accurate for sharp peaks. Linear: faster.")
            )
            grid_mode = g2.selectbox(
                t("Interval grid bersama","Common grid interval"),
                [t("Otomatis (interval terkecil)","Auto (finest interval)"),
                 t("Manual","Manual")],
            )
            if t("Manual","Manual") in grid_mode:
                grid_interval = st.number_input(
                    t("Interval grid (cm⁻¹)","Grid interval (cm⁻¹)"),
                    min_value=0.1, max_value=10.0, value=1.0, step=0.1
                )
            else:
                grid_interval = "auto"

            if "Fingerprint" in window_mode:
                wmode_key = "fingerprint"
                wmin_show, wmax_show = 400, 1800
            elif "Custom" in window_mode:
                wmode_key = "custom"
                wmin_show, wmax_show = wmin_input, wmax_input
            else:
                wmode_key = "full"
                wmin_show = float(np.array(wn).min())
                wmax_show = float(np.array(wn).max())

            wn_arr = np.array(wn)
            n_pts_win = int(np.sum((wn_arr >= wmin_show) & (wn_arr <= wmax_show)))
            wn_interval = float(np.median(np.diff(np.sort(wn_arr)))) if len(wn_arr) > 1 else 1.0
            st.markdown(
                f'<span class="window-chip">{wmin_show:.0f}–{wmax_show:.0f} cm⁻¹</span>'
                f'<span class="window-chip">Δ {wn_interval:.4f} cm⁻¹</span>'
                f'<span style="font-size:0.8rem;color:#475569">{n_pts_win} '
                f'{t("titik · interval MCR","pts · MCR interval")}</span>',
                unsafe_allow_html=True
            )

            if st.button(t("🔍 Jalankan identifikasi","🔍 Run identification"),
                         use_container_width=True, disabled=not ok_deriv_id):
                with st.spinner(t(f"Mencocokkan vs {n_lib:,} spektra library...",
                                  f"Matching vs {n_lib:,} library spectra...")):
                    library_entries = get_all_spectra_for_matching()
                    all_results = []
                    for i in range(nc):
                        res = batch_match(
                            S[i], wn, library_entries,
                            wmode_key, wmin_show, wmax_show,
                            int(top_n), grid_interval, interp_method,
                            rank_by=rank_by, ambiguous_margin=float(ambiguous_margin),
                            deriv_order=deriv_order_id, sg_window=int(sg_window_id),
                            sg_poly=int(sg_poly_id), shift_tolerance=int(shift_tolerance_id)
                        )
                        all_results.append(res)
                    st.session_state["match_results"] = all_results
                st.success(t(f"✅ Identifikasi selesai — {nc} komponen dicocokkan vs {n_lib:,} referensi.",
                             f"✅ Identification complete — {nc} components matched vs {n_lib:,} references."))

            if "match_results" in st.session_state:
                all_results = st.session_state["match_results"]
                colors = px.colors.qualitative.Pastel

                for i, results in enumerate(all_results):
                    with st.expander(f"{t('Komponen','Component')} {i+1}", expanded=(i==0)):
                        if not results:
                            st.warning(t("Tidak ada hasil. Periksa rentang wavenumber.",
                                         "No results. Check wavenumber range."))
                            continue

                        top = results[0]
                        lib_entry = get_spectrum_by_id(top["id"])
                        if lib_entry and i < len(S):
                            fig_ov = go.Figure()
                            fig_ov.add_trace(go.Scatter(
                                x=wn, y=S[i],
                                name=f"{t('Komponen','Component')} {i+1} ({t('diproses','processed') if 'mcr_S_proc' in st.session_state else 'MCR'})",
                                line=dict(color=colors[i%len(colors)], width=1.8)
                            ))
                            sp_interp = interpolate_spectrum(
                                lib_entry["wavenumber"], lib_entry["spectrum"], wn
                            )
                            if S[i].max() > 0 and sp_interp.max() > 0:
                                sp_disp = sp_interp / sp_interp.max() * S[i].max()
                            else:
                                sp_disp = sp_interp
                            fig_ov.add_trace(go.Scatter(
                                x=wn, y=sp_disp,
                                name=f"{top['name']} ({t('referensi','reference')})",
                                line=dict(color="#f97316", width=1.5, dash="dot")
                            ))
                            fig_ov.add_vrect(
                                x0=wmin_show, x1=wmax_show,
                                fillcolor="#7dd3fc", opacity=0.04,
                                annotation_text="window",
                                annotation_position="top left"
                            )
                            pad = (wmax_show - wmin_show) * 0.05
                            fig_ov.update_layout(
                                template="plotly_dark", paper_bgcolor="#0f1117",
                                plot_bgcolor="#0f1117",
                                xaxis=dict(
                                    range=[wmax_show + pad, wmin_show - pad],
                                    gridcolor="#1e293b",
                                    title="Wavenumber (cm⁻¹)"
                                ),
                                yaxis=dict(gridcolor="#1e293b",
                                           title=t("Intensitas (norm.)","Intensity (norm.)")),
                                legend=dict(
                                    title=dict(text=t("Spektra","Spectra"),
                                               font=dict(size=10, color="#7dd3fc")),
                                    bgcolor="#161b27", bordercolor="#2a3142", borderwidth=1,
                                    font=dict(size=10, color="#e2e8f0"),
                                    orientation="v", x=1.02, xanchor="left", y=1
                                ),
                                height=260, margin=dict(l=20,r=160,t=20,b=40)
                            )
                            st.plotly_chart(fig_ov, use_container_width=True)

                        if results:
                            top_r = results[0]
                            grid_warn = top_r.get("grid_warning")
                            ov_w  = top_r.get("overlap_width", 0)
                            n_pts = top_r.get("n_common_points", 0)
                            gi    = top_r.get("grid_interval", 0)
                            i_q   = top_r.get("interval_query", 0)
                            i_l   = top_r.get("interval_lib", 0)
                            im    = top_r.get("interp_method","cubic")
                            grid_html = (
                                f'<div style="background:#0f1829;border:1px solid #1e3a5f;'
                                f'border-radius:8px;padding:8px 14px;margin-bottom:10px;'
                                f'font-family:monospace;font-size:0.76rem;color:#7dd3fc;">'
                                f'<b>{t("Info penyesuaian grid","Grid alignment info")}:</b> '
                                f'Overlap {ov_w:.1f} cm⁻¹ · '
                                f'Grid bersama {gi:.4f} cm⁻¹ · {n_pts} titik · '
                                f'Interpolasi: {im} · '
                                f'ΔMCR {i_q:.4f} cm⁻¹ · Δlib {i_l:.4f} cm⁻¹'
                                f'</div>'
                            )
                            if grid_warn:
                                grid_html += (
                                    f'<div style="background:#1a0f00;border:1px solid #f97316;'
                                    f'border-radius:8px;padding:6px 12px;margin-bottom:10px;'
                                    f'font-size:0.76rem;color:#f97316;">⚠️ {grid_warn}</div>'
                                )
                            st.markdown(grid_html, unsafe_allow_html=True)

                        for rank, r in enumerate(results, 1):
                            pear = r.get("pearson")
                            clabel, cmsg = consensus_label(
                                r["cosine"], r["hqi"], pearson=pear,
                                thresh_cos=thresh_cos, thresh_hqi=thresh_hqi,
                                thresh_pearson=thresh_pearson,
                                ambiguous=r.get("ambiguous", False)
                            )
                            card_cls = {"strong":"m-strong","medium":"m-medium",
                                        "conflict":"m-conflict","weak":"m-weak",
                                        "ambiguous":"m-medium"}.get(clabel,"m-weak")
                            cos_ok = "✓" if r["cosine"] >= thresh_cos else "✗"
                            hqi_ok = "✓" if r["hqi"] >= thresh_hqi else "✗"
                            pear_ok = "✓" if (pear is not None and pear >= thresh_pearson) else "✗"
                            pear_html = (f"Pearson: <b>{pear:.4f}</b> {pear_ok} &nbsp;|&nbsp;"
                                         if pear is not None else "")
                            deriv = r.get("derivative")
                            shift_s = r.get("shift")
                            shift_lag = r.get("shift_lag")
                            comp = r.get("composite")
                            extra_html = ""
                            if deriv is not None:
                                extra_html += f"Turunan: <b>{deriv:.4f}</b> &nbsp;|&nbsp;"
                            if shift_s is not None:
                                extra_html += (f"Geser: <b>{shift_s:.4f}</b> "
                                               f"(lag {shift_lag}) &nbsp;|&nbsp;")
                            if comp is not None:
                                extra_html += f"<b>Komposit: {comp:.4f}</b> &nbsp;|&nbsp;"

                            margin_note = ""
                            if rank == 1 and "margin_to_next" in r:
                                margin_note = (f" · Δ{t('vs #2','vs #2')}: "
                                                f"{r['margin_to_next']:.4f}")

                            st.markdown(f"""
                            <div class="match-card {card_cls}">
                              <span class="m-badge" style="background:#1e293b;color:#94a3b8;
                                font-size:0.68rem;padding:2px 8px;border-radius:4px;float:right;">
                                {cmsg}
                              </span>
                              <span class="m-name">#{rank} &nbsp; {r['name']}</span>
                              <div class="m-scores">
                                Cosine: <b>{r['cosine']:.4f}</b> {cos_ok} &nbsp;|&nbsp;
                                {pear_html}
                                {extra_html}
                                HQI: <b>{r['hqi']:.2f}%</b> {hqi_ok} &nbsp;|&nbsp;
                                {t('Kategori','Category')}: {r['category']}{margin_note}
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.caption(t(
                            "ℹ️ HQI = cosine² × 100 (turunan langsung dari cosine, bukan "
                            "ukuran independen). **Pearson** tahan baseline/envelope "
                            "bersama tapi tidak tahan pergeseran puncak atau rasio "
                            "intensitas relatif yang berubah. **Turunan** (derivative) "
                            "menghilangkan baseline total dan tahan perbedaan rasio "
                            "intensitas antar puncak. **Geser** (shift-tolerant) mencari "
                            "korelasi terbaik pada pergeseran kecil untuk mengakomodasi "
                            "drift kalibrasi. **Komposit** = rata-rata ketiganya — "
                            "jadikan ini acuan utama karena kelemahan masing-masing "
                            "metrik saling menutupi.",
                            "ℹ️ HQI = cosine² × 100 (a direct derivative of cosine, not "
                            "independent). **Pearson** is robust to a shared baseline/"
                            "envelope but not to peak shifts or changing relative peak "
                            "intensity ratios. **Derivative** removes the baseline "
                            "entirely and is robust to relative-intensity-ratio changes. "
                            "**Shift** finds the best correlation over a small shift to "
                            "accommodate calibration drift. **Composite** = average of "
                            "the three — use this as the primary reference since each "
                            "metric's weaknesses offset the others."
                        ))

        # ════════════════════════════════════════════════════════
        # SUB-SECTION ADMIN — Perbandingan Manual
        # Overlay bebas 2 spektra pilihan admin (komponen MCR, acuan
        # library, atau input eksternal manual) + metrik kemiripan,
        # tanpa banner info/peringatan seperti pada hasil auto-match
        # di atas. Hanya tersedia jika role == admin.
        # ════════════════════════════════════════════════════════
        if is_admin() and "mcr_S" in st.session_state:
            st.markdown("---")
            st.markdown(
                f'<p class="sec-hdr">🛠️ {t("Perbandingan Manual (Admin)", "Manual Comparison (Admin)")}</p>',
                unsafe_allow_html=True
            )
            st.caption(t(
                "Bandingkan bebas dua spektra pilihan Anda — komponen hasil MCR, "
                "spektra acuan dari library, atau spektra eksternal yang diinput "
                "manual — lengkap dengan overlay dan metrik kemiripan.",
                "Freely compare two spectra of your choice — an MCR result "
                "component, a library reference spectrum, or a manually entered "
                "external spectrum — complete with overlay and similarity metrics."
            ))

            _wn_manual  = st.session_state["wavenumber"]
            _nc_manual  = st.session_state["mcr_ncomp"]
            _S_manual   = st.session_state.get("mcr_S_proc", st.session_state["mcr_S"])
            _lib_manual = get_all_meta()

            def _manual_spectrum_picker(label, key_prefix):
                """
                Widget untuk memilih SATU sumber spektrum: komponen MCR,
                spektra library, atau input eksternal manual.
                Return (wavenumber_array, spectrum_array, nama_tampilan)
                atau None jika input belum lengkap.
                """
                opt_mcr  = t("Komponen MCR", "MCR Component")
                opt_lib  = t("Spektra Library", "Library Spectrum")
                opt_ext  = t("Input Eksternal", "External Input")
                src_options = [opt_mcr]
                if _lib_manual:
                    src_options.append(opt_lib)
                src_options.append(opt_ext)

                src = st.radio(label, src_options, key=f"{key_prefix}_src", horizontal=True)

                if src == opt_mcr:
                    idx = st.selectbox(
                        t("Pilih komponen", "Select component"),
                        list(range(_nc_manual)),
                        format_func=lambda i: f"{t('Komponen', 'Component')} {i+1}",
                        key=f"{key_prefix}_comp"
                    )
                    return (np.array(_wn_manual, dtype=float),
                            np.array(_S_manual[idx], dtype=float),
                            f"{t('Komponen', 'Component')} {idx+1}")

                elif src == opt_lib:
                    name_map = {f"{m['name']} (#{m['id']})": m["id"] for m in _lib_manual}
                    sel = st.selectbox(
                        t("Pilih spektra acuan", "Select reference spectrum"),
                        list(name_map.keys()), key=f"{key_prefix}_lib"
                    )
                    entry = get_spectrum_by_id(name_map[sel])
                    return (np.array(entry["wavenumber"], dtype=float),
                            np.array(entry["spectrum"], dtype=float),
                            entry["name"])

                else:  # opt_ext — Input Eksternal
                    method = st.radio(
                        t("Metode input", "Input method"),
                        [t("Paste teks", "Paste text"), t("Upload file", "Upload file")],
                        key=f"{key_prefix}_method", horizontal=True
                    )
                    if method == t("Paste teks", "Paste text"):
                        raw = st.text_area(
                            t("Data (wavenumber, intensitas) — satu pasang per baris",
                              "Data (wavenumber, intensity) — one pair per line"),
                            height=140, key=f"{key_prefix}_paste",
                            placeholder="4000,0.012\n3998,0.015\n3996,0.019\n..."
                        )
                        if not raw or not raw.strip():
                            return None
                        try:
                            wn_e, sp_e = _parse_pasted_spectrum(raw)
                        except Exception as e:
                            st.error(f"{t('Gagal membaca data', 'Failed to parse data')}: {e}")
                            return None
                        return wn_e, sp_e, t("Eksternal (paste)", "External (paste)")
                    else:
                        up = st.file_uploader(
                            t("Upload CSV / TXT / Excel (kolom: wavenumber, intensitas)",
                              "Upload CSV / TXT / Excel (columns: wavenumber, intensity)"),
                            type=["csv", "txt", "xlsx", "xls"], key=f"{key_prefix}_upload"
                        )
                        if up is None:
                            return None
                        try:
                            wn_e, sp_e = _parse_uploaded_spectrum(up)
                        except Exception as e:
                            st.error(f"{t('Gagal membaca file', 'Failed to parse file')}: {e}")
                            return None
                        return wn_e, sp_e, f"{t('Eksternal', 'External')} ({up.name})"

            col_manA, col_manB = st.columns(2)
            with col_manA:
                res_manA = _manual_spectrum_picker(t("Spektra A", "Spectrum A"), "manA")
            with col_manB:
                res_manB = _manual_spectrum_picker(t("Spektra B", "Spectrum B"), "manB")

            if res_manA and res_manB:
                wn_mA, sp_mA, name_mA = res_manA
                wn_mB, sp_mB, name_mB = res_manB

                mw1, mw2, mw3 = st.columns([2, 1, 1])
                manual_window_mode = mw1.selectbox(
                    t("Rentang analisis", "Analysis range"),
                    [t("Fingerprint (400–1800 cm⁻¹)", "Fingerprint (400–1800 cm⁻¹)"),
                     t("Full range", "Full range"),
                     t("Custom range", "Custom range")],
                    key="manual_cmp_window_mode"
                )
                manual_wmin = mw2.number_input("Min (cm⁻¹)", value=400, step=50,
                    disabled="Custom" not in manual_window_mode, key="manual_cmp_wmin")
                manual_wmax = mw3.number_input("Max (cm⁻¹)", value=4000, step=50,
                    disabled="Custom" not in manual_window_mode, key="manual_cmp_wmax")

                if "Fingerprint" in manual_window_mode:
                    _wmode, _wmn, _wmx = "fingerprint", 400, 1800
                elif "Custom" in manual_window_mode:
                    _wmode, _wmn, _wmx = "custom", manual_wmin, manual_wmax
                else:
                    _wmode, _wmn, _wmx = "full", None, None

                with st.expander(t("⚙️ Pengaturan lanjutan (Turunan & Geser)",
                                    "⚙️ Advanced settings (Derivative & Shift)")):
                    adv1, adv2, adv3, adv4 = st.columns(4)
                    manual_deriv_order = adv1.selectbox(
                        t("Orde turunan", "Derivative order"), [1, 2], index=0,
                        key="manual_cmp_deriv_order"
                    )
                    manual_sg_window = adv2.number_input(
                        t("Window SG", "SG window"),
                        min_value=5, max_value=51, value=11, step=2,
                        key="manual_cmp_sg_window"
                    )
                    manual_sg_poly = adv3.number_input(
                        "Polynomial order", min_value=1, max_value=6, value=3, step=1,
                        key="manual_cmp_sg_poly"
                    )
                    manual_shift_tol = adv4.number_input(
                        t("Toleransi geser", "Shift tolerance"),
                        min_value=1, max_value=30, value=5, step=1,
                        key="manual_cmp_shift_tol"
                    )
                    ok_manual_deriv, manual_deriv_msgs = validate_derivative_params(
                        manual_deriv_order, int(manual_sg_poly), int(manual_sg_window)
                    )
                    for _msg in manual_deriv_msgs:
                        if _msg.startswith("❌"):
                            st.error(_msg)
                        elif _msg.startswith("⚠️"):
                            st.warning(_msg)
                        else:
                            st.caption(_msg)

                if st.button(t("🔬 Bandingkan", "🔬 Compare"),
                             key="manual_cmp_btn", use_container_width=True,
                             disabled=not ok_manual_deriv):
                    _cmp = compare_two_spectra(
                        wn_mA, sp_mA, wn_mB, sp_mB,
                        window_mode=_wmode, wmin=_wmn, wmax=_wmx,
                        deriv_order=manual_deriv_order,
                        sg_window=int(manual_sg_window), sg_poly=int(manual_sg_poly),
                        shift_tolerance=int(manual_shift_tol)
                    )
                    st.session_state["manual_cmp_result"] = _cmp
                    st.session_state["manual_cmp_names"]  = (name_mA, name_mB)
                    st.session_state["manual_cmp_raw"]    = (wn_mA, sp_mA, wn_mB, sp_mB)

                if "manual_cmp_result" in st.session_state:
                    _cmp = st.session_state["manual_cmp_result"]
                    _name_mA, _name_mB = st.session_state["manual_cmp_names"]

                    if _cmp is None:
                        st.error(t(
                            "Tidak ada overlap rentang wavenumber di antara kedua spektra "
                            "yang dipilih — tidak bisa dibandingkan.",
                            "No wavenumber-range overlap between the two selected spectra "
                            "— cannot be compared."
                        ))
                    else:
                        _wn_mA, _sp_mA, _wn_mB, _sp_mB = st.session_state["manual_cmp_raw"]

                        fig_manual = go.Figure()
                        fig_manual.add_trace(go.Scatter(
                            x=_wn_mA, y=_sp_mA, name=_name_mA,
                            line=dict(color="#7dd3fc", width=1.8)
                        ))
                        _sp_mB_disp = np.array(_sp_mB, dtype=float)
                        if np.max(np.abs(_sp_mA)) > 0 and np.max(np.abs(_sp_mB_disp)) > 0:
                            _sp_mB_disp = (_sp_mB_disp / np.max(np.abs(_sp_mB_disp))
                                           * np.max(np.abs(_sp_mA)))
                        fig_manual.add_trace(go.Scatter(
                            x=_wn_mB, y=_sp_mB_disp, name=f"{_name_mB} ({t('skala disesuaikan','rescaled')})",
                            line=dict(color="#f97316", width=1.5, dash="dot")
                        ))
                        fig_manual.update_layout(
                            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                            xaxis=dict(autorange="reversed", gridcolor="#1e293b",
                                       title="Wavenumber (cm⁻¹)"),
                            yaxis=dict(gridcolor="#1e293b",
                                       title=t("Intensitas", "Intensity")),
                            legend=dict(bgcolor="#161b27", bordercolor="#2a3142", borderwidth=1,
                                        font=dict(size=10, color="#e2e8f0")),
                            height=300, margin=dict(l=20, r=20, t=20, b=40)
                        )
                        st.plotly_chart(fig_manual, use_container_width=True)

                        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                        for _col, _label, _val in [
                            (mc1, "Cosine",                    f"{_cmp['cosine']:.4f}"),
                            (mc2, "Pearson",                   f"{_cmp['pearson']:.4f}"),
                            (mc3, "HQI",                       f"{_cmp['hqi']:.2f}%"),
                            (mc4, t("Turunan", "Derivative"),  f"{_cmp['derivative']:.4f}"),
                            (mc5, t("Geser", "Shift"),         f"{_cmp['shift']:.4f} (lag {_cmp['shift_lag']})"),
                            (mc6, t("Komposit", "Composite"),  f"{_cmp['composite']:.4f}"),
                        ]:
                            _col.markdown(
                                f'<div class="metric-card"><div class="metric-value" '
                                f'style="font-size:1.05rem;">{_val}</div>'
                                f'<div class="metric-label">{_label}</div></div>',
                                unsafe_allow_html=True
                            )

# ════════════════════════════════════════════════════════════════
# TAB 5 — 2D-COS
# ════════════════════════════════════════════════════════════════
with tab_cos:
    if "spectra" not in st.session_state:
        st.info(t("Upload data spektra di tab Input Data terlebih dahulu.",
                  "Please upload spectral data in the Input Data tab first."))
    else:
        wn_cos  = st.session_state["wavenumber"]
        D_cos   = st.session_state["spectra"]
        D_input = D_cos.T

        st.markdown(f'<p class="sec-hdr">{t("Pengaturan perturbasi","Perturbation settings")}</p>',
                    unsafe_allow_html=True)

        perturb_options = list(PERTURBATION_PRESETS.keys())
        col_p1, col_p2 = st.columns(2)
        perturb_type = col_p1.selectbox(
            t("Jenis perturbasi","Perturbation type"), perturb_options,
            key="cos2d_perturb_type"
        )
        preset = PERTURBATION_PRESETS[perturb_type]
        perturb_unit = col_p2.text_input(
            t("Satuan (bisa diubah)","Unit (editable)"),
            value=preset["unit"]
        )

        if "Lainnya" in perturb_type or "Other" in perturb_type:
            perturb_name = st.text_input(
                t("Nama perturbasi","Perturbation name"),
                placeholder=t("mis. Kelembaban, Tegangan, ...","e.g. Humidity, Voltage, ...")
            )
        else:
            perturb_name = perturb_type.split("/")[0].strip()

        n_steps = D_input.shape[0]
        st.markdown(f'<p class="sec-hdr">{t("Nilai perturbasi","Perturbation values")}</p>',
                    unsafe_allow_html=True)
        st.caption(t(
            f"Masukkan {n_steps} nilai perturbasi (satu per baris atau dipisah koma)",
            f"Enter {n_steps} perturbation values (one per line or comma-separated)"
        ))

        default_vals = ", ".join([str(i+1) for i in range(n_steps)])
        perturb_input = st.text_area(
            t("Nilai perturbasi","Perturbation values"),
            value=default_vals, height=68,
            label_visibility="collapsed"
        )
        try:
            raw    = perturb_input.replace("\n", ",").replace(";", ",")
            tokens = [x.strip() for x in raw.split(",") if x.strip()]
            parsed = []
            for tok in tokens:
                try:
                    parsed.append(float(tok))
                except ValueError:
                    try:
                        parsed.append(float(tok.replace(",", ".")))
                    except ValueError:
                        pass
            perturb_vals = parsed if parsed else list(range(1, n_steps + 1))
        except Exception:
            perturb_vals = list(range(1, n_steps + 1))

        if len(perturb_vals) != n_steps:
            st.warning(t(
                f"⚠️ Jumlah nilai perturbasi ({len(perturb_vals)}) tidak sesuai "
                f"dengan jumlah spektra ({n_steps}). Menggunakan nomor urut otomatis.",
                f"⚠️ Perturbation values ({len(perturb_vals)}) don't match "
                f"spectra count ({n_steps}). Using automatic sequence."
            ))
            perturb_vals = list(range(1, n_steps + 1))

        st.markdown(f'<p class="sec-hdr">{t("Rentang wavenumber","Wavenumber range")}</p>',
                    unsafe_allow_html=True)
        wn_arr = np.array(wn_cos)
        cw1, cw2, cw3 = st.columns(3)
        cos_window = cw1.selectbox(
            t("Mode","Mode"),
            [t("Full range","Full range"),
             t("Fingerprint (400–1800 cm⁻¹)","Fingerprint (400–1800 cm⁻¹)"),
             t("Custom","Custom")]
        )
        cos_wmin = cw2.number_input("Min (cm⁻¹)", value=400, step=50,
                                    disabled="Custom" not in cos_window,
                                    key="cos2d_wmin")
        cos_wmax = cw3.number_input("Max (cm⁻¹)", value=1800, step=50,
                                    disabled="Custom" not in cos_window,
                                    key="cos2d_wmax")

        if "Fingerprint" in cos_window:
            wmin_c, wmax_c = 400, 1800
        elif "Custom" in cos_window:
            wmin_c, wmax_c = cos_wmin, cos_wmax
        else:
            wmin_c, wmax_c = None, None

        cscale = st.selectbox(
            t("Skema warna","Color scheme"),
            ["RdBu_r","RdYlBu_r","Spectral_r","Picnic","Portland","Jet"],
            index=0, key="cos2d_colorscale"
        )

        if st.button(t("▶ Jalankan 2D-COS","▶ Run 2D-COS"), use_container_width=True):
            with st.spinner(t("Menghitung 2D-COS...","Computing 2D-COS...")):
                result = compute_2dcos(D_input, wn_cos, wmin_c, wmax_c)
                if result is None:
                    st.error(t("Data tidak cukup untuk 2D-COS (min 3 spektra, 4 titik wavenumber).",
                               "Insufficient data for 2D-COS (min 3 spectra, 4 wavenumber points)."))
                else:
                    st.session_state["cos2d_result"] = result
                    st.session_state["cos2d_perturb"] = perturb_vals
                    st.session_state["cos2d_unit"] = perturb_unit
                    st.session_state["cos2d_name"] = perturb_name
                    st.success(t(
                        f"✅ 2D-COS selesai — {result['n_steps']} spektra · {result['n_points']} titik wavenumber",
                        f"✅ 2D-COS complete — {result['n_steps']} spectra · {result['n_points']} wavenumber points"
                    ))

        if "cos2d_result" in st.session_state:
            res    = st.session_state["cos2d_result"]
            p_vals = st.session_state["cos2d_perturb"]
            p_unit = st.session_state["cos2d_unit"]
            p_name = st.session_state["cos2d_name"]
            wn_r   = res["wn"]
            Phi    = res["Phi"]
            Psi    = res["Psi"]
            Auto   = res["autopower"]

            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-card"><div class="metric-value">{res["n_steps"]}</div>' +
                        f'<div class="metric-label">{t("Langkah perturbasi","Perturbation steps")}</div></div>',
                        unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="metric-value">{res["n_points"]}</div>' +
                        f'<div class="metric-label">{t("Titik wavenumber","Wavenumber points")}</div></div>',
                        unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="metric-value">{Auto.max():.4f}</div>' +
                        f'<div class="metric-label">{t("Autopower maks","Max autopower")}</div></div>',
                        unsafe_allow_html=True)

            st.markdown(f'<p class="sec-hdr">{t("Spektra dinamis (Ã)","Dynamic spectra (Ã)")}</p>',
                        unsafe_allow_html=True)
            fig_dyn = go.Figure()
            colors_d = px.colors.qualitative.Set2
            for i, row in enumerate(res["D_dyn"]):
                p_val = p_vals[i] if i < len(p_vals) else i + 1
                lbl = f"{p_name} {p_val} {p_unit}"
                fig_dyn.add_trace(go.Scatter(
                    x=wn_r, y=row, name=lbl, mode="lines",
                    line=dict(width=1.4, color=colors_d[i % len(colors_d)])
                ))
            fig_dyn.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(autorange="reversed", gridcolor="#1e293b", title="Wavenumber (cm⁻¹)"),
                yaxis=dict(gridcolor="#1e293b", title=t("Intensitas dinamis","Dynamic intensity")),
                legend=dict(bgcolor="#161b27", bordercolor="#2a3142", borderwidth=1,
                    font=dict(size=11, color="#e2e8f0"),
                    title=dict(text=f"{p_name} ({p_unit})",
                               font=dict(size=11, color="#7dd3fc")),
                    orientation="v", x=1.02, y=1, xanchor="left"),
                height=300, margin=dict(l=20, r=160, t=20, b=40)
            )
            st.plotly_chart(fig_dyn, use_container_width=True)

            st.markdown(f'<p class="sec-hdr">{t("Autopower spectrum — puncak aktif","Autopower spectrum — active bands")}</p>',
                        unsafe_allow_html=True)
            fig_auto = go.Figure()
            fig_auto.add_trace(go.Scatter(
                x=wn_r, y=Auto, mode="lines",
                line=dict(color="#7dd3fc", width=1.8),
                fill="tozeroy", fillcolor="rgba(125,211,252,0.08)"
            ))
            fig_auto.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(autorange="reversed", gridcolor="#1e293b", title="Wavenumber (cm⁻¹)"),
                yaxis=dict(gridcolor="#1e293b", title="Autopower"),
                height=220, margin=dict(l=20,r=20,t=10,b=40)
            )
            st.plotly_chart(fig_auto, use_container_width=True)

            st.markdown(f'<p class="sec-hdr">{t("Peta 2D-COS","2D-COS maps")}</p>',
                        unsafe_allow_html=True)
            col_syn, col_asyn = st.columns(2)

            with col_syn:
                st.caption(t("Synchronous (Φ) — perubahan searah","Synchronous (Φ) — in-phase changes"))
                vmax_phi = float(np.abs(Phi).max())
                fig_phi = go.Figure(go.Heatmap(
                    z=Phi, x=wn_r, y=wn_r, colorscale=cscale,
                    zmid=0, zmin=-vmax_phi, zmax=vmax_phi,
                    colorbar=dict(title="Φ", thickness=12, len=0.8)
                ))
                fig_phi.update_layout(
                    template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    xaxis=dict(autorange="reversed", gridcolor="#1e293b", title="ν₁ (cm⁻¹)"),
                    yaxis=dict(autorange="reversed", gridcolor="#1e293b", title="ν₂ (cm⁻¹)"),
                    height=420, margin=dict(l=20,r=20,t=10,b=40)
                )
                st.plotly_chart(fig_phi, use_container_width=True)

            with col_asyn:
                st.caption(t("Asynchronous (Ψ) — urutan kejadian","Asynchronous (Ψ) — sequential order"))
                vmax_psi = float(np.abs(Psi).max())
                fig_psi = go.Figure(go.Heatmap(
                    z=Psi, x=wn_r, y=wn_r, colorscale=cscale,
                    zmid=0, zmin=-vmax_psi, zmax=vmax_psi,
                    colorbar=dict(title="Ψ", thickness=12, len=0.8)
                ))
                fig_psi.update_layout(
                    template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    xaxis=dict(autorange="reversed", gridcolor="#1e293b", title="ν₁ (cm⁻¹)"),
                    yaxis=dict(autorange="reversed", gridcolor="#1e293b", title="ν₂ (cm⁻¹)"),
                    height=420, margin=dict(l=20,r=20,t=10,b=40)
                )
                st.plotly_chart(fig_psi, use_container_width=True)

            # ── Autopower: kontribusi variansi per region ────────────────
            st.markdown(f'<p class="sec-hdr">{t("Analisis kontribusi variansi & justifikasi window","Variance contribution analysis & window justification")}</p>',
                        unsafe_allow_html=True)

            total_var = float(Auto.sum())
            auto_max  = float(Auto.max())

            if total_var > 0 and auto_max > 0:
                regions_def = [
                    {"name": t("Fingerprint","Fingerprint"),
                     "range": "400–1800 cm⁻¹",
                     "min": 400, "max": 1800},
                    {"name": t("C–H stretch","C–H stretch"),
                     "range": "2800–3100 cm⁻¹",
                     "min": 2800, "max": 3100},
                    {"name": t("O–H stretch","O–H stretch"),
                     "range": "3100–3700 cm⁻¹",
                     "min": 3100, "max": 3700},
                    {"name": t("Sisa / Other","Other"),
                     "range": "1800–2800 + 3700–4000 cm⁻¹",
                     "min": None, "max": None},
                ]

                def region_stats(rdef, wn_arr, auto_arr, total_v, auto_mx):
                    wn_arr   = np.array(wn_arr)
                    auto_arr = np.array(auto_arr)
                    if rdef["min"] is None:
                        mask = ((wn_arr > 1800) & (wn_arr < 2800)) | (wn_arr > 3700)
                    else:
                        mask = (wn_arr >= rdef["min"]) & (wn_arr <= rdef["max"])
                    ap_region = auto_arr[mask]
                    var_pct   = float(ap_region.sum() / total_v * 100) if total_v > 0 else 0.0
                    rel_max   = float(ap_region.max() / auto_mx * 100) if len(ap_region) > 0 and auto_mx > 0 else 0.0
                    n_active  = int((ap_region / auto_mx * 100 >= 10).sum())
                    return var_pct, rel_max, n_active

                rows_var = []
                for rd in regions_def:
                    vp, rm, na = region_stats(rd, wn_r, Auto, total_var, auto_max)
                    if vp >= 15 and rm >= 50:
                        badge = "✅"; rec = t("Window utama MCR","Primary MCR window")
                    elif vp >= 5:
                        badge = "🟡"; rec = t("Window tambahan","Secondary window")
                    else:
                        badge = "❌"; rec = t("Tidak direkomendasikan","Not recommended")
                    rows_var.append({
                        t("Region","Region"):          rd["name"],
                        t("Range","Range"):            rd["range"],
                        "Auto_rel maks (%)":           round(rm, 1),
                        t("Kontribusi variansi (%)","Variance contribution (%)"):
                                                       round(vp, 1),
                        t("Rekomendasi","Recommendation"): f"{badge} {rec}",
                    })

                df_var = pd.DataFrame(rows_var)
                st.dataframe(df_var, use_container_width=True, hide_index=True)

                # Justifikasi otomatis untuk window terbaik
                best     = max(rows_var,
                               key=lambda x: x[t("Kontribusi variansi (%)","Variance contribution (%)")])
                best_var  = best[t("Kontribusi variansi (%)","Variance contribution (%)")]
                best_rel  = best["Auto_rel maks (%)"]
                best_name = best[t("Region","Region")]
                best_rng  = best[t("Range","Range")]
                other_var = round(100 - best_var, 1)

                if best_var >= 80:
                    box_color = "#0d2018"; border_color = "#22c55e"; text_color = "#4ade80"
                    strength  = t("sangat kuat (≥80%)","very strong (≥80%)")
                elif best_var >= 60:
                    box_color = "#1a1a08"; border_color = "#eab308"; text_color = "#fde047"
                    strength  = t("cukup kuat (60–80%)","moderate (60–80%)")
                else:
                    box_color = "#1a0a08"; border_color = "#ef4444"; text_color = "#f87171"
                    strength  = t("lemah (<60%)","weak (<60%)")

                justification_id = (
                    f"Pemilihan window analisis <b>{best_rng}</b> ({best_name} region) "
                    f"didasarkan pada distribusi autopower spectrum hasil 2D-COS yang menunjukkan "
                    f"<b>{best_var}%</b> total variansi spektral terkonsentrasi di region ini, "
                    f"dengan autopower relatif maksimum <b>{best_rel}%</b> terhadap keseluruhan spektrum. "
                    f"Band di luar region ini menunjukkan kontribusi variansi kumulatif yang tidak signifikan "
                    f"(&lt;{other_var}%), sehingga tidak memberikan informasi diskriminatif tambahan "
                    f"untuk resolusi MCR-ALS."
                )
                justification_en = (
                    f"The analytical window <b>{best_rng}</b> ({best_name} region) was selected "
                    f"based on the 2D-COS autopower spectrum distribution, which revealed that "
                    f"<b>{best_var}%</b> of total spectral variance is concentrated in this region, "
                    f"with a maximum relative autopower of <b>{best_rel}%</b>. "
                    f"Bands outside this region contribute &lt;{other_var}% of cumulative variance "
                    f"and thus provide insufficient discriminative information for MCR-ALS resolution."
                )
                justification_text = justification_en if lang == "en" else justification_id

                st.markdown(f"""
                <div style="background:{box_color};border:1px solid {border_color};
                  border-radius:10px;padding:1rem 1.2rem;margin-top:0.8rem;">
                  <div style="font-size:0.72rem;color:{text_color};font-family:'DM Mono',monospace;
                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">
                    {t("Justifikasi window otomatis","Auto window justification")}
                    &nbsp;·&nbsp; {t("Kekuatan:","Strength:")} {strength}
                  </div>
                  <div style="font-size:0.85rem;color:#e2e8f0;line-height:1.7;">
                    {justification_text}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Download justifikasi sebagai TXT
                plain_id = (justification_id
                            .replace("<b>","").replace("</b>","").replace("&lt;","<"))
                plain_en = (justification_en
                            .replace("<b>","").replace("</b>","").replace("&lt;","<"))
                dl_text = (
                    f"=== Justifikasi Window (ID) ===\n{plain_id}\n\n"
                    f"=== Window Justification (EN) ===\n{plain_en}"
                )
                st.download_button(
                    t("⬇ Download justifikasi (TXT)","⬇ Download justification (TXT)"),
                    dl_text,
                    f"window_justification_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    "text/plain"
                )

            st.markdown(f'<p class="sec-hdr">{t("Analisis cross-peak & Noda\'s Rules","Cross-peak analysis & Noda\'s Rules")}</p>',
                        unsafe_allow_html=True)
            cp_cols = st.columns(2)
            thr_phi = cp_cols[0].number_input(
                t("Threshold |Φ| minimum","Threshold |Φ| minimum"),
                min_value=0.0, value=0.0, step=0.0001, format="%.4f",
                key="cos2d_thr_phi"
            )
            top_cp = cp_cols[1].number_input(
                t("Tampilkan Top-N cross-peak","Show Top-N cross-peaks"),
                min_value=5, max_value=50, value=15, key="cos2d_top_cp"
            )

            crosspeaks = find_crosspeaks(Phi, Psi, wn_r,
                                         threshold_phi=thr_phi,
                                         threshold_psi=0.0,
                                         top_n=int(top_cp))
            if crosspeaks:
                for cp in crosspeaks:
                    phi_color = "#22c55e" if cp["phi"] > 0 else "#ef4444"
                    psi_color = "#22c55e" if cp["psi"] > 0 else "#ef4444"
                    st.markdown(f"""
                    <div style="background:#161b27;border:1px solid #2a3142;
                      border-radius:8px;padding:0.65rem 1rem;margin-bottom:5px;font-size:0.82rem;">
                      <span style="color:#e2e8f0;font-weight:500;">
                        {cp["wn1"]:.1f} cm⁻¹ &nbsp;↔&nbsp; {cp["wn2"]:.1f} cm⁻¹
                      </span>
                      &nbsp;&nbsp;
                      <span style="color:{phi_color};font-family:monospace;">Φ={cp["phi"]:+.4f}</span>
                      &nbsp;
                      <span style="color:{psi_color};font-family:monospace;">Ψ={cp["psi"]:+.4f}</span>
                      <br>
                      <span style="color:#7dd3fc;">⚖ {cp["noda_rule"]}</span>
                      &nbsp;·&nbsp;
                      <span style="color:#94a3b8;">{t("Urutan:","Order:")} {cp["order"]}</span>
                    </div>
                    """, unsafe_allow_html=True)

                df_cp = pd.DataFrame(crosspeaks)[
                    ["wn1","wn2","phi","psi","sign_phi","sign_psi","noda_rule","order"]
                ]
                df_cp.columns = ["ν₁ (cm⁻¹)","ν₂ (cm⁻¹)","Φ","Ψ","sign(Φ)","sign(Ψ)","Noda Rule","Sequential Order"]
                st.download_button(
                    t("⬇ Export cross-peaks (CSV)","⬇ Export cross-peaks (CSV)"),
                    df_cp.to_csv(index=False), "crosspeaks_2dcos.csv", "text/csv"
                )
            else:
                st.info(t("Tidak ada cross-peak signifikan dengan threshold ini.",
                           "No significant cross-peaks at this threshold."))

            st.markdown(f'<p class="sec-hdr">{t("Export peta 2D-COS","Export 2D-COS maps")}</p>',
                        unsafe_allow_html=True)
            ex1, ex2 = st.columns(2)
            df_phi_exp = pd.DataFrame(Phi, index=wn_r, columns=wn_r)
            df_phi_exp.index.name = "Wavenumber"
            df_psi_exp = pd.DataFrame(Psi, index=wn_r, columns=wn_r)
            df_psi_exp.index.name = "Wavenumber"
            ex1.download_button(
                t("⬇ Synchronous map (CSV)","⬇ Synchronous map (CSV)"),
                df_phi_exp.to_csv(), "synchronous_map.csv", "text/csv", use_container_width=True
            )
            ex2.download_button(
                t("⬇ Asynchronous map (CSV)","⬇ Asynchronous map (CSV)"),
                df_psi_exp.to_csv(), "asynchronous_map.csv", "text/csv", use_container_width=True
            )

# ════════════════════════════════════════════════════════════════
# TAB — ANALISIS TURUNAN (DERIVATIVE ANALYSIS)
# ════════════════════════════════════════════════════════════════
with tab_deriv:
    st.markdown(f'<p class="sec-hdr">{t("Analisis MCR pada domain spektra turunan","MCR analysis on derivative-domain spectra")}</p>',
                unsafe_allow_html=True)
    st.info(t(
        "Fitur ini **terpisah** dari tab Analisis MCR biasa karena domain turunan "
        "memerlukan constraint yang berbeda secara fundamental: non-negativity pada "
        "spektra (S) HARUS dimatikan (nilai negatif adalah bagian valid dari bentuk "
        "turunan), dan library acuan harus berupa spektrum turunan dengan parameter SG "
        "yang identik — bukan spektrum mentah.",
        "This feature is **separate** from the normal MCR tab because the derivative "
        "domain requires fundamentally different constraints: non-negativity on the "
        "spectra (S) MUST be turned off (negative values are a valid part of the "
        "derivative shape), and the reference library must be derivative spectra with "
        "identical SG parameters — not raw spectra."
    ))

    if "spectra" not in st.session_state:
        st.warning(t("Upload data campuran di tab **📂 Input Data** terlebih dahulu.",
                     "Upload mixture data in the **📂 Input Data** tab first."))
    else:
        wavenumber = st.session_state["wavenumber"]
        # PENTING: st.session_state["spectra"] disimpan sebagai
        # (n_wavenumber x n_sampel) — konvensi yang sama dipakai tab_mcr
        # (baris "D = st.session_state['spectra'].T") dan tab 2D-COS
        # (baris "D_input = D_cos.T"). apply_derivative() memproses per
        # BARIS sebagai satu spektrum sepanjang wavenumber, jadi wajib
        # ditranspose dulu ke (n_sampel x n_wavenumber) di sini juga.
        raw_spectra = st.session_state["spectra"].T

        # ── A. Buat spektra turunan ────────────────────────────
        st.markdown(f'<p class="sec-hdr">{t("A. Buat spektra turunan (data campuran)","A. Generate derivative spectra (mixture data)")}</p>',
                    unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        deriv_order = d1.selectbox(t("Derivative order","Derivative order"), [1, 2], index=0,
                                   help=t("2nd derivative memberi resolusi enhancement lebih kuat untuk pita yang saling tumpang tindih, tapi noise jauh lebih tinggi.",
                                          "2nd derivative gives stronger resolution enhancement for overlapping bands, but much higher noise."))
        deriv_poly = d2.number_input(t("Polynomial order","Polynomial order"),
                                     min_value=1, max_value=6, value=deriv_order + 1, step=1)
        deriv_window = d3.number_input(t("Smoothing points","Smoothing points"),
                                       min_value=5, max_value=51, value=15, step=2)
        deriv_symmetric = d4.checkbox(t("Symmetric kernel","Symmetric kernel"), value=True,
                                      disabled=True,
                                      help=t("Selalu simetris di titik interior (standar Savitzky-Golay) — wajib untuk menghindari pergeseran posisi puncak yang merusak validitas matching terhadap library.",
                                             "Always symmetric at interior points (standard Savitzky-Golay) — required to avoid peak-position shift that would invalidate library matching."))

        ok_params, param_messages = validate_derivative_params(deriv_order, int(deriv_poly), int(deriv_window))
        for msg in param_messages:
            if msg.startswith("❌"):
                st.error(msg)
            elif msg.startswith("⚠️"):
                st.warning(msg)
            else:
                st.caption(msg)

        if st.button(t("🧬 Buat Spektra Turunan","🧬 Generate Derivative Spectra"),
                    disabled=not ok_params, use_container_width=True):
            deriv_matrix, deriv_info = apply_derivative(
                raw_spectra, order=deriv_order, poly=int(deriv_poly),
                window=int(deriv_window), symmetric=True
            )
            st.session_state["deriv_spectra"]   = deriv_matrix
            st.session_state["deriv_wavenumber"] = wavenumber
            st.session_state["deriv_order"]      = deriv_order
            st.session_state["deriv_poly"]       = int(deriv_poly)
            st.session_state["deriv_window"]     = deriv_info["window"]
            for _k in ["deriv_mcr_C","deriv_mcr_S","deriv_mcr_lof","deriv_mcr_diag"]:
                st.session_state.pop(_k, None)
            st.success(t(f"Spektra turunan orde-{deriv_order} berhasil dibuat (window={deriv_info['window']}, poly={int(deriv_poly)}).",
                         f"Order-{deriv_order} derivative spectra generated (window={deriv_info['window']}, poly={int(deriv_poly)})."))

        if "deriv_spectra" in st.session_state:
            dwn = st.session_state["deriv_wavenumber"]
            dsp = st.session_state["deriv_spectra"]
            fig_d = go.Figure()
            n_show = min(5, dsp.shape[0])
            for i in range(n_show):
                fig_d.add_trace(go.Scatter(x=dwn, y=dsp[i], mode="lines",
                                           name=st.session_state.get("spec_names", [f"S{i}" for i in range(dsp.shape[0])])[i]))
            fig_d.add_hline(y=0, line_dash="dot", line_color="#475569")
            fig_d.update_layout(template="plotly_dark", height=340,
                                title=t(f"Spektra turunan orde-{st.session_state['deriv_order']} (contoh {n_show} sampel)",
                                        f"Order-{st.session_state['deriv_order']} derivative spectra (sample of {n_show})"),
                                xaxis_title="Wavenumber (cm⁻¹)", yaxis_title=t("Turunan","Derivative"))
            st.plotly_chart(fig_d, use_container_width=True)

            # ── B. Jalankan MCR pada domain turunan ─────────────
            st.markdown(f'<p class="sec-hdr">{t("B. Jalankan MCR-ALS pada domain turunan","B. Run MCR-ALS on derivative domain")}</p>',
                        unsafe_allow_html=True)
            st.caption(t(
                "🔒 Non-negativity pada S otomatis **dimatikan** untuk domain ini (bukan pilihan) — "
                "non-negativity pada C tetap aktif karena konsentrasi fisik tidak pernah negatif.",
                "🔒 Non-negativity on S is automatically **disabled** for this domain (not optional) — "
                "non-negativity on C stays active because physical concentration is never negative."
            ))
            e1, e2, e3, e4 = st.columns(4)
            n_comp_d = e1.number_input(t("Jumlah komponen (k)","Number of components (k)"),
                                       min_value=2, max_value=10, value=3, step=1, key="deriv_ncomp")
            init_method_d = e2.selectbox(t("Metode inisialisasi","Init method"),
                                         ["pca", "library"], index=0, key="deriv_init",
                                         help=t("SIMPLISMA dan NMF tidak ditawarkan di domain turunan karena keduanya mengasumsikan data non-negatif — tidak valid untuk spektrum turunan.",
                                                "SIMPLISMA and NMF are not offered in the derivative domain because both assume non-negative data — not valid for derivative spectra."))
            unimodal_C_d = e3.checkbox(t("Unimodality (C)","Unimodality (C)"), value=False, key="deriv_unimodal_c",
                                       help=t("Aktifkan hanya jika desain sampel Anda memang menghasilkan profil konsentrasi satu-puncak (mis. seri dilusi bertingkat).",
                                              "Enable only if your sample design produces a genuinely single-peak concentration profile (e.g. a stepped dilution series)."))
            max_iter_d = e4.number_input(t("Max iterasi","Max iterations"),
                                         min_value=50, max_value=500, value=150, step=10, key="deriv_maxiter")

            if st.button(t("▶️ Jalankan MCR-ALS (Turunan)","▶️ Run MCR-ALS (Derivative)"), use_container_width=True):
                C_d, S_d, lof_hist_d, r2_d, conv_d, diag_d = run_mcr_als(
                    dsp, int(n_comp_d), max_iter=int(max_iter_d),
                    closure=False, unimodal=False, normalize_S=True,
                    init_method=init_method_d, unimodal_C=unimodal_C_d,
                    s_nonneg=False   # domain turunan — dikunci, bukan pilihan user
                )
                st.session_state["deriv_mcr_C"]    = C_d
                st.session_state["deriv_mcr_S"]    = S_d
                st.session_state["deriv_mcr_lof"]  = lof_hist_d
                st.session_state["deriv_mcr_diag"] = diag_d

            if "deriv_mcr_S" in st.session_state:
                C_d = st.session_state["deriv_mcr_C"]
                S_d = st.session_state["deriv_mcr_S"]
                diag_d = st.session_state["deriv_mcr_diag"]

                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f'<div class="metric-card"><div class="metric-value">{diag_d["lof_final"]:.2f}%</div>'
                           f'<div class="metric-label">LOF</div></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="metric-card"><div class="metric-value">{diag_d["rmse"]:.5f}</div>'
                           f'<div class="metric-label">RMSE</div></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="metric-card"><div class="metric-value">{diag_d["n_iter"]}</div>'
                           f'<div class="metric-label">{t("Iterasi","Iterations")}</div></div>', unsafe_allow_html=True)
                m4.markdown(f'<div class="metric-card"><div class="metric-value">{"✅" if diag_d["converged"] else "⚠️"}</div>'
                           f'<div class="metric-label">{t("Konvergen","Converged")}</div></div>', unsafe_allow_html=True)

                st.caption(t(
                    "ℹ️ Skor NNV (non-negativity violation) tidak bermakna untuk hasil ini — nilai "
                    "negatif pada S di domain turunan adalah normal, bukan pelanggaran.",
                    "ℹ️ NNV (non-negativity violation) score is not meaningful for this result — "
                    "negative values in S in the derivative domain are normal, not a violation."
                ))

                fig_s = go.Figure()
                for i in range(S_d.shape[0]):
                    fig_s.add_trace(go.Scatter(x=dwn, y=S_d[i], mode="lines", name=f"Komponen {i+1}"))
                fig_s.add_hline(y=0, line_dash="dot", line_color="#475569")
                fig_s.update_layout(template="plotly_dark", height=340,
                                    title=t("Spektra murni hasil MCR (domain turunan)","Pure spectra from MCR (derivative domain)"),
                                    xaxis_title="Wavenumber (cm⁻¹)", yaxis_title=t("Turunan","Derivative"))
                st.plotly_chart(fig_s, use_container_width=True)

        # ── C. Library turunan ──────────────────────────────────
        st.markdown(f'<p class="sec-hdr">{t("C. Library spektra turunan (acuan murni)","C. Derivative reference library")}</p>',
                    unsafe_allow_html=True)
        n_lib_d = count_derivative_spectra(st.session_state.get("deriv_order"))
        st.caption(t(
            f"Entri library untuk orde-{st.session_state.get('deriv_order','?')} saat ini: **{n_lib_d}**. "
            "Matching hanya dilakukan terhadap entri dengan orde turunan yang sama persis.",
            f"Library entries for order-{st.session_state.get('deriv_order','?')} currently: **{n_lib_d}**. "
            "Matching only compares against entries with the exact same derivative order."
        ))

        if is_admin():
            with st.expander(t("➕ Tambah acuan turunan ke library","➕ Add a derivative reference to the library")):
                ref_upload = st.file_uploader(
                    t("Upload spektrum murni MENTAH (akan diturunkan otomatis dengan parameter yang sama seperti di atas)",
                      "Upload RAW pure spectrum (will be auto-derivatized with the same parameters as above)"),
                    type=["xlsx","xls","csv","txt"], key="deriv_ref_upload")
                rn1, rn2, rn3 = st.columns(3)
                ref_name = rn1.text_input(t("Nama komponen","Component name"), key="deriv_ref_name")
                ref_cat  = rn2.text_input(t("Kategori","Category"), key="deriv_ref_cat")
                ref_sub  = rn3.text_input(t("Subkategori","Subcategory"), key="deriv_ref_sub")

                if ref_upload and ref_name and "deriv_order" in st.session_state:
                    try:
                        rname = ref_upload.name.lower()
                        rdf = (pd.read_excel(ref_upload) if rname.endswith((".xlsx",".xls"))
                              else pd.read_csv(ref_upload, sep=None, engine="python", comment="#"))
                        r_wn = rdf.iloc[:, 0].values.astype(float)
                        r_sp = rdf.iloc[:, 1].values.astype(float)
                        r_deriv, r_info = apply_derivative(
                            r_sp.reshape(1, -1),
                            order=st.session_state["deriv_order"],
                            poly=st.session_state["deriv_poly"],
                            window=st.session_state["deriv_window"],
                            symmetric=True
                        )
                        if st.button(t("Simpan ke library turunan","Save to derivative library"), key="deriv_ref_save"):
                            new_id = add_derivative_spectrum(
                                name=ref_name, category=ref_cat, subcategory=ref_sub,
                                deriv_order=st.session_state["deriv_order"],
                                sg_window=st.session_state["deriv_window"],
                                sg_poly=st.session_state["deriv_poly"],
                                wavenumber=r_wn, spectrum=r_deriv[0],
                                added_by=st.session_state.get("username", "admin"),
                            )
                            st.success(t(f"Tersimpan (id={new_id}). Parameter SG dicatat otomatis agar konsisten saat matching.",
                                         f"Saved (id={new_id}). SG parameters recorded automatically for matching consistency."))
                    except Exception as e:
                        st.error(t(f"Gagal memproses file: {e}", f"Failed to process file: {e}"))
                elif ref_upload and not ref_name:
                    st.warning(t("Isi nama komponen terlebih dahulu.","Enter a component name first."))

        # Daftar & pratinjau library — TERLIHAT untuk semua pengguna (bukan
        # cuma admin), karena keputusan rentang analisis di bagian D perlu
        # info ini. Hanya tombol tambah/hapus yang tetap admin-only.
        lib_meta_d = get_all_derivative_meta(st.session_state.get("deriv_order"))
        if lib_meta_d:
            for row in lib_meta_d:
                if is_admin():
                    lc1, lc2 = st.columns([5, 1])
                    lc1.markdown(f"**{row['name']}** · {row.get('category','')} "
                                f"· order-{row['deriv_order']} (win={row['sg_window']}, poly={row['sg_poly']})")
                    if lc2.button("🗑️", key=f"del_deriv_{row['id']}"):
                        delete_derivative_spectrum(row['id'])
                        st.rerun()
                else:
                    st.markdown(f"**{row['name']}** · {row.get('category','')} "
                                f"· order-{row['deriv_order']} (win={row['sg_window']}, poly={row['sg_poly']})")

            st.markdown(f'<p class="sec-hdr" style="margin-top:0.8rem;">'
                        f'{t("Pratinjau bentuk spektra turunan","Preview of derivative spectra shape")}</p>',
                        unsafe_allow_html=True)
            st.caption(t(
                "Gunakan pratinjau ini untuk melihat DI MANA pita-pita penciri "
                "setiap acuan sebenarnya berada, sebelum memutuskan rentang "
                "analisis (full/fingerprint/custom) di bagian D — area biru "
                "menandai fingerprint (400–1800 cm⁻¹).",
                "Use this preview to see WHERE each reference's diagnostic bands "
                "actually sit before choosing the analysis range (full/"
                "fingerprint/custom) in section D — the blue band marks the "
                "fingerprint region (400–1800 cm⁻¹)."
            ))
            preview_names = [f"{r['name']} (id={r['id']})" for r in lib_meta_d]
            default_preview = preview_names[:min(3, len(preview_names))]
            picked = st.multiselect(
                t("Pilih acuan untuk ditampilkan","Select references to display"),
                preview_names, default=default_preview, key="deriv_lib_preview_pick"
            )
            if picked:
                fig_libprev = go.Figure()
                colors_lp = px.colors.qualitative.Pastel
                all_wn_min, all_wn_max = [], []
                for k, label in enumerate(picked):
                    rid = int(label.split("id=")[1].rstrip(")"))
                    entry = get_derivative_spectrum_by_id(rid)
                    if not entry:
                        continue
                    wn_e = np.array(entry["wavenumber"], dtype=float)
                    sp_e = np.array(entry["spectrum"], dtype=float)
                    peak_abs = np.max(np.abs(sp_e)) or 1.0
                    fig_libprev.add_trace(go.Scatter(
                        x=wn_e, y=sp_e / peak_abs,
                        name=entry["name"],
                        line=dict(color=colors_lp[k % len(colors_lp)], width=1.5)
                    ))
                    all_wn_min.append(wn_e.min())
                    all_wn_max.append(wn_e.max())
                if all_wn_min:
                    fig_libprev.add_vrect(
                        x0=400, x1=1800, fillcolor="#7dd3fc", opacity=0.06,
                        annotation_text="fingerprint", annotation_position="top left"
                    )
                    span_min, span_max = min(all_wn_min), max(all_wn_max)
                    pad = (span_max - span_min) * 0.03 if span_max > span_min else 10
                    fig_libprev.update_layout(
                        template="plotly_dark", paper_bgcolor="#0f1117",
                        plot_bgcolor="#0f1117", height=340,
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis=dict(range=[span_max + pad, span_min - pad],
                                  gridcolor="#1e293b", title="Wavenumber (cm⁻¹)"),
                        yaxis=dict(gridcolor="#1e293b",
                                  title=t("Intensitas turunan (norm.)","Derivative intensity (norm.)")),
                        legend=dict(orientation="h", y=-0.2)
                    )
                    st.plotly_chart(fig_libprev, use_container_width=True)

        # ── D. Matching domain turunan ──────────────────────────
        st.markdown(f'<p class="sec-hdr">{t("D. Cocokkan komponen MCR dengan library turunan","D. Match MCR component against derivative library")}</p>',
                    unsafe_allow_html=True)
        if "deriv_mcr_S" not in st.session_state:
            st.info(t("Jalankan MCR-ALS pada bagian B terlebih dahulu.","Run MCR-ALS in section B first."))
        else:
            S_d = st.session_state["deriv_mcr_S"]
            dwn = st.session_state["deriv_wavenumber"]
            comp_idx_d = st.selectbox(
                t("Pilih komponen","Select component"),
                list(range(S_d.shape[0])),
                format_func=lambda i: f"Komponen {i+1}", key="deriv_match_comp"
            )

            rd1, rd2, rd3 = st.columns([2, 1, 1])
            range_mode_d = rd1.selectbox(
                t("Rentang analisis","Analysis range"),
                [t("Fingerprint (400–1800 cm⁻¹)","Fingerprint (400–1800 cm⁻¹)"),
                 t("Full range","Full range"),
                 t("Custom range","Custom range")],
                key="deriv_range_mode",
                help=t(
                    "Fingerprint direkomendasikan default: di luar 400–1800 cm⁻¹, "
                    "region domain turunan umumnya ~0 di semua komponen dan bisa "
                    "menggembungkan skor kemiripan secara artifisial jika ikut "
                    "dihitung. Pakai Full/Custom kalau pita penciri diketahui ada "
                    "di luar fingerprint (lihat pratinjau di bagian C).",
                    "Fingerprint is the recommended default: outside 400–1800 cm⁻¹, "
                    "the derivative domain is usually ~0 for every component and "
                    "can artificially inflate similarity scores if included. Use "
                    "Full/Custom if diagnostic bands are known to sit outside the "
                    "fingerprint region (see the preview in section C)."
                )
            )
            wmin_input_d = rd2.number_input("Min (cm⁻¹)", value=400, step=50,
                disabled="Custom" not in range_mode_d, key="deriv_wmin")
            wmax_input_d = rd3.number_input("Max (cm⁻¹)", value=4000, step=50,
                disabled="Custom" not in range_mode_d, key="deriv_wmax")
            if "Fingerprint" in range_mode_d:
                wmode_key_d, wmin_d, wmax_d = "fingerprint", None, None
            elif "Custom" in range_mode_d:
                wmode_key_d, wmin_d, wmax_d = "custom", wmin_input_d, wmax_input_d
            else:
                wmode_key_d, wmin_d, wmax_d = "full", None, None
                st.caption(t(
                    "⚠️ Full range dipilih — pastikan ini disengaja, bukan bawaan; "
                    "region non-pita di luar fingerprint bisa mendominasi skor "
                    "korelasi bentuk kurva (Pearson/shift) kalau proporsinya besar.",
                    "⚠️ Full range selected — make sure this is intentional; "
                    "non-band regions outside the fingerprint can dominate the "
                    "curve-shape scores (Pearson/shift) if they make up a large share."
                ))

            tol_cm = st.slider(t("Toleransi pergeseran posisi pita (cm⁻¹)","Band position shift tolerance (cm⁻¹)"),
                               1.0, 15.0, 6.0, 0.5, key="deriv_tol")

            if st.button(t("🔎 Cocokkan dengan Library Turunan","🔎 Match against Derivative Library"), use_container_width=True):
                lib_entries_d = get_all_derivative_spectra_for_matching(st.session_state["deriv_order"])
                if not lib_entries_d:
                    st.warning(t("Belum ada entri library untuk orde turunan ini.",
                                 "No library entries yet for this derivative order."))
                else:
                    match_results_d = batch_match_derivative(
                        S_d[comp_idx_d], dwn, lib_entries_d,
                        deriv_order=st.session_state["deriv_order"],
                        window_mode=wmode_key_d, wmin=wmin_d, wmax=wmax_d,
                        tolerance_cm=tol_cm
                    )
                    st.session_state["deriv_match_results"] = match_results_d

            if "deriv_match_results" in st.session_state:
                for r in st.session_state["deriv_match_results"]:
                    amb = " 🟠 AMBIGU" if r["ambiguous"] else ""
                    intensity_txt = (f"<b>{r['peak_intensity_agreement']*100:.0f}%</b>"
                                      if r['peak_intensity_agreement'] is not None
                                      else t("tidak ada pita cocok", "no matching bands"))
                    st.markdown(f"""
                    <div class="match-card m-medium">
                      <div class="m-name">{r['name']}{amb}</div>
                      <div class="m-scores">
                        Composite: <b>{r['composite']:.3f}</b> &nbsp;|&nbsp;
                        Pearson: {r['pearson']:.3f} &nbsp;|&nbsp;
                        Shift-corr: {r['shift']:.3f} &nbsp;|&nbsp;
                        {t('Bentuk kurva','Shape score')}: {r['shape_score']:.3f}<br/>
                        {t('Kecocokan posisi pita','Peak position match')}: 
                        <b>{r['peak_match_fraction']*100:.0f}%</b>
                        ({r['n_peaks_query']} vs {r['n_peaks_ref']} 
                        {t('pita terdeteksi','bands detected')})
                        {f", rata-rata shift {r['peak_mean_abs_shift']:.1f} cm⁻¹" if r['peak_mean_abs_shift'] is not None else ""}<br/>
                        {t('Kecocokan intensitas pita','Peak intensity match')}: 
                        {intensity_txt}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 6 — LIBRARY (ADMIN ONLY)
# ════════════════════════════════════════════════════════════════
if is_admin() and tab_lib:
    with tab_lib:
        st.markdown(f'<p class="sec-hdr">{t("Tambah spektra acuan","Add reference spectrum")}</p>',
                    unsafe_allow_html=True)

        with st.form("add_ref"):
            r1,r2 = st.columns(2)
            ref_name   = r1.text_input(t("Nama senyawa","Compound name"))
            ref_cat    = r2.text_input(t("Kategori","Category"))
            r3,r4 = st.columns(2)
            ref_subcat = r3.text_input(t("Sub-kategori","Subcategory"))
            ref_cas    = r4.text_input("CAS Number")
            ref_file   = st.file_uploader(
                t("File spektra (Excel/CSV — 2 kolom: wavenumber, absorbance)",
                  "Spectrum file (Excel/CSV — 2 cols: wavenumber, absorbance)"),
                type=["xlsx","xls","csv","txt"]
            )

            # ── Opsi preprocessing sebelum simpan ────────────
            st.markdown(f'<p style="font-size:0.78rem;color:#7dd3fc;font-family:monospace;'
                        f'text-transform:uppercase;letter-spacing:0.08em;margin:0.8rem 0 0.4rem;">'
                        f'{t("Preprocessing sebelum disimpan (opsional)","Preprocessing before saving (optional)")}</p>',
                        unsafe_allow_html=True)

            pp1, pp2, pp3 = st.columns(3)
            ref_do_baseline = pp1.checkbox(
                t("Koreksi baseline","Baseline correction"),
                value=False, key="ref_baseline",
                help=t("Kurangi nilai minimum (min subtraction)",
                       "Subtract minimum value (min subtraction)")
            )
            ref_norm_method = pp2.selectbox(
                t("Normalisasi","Normalization"),
                ["none", "vector", "max", "area", "minmax"],
                format_func=lambda x: {
                    "none":   t("Tidak ada (simpan apa adanya)","None (save as-is)"),
                    "vector": "Vector L2 — " + t("direkomendasikan untuk matching","recommended for matching"),
                    "max":    "Max intensity = 1",
                    "area":   "Area trapezoid",
                    "minmax": "Min-Max → [0, 1]",
                }.get(x, x),
                key="ref_norm"
            )
            ref_do_smooth = pp3.checkbox(
                t("Smoothing SG","SG Smoothing"),
                value=False, key="ref_smooth",
                help="Savitzky-Golay (window=11, poly=3)"
            )

            # Penjelasan metode normalisasi yang dipilih
            norm_hints = {
                "none":   t("Spektra disimpan persis seperti file input.",
                            "Spectrum saved exactly as in input file."),
                "vector": t("Dibagi L2 norm → unit vector. Optimal untuk cosine similarity & HQI matching.",
                            "Divided by L2 norm → unit vector. Optimal for cosine similarity & HQI matching."),
                "max":    t("Puncak tertinggi = 1. Mudah dibandingkan secara visual.",
                            "Highest peak = 1. Easy visual comparison."),
                "area":   t("Dibagi luas area → sebanding konsentrasi. Cocok jika data kuantitatif.",
                            "Divided by area → proportional to concentration. Good for quantitative data."),
                "minmax": t("Rentang [0,1]. Menghilangkan offset baseline.",
                            "Range [0,1]. Removes baseline offset."),
            }
            st.caption(f"ℹ️ {norm_hints[ref_norm_method]}")

            ref_notes  = st.text_area(t("Catatan","Notes"), height=68)
            submitted  = st.form_submit_button(t("Tambahkan ke Library","Add to Library"))

            if submitted:
                if not ref_name:
                    st.error(t("Nama senyawa wajib diisi.","Compound name is required."))
                elif ref_file is None:
                    st.error(t("File spektra wajib diupload.","Spectrum file is required."))
                else:
                    try:
                        fn = ref_file.name.lower()
                        if fn.endswith((".xlsx",".xls")):
                            df_r = pd.read_excel(ref_file)
                        else:
                            df_r = pd.read_csv(ref_file, sep=None, engine="python")
                        wn_r = np.array(df_r.iloc[:,0].values, dtype=float)
                        sp_r = np.array(df_r.iloc[:,1].values, dtype=float)

                        # ── Terapkan preprocessing ────────────
                        proc_steps = []

                        # 1. Baseline
                        if ref_do_baseline:
                            sp_r = sp_r - sp_r.min()
                            proc_steps.append(t("baseline","baseline"))

                        # 2. Smoothing
                        if ref_do_smooth:
                            from scipy.signal import savgol_filter
                            try:
                                sp_r = savgol_filter(sp_r, 11, 3)
                                sp_r = np.maximum(sp_r, 0)
                                proc_steps.append("SG smoothing")
                            except Exception:
                                pass

                        # 3. Normalisasi
                        if ref_norm_method == "vector":
                            nv = np.linalg.norm(sp_r)
                            if nv > 0: sp_r = sp_r / nv
                            proc_steps.append("vector L2")
                        elif ref_norm_method == "max":
                            mx = sp_r.max()
                            if mx > 0: sp_r = sp_r / mx
                            proc_steps.append("max")
                        elif ref_norm_method == "area":
                            area = (np.trapezoid(np.abs(sp_r), wn_r)
                                    if hasattr(np, "trapezoid")
                                    else np.trapz(np.abs(sp_r), wn_r))
                            if area > 0: sp_r = sp_r / area
                            proc_steps.append("area")
                        elif ref_norm_method == "minmax":
                            mn, mx = sp_r.min(), sp_r.max()
                            if mx > mn: sp_r = (sp_r - mn) / (mx - mn)
                            proc_steps.append("min-max")

                        # Catat preprocessing di notes
                        proc_note = ""
                        if proc_steps:
                            proc_note = (f" | Preprocessing: {', '.join(proc_steps)}")
                        final_notes = (ref_notes + proc_note).strip(" |")

                        new_id = add_spectrum(
                            ref_name, ref_cat, ref_subcat, ref_cas,
                            final_notes, wn_r.tolist(), sp_r.tolist(),
                            added_by=st.session_state.get("username","admin")
                        )

                        proc_msg = (f" ({', '.join(proc_steps)})" if proc_steps
                                    else t(" (tanpa preprocessing)"," (no preprocessing)"))
                        st.success(t(
                            f"✅ '{ref_name}' berhasil ditambahkan (ID: {new_id}){proc_msg}.",
                            f"✅ '{ref_name}' added successfully (ID: {new_id}){proc_msg}."
                        ))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown(f'<p class="sec-hdr">{t("Import batch dari JSON","Batch import from JSON")}</p>',
                    unsafe_allow_html=True)
        imp_file = st.file_uploader(t("Upload file library (.json)","Upload library file (.json)"),
                                    type=["json"], key="imp_json")
        if imp_file:
            import json as _json
            data = _json.load(imp_file)
            st.write(t(f"Ditemukan {len(data)} entri di file.",
                       f"Found {len(data)} entries in file."))
            if st.button(t("Import sekarang","Import now")):
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                    tmp.write(_json.dumps(data).encode())
                    tmp_path = tmp.name
                added = import_from_json(tmp_path, st.session_state.get("username","admin"))
                os.unlink(tmp_path)
                st.success(t(f"✅ {added} spektra berhasil diimport.",
                             f"✅ {added} spectra imported successfully."))
                st.rerun()

        st.markdown(f'<p class="sec-hdr">{t("Daftar library","Library list")} — {count_spectra():,} {t("entri","entries")}</p>',
                    unsafe_allow_html=True)
        cats = ["— " + t("Semua kategori","All categories") + " —"] + get_categories()
        filter_cat = st.selectbox(t("Filter kategori","Filter category"), cats)
        all_meta = get_all_meta()
        if "Semua" not in filter_cat and "All" not in filter_cat:
            all_meta = [e for e in all_meta if e["category"] == filter_cat]

        for entry in all_meta[:100]:
            c_info, c_del = st.columns([6, 1])
            with c_info:
                st.markdown(f"""
                <div style="background:#161b27;border:1px solid #2a3142;border-radius:8px;
                  padding:0.6rem 1rem;margin-bottom:5px;">
                  <span style="font-weight:500;color:#e2e8f0;">{entry['name']}</span>
                  <span style="font-size:0.75rem;color:#475569;margin-left:8px;">{entry['category']}</span>
                  <span style="font-size:0.72rem;color:#334155;float:right;">ID:{entry['id']} · {entry['added_at']}</span>
                  <div style="font-size:0.75rem;color:#64748b;margin-top:2px;">
                    {entry['n_points']} pts · {entry['wavenumber_min']:.0f}–{entry['wavenumber_max']:.0f} cm⁻¹
                    {(' · CAS: '+entry['cas_number']) if entry['cas_number'] else ''}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with c_del:
                if st.button("✕", key=f"del_{entry['id']}",
                             help=t("Hapus entri ini","Delete this entry")):
                    delete_spectrum(entry["id"])
                    st.rerun()

        if len(all_meta) > 100:
            st.caption(t(f"Menampilkan 100 dari {len(all_meta)} entri. Gunakan filter kategori.",
                         f"Showing 100 of {len(all_meta)} entries. Use category filter."))

# ════════════════════════════════════════════════════════════════
# TAB 7 — ADMIN PANEL
# ════════════════════════════════════════════════════════════════
if is_admin() and tab_admin:
    with tab_admin:
        st.markdown(f'<p class="sec-hdr">{t("Manajemen pengguna","User management")}</p>',
                    unsafe_allow_html=True)

        from auth import hash_password as hp
        from database import load_users, save_users
        users = load_users()

        with st.form("add_user"):
            u1,u2 = st.columns(2)
            new_uname = u1.text_input(t("Username baru","New username"))
            new_name  = u2.text_input(t("Nama lengkap","Full name"))
            u3,u4 = st.columns(2)
            new_pw   = u3.text_input(t("Password","Password"), type="password")
            new_role = u4.selectbox(t("Role","Role"), ["user", "admin"])
            if st.form_submit_button(t("Tambah pengguna","Add user")):
                if new_uname and new_pw and new_name:
                    if new_uname in users:
                        st.error(t("Username sudah ada.","Username already exists."))
                    else:
                        users[new_uname] = {"password": hp(new_pw), "role": new_role, "name": new_name}
                        save_users(users)
                        st.success(t(f"✅ User '{new_uname}' berhasil ditambahkan.",
                                     f"✅ User '{new_uname}' added successfully."))
                        st.rerun()

        st.markdown(f'<p class="sec-hdr">{t("Daftar pengguna","User list")}</p>', unsafe_allow_html=True)
        for uname, udata in users.items():
            uc1, uc2 = st.columns([5,1])
            with uc1:
                badge_cls = "badge-admin" if udata["role"] == "admin" else "badge-user"
                st.markdown(f"""
                <div style="background:#161b27;border:1px solid #2a3142;border-radius:8px;
                  padding:0.6rem 1rem;margin-bottom:5px;">
                  <span style="font-weight:500;color:#e2e8f0;">{udata['name']}</span>
                  <span class="badge {badge_cls}">{udata['role']}</span>
                  <span style="font-size:0.75rem;color:#475569;margin-left:8px;">@{uname}</span>
                </div>
                """, unsafe_allow_html=True)
            with uc2:
                cur = st.session_state.get("username","")
                if uname != cur:
                    if st.button("✕", key=f"delusr_{uname}"):
                        del users[uname]
                        save_users(users)
                        st.rerun()

        st.markdown(f'<p class="sec-hdr">{t("Ganti password","Change password")}</p>', unsafe_allow_html=True)
        with st.form("change_pw"):
            cp1,cp2 = st.columns(2)
            cp_user  = cp1.selectbox(t("Pengguna","User"), list(users.keys()))
            cp_newpw = cp2.text_input(t("Password baru","New password"), type="password")
            if st.form_submit_button(t("Ganti password","Change password")):
                if cp_newpw:
                    users[cp_user]["password"] = hp(cp_newpw)
                    save_users(users)
                    st.success(t(f"✅ Password '{cp_user}' berhasil diganti.",
                                 f"✅ Password for '{cp_user}' changed successfully."))

# ════════════════════════════════════════════════════════════════
# TAB — STUDI RISET (simulasi korelasi, robustness sweep, blind validation)
# ════════════════════════════════════════════════════════════════
if is_admin() and tab_research:
    with tab_research:
        st.markdown(f'<p class="sec-hdr">{t("🧪 Studi Riset — Kerangka Identitas Ambiguity-Aware","🧪 Research Studies — Ambiguity-Aware Identity Framework")}</p>',
                    unsafe_allow_html=True)
        st.caption(t(
            "Tiga studi VALIDASI OFFLINE untuk manuskrip (bahan Figure 4 dkk.) — memakai "
            "spektrum SINTETIS dengan ground truth diketahui, TIDAK bergantung pada data yang "
            "sedang dimuat di tab lain. Setiap studi menjalankan MCR-ALS berulang kali (bisa "
            "memakan waktu beberapa menit tergantung parameter) — mulai dari nilai kecil dulu "
            "untuk uji coba sebelum menaikkan jumlah replikat/trial untuk hasil manuskrip.",
            "Three OFFLINE validation studies for the manuscript (Figure 4 material, etc.) — "
            "use SYNTHETIC spectra with known ground truth, NOT dependent on data currently "
            "loaded in other tabs. Each study runs MCR-ALS repeatedly (can take a few minutes "
            "depending on parameters) — start with small values to test before increasing "
            "replicates/trials for manuscript-quality results."
        ))

        sub_sim, sub_rob, sub_blind = st.tabs([
            t("📈 Simulasi Korelasi & Noise", "📈 Correlation & Noise Simulation"),
            t("🔁 Robustness Sweep", "🔁 Robustness Sweep"),
            t("🎯 Blind Validation", "🎯 Blind Validation"),
        ])

        # ── Sub-tab 1: Correlation/noise sweep (similarity_simulator.py) ──
        with sub_sim:
            st.caption(t(
                "Membangun JM-like/JE-like sintetis pada korelasi yang DIATUR EKSAK + MA-like "
                "independen, mensimulasikan campuran, menjalankan MCR-ALS, dan mengukur "
                "seberapa sering similarity-only metric 'over-claim' identitas dibanding "
                "kerangka ambiguity-aware — mengulang beberapa seed per kasus.",
                "Builds synthetic JM-like/JE-like spectra at an EXACTLY controlled correlation "
                "plus an independent MA-like component, simulates mixtures, runs MCR-ALS, and "
                "measures how often the similarity-only metric over-claims identity compared "
                "to the ambiguity-aware framework — repeated over several seeds per case."
            ))

            use_default_cases = st.checkbox(
                t("Pakai tabel kasus default (Easy/Moderate/Difficult/Extreme, sesuai brief)",
                  "Use default case table (Easy/Moderate/Difficult/Extreme, per brief)"),
                value=True, key="sim_use_default_cases",
            )
            if use_default_cases:
                st.dataframe(pd.DataFrame(DEFAULT_CASES), use_container_width=True, hide_index=True)
                sim_cases = DEFAULT_CASES
            else:
                custom_corr = st.multiselect(
                    t("Level korelasi JM-JE", "JM-JE correlation levels"),
                    options=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99],
                    default=[0.7, 0.99], key="sim_custom_corr",
                )
                custom_noise = st.number_input(
                    t("Noise (%) untuk semua kasus custom", "Noise (%) for all custom cases"),
                    min_value=0.0, max_value=20.0, value=1.0, step=0.5, key="sim_custom_noise",
                )
                sim_cases = [{"case_name": f"r={c}", "target_correlation": c, "noise_pct": custom_noise}
                            for c in custom_corr]

            sc1, sc2, sc3 = st.columns(3)
            sim_n_replicates = sc1.number_input(
                t("Jumlah replikat/kasus", "Replicates per case"),
                min_value=1, max_value=200, value=5, step=1, key="sim_n_replicates",
            )
            sim_n_samples = sc2.number_input(
                t("Jumlah sampel campuran/replikat", "Mixture samples per replicate"),
                min_value=6, max_value=100, value=20, step=1, key="sim_n_samples",
            )
            sim_n_points = sc3.number_input(
                t("Jumlah titik wavenumber (sintetis)", "Number of wavenumber points (synthetic)"),
                min_value=50, max_value=1000, value=200, step=50, key="sim_n_points",
            )

            if st.button(t("▶️ Jalankan Studi Simulasi", "▶️ Run Simulation Study"),
                        key="btn_run_sim_study", use_container_width=True):
                if not sim_cases:
                    st.warning(t("Pilih minimal satu level korelasi.", "Select at least one correlation level."))
                else:
                    with st.spinner(t(
                        f"Menjalankan {len(sim_cases)} kasus x {sim_n_replicates} replikat "
                        f"(total {len(sim_cases)*sim_n_replicates} run MCR-ALS)...",
                        f"Running {len(sim_cases)} cases x {sim_n_replicates} replicates "
                        f"(total {len(sim_cases)*sim_n_replicates} MCR-ALS runs)...")):
                        wn_sim = np.linspace(400, 4000, int(sim_n_points))
                        try:
                            sim_results = run_correlation_noise_study(
                                wn_sim, cases=sim_cases, n_replicates=int(sim_n_replicates),
                                n_mixture_samples=int(sim_n_samples), base_seed=0,
                            )
                            st.session_state["sim_study_results"] = sim_results
                        except Exception as e:
                            st.error(t(f"Studi gagal dijalankan: {e}", f"Study failed to run: {e}"))

            if "sim_study_results" in st.session_state:
                df_sim = build_correlation_noise_study_dataframe(st.session_state["sim_study_results"])
                st.dataframe(df_sim, use_container_width=True, hide_index=True)
                st.download_button(
                    t("⬇️ Unduh CSV (bahan Figure 4)", "⬇️ Download CSV (Figure 4 material)"),
                    data=df_sim.to_csv(index=False), file_name="correlation_noise_study.csv",
                    mime="text/csv", key="dl_sim_study",
                )
                st.caption(t(
                    "'Replikat komponen kolaps' menandakan MCR-ALS gagal memisahkan komponen "
                    "pada sebagian replikat (fenomena nyata, bukan bug) — kalau angkanya tinggi, "
                    "naikkan jumlah sampel campuran per replikat.",
                    "'Component-collapsed replicates' indicates MCR-ALS failed to separate "
                    "components in some replicates (a real phenomenon, not a bug) — if this "
                    "number is high, increase mixture samples per replicate."
                ))

        # ── Sub-tab 2: Robustness sweep (robustness_sweep_engine.py) ──
        with sub_rob:
            st.caption(t(
                "Membangun SATU dataset sintetis JM/JE/MA pada korelasi & noise yang "
                "ditentukan, lalu menjalankan grid (init x preprocessing) + noise bootstrap "
                "untuk menguji apakah VERDICT IDENTITAS (bukan cuma LOF) tetap stabil.",
                "Builds ONE synthetic JM/JE/MA dataset at the specified correlation & noise, "
                "then runs a grid (init x preprocessing) + noise bootstrap to test whether the "
                "IDENTITY VERDICT (not just LOF) stays stable."
            ))

            rc1, rc2, rc3 = st.columns(3)
            rob_corr = rc1.slider(
                t("Korelasi JM-JE", "JM-JE correlation"),
                min_value=0.0, max_value=0.999, value=0.85, step=0.01, key="rob_corr",
            )
            rob_noise = rc2.number_input(
                t("Noise (%)", "Noise (%)"), min_value=0.0, max_value=20.0, value=1.0,
                step=0.5, key="rob_noise",
            )
            rob_n_samples = rc3.number_input(
                t("Jumlah sampel campuran", "Mixture samples"),
                min_value=6, max_value=100, value=20, step=1, key="rob_n_samples",
            )

            rob_inits = st.multiselect(
                t("Metode inisialisasi yang diuji", "Initialization methods tested"),
                options=["pca", "simplisma", "nmf"], default=["pca", "simplisma", "nmf"],
                key="rob_inits",
            )
            rob_preps = st.multiselect(
                t("Preprocessing yang diuji", "Preprocessing tested"),
                options=list(PREPROCESSING_VARIANTS.keys()),
                default=["raw", "baseline_norm"], key="rob_preps",
            )
            rob_n_boot = st.number_input(
                t("Jumlah replikat noise bootstrap", "Noise bootstrap replicates"),
                min_value=2, max_value=100, value=10, step=1, key="rob_n_boot",
            )

            if st.button(t("▶️ Jalankan Robustness Sweep", "▶️ Run Robustness Sweep"),
                        key="btn_run_rob_sweep", use_container_width=True):
                if not rob_inits or not rob_preps:
                    st.warning(t("Pilih minimal satu init method dan satu preprocessing.",
                                 "Select at least one init method and one preprocessing."))
                else:
                    n_combo = len(rob_inits) * len(rob_preps)
                    with st.spinner(t(
                        f"Menjalankan {n_combo} kombinasi grid + {rob_n_boot} replikat bootstrap...",
                        f"Running {n_combo} grid combinations + {rob_n_boot} bootstrap replicates...")):
                        try:
                            wn_rob = np.linspace(400, 4000, 200)
                            pair_rob = generate_correlated_pure_spectra(wn_rob, rob_corr, seed=1)
                            S_jm_rob, S_je_rob = pair_rob["S_a"], pair_rob["S_b"]
                            other_rob = generate_independent_spectrum(
                                wn_rob, references=[S_jm_rob, S_je_rob], seed=2)
                            S_ma_rob = other_rob["S_c"]
                            S_true_rob = np.array([S_jm_rob, S_je_rob, S_ma_rob])
                            labels_rob = ["JM", "JE", "MA"]

                            C_true_rob = generate_dirichlet_design(int(rob_n_samples), 3, seed=3)
                            D_rob, _ = simulate_linear_mixture(
                                C_true_rob, S_true_rob, noise_pct=rob_noise, seed=4)

                            grid_res = run_robustness_grid(
                                D_rob, wn_rob, S_true_rob, labels_rob,
                                init_methods=tuple(rob_inits), preprocessing_variants=tuple(rob_preps),
                            )
                            boot_res = run_noise_bootstrap_robustness(
                                D_rob, wn_rob, S_true_rob, labels_rob,
                                n_reps=int(rob_n_boot), noise_pct=rob_noise,
                            )
                            verdict_rob = summarize_robustness(
                                grid_res["summary"], boot_res["summary"], labels_rob)

                            st.session_state["rob_sweep_grid"] = grid_res
                            st.session_state["rob_sweep_boot"] = boot_res
                            st.session_state["rob_sweep_verdict"] = verdict_rob
                            st.session_state["rob_sweep_labels"] = labels_rob
                        except Exception as e:
                            st.error(t(f"Robustness sweep gagal: {e}", f"Robustness sweep failed: {e}"))

            if "rob_sweep_verdict" in st.session_state:
                df_rob = build_robustness_dataframe(
                    st.session_state["rob_sweep_grid"], st.session_state["rob_sweep_boot"],
                    st.session_state["rob_sweep_verdict"], st.session_state["rob_sweep_labels"],
                )
                st.dataframe(df_rob, use_container_width=True, hide_index=True)
                st.download_button(
                    t("⬇️ Unduh CSV", "⬇️ Download CSV"),
                    data=df_rob.to_csv(index=False), file_name="robustness_sweep.csv",
                    mime="text/csv", key="dl_rob_sweep",
                )
                st.caption(t(
                    "robustness_ok di sini bisa dipakai sebagai input parameter robustness_ok "
                    "pada evaluate_component_identity() untuk komponen yang label-nya cocok, "
                    "kalau ingin dikombinasikan manual dengan hasil tab Analisis MCR.",
                    "robustness_ok here can be used as the robustness_ok input to "
                    "evaluate_component_identity() for the matching component label, if you "
                    "want to manually combine it with the MCR Analysis tab result."
                ))

        # ── Sub-tab 3: Blind validation (blind_validation_engine.py) ──
        with sub_blind:
            st.caption(t(
                "Uji diagnostik biner: pada tiap level korelasi impostor-vs-target, jalankan "
                "trial POSITIF (target benar ada) dan NEGATIF (target diganti impostor mirip) "
                "secara 'buta', lalu hitung confusion matrix, sensitivity, specificity, dan CI.",
                "Binary diagnostic test: at each impostor-vs-target correlation level, run "
                "POSITIVE (target genuinely present) and NEGATIVE (target replaced by a "
                "similar impostor) trials 'blindly', then compute confusion matrix, "
                "sensitivity, specificity, and CI."
            ))

            bc1, bc2, bc3 = st.columns(3)
            blind_levels = bc1.multiselect(
                t("Level korelasi impostor", "Impostor correlation levels"),
                options=[0.70, 0.85, 0.95, 0.99], default=[0.70, 0.85, 0.95, 0.99],
                key="blind_levels",
            )
            blind_n_trials = bc2.number_input(
                t("Trial per kelas per level", "Trials per class per level"),
                min_value=2, max_value=100, value=10, step=1, key="blind_n_trials",
            )
            blind_n_samples = bc3.number_input(
                t("Sampel campuran/trial", "Mixture samples/trial"),
                min_value=6, max_value=100, value=15, step=1, key="blind_n_samples",
            )
            blind_noise = st.number_input(
                t("Noise (%)", "Noise (%)"), min_value=0.0, max_value=20.0, value=1.0,
                step=0.5, key="blind_noise",
            )

            if st.button(t("▶️ Jalankan Blind Validation", "▶️ Run Blind Validation"),
                        key="btn_run_blind", use_container_width=True):
                if not blind_levels:
                    st.warning(t("Pilih minimal satu level korelasi.", "Select at least one correlation level."))
                else:
                    n_total_trials = len(blind_levels) * blind_n_trials * 2
                    with st.spinner(t(
                        f"Menjalankan {n_total_trials} trial total (bisa beberapa menit)...",
                        f"Running {n_total_trials} total trials (can take several minutes)...")):
                        try:
                            wn_blind = np.linspace(400, 4000, 180)
                            blind_results = run_blind_validation_study(
                                wn_blind, impostor_correlation_levels=tuple(blind_levels),
                                n_trials_per_class=int(blind_n_trials),
                                n_mixture_samples=int(blind_n_samples), noise_pct=blind_noise,
                                n_bootstrap=500,
                            )
                            st.session_state["blind_study_results"] = blind_results
                        except Exception as e:
                            st.error(t(f"Blind validation gagal: {e}", f"Blind validation failed: {e}"))

            if "blind_study_results" in st.session_state:
                df_blind = build_blind_validation_dataframe(st.session_state["blind_study_results"])
                st.dataframe(df_blind, use_container_width=True, hide_index=True)
                st.download_button(
                    t("⬇️ Unduh CSV", "⬇️ Download CSV"),
                    data=df_blind.to_csv(index=False), file_name="blind_validation.csv",
                    mime="text/csv", key="dl_blind",
                )
                st.caption(t(
                    "Impostor di sini juga sintetis (dibangun pada korelasi yang diatur "
                    "terhadap target) — bukan adulterant kimia riil. Berguna memetakan KAPAN "
                    "pipeline mulai gagal membedakan, tapi validasi akhir tetap perlu sampel "
                    "adulteran riil.",
                    "The impostor here is also synthetic (built at a controlled correlation to "
                    "the target) — not a real chemical adulterant. Useful for mapping WHEN the "
                    "pipeline starts failing to discriminate, but final validation still needs "
                    "real adulterant samples."
                ))

# ════════════════════════════════════════════════════════════════
# TAB LAPORAN / REPORT
# ════════════════════════════════════════════════════════════════
with tab_rep:
    if "mcr_S" not in st.session_state:
        st.info(t("Jalankan MCR-ALS terlebih dahulu.", "Please run MCR-ALS first."))
    else:
        S_raw = st.session_state["mcr_S"]
        S_export = st.session_state.get("mcr_S_proc", S_raw)
        C     = st.session_state["mcr_C"]
        wn    = st.session_state["wavenumber"]
        lof   = st.session_state["mcr_lof"]
        r2    = st.session_state["mcr_r2"]
        nc    = st.session_state["mcr_ncomp"]
        proc_log = st.session_state.get("mcr_proc_log", [])
        snames = st.session_state.get("spec_names", [f"S{i+1}" for i in range(C.shape[0])])
        if len(snames) != C.shape[0]:
            snames = [f"S{i+1}" for i in range(C.shape[0])]

        st.markdown(f'<p class="sec-hdr">{t("Export data","Export data")}</p>',
                    unsafe_allow_html=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Sheet 1 — Pure spectra (processed if available)
            df_S = pd.DataFrame(
                S_export.T, index=wn,
                columns=[f"{t('Komponen','Component')}_{i+1}" for i in range(nc)]
            )
            df_S.index.name = "Wavenumber (cm-1)"
            sheet_name = (t("Spektra Murni (Diproses)","Pure Spectra (Processed)")
                          if proc_log else t("Spektra Murni","Pure Spectra"))
            df_S.to_excel(writer, sheet_name=sheet_name[:31])

            # Sheet 2 — Pure spectra raw (always)
            if proc_log:
                df_S_raw = pd.DataFrame(
                    S_raw.T, index=wn,
                    columns=[f"{t('Komponen','Component')}_{i+1}_raw" for i in range(nc)]
                )
                df_S_raw.index.name = "Wavenumber (cm-1)"
                df_S_raw.to_excel(writer, sheet_name=t("Spektra Murni (Raw)","Pure Spectra (Raw)"))

            # Concentration
            df_C = pd.DataFrame(
                C, index=snames,
                columns=[f"{t('Komponen','Component')}_{i+1}" for i in range(nc)]
            )
            df_C.index.name = t("Sampel","Sample")
            df_C.to_excel(writer, sheet_name=t("Konsentrasi","Concentration"))

            # LOF
            df_lof = pd.DataFrame({
                t("Iterasi","Iteration"): range(1, len(lof)+1),
                "LOF (%)": lof
            })
            df_lof.to_excel(writer, sheet_name="LOF", index=False)

            # Matching results
            if "match_results" in st.session_state:
                rows = []
                for i, results in enumerate(st.session_state["match_results"]):
                    for rank, r in enumerate(results, 1):
                        rows.append({
                            t("Komponen","Component"): i+1,
                            t("Rank","Rank"): rank,
                            t("Nama","Name"): r["name"],
                            t("Kategori","Category"): r["category"],
                            "Cosine": r["cosine"],
                            "Pearson": r.get("pearson"),
                            t("Turunan","Derivative"): r.get("derivative"),
                            t("Geser","Shift"): r.get("shift"),
                            t("Komposit","Composite"): r.get("composite"),
                            "HQI (%)": r["hqi"],
                            t("Ambigu","Ambiguous"): r.get("ambiguous", False),
                            t("Status","Status"): consensus_label(
                                r["cosine"], r["hqi"], pearson=r.get("pearson"),
                                ambiguous=r.get("ambiguous", False)
                            )[1]
                        })
                pd.DataFrame(rows).to_excel(
                    writer, sheet_name=t("Hasil Matching","Matching Results")[:31], index=False
                )

            # Summary
            proc_str = " → ".join(proc_log) if proc_log else t("Tidak ada","None")
            cu = st.session_state.get("mcr_diag", {}).get("constraints_used", {})
            summary_params = [
                t("Jumlah komponen","Number of components"),
                t("Iterasi","Iterations"),
                "LOF akhir / Final LOF (%)",
                "R²",
                t("Post-MCR processing","Post-MCR processing"),
                t("Tanggal analisis","Analysis date"),
                t("Operator","Operator"),
            ]
            summary_values = [
                nc, len(lof), f"{lof[-1]:.4f}", f"{r2:.6f}",
                proc_str,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                st.session_state.get("display_name","—"),
            ]
            # Catat semua constraint MCR-ALS yang dipakai — penting untuk
            # keterlacakan/audit trail (mis. ISO 17025) dan agar hasil bisa
            # direproduksi/dibandingkan ulang dengan software lain.
            if cu:
                constraint_labels = {
                    "init_method":  t("Metode inisialisasi","Initialization method"),
                    "closure":      t("Closure constraint","Closure constraint"),
                    "unimodal_S":   t("Unimodality (S)","Unimodality (S)"),
                    "unimodal_C":   t("Unimodality (C)","Unimodality (C)"),
                    "normalize_S":  t("Normalisasi S","Normalize S"),
                    "smooth_S":     t("Smoothing S per iterasi","Smoothing S per iteration"),
                    "selectivity":  t("Selectivity constraint","Selectivity constraint"),
                    "windowing":    t("Windowing constraint","Windowing constraint"),
                    "library_init": t("Initial guess dari library","Library initial guess"),
                }
                for key, label in constraint_labels.items():
                    if key in cu:
                        summary_params.append(label)
                        summary_values.append(str(cu[key]))
            summary = {
                t("Parameter","Parameter"): summary_params,
                t("Nilai","Value"): summary_values,
            }
            pd.DataFrame(summary).to_excel(
                writer, sheet_name=t("Ringkasan","Summary")[:31], index=False
            )

        output.seek(0)
        fname = f"SpectraVision_Pro_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

        e1,e2 = st.columns(2)
        with e1:
            st.download_button(
                t("⬇ Download laporan Excel (semua sheet)","⬇ Download Excel report (all sheets)"),
                data=output, file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with e2:
            df_S_csv = pd.DataFrame(
                S_export.T, index=wn,
                columns=[f"Component_{i+1}" for i in range(nc)]
            )
            df_S_csv.index.name = "Wavenumber"
            st.download_button(
                t("⬇ Spektra murni (CSV)","⬇ Pure spectra (CSV)"),
                df_S_csv.to_csv(),
                f"pure_spectra_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv", use_container_width=True
            )
        # ── Laporan tambahan: Analisis Keterpisahan Kelompok Replikat ──
        # (independen dari hasil MCR di atas — bisa dipakai walau MCR
        # belum pernah dijalankan, karena analisis replikat murni tidak
        # butuh hasil MCR sama sekali)
        if "replicate_sep_result" in st.session_state:
            st.markdown(f'<p class="sec-hdr">{t("Laporan Analisis Keterpisahan Replikat","Replicate Separability Report")}</p>',
                        unsafe_allow_html=True)

            from mcr_replicate_extension import (
                build_separability_dataframe, build_separability_regions_dataframe
            )

            rep_result = st.session_state["replicate_sep_result"]
            rep_wn     = st.session_state["replicate_sep_wn"]
            rep_name_a, rep_name_b = st.session_state["replicate_sep_names"]

            df_sep_data    = build_separability_dataframe(rep_result, rep_wn, rep_name_a, rep_name_b)
            df_sep_regions = build_separability_regions_dataframe(rep_result)

            output_sep = io.BytesIO()
            with pd.ExcelWriter(output_sep, engine="openpyxl") as writer_sep:
                df_sep_regions.to_excel(writer_sep, sheet_name="Ringkasan Region", index=False)
                df_sep_data.to_excel(writer_sep, sheet_name="Data Perhitungan Lengkap", index=False)
            output_sep.seek(0)

            st.dataframe(df_sep_regions, use_container_width=True)

            st.download_button(
                t("⬇ Download laporan keterpisahan replikat (Excel)",
                  "⬇ Download replicate separability report (Excel)"),
                data=output_sep,
                file_name=f"Separability_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
