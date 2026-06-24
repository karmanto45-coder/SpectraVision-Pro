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
                      count_spectra, get_categories, import_from_json)
from mcr_engine import (preprocess, detect_components, run_mcr_als,
                        run_mcr_multi_k, generate_warnings, compute_scorecard,
                        postprocess_mcr_spectra,
                        batch_match, consensus_label, interpolate_spectrum,
                        apply_window)
from cos2d import (compute_2dcos, find_crosspeaks, apply_nodas_rules,
                   PERTURBATION_PRESETS)

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
     t("📚 Library","📚 Library"),
     t("⚙️ Admin","⚙️ Admin"),
     t("📊 Laporan","📊 Report")]
    if is_admin() else
    [t("📂 Input Data","📂 Input Data"),
     t("🔬 Analisis MCR","🔬 MCR Analysis"),
     t("✨ Proses Spektra MCR","✨ Process MCR Spectra"),
     t("🔍 Identifikasi","🔍 Identification"),
     t("📈 2D-COS","📈 2D-COS"),
     t("📊 Laporan","📊 Report")]
)

tabs = st.tabs(tab_labels)
tab_input   = tabs[0]
tab_mcr     = tabs[1]
tab_postmcr = tabs[2]
tab_match   = tabs[3]
tab_cos     = tabs[4]
tab_lib     = tabs[5] if is_admin() else None
tab_admin   = tabs[6] if is_admin() else None
tab_rep     = tabs[7] if is_admin() else tabs[5]

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
        closure  = a4.checkbox(t("Closure constraint","Closure constraint"), value=False)

        b1, b2, b3 = st.columns(3)

        # ── Metode inisialisasi ───────────────────────────────
        init_method = b1.radio(
            t("Metode inisialisasi","Initialization method"),
            ["pca", "nmf"],
            format_func=lambda x: {
                "pca": t("PCA — umum, cepat","PCA — general, fast"),
                "nmf": t("NMF-NNDSVD — lebih baik untuk spektroskopi",
                         "NMF-NNDSVD — better for spectroscopy"),
            }[x],
            key="mcr_init_method",
            help=t(
                "PCA: inisialisasi standar, cepat. "
                "NMF-NNDSVD: non-negatif dari awal tanpa koreksi, "
                "lebih sesuai untuk data spektra, "
                "cenderung menghasilkan spektra murni yang lebih akurat.",
                "PCA: standard initialization, fast. "
                "NMF-NNDSVD: non-negative from the start without correction, "
                "more appropriate for spectral data, "
                "tends to produce more accurate pure spectra."
            )
        )

        # ── Constraint tambahan ───────────────────────────────
        unimodal    = b1.checkbox(
            t("Unimodality constraint","Unimodality constraint"), value=False,
            help=t("Paksa spektra murni memiliki satu puncak saja",
                   "Force pure spectra to have a single peak"))
        normalize_S = b1.checkbox(
            t("Normalisasi S per iterasi","Normalize S per iteration"), value=False,
            help=t("Normalisasi unit vector S setiap iterasi. "
                   "Aktifkan hanya jika skala komponen sangat berbeda.",
                   "Unit-vector normalize S each iteration. "
                   "Enable only if component scales differ greatly."))

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
                    fixed_conc_zero=fixed_conc_zero_list if use_window else None
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

            m1,m2,m3,m4 = st.columns(4)
            for col, val, lbl in zip(
                [m1,m2,m3,m4],
                [f"{lof_h[-1]:.3f}%", f"{r2:.4f}", len(lof_h), nc],
                ["LOF", "R²", t("Iterasi","Iterations"), t("Komponen","Components")]
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
                         use_container_width=True):
                with st.spinner(t(f"Mencocokkan vs {n_lib:,} spektra library...",
                                  f"Matching vs {n_lib:,} library spectra...")):
                    library_entries = get_all_spectra_for_matching()
                    all_results = []
                    for i in range(nc):
                        res = batch_match(
                            S[i], wn, library_entries,
                            wmode_key, wmin_show, wmax_show,
                            int(top_n), grid_interval, interp_method
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
                            clabel, cmsg = consensus_label(r["cosine"], r["hqi"],
                                                           thresh_cos, thresh_hqi)
                            card_cls = {"strong":"m-strong","medium":"m-medium",
                                        "conflict":"m-conflict","weak":"m-weak"}.get(clabel,"m-weak")
                            cos_ok = "✓" if r["cosine"] >= thresh_cos else "✗"
                            hqi_ok = "✓" if r["hqi"] >= thresh_hqi else "✗"

                            conflict_note = ""
                            if rank > 1:
                                prev = results[rank-2]
                                if (r["cosine"] > prev["cosine"]) != (r["hqi"] > prev["hqi"]):
                                    conflict_note = f" · ⚠ {t('konflik ranking','rank conflict')}"

                            st.markdown(f"""
                            <div class="match-card {card_cls}">
                              <span class="m-badge" style="background:#1e293b;color:#94a3b8;
                                font-size:0.68rem;padding:2px 8px;border-radius:4px;float:right;">
                                {cmsg}
                              </span>
                              <span class="m-name">#{rank} &nbsp; {r['name']}</span>
                              <div class="m-scores">
                                Cosine: <b>{r['cosine']:.4f}</b> {cos_ok} &nbsp;|&nbsp;
                                HQI: <b>{r['hqi']:.2f}%</b> {hqi_ok} &nbsp;|&nbsp;
                                {t('Kategori','Category')}: {r['category']}{conflict_note}
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

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

        from auth import load_users, save_users, hash_password as hp
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
                            "HQI (%)": r["hqi"],
                            t("Status","Status"): consensus_label(r["cosine"], r["hqi"])[1]
                        })
                pd.DataFrame(rows).to_excel(
                    writer, sheet_name=t("Hasil Matching","Matching Results")[:31], index=False
                )

            # Summary
            proc_str = " → ".join(proc_log) if proc_log else t("Tidak ada","None")
            summary = {
                t("Parameter","Parameter"): [
                    t("Jumlah komponen","Number of components"),
                    t("Iterasi","Iterations"),
                    "LOF akhir / Final LOF (%)",
                    "R²",
                    t("Post-MCR processing","Post-MCR processing"),
                    t("Tanggal analisis","Analysis date"),
                    t("Operator","Operator")
                ],
                t("Nilai","Value"): [
                    nc, len(lof), f"{lof[-1]:.4f}", f"{r2:.6f}",
                    proc_str,
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    st.session_state.get("display_name","—")
                ]
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
