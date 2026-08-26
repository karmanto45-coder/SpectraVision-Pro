"""
snippet_ambiguity_ui.py
========================
Potongan kode untuk ditempel di app.py, TEPAT SETELAH blok expander
"Validation Scorecard" berakhir (yaitu setelah bagian "Residual per
sampel" selesai, masih di dalam `if "mcr_S" in st.session_state:` dan
`if "mcr_diag" in st.session_state:`).

Import tambahan yang perlu ditaruh di bagian atas app.py, sejajar dengan
import mcr_engine:

    from ambiguity_engine import compute_rotational_ambiguity, K_EXACT_LIMIT

Variabel yang diasumsikan sudah ada di scope ini (mengikuti konvensi yang
sudah dipakai di sekitar blok scorecard):
    C_res, S_res   -> st.session_state["mcr_C"], ["mcr_S"]
    diag_stored    -> st.session_state["mcr_diag"]
    nc             -> st.session_state["mcr_ncomp"]
    t(id, en)      -> helper bilingual yang sudah ada
    lang           -> bahasa aktif ("id"/"en")
"""

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

