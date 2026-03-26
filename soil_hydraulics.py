"""
soil_hydraulics.py
==================
Curva di ritenzione idrica (SWRC van Genuchten), conducibilità idraulica
insatura K(θ), potenziale idrico ψ(θ) mensile, profilo pH e materia
organica da SoilGrids, temperatura del suolo da ERA5-Land.

Modello van Genuchten (1980) / Mualem (1976):
    θ(h)  = θ_r + (θ_s − θ_r) · [1 + (α·h)^n]^(−m)
    K(θ)  = K_sat · Se^l · [1 − (1 − Se^(1/m))^m]²
dove m = 1 − 1/n, l = 0.5, Se = (θ − θ_r)/(θ_s − θ_r)

Fit dei parametri α, n:
  I due punti noti sono FC (−33 kPa) e WP (−1500 kPa) da Saxton & Rawls
  già presenti in soil_res.  L'iterazione su m = 1 − 1/n converge in
  <20 passi.

K_sat stimata da Saxton & Rawls (2006):
    K_sat [mm/h] = 1930 · (θ_sat − θ_FC)^3.1

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import numpy as np

# ---------------------------------------------------------------------------
# Costanti fisiche
# ---------------------------------------------------------------------------

H_FC_CM      = 333.0       # |matric head| a Field Capacity  (−33 kPa)  [cm]
H_WP_CM      = 15000.0     # |matric head| a Wilting Point (−1500 kPa)  [cm]
CM_TO_MPA    = 9.81e-5     # 1 cm H₂O = 9.81×10⁻⁵ MPa
MUALEM_L     = 0.5         # parametro di connettività dei pori (Mualem 1976)

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

DEPTH_LABELS = ["0-5cm","5-15cm","15-30cm","30-60cm","60-100cm","100-200cm"]
DEPTH_MIDS   = np.array([2.5, 10.0, 22.5, 45.0, 80.0, 150.0])


# ---------------------------------------------------------------------------
# Fit van Genuchten da WP / FC
# ---------------------------------------------------------------------------

def _fit_van_genuchten(theta_r, theta_s, theta_fc, theta_wp,
                       max_iter=60, tol=1e-7):
    """
    Stima i parametri van Genuchten α [1/cm] e n [−] fissando θ(h) a:
      h = H_FC_CM → θ = θ_FC
      h = H_WP_CM → θ = θ_WP

    Iterazione a punto fisso su m = 1 − 1/n.

    Returns
    -------
    alpha : float [1/cm]
    n     : float [−]
    m     : float [−]
    """
    # Garantisce ordine fisico
    theta_r = float(np.clip(theta_r, 0.005, theta_wp - 0.005))
    theta_s = float(np.clip(theta_s, theta_fc + 0.01, 0.70))

    Se_FC = np.clip((theta_fc - theta_r) / (theta_s - theta_r), 0.01, 0.999)
    Se_WP = np.clip((theta_wp - theta_r) / (theta_s - theta_r), 0.001,
                     Se_FC - 0.001)

    n = 1.50  # valore iniziale tipico
    for _ in range(max_iter):
        m   = 1.0 - 1.0 / n
        A_FC = Se_FC ** (-1.0 / m) - 1.0
        A_WP = Se_WP ** (-1.0 / m) - 1.0
        if A_WP <= 0.0 or A_FC <= 0.0:
            break
        n_new = np.log(A_FC / A_WP) / np.log(H_FC_CM / H_WP_CM)
        n_new = float(np.clip(n_new, 1.01, 12.0))
        if abs(n_new - n) < tol:
            n = n_new
            break
        n = n_new

    m     = 1.0 - 1.0 / n
    A_FC  = np.clip(Se_FC ** (-1.0 / m) - 1.0, 1e-12, None)
    alpha = A_FC ** (1.0 / n) / H_FC_CM

    return float(alpha), float(n), float(m)


# ---------------------------------------------------------------------------
# Funzioni SWRC
# ---------------------------------------------------------------------------

def theta_from_h(h_cm, alpha, n, m, theta_r, theta_s):
    """θ(h) van Genuchten; h = |matric head| [cm], h > 0."""
    h = np.maximum(np.asarray(h_cm, dtype=float), 0.0)
    Se = 1.0 / (1.0 + (alpha * h) ** n) ** m
    return theta_r + (theta_s - theta_r) * Se


def h_from_theta(theta, alpha, n, m, theta_r, theta_s):
    """h(θ) van Genuchten [cm], h > 0 = tensione."""
    Se = np.clip(
        (np.asarray(theta, dtype=float) - theta_r) / (theta_s - theta_r),
        1e-6, 1.0 - 1e-6,
    )
    return (1.0 / alpha) * (Se ** (-1.0 / m) - 1.0) ** (1.0 / n)


def psi_mpa(theta, alpha, n, m, theta_r, theta_s):
    """ψ [MPa] negativo (tensione)."""
    h = h_from_theta(theta, alpha, n, m, theta_r, theta_s)
    return -h * CM_TO_MPA


def k_unsat_cmd(theta, alpha, n, m, theta_r, theta_s, k_sat_cmd):
    """
    K(θ) [cm/d] Mualem-van Genuchten.
    K = K_sat · Se^l · [1 − (1 − Se^(1/m))^m]²
    """
    Se = np.clip(
        (np.asarray(theta, dtype=float) - theta_r) / (theta_s - theta_r),
        1e-9, 1.0,
    )
    K = k_sat_cmd * Se ** MUALEM_L * (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2
    return K


def _ksat_saxton(theta_sat, theta_fc):
    """K_sat [cm/d] da Saxton & Rawls (2006)."""
    ksat_mm_h = 1930.0 * float(np.clip(theta_sat - theta_fc, 1e-4, 1.0)) ** 3.1
    return ksat_mm_h * 2.4  # mm/h → cm/d


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def compute_soil_hydraulics(soil_res, era5_sm_res, sm_fused_res=None,
                             soil_temp_res=None):
    """
    Calcola SWRC van Genuchten, K(θ), ψ(θ) mensile, pH, OM, T_suolo.

    Parameters
    ----------
    soil_res     : dict da get_soil_properties()
    era5_sm_res  : dict da get_era5_soil_moisture()
    sm_fused_res : dict da fuse_soil_moisture() (opzionale)
    soil_temp_res: dict da get_era5_soil_temperature() (opzionale)

    Returns
    -------
    dict con:
      vg_alpha, vg_n, vg_m, vg_theta_r, vg_theta_s,
      vg_theta_fc, vg_theta_wp, k_sat_cmd
      swrc_theta[200], swrc_psi_mpa[200]   — curva SWRC log(h)
      theta_monthly[12], psi_monthly_mpa[12], k_monthly_cmd[12]
      ph_mean[6], ph_mid_cm[6]             — profilo pH SoilGrids
      om_pct_mean[6], om_mid_cm[6]         — profilo OM [%]
      st_depths_cm[4], st_monthly[4,12]    — T suolo mensile per strato
      depth_labels[6]
    """
    # ------------------------------------------------------------------ #
    # 1. Parametri idrologici da Saxton & Rawls (già in soil_res)
    # ------------------------------------------------------------------ #
    theta_wp  = float(soil_res.get('theta_wp',  0.10))
    theta_fc  = float(soil_res.get('theta_fc',  0.30))
    theta_sat = float(soil_res.get('theta_sat', 0.45))

    if any(np.isnan(v) for v in [theta_wp, theta_fc, theta_sat]):
        # Valori di fallback se SoilGrids non è disponibile
        theta_wp, theta_fc, theta_sat = 0.10, 0.28, 0.42

    theta_r = float(np.clip(theta_wp - 0.02, 0.005, theta_wp - 0.005))

    alpha, n, m = _fit_van_genuchten(theta_r, theta_sat, theta_fc, theta_wp)
    k_sat       = _ksat_saxton(theta_sat, theta_fc)

    # ------------------------------------------------------------------ #
    # 2. Curva SWRC (200 punti, h da 1 a 10⁶ cm — log-uniforme)
    # ------------------------------------------------------------------ #
    h_curve     = np.logspace(0, 6, 200)
    swrc_theta  = theta_from_h(h_curve, alpha, n, m, theta_r, theta_sat)
    swrc_psi    = -h_curve * CM_TO_MPA       # MPa, negativo

    # ------------------------------------------------------------------ #
    # 3. θ mensile (usa sm_fused se disponibile, altrimenti ERA5 0-7 cm)
    # ------------------------------------------------------------------ #
    if sm_fused_res is not None and 'theta_monthly' in sm_fused_res:
        theta_monthly = np.array(sm_fused_res['theta_monthly'], dtype=float)
    else:
        theta_monthly = np.array(
            era5_sm_res.get('sm0_7_monthly_mean', np.full(12, np.nan)),
            dtype=float,
        )

    # ------------------------------------------------------------------ #
    # 4. ψ(θ) e K(θ) mensili
    # ------------------------------------------------------------------ #
    psi_monthly = np.array([
        float(psi_mpa(th, alpha, n, m, theta_r, theta_sat))
        if not np.isnan(th) else np.nan
        for th in theta_monthly
    ])
    k_monthly = np.array([
        float(k_unsat_cmd(th, alpha, n, m, theta_r, theta_sat, k_sat))
        if not np.isnan(th) else np.nan
        for th in theta_monthly
    ])

    # ------------------------------------------------------------------ #
    # 5. Profilo pH (da SoilGrids phh2o_profile)
    # ------------------------------------------------------------------ #
    ph_prof  = soil_res.get('phh2o_profile', {})
    ph_mean  = np.asarray(ph_prof.get('mean', np.full(6, np.nan)), dtype=float)
    ph_mid   = np.asarray(ph_prof.get('mid_cm', DEPTH_MIDS), dtype=float)

    # ------------------------------------------------------------------ #
    # 6. Profilo materia organica — SOC [g/kg] × 1.724 / 10 → OM [%]
    # ------------------------------------------------------------------ #
    soc_prof = soil_res.get('soc_profile', {})
    soc_mean = np.asarray(soc_prof.get('mean', np.full(6, np.nan)), dtype=float)
    om_mean  = soc_mean / 10.0 * 1.724     # g/kg → %
    om_mid   = np.asarray(soc_prof.get('mid_cm', DEPTH_MIDS), dtype=float)

    # ------------------------------------------------------------------ #
    # 7. Temperatura del suolo (ERA5-Land 4 profondità)
    # ------------------------------------------------------------------ #
    ST_DEPTHS = np.array([3.5, 17.5, 64.0, 177.5])   # cm mid-point
    ST_LABELS = ["0-7 cm", "7-28 cm", "28-100 cm", "100-255 cm"]

    if soil_temp_res is not None:
        st_monthly = np.full((4, 12), np.nan)
        for i, key in enumerate(["st0_7", "st7_28", "st28_100", "st100_255"]):
            arr = soil_temp_res.get(f"{key}_monthly_mean")
            if arr is not None:
                st_monthly[i, :] = np.asarray(arr, dtype=float)
    else:
        st_monthly = np.full((4, 12), np.nan)

    return dict(
        # van Genuchten
        vg_alpha    = alpha,
        vg_n        = n,
        vg_m        = m,
        vg_theta_r  = theta_r,
        vg_theta_s  = theta_sat,
        vg_theta_fc = theta_fc,
        vg_theta_wp = theta_wp,
        k_sat_cmd   = k_sat,
        # SWRC
        swrc_theta  = swrc_theta,
        swrc_psi_mpa= swrc_psi,
        # Mensili
        theta_monthly    = theta_monthly,
        psi_monthly_mpa  = psi_monthly,
        k_monthly_cmd    = k_monthly,
        # Profili
        ph_mean          = ph_mean,
        ph_mid_cm        = ph_mid,
        om_pct_mean      = om_mean,
        om_mid_cm        = om_mid,
        depth_labels     = DEPTH_LABELS,
        # Temperatura suolo
        st_depths_cm     = ST_DEPTHS,
        st_labels        = ST_LABELS,
        st_monthly       = st_monthly,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report_soil_hydraulics(res):
    """Testo formattato per il report principale."""
    w = 72
    L = [
        "=" * w,
        "SOIL HYDRAULICS  (van Genuchten 1980 / Mualem 1976)",
        "=" * w,
    ]

    # --- vG parameters ---
    L += [
        "  van Genuchten parameters (fit da WP/FC Saxton & Rawls 2006):",
        f"    θ_r  = {res['vg_theta_r']:.4f} m³/m³   (residual)",
        f"    θ_s  = {res['vg_theta_s']:.4f} m³/m³   (saturated = θ_sat)",
        f"    θ_FC = {res['vg_theta_fc']:.4f} m³/m³   (−33 kPa)",
        f"    θ_WP = {res['vg_theta_wp']:.4f} m³/m³   (−1500 kPa)",
        f"    α    = {res['vg_alpha']:.5f} cm⁻¹",
        f"    n    = {res['vg_n']:.4f}",
        f"    m    = {res['vg_m']:.4f}  (= 1 − 1/n)",
        f"    K_sat= {res['k_sat_cmd']:.2f} cm/d  (Saxton & Rawls 2006)",
        "",
    ]

    # --- ψ/K mensile ---
    M = MONTHS
    hdr = "  " + " " * 12 + "  ".join(f"{m:>5}" for m in M)
    L.append(hdr)
    L.append("  " + "-" * (w - 2))

    th = res['theta_monthly']
    ps = res['psi_monthly_mpa']
    kk = res['k_monthly_cmd']

    th_s = "  ".join(f"{v:.3f}" if not np.isnan(v) else "  N/A" for v in th)
    ps_s = "  ".join(f"{v:.3f}" if not np.isnan(v) else "  N/A" for v in ps)
    kk_s = "  ".join(f"{v:.2f}" if not np.isnan(v) else "  N/A" for v in kk)

    L += [
        f"  {'θ [m³/m³]':<12} {th_s}",
        f"  {'ψ [MPa]':<12} {ps_s}",
        f"  {'K [cm/d]':<12} {kk_s}",
        "",
    ]

    # --- pH ---
    L.append("  pH profile (SoilGrids phH₂O):")
    for lbl, ph in zip(res['depth_labels'], res['ph_mean']):
        bar = "█" * int(ph * 3) if not np.isnan(ph) else ""
        val = f"{ph:.2f}" if not np.isnan(ph) else "N/A"
        L.append(f"    {lbl:<10}  {bar:<25} {val}")
    L.append("")

    # --- OM ---
    L.append("  Organic matter [%]  (SOC × 1.724):")
    for lbl, om in zip(res['depth_labels'], res['om_pct_mean']):
        bar = "█" * int(om * 4) if not np.isnan(om) else ""
        val = f"{om:.2f} %" if not np.isnan(om) else "N/A"
        L.append(f"    {lbl:<10}  {bar:<25} {val}")
    L.append("")

    # --- Soil temperature ---
    st = res['st_monthly']
    any_t = not np.all(np.isnan(st))
    L.append("  Soil temperature [°C]  (ERA5-Land):")
    if any_t:
        L.append("  " + " " * 12 + "  ".join(f"{m:>5}" for m in M))
        L.append("  " + "-" * (w - 2))
        for i, lbl in enumerate(res['st_labels']):
            row = st[i]
            rs  = "  ".join(f"{v:5.1f}" if not np.isnan(v) else "  N/A"
                            for v in row)
            L.append(f"  {lbl:<12} {rs}")
    else:
        L.append("    [non disponibile — eseguire con soil_temp_res]")
    L.append("=" * w)

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_soil_hydraulics(res, path, site_name=""):
    """
    6 pannelli:
      [0,0] SWRC θ(ψ) in scala log
      [0,1] K(θ) Mualem-vG in scala log
      [0,2] θ mensile con soglie FC/WP
      [1,0] ψ mensile [MPa]
      [1,1] Profilo pH
      [1,2] Profilo materia organica [%]
    Se soil_temp_res è presente, aggiunge riga con temperatura suolo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    STYLE = {
        "figure.dpi": 100, "figure.facecolor": "white",
        "axes.facecolor": "#f8f8f6", "axes.grid": True,
        "grid.color": "white", "grid.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    }

    has_temp = not np.all(np.isnan(res['st_monthly']))
    nrows    = 3 if has_temp else 2
    figH     = 15 if has_temp else 11

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(16, figH), facecolor="white")
        gs  = GridSpec(nrows, 3, figure=fig, hspace=0.44, wspace=0.38)

        x_mo = np.arange(1, 13)
        M    = MONTHS

        col_fc  = "#2166ac"
        col_wp  = "#d01c8b"
        col_sat = "#4dac26"
        col_th  = "#2c7bb6"
        col_psi = "#b2182b"
        col_k   = "#1a7837"

        # ---------------------------------------------------------- #
        # [0,0] SWRC: θ vs |ψ| [kPa] — scala semi-log x
        # ---------------------------------------------------------- #
        ax = fig.add_subplot(gs[0, 0])
        psi_kpa = np.abs(res['swrc_psi_mpa']) * 1e4  # MPa → kPa (×10000/10)
        # actually 1 MPa = 1000 kPa
        psi_kpa = np.abs(res['swrc_psi_mpa']) * 1000.0  # MPa→kPa
        ax.semilogx(psi_kpa, res['swrc_theta'], color=col_th, lw=2.5)
        ax.axvline(33,   color=col_fc,  ls="--", lw=1.3,
                   label=f"FC = {res['vg_theta_fc']:.3f}")
        ax.axvline(1500, color=col_wp,  ls="--", lw=1.3,
                   label=f"WP = {res['vg_theta_wp']:.3f}")
        ax.axhline(res['vg_theta_r'],  color="gray", ls=":", lw=1.1,
                   label=f"θ_r = {res['vg_theta_r']:.3f}")
        ax.axhline(res['vg_theta_s'],  color=col_sat, ls=":", lw=1.1,
                   label=f"θ_sat = {res['vg_theta_s']:.3f}")
        ax.set_xlabel("|ψ| [kPa]")
        ax.set_ylabel("θ [m³/m³]")
        ax.set_title(f"SWRC van Genuchten\nα={res['vg_alpha']:.5f} cm⁻¹  n={res['vg_n']:.3f}")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, None)
        ax.set_xlim(1, 1e6)

        # ---------------------------------------------------------- #
        # [0,1] K(θ) — scala log y
        # ---------------------------------------------------------- #
        ax2 = fig.add_subplot(gs[0, 1])
        theta_rng = np.linspace(res['vg_theta_r'] * 1.01, res['vg_theta_s'], 200)
        k_rng     = k_unsat_cmd(theta_rng, res['vg_alpha'], res['vg_n'],
                                 res['vg_m'], res['vg_theta_r'],
                                 res['vg_theta_s'], res['k_sat_cmd'])
        ax2.semilogy(theta_rng, k_rng, color=col_k, lw=2.5)
        ax2.axvline(res['vg_theta_fc'], color=col_fc, ls="--", lw=1.3,
                    label="FC")
        ax2.axvline(res['vg_theta_wp'], color=col_wp, ls="--", lw=1.3,
                    label="WP")
        ax2.set_xlabel("θ [m³/m³]")
        ax2.set_ylabel("K(θ) [cm/d]")
        ax2.set_title(f"Conducibilità idraulica insatura\nK_sat = {res['k_sat_cmd']:.2f} cm/d")
        ax2.legend(fontsize=9)
        ax2.set_xlim(res['vg_theta_r'], res['vg_theta_s'] * 1.02)

        # ---------------------------------------------------------- #
        # [0,2] θ mensile con soglie FC / WP
        # ---------------------------------------------------------- #
        ax3 = fig.add_subplot(gs[0, 2])
        th  = res['theta_monthly']
        valid = ~np.isnan(th)
        ax3.bar(x_mo[valid], th[valid], color=col_th, alpha=0.75,
                edgecolor="white")
        ax3.axhline(res['vg_theta_fc'], color=col_fc, ls="--", lw=1.5,
                    label=f"FC = {res['vg_theta_fc']:.3f}")
        ax3.axhline(res['vg_theta_wp'], color=col_wp, ls="--", lw=1.5,
                    label=f"WP = {res['vg_theta_wp']:.3f}")
        ax3.set_xticks(x_mo)
        ax3.set_xticklabels(M, fontsize=8)
        ax3.set_ylabel("θ [m³/m³]")
        ax3.set_title("Soil moisture mensile\n(ERA5-Land / fused)")
        ax3.legend(fontsize=8)
        ax3.set_xlim(0.4, 12.6)
        ax3.set_ylim(0, None)

        # ---------------------------------------------------------- #
        # [1,0] ψ mensile [MPa]
        # ---------------------------------------------------------- #
        ax4 = fig.add_subplot(gs[1, 0])
        ps  = res['psi_monthly_mpa']
        valid = ~np.isnan(ps)
        colors_psi = [col_wp if v < -1.5 else
                      (col_fc if v < -0.033 else col_sat)
                      for v in ps]
        ax4.bar(x_mo[valid], ps[valid],
                color=[colors_psi[i] for i in np.where(valid)[0]],
                alpha=0.8, edgecolor="white")
        ax4.axhline(-0.033, color=col_fc, ls="--", lw=1.3,
                    label="FC (−0.033 MPa)")
        ax4.axhline(-1.5,   color=col_wp, ls="--", lw=1.3,
                    label="WP (−1.5 MPa)")
        ax4.set_xticks(x_mo)
        ax4.set_xticklabels(M, fontsize=8)
        ax4.set_ylabel("ψ [MPa]")
        ax4.set_title("Potenziale idrico mensile ψ(θ)")
        ax4.legend(fontsize=8)
        ax4.set_xlim(0.4, 12.6)

        # ---------------------------------------------------------- #
        # [1,1] Profilo pH
        # ---------------------------------------------------------- #
        ax5 = fig.add_subplot(gs[1, 1])
        ph   = res['ph_mean']
        mid  = res['ph_mid_cm']
        valid = ~np.isnan(ph)
        if valid.any():
            ax5.barh(-mid[valid], ph[valid], height=np.diff(
                np.concatenate([[0], mid]))[valid] * 0.7,
                     color="#8073ac", alpha=0.8, edgecolor="white")
            ax5.plot(ph[valid], -mid[valid], "ko-", ms=6, zorder=5)
            for d, v in zip(mid[valid], ph[valid]):
                ax5.text(v + 0.05, -d, f"{v:.1f}", va="center", fontsize=9)
            ax5.axvline(7.0, color="gray", ls=":", lw=1.1, label="pH 7 (neutro)")
            ax5.set_xlabel("pH")
            ax5.set_ylabel("Profondità [cm]")
            yticks = -mid[valid]
            ax5.set_yticks(yticks)
            ax5.set_yticklabels([f"{d:.0f}" for d in mid[valid]], fontsize=9)
            ax5.legend(fontsize=8)
            ax5.set_xlim(max(0, ph[valid].min() - 0.5),
                          min(14, ph[valid].max() + 0.5))
        ax5.set_title("Profilo pH (SoilGrids)")

        # ---------------------------------------------------------- #
        # [1,2] Profilo OM [%]
        # ---------------------------------------------------------- #
        ax6 = fig.add_subplot(gs[1, 2])
        om   = res['om_pct_mean']
        mid_om = res['om_mid_cm']
        valid = ~np.isnan(om)
        if valid.any():
            ax6.barh(-mid_om[valid], om[valid], height=np.diff(
                np.concatenate([[0], mid_om]))[valid] * 0.7,
                     color="#d6604d", alpha=0.8, edgecolor="white")
            ax6.plot(om[valid], -mid_om[valid], "ko-", ms=6, zorder=5)
            for d, v in zip(mid_om[valid], om[valid]):
                ax6.text(v + 0.02, -d, f"{v:.2f}%", va="center", fontsize=9)
            ax6.set_xlabel("Materia organica [%]")
            ax6.set_ylabel("Profondità [cm]")
            yticks = -mid_om[valid]
            ax6.set_yticks(yticks)
            ax6.set_yticklabels([f"{d:.0f}" for d in mid_om[valid]], fontsize=9)
            ax6.set_xlim(0, None)
        ax6.set_title("Profilo materia organica\n(SOC × 1.724)")

        # ---------------------------------------------------------- #
        # [2,:] Temperatura del suolo — solo se disponibile
        # ---------------------------------------------------------- #
        if has_temp:
            ax_t = fig.add_subplot(gs[2, :])
            colors_t = ["#d7191c", "#fdae61", "#2c7bb6", "#1a9641"]
            st = res['st_monthly']
            for i, (lbl, col) in enumerate(zip(res['st_labels'], colors_t)):
                row   = st[i]
                valid = ~np.isnan(row)
                if valid.any():
                    ax_t.plot(x_mo[valid], row[valid], "o-",
                              color=col, lw=2, ms=5, label=lbl)
                    ax_t.fill_between(x_mo[valid],
                                      row[valid] - 2, row[valid] + 2,
                                      alpha=0.08, color=col)
            ax_t.axhline(0, color="gray", ls="--", lw=1.0)
            ax_t.set_xticks(x_mo)
            ax_t.set_xticklabels(M, fontsize=9)
            ax_t.set_ylabel("Temperatura suolo [°C]")
            ax_t.set_title("Temperatura del suolo mensile — ERA5-Land",
                           fontsize=12)
            ax_t.legend(fontsize=9, ncol=4, loc="upper right")
            ax_t.set_xlim(0.5, 12.5)

        fig.suptitle(
            f"Soil Hydraulics & Chemistry  |  {site_name}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(path, dpi=100)
        plt.close(fig)

    print(f"  Saved: {path}")
