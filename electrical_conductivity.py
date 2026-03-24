"""
electrical_conductivity.py
==========================
Stima la conducibilità elettrica apparente del suolo (ECa) nel footprint
CRNS usando proprietà SoilGrids + umidità mensile ERA5.

Modello fisico:
  ECa = ECw × (θᵥ/φ)^1.5 × φ  +  ECclay × θᵥ
  dove ECw e ECclay derivano da texture, CEC, SOC via PTF.
  Correzione di temperatura a 25°C (Corwin & Lesch, 2005).

ECe (pasta di saturazione) stimato con PTF da clay, sand, CEC, SOC
(Brevik et al., 2006).

Variabilità stagionale: calcolata usando SM ERA5 mensile (0-7 cm)
come proxy di θᵥ nel top-soil dominato dal CRNS.

Riferimenti:
  Rhoades J.D. et al. (1989) Soil Sci. Soc. Am. J. 53, 433-439.
  Corwin D.L. & Lesch S.M. (2005) Computers and Electronics in Agric. 46.
  Brevik E.C. et al. (2006) Soil Sci. Soc. Am. J. 70, 381-388.
"""

import numpy as np

_MONTHS_IT = ["Gen","Feb","Mar","Apr","Mag","Giu",
              "Lug","Ago","Set","Ott","Nov","Dic"]


# ---------------------------------------------------------------------------
# Calcolo EC
# ---------------------------------------------------------------------------

def compute_ec(soil, era5_sm, site_climate):
    """
    Stima ECa mensile nel footprint CRNS.

    Parameters
    ----------
    soil         : dict da get_soil_properties()
    era5_sm      : dict da get_era5_soil_moisture()
    site_climate : dict da get_site_climate()

    Returns
    -------
    dict con ECa mensile, ECe, curva ECa(θᵥ), variabilità stagionale
    """
    # --- Proprietà del suolo (medie pesate CRNS, già nel footprint) ---
    clay_pct  = float(soil.get('clay_crns', 20.0))
    sand_pct  = float(soil.get('sand_crns', 40.0))
    silt_pct  = max(0.0, 100.0 - clay_pct - sand_pct)
    soc       = float(soil.get('soc_crns', 10.0))     # g/kg
    cec       = float(soil.get('cec_crns', 15.0))     # cmol/kg
    bdod      = float(soil.get('bdod_crns', 1.3))     # g/cm³
    theta_wp  = float(soil.get('theta_wp',  0.10))
    theta_fc  = float(soil.get('theta_fc',  0.30))
    theta_sat = float(soil.get('theta_sat', 0.45))

    clay_dec = clay_pct / 100.0
    soc_pct  = soc / 10.0          # g/kg → %

    # Porosità totale (ρ_s = 2.65 g/cm³ per minerali silicatici)
    rho_s    = 2.65
    porosity = max(0.30, 1.0 - bdod / rho_s)

    # --- ECe stima — pasta di saturazione [dS/m] ---
    # PTF empirica (Brevik et al. 2006 riadattata per suoli non salini)
    ECe = max(0.01,
              0.10
              + 0.032 * clay_pct
              + 0.006 * silt_pct
              + 0.012 * soc_pct * 10
              + 0.022 * cec)

    # --- ECw — conducibilità acqua nei pori [dS/m] ---
    # ECw = ECe × θ_sat / (φ × τ), tortuosità τ ≈ 0.45
    ECw = max(0.05, ECe * theta_sat / (porosity * 0.45))

    # --- Contributo superficiale delle argille [dS/m] ---
    # Proporzionale a CEC (cariche superficiali) e clay fraction
    ECclay = 0.018 * cec * clay_dec

    # --- Funzione ECa(θᵥ) con correzione temperatura ---
    def _ECa(theta_v, T_C=25.0):
        """
        ECa [dS/m] secondo Rhoades (1989).
        n = 1.5 (tipico terreno agrario).
        Correzione T → 25°C: ECa25 = ECa / (1 + 0.0191×(T−25)).
        """
        theta_v = np.asarray(theta_v, dtype=float)
        T_C     = np.asarray(T_C,     dtype=float)
        theta_v = np.clip(theta_v, 0.0, porosity)
        ECa = ECw * (theta_v / porosity) ** 1.5 * porosity + ECclay * theta_v
        ECa = np.maximum(ECa, 0.0)
        corr = 1.0 + 0.0191 * (T_C - 25.0)
        return ECa / np.where(corr > 0.1, corr, 0.1)

    # --- Valori di riferimento a 25°C ---
    ECa_wp  = float(_ECa(theta_wp))
    ECa_fc  = float(_ECa(theta_fc))
    ECa_sat = float(_ECa(theta_sat))

    # --- Curva ECa vs θᵥ a 25°C (per plot) ---
    theta_curve = np.linspace(theta_wp, theta_sat, 120)
    ECa_curve   = _ECa(theta_curve)

    # --- Variazione mensile usando ERA5 SM 0-7 cm ---
    Tmean      = np.asarray(site_climate['T_mean_monthly_C'], dtype=float)
    sm_monthly = np.asarray(
        era5_sm.get('sm0_7_monthly_mean', np.full(12, theta_fc)), dtype=float)
    sm_monthly = np.clip(sm_monthly, theta_wp * 0.7, theta_sat)

    ECa_monthly = _ECa(sm_monthly, Tmean)

    # --- Sensibilità: ΔECa / Δθᵥ (derivata numerica al centro) ---
    dECa_dtheta = float((_ECa(theta_fc * 1.05) - _ECa(theta_fc * 0.95))
                        / (theta_fc * 0.10))

    return {
        # Valori di riferimento
        "ECe_dSm"          : round(ECe,  3),
        "ECw_dSm"          : round(ECw,  3),
        "ECclay_dSm"       : round(ECclay, 4),
        "ECa_wp_dSm"       : round(ECa_wp,  4),
        "ECa_fc_dSm"       : round(ECa_fc,  4),
        "ECa_sat_dSm"      : round(ECa_sat, 4),
        # Variazione mensile
        "ECa_monthly_dSm"  : [round(float(v), 4) for v in ECa_monthly],
        "sm_monthly_m3m3"  : [round(float(v), 4) for v in sm_monthly],
        "ECa_mean_dSm"     : round(float(np.nanmean(ECa_monthly)), 4),
        "ECa_min_dSm"      : round(float(np.nanmin(ECa_monthly)),  4),
        "ECa_max_dSm"      : round(float(np.nanmax(ECa_monthly)),  4),
        "ECa_range_dSm"    : round(float(np.nanmax(ECa_monthly) - np.nanmin(ECa_monthly)), 4),
        # Curva
        "theta_curve"      : theta_curve.tolist(),
        "ECa_curve"        : ECa_curve.tolist(),
        # Soglie θᵥ
        "theta_wp"         : theta_wp,
        "theta_fc"         : theta_fc,
        "theta_sat"        : theta_sat,
        "porosity"         : round(porosity, 3),
        # Proprietà suolo usate
        "clay_pct"         : clay_pct,
        "sand_pct"         : sand_pct,
        "silt_pct"         : silt_pct,
        "soc_gkg"          : soc,
        "cec_cmolkg"       : cec,
        "bdod_gcm3"        : bdod,
        "dECa_dtheta"      : round(dECa_dtheta, 3),
    }


# ---------------------------------------------------------------------------
# Report testuale
# ---------------------------------------------------------------------------

def report_ec(ec):
    """Blocco testuale per la sezione EC nel report."""
    if ec is None:
        return "  [non disponibile]"

    L = []
    def s(x=""): L.append(x)

    s(f"  Modello         : Rhoades (1989) + Corwin & Lesch (2005) corr. T")
    s(f"  PTF ECe         : Brevik et al. (2006) per suoli non salini")
    s()
    s(f"  Proprietà del suolo (medie pesate CRNS nel footprint):")
    s(f"    Clay={ec['clay_pct']:.0f}%  Sand={ec['sand_pct']:.0f}%  "
      f"Silt={ec['silt_pct']:.0f}%")
    s(f"    CEC={ec['cec_cmolkg']:.1f} cmol/kg  "
      f"SOC={ec['soc_gkg']:.1f} g/kg  "
      f"ρb={ec['bdod_gcm3']:.2f} g/cm³  "
      f"φ={ec['porosity']:.3f}")
    s()
    s(f"  ECe (pasta sat.) : {ec['ECe_dSm']:.3f} dS/m  "
      f"  (riferimento agronomico)")
    s(f"  ECw (acqua pori) : {ec['ECw_dSm']:.3f} dS/m")
    s()
    s(f"  ECa apparente vs θᵥ (T=25°C):")
    s(f"    @ θ_wp  = {ec['theta_wp']:.3f} m³/m³ : {ec['ECa_wp_dSm']:.4f} dS/m")
    s(f"    @ θ_fc  = {ec['theta_fc']:.3f} m³/m³ : {ec['ECa_fc_dSm']:.4f} dS/m")
    s(f"    @ θ_sat = {ec['theta_sat']:.3f} m³/m³ : {ec['ECa_sat_dSm']:.4f} dS/m")
    s(f"    Sensibilità  ∂ECa/∂θᵥ ≈ {ec['dECa_dtheta']:.3f} dS/m per m³/m³")
    s()
    s(f"  Variazione stagionale (ERA5 SM 0-7cm):")
    s(f"    Media: {ec['ECa_mean_dSm']:.4f} dS/m  "
      f"Min: {ec['ECa_min_dSm']:.4f}  "
      f"Max: {ec['ECa_max_dSm']:.4f}  "
      f"Range: {ec['ECa_range_dSm']:.4f} dS/m")
    s()
    s(f"  {'Mese':<5}  {'θᵥ ERA5[m³/m³]':>15}  {'ECa[dS/m]':>10}")
    s(f"  {'-'*5}  {'-'*15}  {'-'*10}")
    for i, m in enumerate(_MONTHS_IT):
        s(f"  {m:<5}  {ec['sm_monthly_m3m3'][i]:>15.4f}  "
          f"{ec['ECa_monthly_dSm'][i]:>10.4f}")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_ec(ec, path, site_name=""):
    """
    Figura EC: 2 pannelli
      sx : curva ECa vs θᵥ con punti mensili sovrapposti
      dx : ECa mensile (barre) colorato per θᵥ
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    STYLE = {"figure.dpi": 150, "figure.facecolor": "white",
             "axes.grid": True, "grid.alpha": 0.3}

    theta_curve = np.array(ec['theta_curve'])
    ECa_curve   = np.array(ec['ECa_curve'])
    sm_mon      = np.array(ec['sm_monthly_m3m3'])
    ECa_mon     = np.array(ec['ECa_monthly_dSm'])

    cmap = plt.cm.RdYlBu
    norm = Normalize(vmin=ec['theta_wp'], vmax=ec['theta_sat'])
    cols_mon = cmap(norm(sm_mon))

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")

        # --- Pannello 1: curva ECa(θᵥ) ---
        ax = axes[0]
        ax.plot(theta_curve, ECa_curve, "k-", lw=2.5, label="ECa(θᵥ) T=25°C")
        ax.axvline(ec['theta_wp'],  color="#d62728", ls="--", lw=1.5,
                   label=f"θ_wp={ec['theta_wp']:.3f}")
        ax.axvline(ec['theta_fc'],  color="#ff7f0e", ls="--", lw=1.5,
                   label=f"θ_fc={ec['theta_fc']:.3f}")
        ax.axvline(ec['theta_sat'], color="#1f77b4", ls="--", lw=1.5,
                   label=f"θ_sat={ec['theta_sat']:.3f}")
        # Punti mensili
        sc = ax.scatter(sm_mon, ECa_mon, c=sm_mon, cmap=cmap, norm=norm,
                        s=80, zorder=5, edgecolors="k", linewidths=0.7,
                        label="ERA5 mensile")
        for i, m in enumerate(_MONTHS_IT):
            ax.annotate(m, (sm_mon[i], ECa_mon[i]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7.5)
        ax.set_xlabel("θᵥ [m³/m³]", fontsize=11)
        ax.set_ylabel("ECa [dS/m]", fontsize=11)
        ax.set_title(
            f"Curva ECa vs θᵥ  (T=25°C)\n"
            f"ECe={ec['ECe_dSm']:.3f} dS/m  "
            f"Clay={ec['clay_pct']:.0f}%  CEC={ec['cec_cmolkg']:.1f} cmol/kg",
            fontsize=11)
        ax.legend(fontsize=9)
        plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     label="θᵥ [m³/m³]", fraction=0.04)

        # --- Pannello 2: ECa mensile ---
        ax = axes[1]
        x = np.arange(12)
        bars = ax.bar(x, ECa_mon, color=cols_mon, edgecolor="k",
                      linewidth=0.6, alpha=0.88)
        ax.axhline(ec['ECa_fc_dSm'],  color="#ff7f0e", ls="--", lw=1.5,
                   label=f"ECa@θ_fc={ec['ECa_fc_dSm']:.4f} dS/m")
        ax.axhline(ec['ECa_mean_dSm'], color="k", ls=":", lw=1.2,
                   label=f"Media={ec['ECa_mean_dSm']:.4f} dS/m")
        ax.set_xticks(x)
        ax.set_xticklabels(_MONTHS_IT, fontsize=9)
        ax.set_ylabel("ECa [dS/m]", fontsize=11)
        ax.set_title(
            f"Variazione stagionale ECa\n"
            f"Min={ec['ECa_min_dSm']:.4f}  "
            f"Max={ec['ECa_max_dSm']:.4f}  "
            f"Range={ec['ECa_range_dSm']:.4f} dS/m",
            fontsize=11)
        ax.legend(fontsize=9)
        sm2 = ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(sm2, ax=ax, label="θᵥ ERA5 [m³/m³]", fraction=0.04)

        fig.suptitle(
            f"Conducibilità Elettrica Apparente (ECa)  |  {site_name}",
            fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
