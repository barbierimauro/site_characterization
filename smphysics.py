"""
sm_downscaling_physics.py
=========================
Funzioni fisiche per il downscaling della soil moisture ERA5.

Contenuto:
  - Pedotransfer functions (Saxton & Rawls 2006)
    -> Field Capacity (FC) e Wilting Point (WP) da texture
  - Correzione topografica (TWI + PISR)
  - Correzione pedologica (rescaling al range locale)
  - Correzione LULC (f_H from lulc_constants)

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pedotransfer functions — Saxton & Rawls 2006
# Input: sabbia e argilla in frazione [0-1]
# Output: theta [m³/m³]
# ---------------------------------------------------------------------------

def saxton_rawls(sand_frac, clay_frac, om_frac=0.02):
    """
    Stima Field Capacity (FC) e Wilting Point (WP) con
    Saxton & Rawls 2006 (SSSAJ 70:1569-1578).

    Parameters
    ----------
    sand_frac : float [0-1]  frazione di sabbia
    clay_frac : float [0-1]  frazione di argilla
    om_frac   : float [0-1]  frazione materia organica (default 0.02)

    Returns
    -------
    FC  : float [m³/m³]  capacità di campo (~-33 kPa)
    WP  : float [m³/m³]  punto di appassimento (~-1500 kPa)
    SAT : float [m³/m³]  saturazione
    """
    S = sand_frac
    C = clay_frac
    OM = om_frac

    # Wilting Point (theta at -1500 kPa)
    t1500t = (-0.024*S + 0.487*C + 0.006*OM
              + 0.005*S*OM - 0.013*C*OM
              + 0.068*S*C + 0.031)
    WP = t1500t + (0.14*t1500t - 0.02)
    WP = float(np.clip(WP, 0.01, 0.50))

    # Field Capacity (theta at -33 kPa)
    t33t = (-0.251*S + 0.195*C + 0.011*OM
            + 0.006*S*OM - 0.027*C*OM
            + 0.452*S*C + 0.299)
    FC = t33t + (1.283*t33t**2 - 0.374*t33t - 0.015)
    FC = float(np.clip(FC, WP + 0.01, 0.60))

    # Saturated water content
    tSt = (0.278*S + 0.034*C + 0.022*OM
           - 0.018*S*OM - 0.027*C*OM
           - 0.584*S*C + 0.078)
    SAT = tSt + (0.636*tSt - 0.107)
    SAT = float(np.clip(SAT, FC + 0.01, 0.70))

    return FC, WP, SAT


def fc_wp_from_soilgrids(soil_res):
    """
    Calcola FC e WP dai risultati di get_soil_properties().
    Usa i valori pesati CRNS (media sui primi strati).

    Parameters
    ----------
    soil_res : dict da get_soil_properties()

    Returns
    -------
    FC, WP, SAT : float [m³/m³]
    """
    sand = soil_res.get("sand_crns", np.nan)
    clay = soil_res.get("clay_crns", np.nan)
    soc  = soil_res.get("soc_crns",  np.nan)   # g/kg

    if np.isnan(sand) or np.isnan(clay):
        return 0.30, 0.12, 0.45   # valori di default

    sand_f = sand / 100.0
    clay_f = clay / 100.0
    # SOC g/kg -> OM fraction (fattore 1.724 per C->OM)
    om_f   = (soc / 1000.0 * 1.724) if not np.isnan(soc) else 0.02
    om_f   = float(np.clip(om_f, 0.001, 0.20))

    return saxton_rawls(sand_f, clay_f, om_f)


# ---------------------------------------------------------------------------
# Correzione topografica
# ---------------------------------------------------------------------------

def topo_correction(twi_site, twi_mean_fp,
                    pisr_site_kWh, pisr_mean_kWh,
                    beta_twi=0.008, alpha_pisr=0.12):
    """
    Fattore moltiplicativo di correzione topografica per SM.

    Due componenti:
      1. TWI: punti con TWI > TWI_mean -> più umidi
         delta_theta = beta_twi * (TWI_site - TWI_mean)
         (additivo, non moltiplicativo)

      2. PISR: più radiazione -> più ET -> meno SM
         f_pisr = exp(-alpha_pisr * dPISR_rel)
         dove dPISR_rel = (PISR_site - PISR_mean) / PISR_mean

    Parameters
    ----------
    twi_site      : float, TWI del punto sensore
    twi_mean_fp   : float, TWI medio nel footprint
    pisr_site_kWh : float, PISR annua al sito [kWh/m²]
    pisr_mean_kWh : float, PISR media cella ERA5 [kWh/m²]
    beta_twi      : float [m³/m³ per unità TWI]  default 0.008
    alpha_pisr    : float sensitività SM-PISR     default 0.12

    Returns
    -------
    delta_theta_twi : float [m³/m³] correzione additiva TWI
    f_pisr          : float [-]     fattore moltiplicativo PISR
    """
    delta_theta_twi = float(beta_twi * (twi_site - twi_mean_fp))
    delta_theta_twi = float(np.clip(delta_theta_twi, -0.15, 0.15))

    if pisr_mean_kWh > 0:
        dpisr_rel = (pisr_site_kWh - pisr_mean_kWh) / pisr_mean_kWh
    else:
        dpisr_rel = 0.0
    f_pisr = float(np.exp(-alpha_pisr * dpisr_rel))
    f_pisr = float(np.clip(f_pisr, 0.5, 2.0))

    return delta_theta_twi, f_pisr


# ---------------------------------------------------------------------------
# Correzione pedologica
# ---------------------------------------------------------------------------

def pedological_rescaling(theta_era5_monthly,
                            FC_local, WP_local,
                            FC_era5=0.32, WP_era5=0.12):
    """
    Rescaling lineare dal range idrologico ERA5 al range locale.

    ERA5 usa un modello idrologico con parametri globali (FC_era5,
    WP_era5). Il suolo locale ha FC e WP diversi da Saxton & Rawls.
    La correzione porta theta dal range ERA5 al range locale:

        theta_local = WP_local + (theta_era5 - WP_era5)
                      / (FC_era5 - WP_era5)
                      * (FC_local - WP_local)

    Capped a [WP_local, FC_local].

    Parameters
    ----------
    theta_era5_monthly : array (12,) [m³/m³]
    FC_local, WP_local : float [m³/m³] da Saxton & Rawls
    FC_era5, WP_era5   : float [m³/m³] parametri tipici ERA5-Land

    Returns
    -------
    theta_rescaled : array (12,) [m³/m³]
    """
    theta = np.asarray(theta_era5_monthly, dtype=float)
    denom = FC_era5 - WP_era5
    if denom < 0.01:
        return theta.copy()

    rescaled = WP_local + (theta - WP_era5) / denom \
               * (FC_local - WP_local)
    return np.clip(rescaled, WP_local, FC_local)


# ---------------------------------------------------------------------------
# Correzione LULC
# ---------------------------------------------------------------------------

def lulc_correction(theta_monthly, lulc_res):
    """
    Correzione categorica per uso del suolo.

    Regole:
      - Se frazione impermeabile (built-up + bare) > 0.5:
        theta *= 0.3  (suolo compattato / asfalto)
      - Se frazione wetland/forest > 0.4 e kappa_lulc > 1.2:
        theta = max(theta, FC * 0.7)  (suolo sempre tendenzialmente umido)
      - Altrimenti: nessuna correzione

    Parameters
    ----------
    theta_monthly : array (12,)
    lulc_res      : dict da get_lulc()

    Returns
    -------
    theta_corrected : array (12,)
    f_lulc          : float, fattore applicato
    lulc_note       : str, descrizione della correzione
    """
    theta = np.asarray(theta_monthly, dtype=float).copy()
    fracs = lulc_res.get("wc_class_fractions", {})

    # Frazioni WorldCover
    f_built  = fracs.get(50, {}).get("fraction", 0.0)  # built-up
    f_bare   = fracs.get(60, {}).get("fraction", 0.0)  # bare
    f_wet    = fracs.get(90, {}).get("fraction", 0.0)  # wetland
    f_forest = fracs.get(10, {}).get("fraction", 0.0)  # tree cover
    f_snow   = fracs.get(70, {}).get("fraction", 0.0)  # snow/ice
    kappa    = lulc_res.get("wc_kappa", 1.0)

    f_impervious = f_built + f_bare

    if f_snow > 0.3:
        # Sito prevalentemente nevoso -> SM bassa (neve != acqua liquida)
        theta *= 0.4
        return theta, 0.4, "Snow/ice dominant: theta reduced"

    if f_impervious > 0.5:
        theta *= 0.30
        return theta, 0.30, "Impervious dominant: theta strongly reduced"

    if f_impervious > 0.2:
        f = 1.0 - 0.5 * f_impervious
        theta *= f
        return theta, float(f), f"Partial impervious: f={f:.2f}"

    if (f_wet + f_forest) > 0.4 and kappa > 1.2:
        # Suolo umido strutturale: SM tendenzialmente alta
        f = 1.10
        theta = np.minimum(theta * f, 0.55)
        return theta, f, "Wet/forest dominant: theta slightly increased"

    return theta, 1.0, "No significant LULC correction"


# ---------------------------------------------------------------------------
# Incertezza composta
# ---------------------------------------------------------------------------

def combined_uncertainty(sigma_era5=0.05,
                          sigma_twi=0.02,
                          sigma_pedol=0.03,
                          sigma_lulc=0.02):
    """
    Incertezza totale per propagazione gaussiana (somma quadratica).
    Valori di default dalla letteratura.

    Returns
    -------
    sigma_total : float [m³/m³]
    components  : dict con singoli contributi
    """
    sigma_total = float(np.sqrt(sigma_era5**2 + sigma_twi**2
                                + sigma_pedol**2 + sigma_lulc**2))
    return sigma_total, {
        "era5_resolution": sigma_era5,
        "topo_correction" : sigma_twi,
        "pedological"     : sigma_pedol,
        "lulc"            : sigma_lulc,
        "total"           : sigma_total,
    }




"""
sm_fusion.py
============
Fusione ERA5 + dati locali per stima soil moisture al sito CRNS.

Integra:
  era5_soil_moisture.py    -> SM grezza ERA5-Land
  sm_downscaling_physics.py -> correzioni fisiche locali

Input richiesti (già calcolati dalla pipeline):
  era5_res    : da get_era5_soil_moisture()
  soil_res    : da get_soil_properties()
  twi_res     : da compute_twi()
  climate_res : da get_site_climate()  (POA mensile)
  lulc_res    : da get_lulc()

Output: theta_v mensile stimata con incertezza.

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

# Parametri ERA5-Land idrologici tipici (globali)
FC_ERA5_DEFAULT = 0.32
WP_ERA5_DEFAULT = 0.12


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def fuse_soil_moisture(
    era5_res,
    soil_res,
    twi_res,
    climate_res,
    lulc_res,
    pisr_era5_annual_kWh = None,
    FC_era5              = FC_ERA5_DEFAULT,
    WP_era5              = WP_ERA5_DEFAULT,
    beta_twi             = 0.008,
    alpha_pisr           = 0.12,
    verbose              = True,
):
    """
    Stima theta_v mensile al sito fondendo ERA5 con dati locali.

    Sequenza di correzioni:
      1. ERA5 SM grezzo (strato 0-7 cm come proxy per z86)
      2. Rescaling pedologico -> range [WP_local, FC_local]
      3. Correzione topografica (TWI + PISR)
      4. Correzione LULC categorica

    Parameters
    ----------
    era5_res    : dict da get_era5_soil_moisture()
    soil_res    : dict da get_soil_properties()
    twi_res     : dict da compute_twi()
    climate_res : dict da get_site_climate()
    lulc_res    : dict da get_lulc()
    pisr_era5_annual_kWh : float, PISR media cella ERA5 [kWh/m²/anno]
                  Se None, stima da climate_res senza correzione orizzonte
    FC_era5, WP_era5 : parametri idrologici ERA5 globali
    beta_twi    : sensitività SM-TWI  [m³/m³ per unità TWI]
    alpha_pisr  : sensitività SM-PISR [-]

    Returns
    -------
    dict con:
        theta_monthly       : array (12,) SM stimata al sito [m³/m³]
        theta_monthly_low   : array (12,) limite inferiore (- sigma)
        theta_monthly_high  : array (12,) limite superiore (+ sigma)
        theta_annual_mean   : float
        theta_current       : float, stima per il momento attuale
        FC_local, WP_local  : float, limiti pedologici
        sigma_total         : float, incertezza composta [m³/m³]
        sigma_components    : dict
        corrections         : dict con dettaglio correzioni applicate
        era5_raw_monthly    : array (12,) SM ERA5 non corretta
        months              : lista nomi mesi
    """

    # ------------------------------------------------------------------ #
    # 1. SM ERA5 grezza — strato 0-7cm come riferimento
    # ------------------------------------------------------------------ #
    theta_era5 = era5_res.get("sm0_7_monthly_mean",
                               np.full(12, np.nan)).copy()
    theta_era5_current = float(era5_res.get("sm0_7_current", np.nan))

    if verbose:
        print(f"   ERA5 raw SM: mean={np.nanmean(theta_era5):.3f} m³/m³",
              flush=True)

    # ------------------------------------------------------------------ #
    # 2. FC e WP locali da SoilGrids
    # ------------------------------------------------------------------ #
    FC_local, WP_local, SAT_local = fc_wp_from_soilgrids(soil_res)

    if verbose:
        print(f"   Pedology: FC={FC_local:.3f}  WP={WP_local:.3f}  "
              f"SAT={SAT_local:.3f}", flush=True)

    # ------------------------------------------------------------------ #
    # 3. Rescaling pedologico
    # ------------------------------------------------------------------ #
    theta_pedol = pedological_rescaling(
        theta_era5, FC_local, WP_local, FC_era5, WP_era5)

    # Stesso rescaling per valore corrente
    theta_current_pedol = float(WP_local +
        (theta_era5_current - WP_era5) / max(FC_era5 - WP_era5, 0.01)
        * (FC_local - WP_local)) if not np.isnan(theta_era5_current) \
        else np.nan
    theta_current_pedol = float(np.clip(
        theta_current_pedol, WP_local, FC_local)) \
        if not np.isnan(theta_current_pedol) else np.nan

    if verbose:
        print(f"   After pedol: mean={np.nanmean(theta_pedol):.3f}",
              flush=True)

    # ------------------------------------------------------------------ #
    # 4. Correzione topografica
    # ------------------------------------------------------------------ #
    twi_site    = twi_res.get("twi_weighted", twi_res.get("twi_mean_fp", 7.0))
    twi_mean    = twi_res.get("twi_mean_fp", 7.0)

    # PISR al sito: somma annua POA da climate_res
    pisr_site = float(np.nansum(
        climate_res.get("POA_monthly_kWh_m2",
                        np.full(12, 100.0))))

    # PISR ERA5: approssimazione — POA senza correzione orizzonte
    # = pisr_site / SVF (se disponibile da thermal_index)
    if pisr_era5_annual_kWh is not None:
        pisr_era5_val = float(pisr_era5_annual_kWh)
    else:
        # Stima: GHI annuo (orizzonte piatto, nessuna ostruzione)
        pisr_era5_val = float(np.nansum(
            climate_res.get("GHI_monthly_kWh_m2",
                            np.full(12, 90.0))))

    delta_twi, f_pisr = topo_correction(
        twi_site, twi_mean, pisr_site, pisr_era5_val,
        beta_twi, alpha_pisr)

    # Applica: additivo per TWI, moltiplicativo per PISR
    theta_topo = (theta_pedol + delta_twi) * f_pisr
    theta_topo = np.clip(theta_topo, WP_local, SAT_local)

    # Stesso per corrente
    theta_current_topo = ((theta_current_pedol + delta_twi) * f_pisr
                           if not np.isnan(theta_current_pedol)
                           else np.nan)

    if verbose:
        print(f"   After topo: delta_TWI={delta_twi:+.3f}  "
              f"f_PISR={f_pisr:.3f}  "
              f"mean={np.nanmean(theta_topo):.3f}", flush=True)

    # ------------------------------------------------------------------ #
    # 5. Correzione LULC
    # ------------------------------------------------------------------ #
    theta_lulc, f_lulc, lulc_note = lulc_correction(
        theta_topo, lulc_res)
    theta_lulc = np.clip(theta_lulc, WP_local, SAT_local)

    theta_current_final = (float(theta_current_topo * f_lulc)
                            if not np.isnan(theta_current_topo)
                            else np.nan)

    if verbose:
        print(f"   After LULC: f={f_lulc:.3f}  {lulc_note}", flush=True)
        print(f"   Final SM: mean={np.nanmean(theta_lulc):.3f}  "
              f"range=[{np.nanmin(theta_lulc):.3f}, "
              f"{np.nanmax(theta_lulc):.3f}]", flush=True)

    # ------------------------------------------------------------------ #
    # 6. Incertezza
    # ------------------------------------------------------------------ #
    sigma_total, sigma_components = combined_uncertainty()
    theta_low  = np.clip(theta_lulc - sigma_total,
                          WP_local, SAT_local)
    theta_high = np.clip(theta_lulc + sigma_total,
                          WP_local, SAT_local)

    return dict(
        theta_monthly      = theta_lulc,
        theta_monthly_low  = theta_low,
        theta_monthly_high = theta_high,
        theta_annual_mean  = float(np.nanmean(theta_lulc)),
        theta_current      = theta_current_final,
        FC_local           = FC_local,
        WP_local           = WP_local,
        SAT_local          = SAT_local,
        sigma_total        = sigma_total,
        sigma_components   = sigma_components,
        era5_raw_monthly   = theta_era5,
        corrections        = {
            "delta_twi_m3m3"   : float(delta_twi),
            "f_pisr"           : float(f_pisr),
            "f_lulc"           : float(f_lulc),
            "lulc_note"        : lulc_note,
            "twi_site"         : float(twi_site),
            "twi_mean_fp"      : float(twi_mean),
            "pisr_site_kWh"    : float(pisr_site),
            "pisr_era5_kWh"    : float(pisr_era5_val),
        },
        months             = MONTHS,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report_sm_fusion(res):
    w = 72
    M = res["months"]
    L = ["="*w,
         "SOIL MOISTURE — Downscaled Estimate",
         "="*w,
         f"  FC={res['FC_local']:.3f}  WP={res['WP_local']:.3f}  "
         f"SAT={res['SAT_local']:.3f}  [m³/m³]  (Saxton & Rawls 2006)",
         f"  Uncertainty: ±{res['sigma_total']:.3f} m³/m³  "
         f"(ERA5 + topo + pedol + LULC)",
         ""]

    c = res["corrections"]
    L += ["  Corrections applied:",
          f"    TWI delta     : {c['delta_twi_m3m3']:+.3f} m³/m³  "
          f"(TWI_site={c['twi_site']:.1f}  "
          f"TWI_mean={c['twi_mean_fp']:.1f})",
          f"    PISR factor   : {c['f_pisr']:.3f}  "
          f"(site={c['pisr_site_kWh']:.0f} kWh/m²  "
          f"ERA5={c['pisr_era5_kWh']:.0f} kWh/m²)",
          f"    LULC factor   : {c['f_lulc']:.3f}  "
          f"({c['lulc_note']})",
          ""]

    hdr = "  " + " "*10 + "  ".join(f"{m:>5}" for m in M)
    L.append(hdr)
    L.append("  " + "-"*(w-2))

    th  = res["theta_monthly"]
    era = res["era5_raw_monthly"]
    lo  = res["theta_monthly_low"]
    hi  = res["theta_monthly_high"]

    L.append("  ERA5 raw  " + "  ".join(
        f"{v:.3f}" if not np.isnan(v) else "  N/A" for v in era))
    L.append("  Fused     " + "  ".join(
        f"{v:.3f}" if not np.isnan(v) else "  N/A" for v in th))
    L.append("  Low  (-σ) " + "  ".join(
        f"{v:.3f}" if not np.isnan(v) else "  N/A" for v in lo))
    L.append("  High (+σ) " + "  ".join(
        f"{v:.3f}" if not np.isnan(v) else "  N/A" for v in hi))

    cur = res["theta_current"]
    L += ["",
          f"  Annual mean : {res['theta_annual_mean']:.3f} m³/m³",
          f"  Current     : {cur:.3f} m³/m³"
          if cur is not None and not np.isnan(cur)
          else "  Current     : N/A",
          ""]

    sig = res["sigma_components"]
    L += ["  Uncertainty breakdown:",
          f"    ERA5 resolution : ±{sig['era5_resolution']:.3f}",
          f"    Topo correction : ±{sig['topo_correction']:.3f}",
          f"    Pedological     : ±{sig['pedological']:.3f}",
          f"    LULC            : ±{sig['lulc']:.3f}",
          f"    Total (RSS)     : ±{sig['total']:.3f}",
          "="*w]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sm_fusion(res, path, site_name=""):
    """
    Due pannelli:
      Left  : confronto ERA5 grezzo vs SM fusa, con banda incertezza
              + linee FC, WP, SAT
      Right : barchart correzioni applicate
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STYLE = {"figure.facecolor": "white",
             "axes.facecolor": "#f8f8f6",
             "axes.grid": True, "grid.color": "white",
             "axes.spines.top": False, "axes.spines.right": False}

    M    = res["months"]
    x    = np.arange(1, 13)
    th   = res["theta_monthly"]
    era  = res["era5_raw_monthly"]
    lo   = res["theta_monthly_low"]
    hi   = res["theta_monthly_high"]
    FC   = res["FC_local"]
    WP   = res["WP_local"]
    SAT  = res["SAT_local"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                                  facecolor="white")

        # --- Left: ciclo annuale ---
        ax = axes[0]
        valid = ~np.isnan(th)

        ax.fill_between(x[valid], lo[valid], hi[valid],
                        alpha=0.25, color="#2166ac",
                        label=f"±σ = ±{res['sigma_total']:.3f}")
        ax.plot(x[valid], th[valid], "o-",
                color="#2166ac", lw=2.5, ms=6,
                label="Fused SM estimate")

        valid_e = ~np.isnan(era)
        ax.plot(x[valid_e], era[valid_e], "s--",
                color="#d01c8b", lw=1.5, ms=5, alpha=0.7,
                label="ERA5 raw (0-7cm)")

        ax.axhline(FC,  color="#e69f00", ls="-.",
                   lw=1.5, label=f"FC  = {FC:.3f}")
        ax.axhline(WP,  color="#cc79a7", ls=":",
                   lw=1.5, label=f"WP  = {WP:.3f}")
        ax.axhline(SAT, color="#56b4e9", ls="--",
                   lw=1, alpha=0.6, label=f"SAT = {SAT:.3f}")

        # Valore corrente
        cur = res.get("theta_current")
        if cur is not None and not np.isnan(cur):
            ax.axhline(cur, color="#009e73", ls="-",
                       lw=2, label=f"Current = {cur:.3f}")

        ax.set_xlim(0.5, 12.5)
        ax.set_xticks(x)
        ax.set_xticklabels(M, fontsize=9)
        ax.set_ylim(0, SAT * 1.15)
        ax.set_xlabel("Month")
        ax.set_ylabel("θ_v  [m³/m³]")
        ax.set_title("Soil Moisture — ERA5 vs Downscaled",
                     fontsize=12)
        ax.legend(fontsize=9, loc="upper right")

        # --- Right: barchart correzioni ---
        ax2 = axes[1]
        c    = res["corrections"]
        corr_names = ["ERA5 base\n(annual mean)",
                       "TWI delta",
                       "PISR factor\n(effect)",
                       "LULC factor\n(effect)"]
        era_mean = float(np.nanmean(era))
        fused_mean = res["theta_annual_mean"]

        # Contributi relativi in m³/m³
        delta_twi   = c["delta_twi_m3m3"]
        delta_pisr  = era_mean * (c["f_pisr"] - 1.0)
        delta_lulc  = fused_mean - era_mean - delta_twi - delta_pisr
        values = [era_mean, delta_twi, delta_pisr, delta_lulc]
        colors = ["#2166ac",
                  "#27ae60" if delta_twi >= 0 else "#c0392b",
                  "#27ae60" if delta_pisr >= 0 else "#c0392b",
                  "#27ae60" if delta_lulc >= 0 else "#c0392b"]

        bars = ax2.bar(corr_names, values, color=colors,
                       edgecolor="white", width=0.6)
        ax2.axhline(0, color="gray", lw=1)
        ax2.axhline(fused_mean, color="#2166ac", ls="--",
                    lw=2, label=f"Fused mean = {fused_mean:.3f}")
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     val + (0.003 if val >= 0 else -0.008),
                     f"{val:+.3f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
        ax2.set_ylabel("θ_v contribution [m³/m³]")
        ax2.set_title("Corrections breakdown\n"
                      "(annual mean contribution)", fontsize=12)
        ax2.legend(fontsize=9)

        fig.suptitle(f"Soil Moisture Downscaling  |  {site_name}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {path}")
