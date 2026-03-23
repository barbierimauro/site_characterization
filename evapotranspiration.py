"""
evapotranspiration.py
=====================
Calcola l'evapotraspirazione di riferimento (ET₀) con il metodo
FAO-56 Penman-Monteith, usando esclusivamente dati già disponibili
nel pipeline (site_climate + crns_corr).

Output: ET₀ mensile, bilancio idrico P-ET₀, indice di aridità UNESCO,
        VPD, Rn, grafici mensili.

Riferimento:
  Allen R.G. et al. (1998) FAO Irrigation and Drainage Paper No. 56.
  Thornthwaite C.W. (1948) — solo per confronto/validazione.
"""

import numpy as np
import os

# Giorni per mese (anno non bisestile)
_DAYS  = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
# Giorno giuliano al centro di ogni mese
_J_MID = np.cumsum(_DAYS) - _DAYS / 2.0

# Costante solare [MJ/m²/min]
_GSC = 0.0820

_MONTHS_IT = ["Gen","Feb","Mar","Apr","Mag","Giu",
              "Lug","Ago","Set","Ott","Nov","Dic"]


# ---------------------------------------------------------------------------
# Funzioni fisiche
# ---------------------------------------------------------------------------

def _es_kPa(T_C):
    """Pressione di vapore saturo [kPa] — formula Tetens (FAO-56 eq. 11)."""
    return 0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))


def _Ra_MJ_m2_day(lat_deg, J):
    """
    Radiazione extraterrestre Ra [MJ/m²/giorno] per giorno giuliano J.
    FAO-56 eq. 21.
    """
    phi    = np.radians(lat_deg)
    dr     = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    delta  = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    omegas = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    Ra = (24 * 60 / np.pi) * _GSC * dr * (
        omegas * np.sin(phi) * np.sin(delta)
        + np.cos(phi) * np.cos(delta) * np.sin(omegas))
    return np.maximum(Ra, 0.0)


# ---------------------------------------------------------------------------
# Calcolo ET₀
# ---------------------------------------------------------------------------

def compute_et(site_climate, lat, elev_m, crns_corr=None):
    """
    Calcola ET₀ FAO-56 Penman-Monteith su base mensile.

    Parameters
    ----------
    site_climate : dict da get_site_climate()
    lat          : latitudine WGS84 [deg]
    elev_m       : quota del sito [m a.s.l.]
    crns_corr    : dict da get_crns_corrections() — per riduzione SWE (opt.)

    Returns
    -------
    dict con valori mensili [12] e scalari riassuntivi
    """
    Tmean  = np.asarray(site_climate['T_mean_monthly_C'],    dtype=float)
    Tmax   = np.asarray(site_climate['T_max_monthly_C'],     dtype=float)
    Tmin   = np.asarray(site_climate['T_min_monthly_C'],     dtype=float)
    RH     = np.asarray(site_climate['RH_mean_monthly_pct'], dtype=float)
    u2     = np.asarray(site_climate['WS_mean_monthly_ms'],  dtype=float)
    GHI    = np.asarray(site_climate['GHI_monthly_kWh_m2'],  dtype=float)
    P_mm   = np.asarray(site_climate['precip_monthly_mm'],   dtype=float)
    SP_hPa = np.asarray(site_climate['SP_mean_monthly_hPa'], dtype=float)

    # Radiazione solare [MJ/m²/giorno]: kWh/m²/mese → MJ/m²/giorno
    Rs = GHI * 3.6 / _DAYS

    # Radiazione extraterrestre
    Ra = _Ra_MJ_m2_day(lat, _J_MID)

    # Radiazione solare netta (albedo = 0.23 — prato FAO)
    Rns = (1.0 - 0.23) * Rs

    # Pressioni di vapore
    es_max = _es_kPa(Tmax)
    es_min = _es_kPa(Tmin)
    es     = (es_max + es_min) / 2.0          # pressione vapore saturazione
    ea     = np.clip(es * RH / 100.0, 1e-4, None)  # pressione vapore effettiva

    # Radiazione cielo sereno e rapporto Rs/Rso (FAO-56 eq. 37)
    Rso    = (0.75 + 2e-5 * elev_m) * Ra
    Rs_Rso = np.where(Rso > 0, np.clip(Rs / Rso, 0.25, 1.0), 0.5)

    # Radiazione onda lunga netta Rnl (FAO-56 eq. 39)
    sigma  = 4.903e-9   # MJ/m²/K⁴/giorno
    Rnl    = sigma * (((Tmax + 273.16)**4 + (Tmin + 273.16)**4) / 2.0) \
             * (0.34 - 0.14 * np.sqrt(ea)) \
             * (1.35 * Rs_Rso - 0.35)
    Rnl    = np.maximum(Rnl, 0.0)

    Rn = np.maximum(Rns - Rnl, 0.0)

    # Costante psicrometrica γ [kPa/°C] (FAO-56 eq. 8)
    gamma = 0.665e-3 * (SP_hPa / 10.0)

    # Pendenza curva pressione vapore Δ [kPa/°C] (FAO-56 eq. 13)
    Delta = 4098.0 * _es_kPa(Tmean) / (Tmean + 237.3) ** 2

    # ET₀ Penman-Monteith [mm/giorno] (FAO-56 eq. 6)
    num   = 0.408 * Delta * Rn + gamma * (900.0 / (Tmean + 273.0)) * u2 * (es - ea)
    den   = Delta + gamma * (1.0 + 0.34 * u2)
    ET0_d = np.maximum(num / den, 0.0)

    # Mensilizza [mm/mese]
    ET0_monthly = ET0_d * _DAYS

    # Riduzione per copertura nevosa (SWE)
    snow_factor = np.ones(12)
    if crns_corr is not None:
        swe = np.asarray(crns_corr.get('swe_mm_monthly', np.zeros(12)), dtype=float)
        snow_factor = np.where(swe > 100, 0.05,
                      np.where(swe > 30,  0.30,
                      np.where(swe > 10,  0.60,
                               1.00)))
    ET0_monthly = ET0_monthly * snow_factor

    # Bilancio idrico mensile
    water_balance = P_mm - ET0_monthly

    # Indice di aridità UNESCO (P/ET₀ sul periodo annuo)
    ET0_annual = float(ET0_monthly.sum())
    P_annual   = float(P_mm.sum())
    ai = P_annual / ET0_annual if ET0_annual > 0 else float('nan')

    aridity_class = (
        "iper-arido"  if ai < 0.05 else
        "arido"       if ai < 0.20 else
        "semi-arido"  if ai < 0.50 else
        "sub-umido"   if ai < 0.65 else
        "umido"       if ai < 1.00 else
        "iper-umido"
    )

    return {
        "ET0_monthly_mm"        : ET0_monthly.tolist(),
        "ET0_annual_mm"         : ET0_annual,
        "water_balance_monthly" : water_balance.tolist(),
        "water_balance_annual"  : float(water_balance.sum()),
        "precip_annual_mm"      : P_annual,
        "aridity_index"         : float(ai),
        "aridity_class"         : aridity_class,
        "Rn_monthly_MJ_m2"     : Rn.tolist(),
        "Ra_monthly_MJ_m2"     : Ra.tolist(),
        "Rs_monthly_MJ_m2"     : Rs.tolist(),
        "es_monthly_kPa"        : es.tolist(),
        "ea_monthly_kPa"        : ea.tolist(),
        "VPD_monthly_kPa"       : (es - ea).tolist(),
        "snow_factor_monthly"   : snow_factor.tolist(),
        "lat"                   : lat,
        "elev_m"                : elev_m,
    }


# ---------------------------------------------------------------------------
# Report testuale
# ---------------------------------------------------------------------------

def report_et(et):
    """Blocco testuale per la sezione ET nel report."""
    if et is None:
        return "  [non disponibile]"

    L = []
    def s(x=""): L.append(x)

    ET0 = et['ET0_monthly_mm']
    P   = [0.0] * 12   # placeholder — P viene da site_climate, non salvata qui
    WB  = et['water_balance_monthly']
    VPD = et['VPD_monthly_kPa']
    Rn  = et['Rn_monthly_MJ_m2']
    sf  = et['snow_factor_monthly']

    s(f"  Metodo          : FAO-56 Penman-Monteith (Allen et al., 1998)")
    s(f"  Sito            : {et['lat']:.4f}°N  {et['elev_m']:.0f} m a.s.l.")
    s(f"  ET₀ annua       : {et['ET0_annual_mm']:.0f} mm")
    s(f"  P annua         : {et['precip_annual_mm']:.0f} mm")
    s(f"  Bilancio P-ET₀  : {et['water_balance_annual']:+.0f} mm")
    s(f"  Indice aridità  : {et['aridity_index']:.3f}  → {et['aridity_class']}")
    s()
    s(f"  {'Mese':<5}  {'ET₀[mm]':>8}  {'P-ET₀[mm]':>10}  "
      f"{'VPD[kPa]':>9}  {'Rn[MJ/m²]':>10}  {'f_neve':>7}")
    s(f"  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*7}")
    for i, m in enumerate(_MONTHS_IT):
        s(f"  {m:<5}  {ET0[i]:>8.1f}  {WB[i]:>+10.1f}  "
          f"{VPD[i]:>9.3f}  {Rn[i]:>10.2f}  {sf[i]:>7.2f}")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_et(et, site_climate, path, site_name=""):
    """
    Figura ET: 3 pannelli
      sx  : P vs ET₀ mensile (barre) + bilancio cumulato (linea)
      ctr : VPD mensile
      dx  : Rn mensile
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    STYLE = {"figure.dpi": 150, "figure.facecolor": "white",
             "axes.grid": True, "grid.alpha": 0.3}

    ET0 = np.array(et['ET0_monthly_mm'])
    P   = np.asarray(site_climate['precip_monthly_mm'])
    WB  = np.array(et['water_balance_monthly'])
    VPD = np.array(et['VPD_monthly_kPa'])
    Rn  = np.array(et['Rn_monthly_MJ_m2'])

    x   = np.arange(12)
    w   = 0.38

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor="white")

        # --- Pannello 1: P vs ET₀ ---
        ax = axes[0]
        ax.bar(x - w/2, P,   width=w, color="#4292c6", label="P [mm]",    alpha=0.85)
        ax.bar(x + w/2, ET0, width=w, color="#fc8d59", label="ET₀ [mm]",  alpha=0.85)
        ax2 = ax.twinx()
        wb_cum = np.cumsum(WB)
        col_wb = np.where(WB >= 0, "#2ca02c", "#d62728")
        ax2.bar(x, WB, width=0.8, color=col_wb, alpha=0.30, label="P−ET₀")
        ax2.axhline(0, color="k", lw=0.8, ls="--")
        ax2.set_ylabel("Bilancio P−ET₀ [mm]", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(_MONTHS_IT, fontsize=9)
        ax.set_ylabel("mm", fontsize=10)
        ax.set_title(
            f"Precipitazione vs ET₀\n"
            f"P={et['precip_annual_mm']:.0f} mm/a  "
            f"ET₀={et['ET0_annual_mm']:.0f} mm/a  "
            f"AI={et['aridity_index']:.2f} ({et['aridity_class']})",
            fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        handles2 = [mpatches.Patch(color="#2ca02c", alpha=0.5, label="Surplus"),
                    mpatches.Patch(color="#d62728", alpha=0.5, label="Deficit")]
        ax2.legend(handles=handles2, loc="upper left", fontsize=9)

        # --- Pannello 2: VPD ---
        ax = axes[1]
        colors_vpd = plt.cm.YlOrRd(VPD / max(VPD.max(), 0.01))
        ax.bar(x, VPD, color=colors_vpd, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(_MONTHS_IT, fontsize=9)
        ax.set_ylabel("VPD [kPa]", fontsize=10)
        ax.set_title("Deficit di pressione vapore (VPD)\n"
                     f"Media annua: {np.mean(VPD):.3f} kPa", fontsize=11)

        # --- Pannello 3: Rn ---
        ax = axes[2]
        colors_rn = plt.cm.plasma(Rn / max(Rn.max(), 0.01))
        ax.bar(x, Rn, color=colors_rn, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(_MONTHS_IT, fontsize=9)
        ax.set_ylabel("Rn [MJ/m²/giorno]", fontsize=10)
        ax.set_title("Radiazione netta (Rn)\n"
                     f"Media annua: {np.mean(Rn):.2f} MJ/m²/giorno", fontsize=11)

        fig.suptitle(
            f"Evapotraspirazione FAO-56 Penman-Monteith  |  {site_name}  |  "
            f"{et['lat']:.4f}°N  {et['elev_m']:.0f} m",
            fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
