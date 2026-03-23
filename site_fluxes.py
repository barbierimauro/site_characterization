"""
compute_site_fluxes
===================
Stima i conteggi teorici attesi di muoni e neutroni epitermali per un
sensore CRNS a una data quota e latitudine, e calcola N0 teorico via
formula di Desilets (suolo completamente secco, theta_v = 0).

Fisica
------
Muoni:
    N_muon(H) = N_muon_sl * exp(beta_muon * (P0 - P(H))) * kappa_muon

Neutroni epitermali:
    N_neut(H) = N_neut_sl * f_Rc * exp(beta_neut * (P0 - P(H))) * kappa_topo

    f_Rc: fattore di correzione per rigidity cutoff rispetto al livello
    del mare a latitudine geomagnetically corrected (usando tabella
    Smart & Shea 2019 tramite crnpy).
    Al livello del mare f_Rc = Rc_sl_ref / Rc_site (normalizzato a Rc=0
    che è il polo, massimo flusso). In pratica si normalizza a un sito
    di riferimento noto (default: polo, Rc=0 -> f_Rc massimo).

N0 teorico (Desilets, suolo secco):
    Dalla formula di Desilets: N/N0 = a0 + a1/(theta_v + a2)
    Per theta_v = 0:  N0 = N_neut_site / (a0 + a1/a2)

Parametri Desilets (Desilets et al. 2010):
    a0 = 0.0808
    a1 = 0.372
    a2 = 0.115

Rigidity cutoff:
    Da Smart & Shea 2019 via crnpy.cutoff_rigidity(lat, lon).
    Accuratezza: ±0.3 GV.

    La correzione di flusso per Rc è basata su Hawdon et al. 2014:
        f_Rc = exp(alpha * (Rc_ref - Rc_site))
    con alpha ≈ -0.075 (riduzione del flusso all'aumentare di Rc).
    Rc_ref = 0 GV (polo geomagnetico, flusso massimo).
    Per siti italiani (Rc ~ 5 GV): f_Rc ~ exp(-0.075 * 5) ~ 0.69.

Nota: i valori N_muon_sl e N_neut_sl sono specifici del rivelatore
(area efficace, efficienza, moderatore). I valori di default (4000 e 900
cph) devono essere calibrati per il rivelatore Finapp.

coefficienti per i muoni in funzione dell'altezza
  [Regime basso (< 1000 m)]  n = 51
    slope  b = (-1.0584e-06 ± 4.17e-08) hPa^-1 m^-1
    interc a = (-1.0478e-03 ± 1.91e-05) hPa^-1
  [Regime alto  (>= 1000 m)]  n = 7
    slope  b = (-1.1067e-06 ± 9.43e-09) hPa^-1 m^-1
    interc a = (-1.4842e-03 ± 2.54e-05) hPa^-1

    Nota: beta_muon non è più un parametro fisso — viene calcolato
    internamente con fit lineare a due regimi (< 1000 m, >= 1000 m):
        beta_muon(alt) = a + b * alt_m   [hPa^-1]
    con segno negativo perché il flusso cresce salendo (dP < 0 implica
    che l'esponente deve essere positivo → beta è definito con segno
    tale che exp(beta * dP) > 1 quando dP > 0).

    Fit empirico da dati stazione:
      basso (< 1000 m): a = -1.0478e-3, b = -1.0584e-6  hPa^-1 m^-1
      alto  (>= 1000 m): a = -1.4842e-3, b = -1.1067e-6  hPa^-1 m^-1


Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Costanti fisiche
# ---------------------------------------------------------------------------
P0_HPA      = 1013.25   # pressione di riferimento [hPa]
H_SCALE_M   = 8500.0    # scala di altezza atmosferica [m]

# Parametri Desilets 2010  ← FORMULA DI RIFERIMENTO
# N/N0 = a0 + a1/(theta_v + a2)   (invertibile: N0 = N_dry / (a0 + a1/(lw+a2)))
DESILETS_A0 = 0.0808
DESILETS_A1 = 0.372
DESILETS_A2 = 0.115

# Fattore Hawdon 2014 per correzione rigidity cutoff
HAWDON_ALPHA = -0.075   # [GV^-1]  dN/dRc / N
RC_REF_GV    = 0.0      # riferimento polo (flusso massimo)

# ---------------------------------------------------------------------------
# Correzione vapore acqueo (Zreda et al. 2012)
# f_WV = exp(-alpha_WV * rho_WV)   [rho_WV in g/m³]
# ---------------------------------------------------------------------------
WV_ALPHA = 0.0054   # m³/g  (Zreda 2012, Rosolem 2013)

# ---------------------------------------------------------------------------
# SOC hydrogen equivalent (Bogena 2017)
# theta_v_soc = 0.55 * rho_b * (SOC_gkg / 1000 * 1.724)  [m³/m³ water equiv]
# ---------------------------------------------------------------------------
SOC_TO_SOM     = 1.724   # fattore SOC -> SOM
SOM_H_FRACTION = 0.55    # frazione H nella SOM rispetto all'acqua


def _pressure(alt_m):
    """Pressione barometrica alla quota alt_m [hPa]."""
    return P0_HPA * np.exp(-alt_m / H_SCALE_M)


def _cutoff_rigidity(lat, lon):
    """
    Rigidity cutoff [GV] da tabella Smart & Shea 2019.
    Richiede crnpy installato.
    """
    try:
        import crnpy
        # crnpy vuole lon in [0, 360]
        lon360 = lon % 360.0
        return float(crnpy.cutoff_rigidity(lat, lon360))
    except ImportError:
        raise ImportError(
            "crnpy non installato. Installa con: pip install crnpy")


def _f_rigidity(Rc_site, Rc_ref=RC_REF_GV, alpha=HAWDON_ALPHA):
    """
    Fattore di correzione del flusso di neutroni per rigidity cutoff.
    Hawdon et al. 2014:
        f_Rc = exp(alpha * (Rc_site - Rc_ref))
    alpha < 0: flusso diminuisce all'aumentare di Rc (latitudini basse).
    """
    return float(np.exp(alpha * (Rc_site - Rc_ref)))


def compute_wv_correction(rho_WV_gm3):
    """
    Correzione per vapore acqueo atmosferico (Zreda et al. 2012, Rosolem 2013).

    Il vapore acqueo modera i neutroni epitermali in atmosfera, riducendo il
    segnale in modo proporzionale alla densità assoluta dell'umidità.

    Formula:
        f_WV = exp(-alpha_WV * rho_WV)

    dove rho_WV è l'umidità assoluta [g/m³] e alpha_WV = 0.0054 m³/g.

    NOTA: questa è una correzione OPERATIVA (da applicare ai conteggi misurati),
    non una caratteristica fissa del sito. Per la site characterization si
    fornisce come climatologia mensile attesa, utile per stimare la variabilità
    stagionale del segnale.

    Parameters
    ----------
    rho_WV_gm3 : float or array — umidità assoluta [g/m³]
        Per T=20°C, RH=50%: rho_WV ≈ 8.7 g/m³
        Per T=10°C, RH=80%: rho_WV ≈ 7.5 g/m³
        Per T=0°C,  RH=80%: rho_WV ≈ 3.7 g/m³

    Returns
    -------
    float or array — fattore correttivo f_WV [0..1]
    """
    return float(np.exp(-WV_ALPHA * rho_WV_gm3))


def compute_agbh_theta_equiv(lai_m2m2, litter_water_mm=0.0,
                              leaf_water_content_g_per_m2=150.0):
    """
    Stima del contributo della biomassa aerea all'idrogeno equivalente.

    La biomassa sopra il suolo (AGBH: Above-Ground Biomass Hydrogen) modera
    i neutroni epitermali. Viene espresso come volume d'acqua equivalente
    distribuito sull'area del footprint [mm = kg/m²], e poi convertito in
    theta_v equivalente dividendo per z86.

    Formula semplificata (Baatz et al. 2015, Schrön 2017):
        AGBH_mm = LAI * leaf_water_content [mm]
        theta_v_agbh = AGBH_mm / (z86_cm * 10)  [m³/m³ equiv.]

    NOTA: è una correzione SUPPLEMENTARE rispetto a Desilets, utile per stimare
    l'entità del bias in siti forestati. La formula Desilets rimane la formula
    primaria: l'effetto AGBH si manifesta come un bias apparente su N0 e sul
    theta_v misurato.

    Parameters
    ----------
    lai_m2m2             : LAI medio annuo [m²/m²]
    litter_water_mm      : acqua nel lettiera/strato erbaceo [mm]
    leaf_water_content_g_per_m2 : acqua per unità di LAI [g/m²] (default 150 g/m²)

    Returns
    -------
    dict con:
        agbh_mm      : acqua equivalente biomassa aerea [mm]
        lai_used     : LAI usato
    """
    agbh_mm = float(lai_m2m2 * leaf_water_content_g_per_m2 / 1000.0 * 1000.0 +
                    litter_water_mm)
    return dict(agbh_mm=agbh_mm, lai_used=float(lai_m2m2),
                litter_mm=float(litter_water_mm))


def compute_site_fluxes(
    lat,
    lon,
    alt_m,
    kappa_topo,
    kappa_muon,
    N_muon_sl  = 4000.0,   # conteggio muoni al livello del mare [cph]
    N_neut_sl  = 900.0,    # conteggio neutroni al livello del mare [cph]
    beta_muon  = 0.005,     # coefficiente barometrico muoni [-]
    beta_neut  = 0.0077,    # coefficiente barometrico neutroni [-]
    lw         = 0.0,       # lattice water [g/g] — Köhli 2021
    soc_gkg    = 0.0,       # SOC [g/kg] — per inventario H supplementare
    rho_b      = 1.4,       # bulk density [g/cm³] — per conversione SOC→H
):
    """
    Calcola i conteggi teorici attesi al sito e N0 teorico.

    N0 è calcolato con la formula di Desilets 2010, che rimane la formula
    di riferimento primaria per la sua invertibilità.

    Con correzione lattice water (lw):
        N0 = N_neut_site / (a0 + a1 / (lw + a2))
    Con lw=0 si recupera la formula originale N0 = N_neut_site / (a0 + a1/a2).

    Fisicamente: lw è l'idrogeno strutturalmente legato nei minerali argillosi
    (Köhli 2021) e non varia con l'umidità del suolo. È sempre presente anche
    nel "suolo secco" e quindi riduce il conteggio di riferimento N_dry rispetto
    a lw=0. Includere lw corregge questo bias sistematico in modo consistente
    con la formula di Desilets.

    Parameters
    ----------
    lat, lon    : coordinate geografiche WGS84 [deg]
    alt_m       : quota del sensore [m a.s.l.]
    kappa_topo  : correzione topografica neutroni (da compute_kappa_topo_3d)
    kappa_muon  : correzione FOV muoni (da compute_kappa_muon)
    N_muon_sl   : conteggio muoni al livello del mare [cph]
    N_neut_sl   : conteggio neutroni al livello del mare [cph]
    lw          : acqua di reticolo [g/g] — da Köhli 2021 tramite clay%
    soc_gkg     : carbonio organico del suolo [g/kg] — info supplementare
    rho_b       : bulk density [g/cm³] — usata solo per conversione SOC→H

    Returns
    -------
    dict con tutte le quantità calcolate
    """

    # ------------------------------------------------------------------ #
    # 1. Pressione al sito
    # ------------------------------------------------------------------ #
    P_site = _pressure(alt_m)
    dP_rel = (P0_HPA - P_site)   # > 0 in quota

    # ------------------------------------------------------------------ #
    # 2. Rigidity cutoff
    # ------------------------------------------------------------------ #
    Rc = _cutoff_rigidity(lat, lon)
    f_Rc = _f_rigidity(Rc)


    # ------------------------------------------------------------------ #
    # 2. Beta muoni — fit lineare a due regimi
    #    beta(alt) = a + b * alt_m  [hPa^-1]
    #    Il fit restituisce valori negativi; per avere
    #    exp(beta_muon * dP) > 1 con dP > 0 usiamo il valore assoluto.
    # ------------------------------------------------------------------ #
    if alt_m < 1000.0:
        a_muon = -1.0478e-3   # hPa^-1
        b_muon = -1.0584e-6   # hPa^-1 m^-1
    else:
        a_muon = -1.4842e-3   # hPa^-1
        b_muon = -1.1067e-6   # hPa^-1 m^-1

    beta_muon = abs(a_muon + b_muon * alt_m)   # hPa^-1, sempre > 0

    # ------------------------------------------------------------------ #
    # 3. Fattori di quota
    # ------------------------------------------------------------------ #
    alt_factor_muon = float(np.exp( beta_muon * dP_rel))
    alt_factor_neut = float(np.exp( beta_neut * dP_rel))


    # ------------------------------------------------------------------ #
    # 4. Conteggi attesi al sito
    #
    # Muoni: quota + topografia (kappa_muon)
    # Neutroni: quota + rigidity + topografia (kappa_topo)
    # ------------------------------------------------------------------ #
    N_muon_site = N_muon_sl * alt_factor_muon * kappa_muon
    N_neut_site = N_neut_sl * f_Rc * alt_factor_neut * kappa_topo

    # ------------------------------------------------------------------ #
    # 5. N0 teorico — Desilets 2010 con correzione lattice water
    #
    # Formula Desilets (invertibile):
    #   N(θ_v) = N0 * (a0 + a1 / (θ_v + lw + a2))
    #
    # "Suolo secco" = θ_v_liquida = 0, ma lw è SEMPRE presente:
    #   N_dry = N0 * (a0 + a1 / (lw + a2))
    #   → N0 = N_neut_site / (a0 + a1 / (lw + a2))
    #
    # Con lw=0 si recupera la formula originale: N0 = N_neut_site / (a0+a1/a2)
    # ------------------------------------------------------------------ #
    lw = max(0.0, float(lw))
    desilets_denom_lw = DESILETS_A0 + DESILETS_A1 / (lw + DESILETS_A2)
    N0_theoretical    = N_neut_site / desilets_denom_lw

    # ------------------------------------------------------------------ #
    # 6. Contributo idrogeno da SOC (supplementare, non modifica N0)
    # Bogena 2017: H_SOC ~ 0.55 * rho_b * SOM_fraction [m³/m³ water equiv]
    # ------------------------------------------------------------------ #
    som_fraction  = float(soc_gkg) / 1000.0 * SOC_TO_SOM
    theta_v_soc   = SOM_H_FRACTION * float(rho_b) * som_fraction

    return dict(
        # Geometria e correzioni
        pressure_hpa      = float(P_site),
        dP_relative       = float(dP_rel),
        Rc_gv             = float(Rc),
        f_Rc              = float(f_Rc),
        alt_factor_muon   = alt_factor_muon,
        alt_factor_neut   = alt_factor_neut,
        kappa_muon        = float(kappa_muon),
        kappa_topo        = float(kappa_topo),
        # Conteggi attesi
        N_muon_site       = float(N_muon_site),
        N_neut_site       = float(N_neut_site),
        # N0 teorico (Desilets + lw)
        N0_theoretical    = float(N0_theoretical),
        desilets_denom_lw = float(desilets_denom_lw),
        lw                = lw,
        theta_v_soc       = float(theta_v_soc),
        # Parametri usati
        N_muon_sl         = float(N_muon_sl),
        N_neut_sl         = float(N_neut_sl),
        beta_muon         = float(beta_muon),
        beta_neut         = float(beta_neut),
    )


def compute_desilets_curve(N0_theoretical, theta_wp=0.05, theta_fc=0.35,
                           n_points=12, lw=0.0, soc_equiv=0.0):
    """
    Curva di Desilets N(θ_v) sull'intervallo [θ_WP, θ_FC] del sito.

    Formula PRIMARIA — Desilets et al. 2010 con correzione lattice water:
        N(θ_v) = N0 * (a0 + a1 / (θ_v + lw + a2))
        ∂N/∂θ_v = −N0 * a1 / (θ_v + lw + a2)²

    dove θ_v è SOLO l'acqua liquida, lw è l'acqua di reticolo (costante).
    Con lw=0 la formula coincide con Desilets 2010 originale.
    La formula rimane pienamente invertibile:
        θ_v = a1 / (N/N0 - a0) - lw - a2

    Curva SUPPLEMENTARE con contributo SOC:
        θ_v_eff = θ_v + lw + soc_equiv   (mostra l'impatto dell'inventario H)

    Parameters
    ----------
    N0_theoretical : float — N0 teorico [cph] da compute_site_fluxes
    theta_wp       : float — punto di appassimento [m³/m³] (solo acqua liquida)
    theta_fc       : float — capacità di campo [m³/m³]
    n_points       : int   — punti della curva (default 12)
    lw             : float — acqua di reticolo [g/g] (Köhli 2021, da clay%)
    soc_equiv      : float — contributo SOC come θ_v equivalente [m³/m³]

    Returns
    -------
    dict con:
        theta_v        array (n,)  range SM liquido usato per la curva [m³/m³]
        N_counts       array (n,)  conteggi Desilets+lw [cph]
        N_counts_soc   array (n,)  conteggi Desilets+lw+SOC (supplementare) [cph]
        dN_dtheta      array (n,)  sensibilità ∂N/∂θ_v [cph / (m³/m³)]
        N_at_wp, N_at_fc, dN_at_wp, dN_at_fc, delta_N
        lw             float       lattice water usata
        soc_equiv      float       SOC equiv usato
        N0_theoretical float       input N0
        theta_wp, theta_fc
    """
    wp = max(0.01, float(theta_wp))
    fc = min(0.60, float(theta_fc))
    if fc <= wp:
        fc = wp + 0.10
    lw       = max(0.0, float(lw))
    soc_eq   = max(0.0, float(soc_equiv))

    theta_v = np.linspace(wp, fc, n_points)

    # Formula primaria Desilets + lw
    def _N(th):
        return N0_theoretical * (DESILETS_A0
                                 + DESILETS_A1 / (th + lw + DESILETS_A2))

    def _dN(th):
        return -N0_theoretical * DESILETS_A1 / (th + lw + DESILETS_A2)**2

    N_counts  = _N(theta_v)
    dN_dtheta = _dN(theta_v)

    # Curva supplementare con SOC (informativa)
    def _N_soc(th):
        return N0_theoretical * (DESILETS_A0
                                 + DESILETS_A1 / (th + lw + soc_eq + DESILETS_A2))

    N_counts_soc = _N_soc(theta_v)

    return dict(
        theta_v        = theta_v,
        N_counts       = N_counts,
        N_counts_soc   = N_counts_soc,
        dN_dtheta      = dN_dtheta,
        N_at_wp        = float(_N(wp)),
        N_at_fc        = float(_N(fc)),
        dN_at_wp       = float(_dN(wp)),
        dN_at_fc       = float(_dN(fc)),
        delta_N        = float(_N(wp) - _N(fc)),
        theta_wp       = wp,
        theta_fc       = fc,
        lw             = lw,
        soc_equiv      = soc_eq,
        N0_theoretical = float(N0_theoretical),
    )


def report_desilets_curve(dc):
    """Stampa tabellare della curva di Desilets N(θ_v)."""
    w = 72
    lw     = dc.get('lw', 0.0)
    soc_eq = dc.get('soc_equiv', 0.0)
    L = [
        "=" * w,
        "DESILETS CURVE — Expected N counts vs Soil Moisture",
        "  Formula: N(θ_v) = N0 × (a0 + a1/(θ_v + lw + a2))  [Desilets 2010 + lw]",
        "=" * w,
        f"  N0 theoretical  : {dc['N0_theoretical']:.0f} cph",
        f"  θ_WP (wilting)  : {dc['theta_wp']:.3f} m³/m³  (acqua liquida)",
        f"  θ_FC (field cap): {dc['theta_fc']:.3f} m³/m³  (acqua liquida)",
        f"  lw (lattice H₂O): {lw:.4f} g/g  (offset costante Köhli 2021)",
        f"  SOC equiv       : {soc_eq:.4f} m³/m³  (informativo, non incluso nella curva primaria)",
        "",
        f"  N at θ_WP (dry) : {dc['N_at_wp']:.0f} cph   (max expected counts)",
        f"  N at θ_FC (wet) : {dc['N_at_fc']:.0f} cph   (min expected counts)",
        f"  ΔN dynamic range: {dc['delta_N']:.0f} cph",
        "",
        f"  Sensitivity ∂N/∂θ_v:",
        f"    at θ_WP : {dc['dN_at_wp']:.0f} cph / (m³/m³)  "
        f"= {dc['dN_at_wp']/100:.1f} cph / 1% SM",
        f"    at θ_FC : {dc['dN_at_fc']:.0f} cph / (m³/m³)  "
        f"= {dc['dN_at_fc']/100:.1f} cph / 1% SM",
        "",
        "  θ_v [m³/m³]  N+lw [cph]  N+lw+SOC [cph]  ∂N/∂θ",
        "  " + "-" * 56,
    ]
    nc_soc = dc.get('N_counts_soc', dc['N_counts'])
    for th, nc, ns, dn in zip(dc['theta_v'], dc['N_counts'], nc_soc, dc['dN_dtheta']):
        L.append(f"  {th:.3f}        {nc:7.0f}         {ns:7.0f}    {dn:8.0f}")
    L.append("=" * w)
    return "\n".join(L)


def report_site_fluxes(res):
    """Stampa leggibile dei risultati di compute_site_fluxes."""
    w = 72
    L = [
        "=" * w,
        "SITE FLUX ESTIMATES  &  N0 THEORETICAL",
        "=" * w,
        "",
        "  --- Atmospheric & geomagnetic ---",
        f"  Pressure at site   : {res['pressure_hpa']:.2f} hPa",
        f"  dP/P0              : {res['dP_relative']:.4f}",
        f"  Cutoff rigidity Rc : {res['Rc_gv']:.2f} GV  (Smart & Shea 2019)",
        f"  f_Rc               : {res['f_Rc']:.4f}  (Hawdon 2014)",
        "",
        "  --- Altitude scaling factors ---",
        f"  alt_factor_muon    : {res['alt_factor_muon']:.4f}"
        f"  (exp({res['beta_muon']:.2f} * {res['dP_relative']:.4f}))",
        f"  alt_factor_neut    : {res['alt_factor_neut']:.4f}"
        f"  (exp({res['beta_neut']:.2f} * {res['dP_relative']:.4f}))",
        "",
        "  --- Topographic corrections ---",
        f"  kappa_topo         : {res['kappa_topo']:.4f}",
        f"  kappa_muon         : {res['kappa_muon']:.4f}",
        "",
        "  --- Expected counts at site ---",
        f"  N_muon_site        : {res['N_muon_site']:.0f} cph",
        f"    = {res['N_muon_sl']:.0f} (sl)"
        f" x {res['alt_factor_muon']:.4f} (alt)"
        f" x {res['kappa_muon']:.4f} (kappa_muon)",
        f"  N_neut_site        : {res['N_neut_site']:.0f} cph",
        f"    = {res['N_neut_sl']:.0f} (sl)"
        f" x {res['f_Rc']:.4f} (Rc)"
        f" x {res['alt_factor_neut']:.4f} (alt)"
        f" x {res['kappa_topo']:.4f} (kappa_topo)",
        "",
        "  --- N0 theoretical (Desilets 2010 + lattice water) ---",
        f"  lw (lattice water)    : {res.get('lw', 0.0):.4f} g/g",
        f"  θ_v_SOC (supplementare): {res.get('theta_v_soc', 0.0):.4f} m³/m³",
        f"  N0 = N_neut_site / (a0 + a1/(lw + a2))",
        f"     = {res['N_neut_site']:.1f} / {res['desilets_denom_lw']:.4f}",
        f"     = {res['N0_theoretical']:.0f} cph",
        f"  Inversione: θ_v = a1/(N/N0 - a0) - lw - a2",
        "=" * w,
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Siti di test
    sites = [
        ("LIMENA (pianura)",         45.467, 11.851,   25.0, 1.00, 1.00),
        ("Malga Fadner (alpino)",     46.925, 11.861, 1100.0, 0.92, 0.88),
        ("Cima Pradazzo (alta quota)",46.356, 11.822, 1800.0, 0.85, 0.82),
    ]

    for name, lat, lon, alt, kt, km in sites:
        print(f"\n{'='*55}")
        print(f"SITO: {name}")
        print(f"  lat={lat}  lon={lon}  alt={alt}m")
        print(f"  kappa_topo={kt}  kappa_muon={km}  (valori dummy)")
        res = compute_site_fluxes(lat, lon, alt, kt, km)
        print(report_site_fluxes(res))
