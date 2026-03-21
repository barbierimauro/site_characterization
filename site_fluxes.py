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

# Parametri Desilets 2010
DESILETS_A0 = 0.0808
DESILETS_A1 = 0.372
DESILETS_A2 = 0.115

# Fattore Hawdon 2014 per correzione rigidity cutoff
HAWDON_ALPHA = -0.075   # [GV^-1]  dN/dRc / N
RC_REF_GV    = 0.0      # riferimento polo (flusso massimo)


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


def compute_site_fluxes(
    lat,
    lon,
    alt_m,
    kappa_topo,
    kappa_muon,
    N_muon_sl  = 4000.0,   # conteggio muoni al livello del mare [cph]
    N_neut_sl  = 900.0,    # conteggio neutroni al livello del mare [cph]
    beta_muon  = 0.005,     # coefficiente barometrico muoni [-]
    beta_neut  = 0.0077,      # coefficiente barometrico neutroni [-]
):
    """
    Calcola i conteggi teorici attesi al sito e N0 teorico.

    Parameters
    ----------
    lat, lon    : coordinate geografiche WGS84 [deg]
    alt_m       : quota del sensore [m a.s.l.]
    kappa_topo  : correzione topografica neutroni (da compute_kappa_topo_3d)
    kappa_muon  : correzione FOV muoni (da compute_kappa_muon)
    N_muon_sl   : conteggio muoni al livello del mare [cph]
    N_neut_sl   : conteggio neutroni al livello del mare [cph]
    beta_muon   : coefficiente barometrico muoni (default 1.15)
    beta_neut   : coefficiente barometrico neutroni (default 2.3)

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
    # 5. N0 teorico — Desilets, theta_v = 0
    #
    # N/N0 = a0 + a1/(theta_v + a2)
    # Per theta_v = 0:
    #   N0 = N_neut_site / (a0 + a1/a2)
    # ------------------------------------------------------------------ #
    desilets_denom = DESILETS_A0 + DESILETS_A1 / DESILETS_A2
    N0_theoretical = N_neut_site / desilets_denom

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
        # N0 teorico
        N0_theoretical    = float(N0_theoretical),
        desilets_denom    = float(desilets_denom),
        # Parametri usati
        N_muon_sl         = float(N_muon_sl),
        N_neut_sl         = float(N_neut_sl),
        beta_muon         = float(beta_muon),
        beta_neut         = float(beta_neut),
    )


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
        "  --- N0 theoretical (Desilets, theta_v=0) ---",
        f"  N0 = N_neut_site / (a0 + a1/a2)",
        f"     = {res['N_neut_site']:.1f} / {res['desilets_denom']:.4f}",
        f"     = {res['N0_theoretical']:.0f} cph",
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
