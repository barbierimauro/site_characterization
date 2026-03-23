"""
crns_corrections.py
===================
Correzioni CRNS supplementari per la site characterization:

  - Vapore acqueo atmosferico (κ_WV) — stagionalità mensile da ERA5
  - AGBH — acqua equivalente biomassa aerea dal LAI
  - SWE  — Snow Water Equivalent stagionale da Open-Meteo (ERA5-Land)

Queste correzioni sono SUPPLEMENTARI rispetto alla formula primaria di Desilets
2010 e a quella topografica (kappa_topo/kappa_muon).  Non modificano il calcolo
di N0, ma forniscono:
  1. La stima della variabilità stagionale del segnale dovuta a WV
  2. L'entità del bias da biomassa aerea (AGBH) su N0 stimato e sul footprint
  3. La presenza e intensità del manto nevoso che, durante i mesi coperti,
     rende la sonda CRNS insensibile all'umidità del suolo sottostante.

Sorgenti dati
-------------
  - Open-Meteo Archive API (ERA5): precipitazione, umidità, neve
    URL: https://archive-api.open-meteo.com/v1/archive
    Periodo: 5 anni (2019-2023), media mensile
    Nessuna registrazione richiesta.

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import numpy as np
import requests
from net_utils import http_get
import hashlib
import json
import os

# ---------------------------------------------------------------------------
# Costanti fisiche
# ---------------------------------------------------------------------------

WV_ALPHA = 0.0054     # m³/g  (Zreda 2012, Rosolem 2013)
                       # f_WV = exp(-WV_ALPHA * rho_WV)

# Densità del manto nevoso fresco/medio — [kg/m³]
SNOW_DENSITY_FRESH   = 100.0   # neve fresca
SNOW_DENSITY_MEAN    = 250.0   # neve media (stagionale)

# Soglia SWE oltre la quale la sonda CRNS non vede il suolo
# Il manto nevoso modera totalmente i neutroni per SWE > ~150 mm
SWE_OPAQUE_MM = 150.0


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(lat, lon, extra=""):
    tag = f"{lat:.5f}_{lon:.5f}_{extra}"
    return hashlib.sha256(tag.encode()).hexdigest()[:16]


def _corr_cache_path(cache_dir, lat, lon):
    return os.path.join(cache_dir, f"crns_corr_{_cache_key(lat, lon)}.json")


def _save_corr_cache(data, cache_dir, lat, lon):
    os.makedirs(cache_dir, exist_ok=True)
    path = _corr_cache_path(cache_dir, lat, lon)

    def _serial(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    payload = {k: _serial(v) for k, v in data.items()}
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"   CRNS corrections cached -> {path}", flush=True)


def _load_corr_cache(cache_dir, lat, lon):
    path = _corr_cache_path(cache_dir, lat, lon)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        # Riconverti liste in array
        for k in list(d.keys()):
            if isinstance(d[k], list):
                d[k] = np.array(d[k])
        print(f"   CRNS corrections from cache: {path}", flush=True)
        return d
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Download ERA5 via Open-Meteo
# ---------------------------------------------------------------------------

def _fetch_openmeteo_era5(lat, lon, start_year=2019, end_year=2023,
                           timeout_s=60):
    """
    Scarica dati ERA5 giornalieri da Open-Meteo per calcolare la
    stagionalità mensile di umidità atmosferica (vapore acqueo) e SWE.

    Variabili richieste:
        temperature_2m        [°C]     — per calcolo rho_WV
        relative_humidity_2m  [%]      — per calcolo rho_WV
        snowfall              [cm]     — accumulo giornaliero neve (water equiv)
        snow_depth            [m]      — profondità del manto nevoso

    Parameters
    ----------
    lat, lon    : coordinate WGS84
    start_year  : anno inizio (incluso)
    end_year    : anno fine (incluso)

    Returns
    -------
    dict con array mensili (12 valori, indice 0=gennaio)
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"  : round(lat, 5),
        "longitude" : round(lon, 5),
        "start_date": f"{start_year}-01-01",
        "end_date"  : f"{end_year}-12-31",
        "daily"     : ("temperature_2m_mean,relative_humidity_2m_mean,"
                       "snowfall_sum,snow_depth_mean"),
        "timezone"  : "UTC",
    }
    resp = http_get(url, params=params, timeout=timeout_s)
    d = resp.json()["daily"]

    dates = d["time"]
    T_arr = np.array(d["temperature_2m_mean"],    dtype=float)
    RH_arr = np.array(d["relative_humidity_2m_mean"], dtype=float)
    SF_arr = np.array(d["snowfall_sum"],           dtype=float)   # cm/day water equiv
    SD_arr = np.array(d["snow_depth_mean"],        dtype=float)   # m

    # Sostituisci NaN con 0 per precipitazione/neve
    SF_arr = np.where(np.isnan(SF_arr), 0.0, SF_arr)
    SD_arr = np.where(np.isnan(SD_arr), 0.0, SD_arr)

    months = np.array([int(s[5:7]) for s in dates])

    # Per ogni mese: media multi-annuale
    rho_WV_monthly   = np.zeros(12)
    f_WV_monthly     = np.zeros(12)
    snowfall_monthly = np.zeros(12)   # mm/mese water equiv (media anni)
    swe_monthly      = np.zeros(12)   # mm SWE manto nevoso medio

    for m in range(1, 13):
        mask = months == m
        T_m  = T_arr[mask]
        RH_m = RH_arr[mask]
        SF_m = SF_arr[mask]   # cm/day
        SD_m = SD_arr[mask]   # m

        # Densità vapor acqueo: rho_WV = RH/100 * e_sat / (Rv * T_K)
        # Approssimazione Magnus: e_sat [Pa] = 611.2 * exp(17.67*T/(T+243.5))
        T_K  = T_m + 273.15
        e_sat = 611.2 * np.exp(17.67 * T_m / (T_m + 243.5))
        e_sat = np.where(np.isnan(e_sat), 611.2, e_sat)
        Rv   = 461.5   # J/(kg·K)
        rho  = (RH_m / 100.0) * e_sat / (Rv * T_K)  # kg/m³
        rho  = np.where(np.isnan(rho), 0.0, rho)
        rho_gm3 = rho * 1000.0   # g/m³

        rho_WV_monthly[m-1] = float(np.nanmean(rho_gm3))
        f_WV_monthly[m-1]   = float(np.mean(np.exp(-WV_ALPHA * rho_gm3)))

        # Neve: snowfall in cm water equiv / day → mm/mese
        n_days = np.sum(mask) / (end_year - start_year + 1)
        snowfall_monthly[m-1] = float(np.nanmean(SF_m) * 10.0 * n_days)  # cm→mm * giorni

        # SWE: snow_depth [m] * density_media [kg/m³] → mm
        swe_monthly[m-1] = float(np.nanmean(SD_m) * SNOW_DENSITY_MEAN)  # m * kg/m³ = mm

    return dict(
        rho_WV_gm3_monthly   = rho_WV_monthly,
        f_WV_monthly         = f_WV_monthly,
        snowfall_mm_monthly  = snowfall_monthly,
        swe_mm_monthly       = swe_monthly,
        era5_years           = f"{start_year}-{end_year}",
    )


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def get_crns_corrections(lat, lon, lai_annual_m2m2=0.0,
                          litter_water_mm=0.0,
                          start_year=2019, end_year=2023,
                          cache_dir=None, timeout_s=90):
    """
    Calcola le correzioni CRNS supplementari per la site characterization.

    Parameters
    ----------
    lat, lon         : coordinate WGS84
    lai_annual_m2m2  : LAI annuo medio [m²/m²] — per stima AGBH
    litter_water_mm  : acqua nel lettiera/erbaceo [mm] — supplementare AGBH
    start_year       : primo anno dati ERA5
    end_year         : ultimo anno dati ERA5
    cache_dir        : directory cache locale
    timeout_s        : timeout HTTP [s]

    Returns
    -------
    dict con:
        rho_WV_gm3_monthly   array(12) — umidità assoluta mensile [g/m³]
        f_WV_monthly         array(12) — fattore correttivo WV mensile [0..1]
        rho_WV_annual_mean   float     — media annua rho_WV [g/m³]
        f_WV_annual_mean     float     — media annua f_WV
        snowfall_mm_monthly  array(12) — snowfall mensile [mm water equiv]
        swe_mm_monthly       array(12) — SWE medio mensile [mm]
        snow_months          list      — mesi con SWE > soglia (1-based)
        agbh_mm              float     — AGBH stimato [mm]
        agbh_theta_equiv_per_cm float  — theta_v equiv per cm di z86
        era5_years           str
    """
    if cache_dir is not None:
        cached = _load_corr_cache(cache_dir, lat, lon)
        if cached is not None:
            # Aggiorna AGBH con i valori attuali (non dipende da ERA5)
            agbh = _compute_agbh(lai_annual_m2m2, litter_water_mm)
            cached.update(agbh)
            return cached

    print("   CRNS corrections: fetching ERA5 from Open-Meteo ...", flush=True)
    era5 = _fetch_openmeteo_era5(lat, lon, start_year, end_year, timeout_s)

    rho_WV = era5['rho_WV_gm3_monthly']
    swe    = era5['swe_mm_monthly']

    result = dict(
        rho_WV_gm3_monthly  = rho_WV,
        f_WV_monthly        = era5['f_WV_monthly'],
        rho_WV_annual_mean  = float(np.mean(rho_WV)),
        f_WV_annual_mean    = float(np.mean(era5['f_WV_monthly'])),
        snowfall_mm_monthly = era5['snowfall_mm_monthly'],
        swe_mm_monthly      = swe,
        snow_months         = [int(m+1) for m in range(12)
                               if swe[m] >= SWE_OPAQUE_MM * 0.3],
        era5_years          = era5['era5_years'],
    )

    if cache_dir is not None:
        _save_corr_cache(result, cache_dir, lat, lon)

    agbh = _compute_agbh(lai_annual_m2m2, litter_water_mm)
    result.update(agbh)
    return result


def _compute_agbh(lai_annual_m2m2, litter_water_mm=0.0,
                   leaf_water_g_per_m2=150.0):
    """
    Stima del contributo AGBH (Above-Ground Biomass Hydrogen).

    Formula (Baatz et al. 2015, Schrön 2017):
        AGBH_mm = LAI × water_per_m2_leaf + litter_mm

    L'AGBH si comporta come un'ulteriore sorgente di idrogeno sopra il suolo
    e riduce il footprint r86 e la profondità effettiva z86 misurata.
    Il suo effetto netto è: il sensore CRNS vede più idrogeno del solo suolo,
    quindi N0 appare più basso e theta_v misurato appare sovrastimato.

    Consiglio operativo: per siti forestati con LAI > 2, fare la calibrazione
    in condizioni di LAI minimo (inverno, foglie cadute) o misurare AGBH
    direttamente con survey fitomassa.
    """
    agbh_mm = float(lai_annual_m2m2 * leaf_water_g_per_m2 / 1000.0 * 1000.0
                    + litter_water_mm)
    # Conversione in theta_v equivalente per cm di z86
    # AGBH_mm / (z86_cm * 10) [adimensionale] — da usare come offset su theta_v
    # Riportiamo il valore grezzo in mm; la conversione richiede z86 (esterno)
    return dict(
        agbh_mm                 = agbh_mm,
        lai_used_m2m2           = float(lai_annual_m2m2),
        litter_water_mm         = float(litter_water_mm),
    )


# ---------------------------------------------------------------------------
# Report testuale
# ---------------------------------------------------------------------------

MONTHS = ['Gen','Feb','Mar','Apr','Mag','Giu',
          'Lug','Ago','Set','Ott','Nov','Dic']


def report_crns_corrections(res, z86_cm=15.0):
    """
    Stampa leggibile delle correzioni CRNS supplementari.

    Parameters
    ----------
    res    : dict da get_crns_corrections()
    z86_cm : profondità penetrazione neutroni [cm] — per convertire AGBH in theta_v
    """
    w = 72
    agbh_mm = res.get('agbh_mm', 0.0)
    agbh_theta = agbh_mm / (z86_cm * 10.0) if z86_cm > 0 else 0.0

    snow_months = res.get('snow_months', [])
    swe         = res.get('swe_mm_monthly', np.zeros(12))
    rho_wv      = res.get('rho_WV_gm3_monthly', np.zeros(12))
    f_wv        = res.get('f_WV_monthly', np.ones(12))
    sf          = res.get('snowfall_mm_monthly', np.zeros(12))

    snow_months = list(snow_months) if hasattr(snow_months, '__iter__') else []
    snow_names = [MONTHS[m-1] for m in snow_months] if snow_months else ["nessuno"]

    L = [
        "=" * w,
        "CRNS SUPPLEMENTARY CORRECTIONS",
        "=" * w,
        "",
        "  A) VAPORE ACQUEO ATMOSFERICO (κ_WV = exp(-0.0054 × ρ_WV))",
        f"     Sorgente: ERA5 Open-Meteo {res.get('era5_years','')}",
        f"     ρ_WV annua media : {res.get('rho_WV_annual_mean',0):.1f} g/m³",
        f"     κ_WV annuo medio : {res.get('f_WV_annual_mean',1):.4f}",
        f"     Impatto: riduce N osservato di ~{(1-res.get('f_WV_annual_mean',1))*100:.1f}% in media",
        "",
        f"     {'Mese':<6} {'ρ_WV [g/m³]':>12} {'κ_WV':>8} {'SWE [mm]':>10} {'Snowfall [mm]':>14}",
        "     " + "-" * 52,
    ]
    for i in range(12):
        L.append(f"     {MONTHS[i]:<6} {rho_wv[i]:12.1f} {f_wv[i]:8.4f}"
                 f" {swe[i]:10.1f} {sf[i]:14.1f}")
    L += [
        "",
        "  B) BIOMASSA AEREA (AGBH)",
        f"     LAI usato       : {res.get('lai_used_m2m2',0):.2f} m²/m²",
        f"     Lettiera        : {res.get('litter_water_mm',0):.1f} mm",
        f"     AGBH stimato    : {agbh_mm:.1f} mm  (acqua equiv. biomassa aerea)",
        f"     θ_v equiv AGBH  : {agbh_theta:.4f} m³/m³  (per z86={z86_cm:.1f} cm)",
        "     NOTA: LAI=0 → nessun bias. LAI>2 (foresta) → considerare",
        "     calibrazione in periodo a foglie cadute o misura diretta AGBH.",
        "",
        "  C) MANTO NEVOSO (SWE stagionale)",
        f"     SWE > 45 mm (30% soglia opaca) nei mesi: {', '.join(snow_names)}",
        f"     SWE > {SWE_OPAQUE_MM:.0f} mm → sonda non vede il suolo sottostante.",
        f"     Pianificare installazione/calibrazione fuori dalla stagione nevosa.",
        "=" * w,
    ]
    return "\n".join(L)
