"""
get_site_climate
================
Caratterizzazione climatica completa di un sito CRNS.

Sorgenti dati
-------------
- PVGIS TMY (JRC/EC) via pvlib.iotools.get_pvgis_tmy:
    radiazione solare, temperatura, umidità relativa, vento, pressione.
    Dati ERA5/SARAH3, periodo tipico 2005-2020.
    Se fornito horizon[], viene passato a PVGIS come userhorizon
    (profilo dell'orizzonte dal tuo DEM 30m — più preciso del DEM interno PVGIS).

- Open-Meteo Climate API (ERA5 ~31 km):
    precipitazione giornaliera storica → medie mensili e annuali.
    Nessuna registrazione richiesta.

Output
------
Tutto a risoluzione mensile (array 12 valori, indice 0=gennaio) + scalari annuali.
Vedi docstring di get_site_climate() per la struttura completa del dizionario.

Dipendenze
----------
    pip install pvlib requests numpy

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import numpy as np
import requests
from datetime import datetime, date


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

# Parametri modello temperatura cella Faiman 1994
# (usati internamente da pvlib.temperature.faiman)
FAIMAN_U0 = 25.0   # W m-2 K-1
FAIMAN_U1 = 6.84   # W m-2 K-1 (m/s)-1


# ---------------------------------------------------------------------------
# Helpers interni
# ---------------------------------------------------------------------------

def _monthly_mean(series, agg='mean'):
    """
    Aggrega una serie oraria pandas in 12 valori mensili.
    agg: 'mean', 'sum', 'max', 'min', 'p95'
    """
    grp = series.groupby(series.index.month)
    if agg == 'mean':
        return grp.mean().values
    elif agg == 'sum':
        # sum orario -> kWh/mese se valori in W/m2 (dividi per 1000)
        return grp.sum().values
    elif agg == 'max':
        return grp.max().values
    elif agg == 'min':
        return grp.min().values
    elif agg == 'p95':
        return grp.quantile(0.95).values
    else:
        raise ValueError(f"agg non riconosciuto: {agg}")


def _monthly_count(series, condition_func):
    """
    Conta ore/giorni per mese che soddisfano condition_func(series).
    condition_func: lambda x: x < 0
    """
    mask = condition_func(series)
    return mask.groupby(mask.index.month).sum().values


def _annual(monthly, agg='mean'):
    """Scalare annuale da array mensile."""
    if agg == 'mean':
        return float(np.mean(monthly))
    elif agg == 'sum':
        return float(np.sum(monthly))
    elif agg == 'max':
        return float(np.max(monthly))
    elif agg == 'min':
        return float(np.min(monthly))


def _reample_horizon_for_pvgis(horizon_deg, azimuths_deg, n_pvgis=36):
    """
    Resampla il profilo di orizzonte al passo uniforme richiesto da PVGIS.
    PVGIS accetta liste di angoli a passo uniforme partendo da nord (0°),
    senso orario. Default 36 valori = passo 10°.

    Parameters
    ----------
    horizon_deg  : array, angoli di elevazione dell'orizzonte [deg]
    azimuths_deg : array, azimuths corrispondenti [deg], 0=N, CW
    n_pvgis      : numero di punti per PVGIS (default 36 -> passo 10°)

    Returns
    -------
    list of float, lunghezza n_pvgis
    """
    az_pvgis = np.linspace(0, 360, n_pvgis, endpoint=False)
    h_interp = np.interp(az_pvgis, azimuths_deg,
                         horizon_deg, period=360)
    return [round(float(h), 2) for h in h_interp]


def _get_pvgis_tmy(lat, lon, userhorizon=None, startyear=2005, endyear=2020):
    """
    Scarica TMY da PVGIS via pvlib.
    Ritorna DataFrame orario con colonne pvlib standard.
    """
    import pvlib
    kwargs = dict(
        map_variables=True,
        usehorizon=True,
        startyear=startyear,
        endyear=endyear,
        timeout=60,
    )
    if userhorizon is not None:
        kwargs['userhorizon'] = userhorizon

    tmy, meta = pvlib.iotools.get_pvgis_tmy(lat, lon, **kwargs)
    return tmy, meta


def _compute_poa(tmy, lat, panel_tilt_deg, panel_azimuth_deg):
    """
    Calcola irradianza sul piano del pannello (POA) da GHI/DNI/DHI
    usando modello di trasposizione Perez via pvlib.

    Returns
    -------
    poa_global : Series oraria [W/m2]
    """
    import pvlib

    loc = pvlib.location.Location(latitude=lat, longitude=0,
                                   altitude=0)
    # Posizione solare
    solpos = loc.get_solarposition(tmy.index)

    # Trasposizione GHI -> POA (modello Perez)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt    = panel_tilt_deg,
        surface_azimuth = panel_azimuth_deg,
        solar_zenith    = solpos['apparent_zenith'],
        solar_azimuth   = solpos['azimuth'],
        dni  = tmy['dni'],
        ghi  = tmy['ghi'],
        dhi  = tmy['dhi'],
        model='perez',
    )
    return poa['poa_global']


def _compute_cell_temp(poa_global, temp_air, wind_speed):
    """
    Temperatura di cella con modello Faiman 1994.
    T_cell = T_air + POA / (U0 + U1 * WS)
    """
    return temp_air + poa_global / (FAIMAN_U0 + FAIMAN_U1 * wind_speed)


def _get_precipitation_openmeteo(lat, lon, startyear=2005, endyear=2020):
    """
    Scarica precipitazioni giornaliere storiche da Open-Meteo Climate API
    (sorgente ERA5, risoluzione ~31 km).

    Ritorna array (12,) con precipitazione mensile media [mm/mese],
    array (12,) con giorni medi di pioggia per mese (precip > 1 mm),
    e quota della cella ERA5 [m] (campo 'elevation' nella risposta JSON).
    """
    url = "https://climate-api.open-meteo.com/v1/climate"
    params = {
        "latitude"         : lat,
        "longitude"        : lon,
        "start_date"       : f"{startyear}-01-01",
        "end_date"         : f"{endyear}-12-31",
        "models"           : "ERA5",
        "daily"            : "precipitation_sum",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    era5_elevation_m = float(data.get('elevation', float('nan')))

    import pandas as pd
    dates  = pd.to_datetime(data['daily']['time'])
    precip = np.array(data['daily']['precipitation_sum'], dtype=float)
    precip = np.nan_to_num(precip, nan=0.0)

    s = pd.Series(precip, index=dates)

    # Media mensile del totale mensile (mm/mese)
    monthly_total = s.resample('ME').sum()           # mm per ogni mese
    monthly_mean  = monthly_total.groupby(
                        monthly_total.index.month).mean().values  # 12 valori

    # Giorni medi di pioggia per mese
    rainy         = (s > 1.0).astype(float)
    rainy_monthly = rainy.resample('ME').sum()
    rainy_mean    = rainy_monthly.groupby(
                        rainy_monthly.index.month).mean().values

    return monthly_mean, rainy_mean, era5_elevation_m


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def get_site_climate(
    lat,
    lon,
    alt_m,
    horizon_deg       = None,   # array elevazione orizzonte [deg], dal tuo DEM
    azimuths_deg      = None,   # array azimuths [deg] corrispondenti
    panel_efficiency  = 0.20,   # efficienza pannello [-]
    panel_tilt_deg    = None,   # None = usa lat come inclinazione ottimale
    panel_azimuth_deg = 180.0,  # 180 = sud
    startyear         = 2005,
    endyear           = 2020,
):
    """
    Caratterizzazione climatica completa di un sito CRNS.

    Parameters
    ----------
    lat, lon          : coordinate WGS84 [deg]
    alt_m             : quota sito [m a.s.l.]
    horizon_deg       : profilo orizzonte dal DEM 30m [deg elevazione]
                        Se None, PVGIS usa il suo DEM interno (~90m)
    azimuths_deg      : azimuths corrispondenti a horizon_deg [deg]
    panel_efficiency  : efficienza pannello fotovoltaico [-]  default 0.20
    panel_tilt_deg    : inclinazione pannello [deg]  None -> usa lat
    panel_azimuth_deg : azimuth pannello [deg]  180=sud
    startyear         : primo anno del periodo TMY
    endyear           : ultimo anno del periodo TMY

    Returns
    -------
    dict con chiavi descritte sotto. Tutti gli array mensili hanno
    lunghezza 12, indice 0 = gennaio.

    Radiazione (da PVGIS TMY):
        GHI_monthly_kWh_m2      [12]  irradianza globale orizzontale
        DNI_monthly_kWh_m2      [12]  diretta normale
        DHI_monthly_kWh_m2      [12]  diffusa orizzontale
        POA_monthly_kWh_m2      [12]  sul piano pannello
        sunshine_hours_monthly  [12]  ore equivalenti di sole
        GHI_annual_kWh_m2       scalare annuale
        POA_annual_kWh_m2       scalare annuale

    Produzione FV:
        energy_monthly_kWh      [12]  produzione mensile pannello 1m²
        energy_annual_kWh       scalare annuale
        power_peak_W            potenza di picco stimata

    Temperatura (da PVGIS TMY):
        T_mean_monthly_C        [12]  media mensile
        T_min_monthly_C         [12]  minima mensile
        T_max_monthly_C         [12]  massima mensile
        frost_days_monthly      [12]  giorni con almeno 1h sotto 0°C
        T_mean_annual_C         scalare
        T_min_annual_C          scalare (minima assoluta annua)
        T_max_annual_C          scalare (massima assoluta annua)
        frost_days_annual       scalare

    Umidità relativa (da PVGIS TMY):
        RH_mean_monthly_pct     [12]
        RH_min_monthly_pct      [12]
        RH_max_monthly_pct      [12]
        RH_mean_annual_pct      scalare

    Pressione (da PVGIS TMY):
        SP_mean_monthly_hPa     [12]
        SP_mean_annual_hPa      scalare

    Vento (da PVGIS TMY):
        WS_mean_monthly_ms      [12]  velocità media mensile
        WS_max_monthly_ms       [12]  massima oraria mensile
        WS_p95_monthly_ms       [12]  percentile 95 mensile
        WS_mean_annual_ms       scalare
        WS_max_annual_ms        scalare
        WS_p95_annual_ms        scalare

    Precipitazioni (da Open-Meteo ERA5):
        precip_monthly_mm       [12]  totale mensile medio
        rainy_days_monthly      [12]  giorni medi con precip > 1mm
        precip_annual_mm        scalare
        rainy_days_annual       scalare
        dry_months              lista mesi (nome) con precip < 30mm
        wet_months              lista mesi (nome) con precip > 100mm

    Metadata:
        panel_tilt_deg          inclinazione pannello usata
        panel_azimuth_deg       azimuth pannello usato
        panel_efficiency        efficienza usata
        horizon_source          'user_dem_30m' o 'pvgis_internal'
        data_source_radiation   stringa database PVGIS
        data_source_precip      'ERA5 via Open-Meteo ~31km'
        spatial_warning_alpine  True se alt_m > 800m
        startyear, endyear
        months                  lista nomi mesi ['Jan',...,'Dec']
    """

    # ------------------------------------------------------------------ #
    # 0. Setup
    # ------------------------------------------------------------------ #
    if panel_tilt_deg is None:
        panel_tilt_deg = round(abs(lat))   # inclinazione ottimale ≈ lat

    userhorizon   = None
    horizon_source = 'pvgis_internal'
    if horizon_deg is not None and azimuths_deg is not None:
        userhorizon    = _reample_horizon_for_pvgis(
                             horizon_deg, azimuths_deg)
        horizon_source = 'user_dem_30m'

    # ------------------------------------------------------------------ #
    # 1. PVGIS TMY
    # ------------------------------------------------------------------ #
    tmy, meta = _get_pvgis_tmy(lat, lon,
                                userhorizon=userhorizon,
                                startyear=startyear,
                                endyear=endyear)

    # Pressione: PVGIS restituisce Pa -> converti in hPa
    tmy['pressure_hpa'] = tmy['pressure'] / 100.0

    # ------------------------------------------------------------------ #
    # 2. POA e produzione FV
    # ------------------------------------------------------------------ #
    poa = _compute_poa(tmy, lat, panel_tilt_deg, panel_azimuth_deg)

    # Temperatura cella
    t_cell = _compute_cell_temp(poa, tmy['temp_air'], tmy['wind_speed'])

    # Efficienza corretta per temperatura (coefficiente tipico -0.004 /°C)
    # riferimento STC = 25°C
    gamma    = -0.004
    eta_cell = panel_efficiency * (1 + gamma * (t_cell - 25.0))
    eta_cell = eta_cell.clip(lower=0.0)

    # Produzione oraria [Wh] per 1 m²
    power_hourly = poa * eta_cell * 1.0   # area = 1 m²

    # ------------------------------------------------------------------ #
    # 3. Aggregazioni mensili — RADIAZIONE
    # ------------------------------------------------------------------ #
    # Da W/m2 orari a kWh/m2/mese: sum(W/m2 * 1h) / 1000
    GHI_monthly = _monthly_mean(tmy['ghi'],  agg='sum') / 1000.0
    DNI_monthly = _monthly_mean(tmy['dni'],  agg='sum') / 1000.0
    DHI_monthly = _monthly_mean(tmy['dhi'],  agg='sum') / 1000.0
    POA_monthly = _monthly_mean(poa,         agg='sum') / 1000.0

    # Ore di sole equivalenti = GHI mensile [kWh/m2] / 1 kW/m2
    sunshine_monthly = GHI_monthly.copy()   # numericamente identico

    # Produzione FV mensile [kWh]
    energy_monthly = _monthly_mean(power_hourly, agg='sum') / 1000.0

    # ------------------------------------------------------------------ #
    # 4. Aggregazioni mensili — TEMPERATURA
    # ------------------------------------------------------------------ #
    T_mean_monthly = _monthly_mean(tmy['temp_air'], agg='mean')
    T_min_monthly  = _monthly_mean(tmy['temp_air'], agg='min')
    T_max_monthly  = _monthly_mean(tmy['temp_air'], agg='max')

    # Giorni con almeno 1 ora sotto 0°C nel mese
    # Conta ore sotto 0 per mese, poi converti in giorni (/24)
    frost_hours   = _monthly_count(tmy['temp_air'], lambda x: x < 0.0)
    frost_days    = frost_hours / 24.0

    # ------------------------------------------------------------------ #
    # 5. Aggregazioni mensili — UMIDITA'
    # ------------------------------------------------------------------ #
    RH_mean_monthly = _monthly_mean(tmy['relative_humidity'], agg='mean')
    RH_min_monthly  = _monthly_mean(tmy['relative_humidity'], agg='min')
    RH_max_monthly  = _monthly_mean(tmy['relative_humidity'], agg='max')

    # ------------------------------------------------------------------ #
    # 6. Aggregazioni mensili — PRESSIONE
    # ------------------------------------------------------------------ #
    SP_mean_monthly = _monthly_mean(tmy['pressure_hpa'], agg='mean')

    # ------------------------------------------------------------------ #
    # 7. Aggregazioni mensili — VENTO
    # ------------------------------------------------------------------ #
    WS_mean_monthly = _monthly_mean(tmy['wind_speed'], agg='mean')
    WS_max_monthly  = _monthly_mean(tmy['wind_speed'], agg='max')
    WS_p95_monthly  = _monthly_mean(tmy['wind_speed'], agg='p95')

    # ------------------------------------------------------------------ #
    # 8. Precipitazioni — Open-Meteo
    # ------------------------------------------------------------------ #
    precip_monthly, rainy_days_monthly, era5_elevation_m = \
        _get_precipitation_openmeteo(lat, lon, startyear=startyear, endyear=endyear)

    # ------------------------------------------------------------------ #
    # 9. Scalari annuali
    # ------------------------------------------------------------------ #
    result = dict(

        # --- Radiazione ---
        GHI_monthly_kWh_m2      = GHI_monthly,
        DNI_monthly_kWh_m2      = DNI_monthly,
        DHI_monthly_kWh_m2      = DHI_monthly,
        POA_monthly_kWh_m2      = POA_monthly,
        sunshine_hours_monthly  = sunshine_monthly,
        GHI_annual_kWh_m2       = _annual(GHI_monthly, 'sum'),
        POA_annual_kWh_m2       = _annual(POA_monthly, 'sum'),

        # --- Produzione FV ---
        energy_monthly_kWh      = energy_monthly,
        energy_annual_kWh       = _annual(energy_monthly, 'sum'),
        power_peak_W            = float(panel_efficiency * 1000.0),

        # --- Temperatura ---
        T_mean_monthly_C        = T_mean_monthly,
        T_min_monthly_C         = T_min_monthly,
        T_max_monthly_C         = T_max_monthly,
        frost_days_monthly      = frost_days,
        T_mean_annual_C         = _annual(T_mean_monthly, 'mean'),
        T_min_annual_C          = _annual(T_min_monthly,  'min'),
        T_max_annual_C          = _annual(T_max_monthly,  'max'),
        frost_days_annual       = _annual(frost_days,     'sum'),

        # --- Umidità relativa ---
        RH_mean_monthly_pct     = RH_mean_monthly,
        RH_min_monthly_pct      = RH_min_monthly,
        RH_max_monthly_pct      = RH_max_monthly,
        RH_mean_annual_pct      = _annual(RH_mean_monthly, 'mean'),

        # --- Pressione ---
        SP_mean_monthly_hPa     = SP_mean_monthly,
        SP_mean_annual_hPa      = _annual(SP_mean_monthly, 'mean'),

        # --- Vento ---
        WS_mean_monthly_ms      = WS_mean_monthly,
        WS_max_monthly_ms       = WS_max_monthly,
        WS_p95_monthly_ms       = WS_p95_monthly,
        WS_mean_annual_ms       = _annual(WS_mean_monthly, 'mean'),
        WS_max_annual_ms        = _annual(WS_max_monthly,  'max'),
        WS_p95_annual_ms        = _annual(WS_p95_monthly,  'mean'),

        # --- Precipitazioni ---
        precip_monthly_mm       = precip_monthly,
        rainy_days_monthly      = rainy_days_monthly,
        precip_annual_mm        = _annual(precip_monthly,    'sum'),
        rainy_days_annual       = _annual(rainy_days_monthly,'sum'),
        dry_months              = [MONTHS[i] for i,v in
                                   enumerate(precip_monthly) if v < 30],
        wet_months              = [MONTHS[i] for i,v in
                                   enumerate(precip_monthly) if v > 100],

        # --- ERA5 grid ---
        era5_elevation_m        = era5_elevation_m,

        # --- Metadata ---
        panel_tilt_deg          = panel_tilt_deg,
        panel_azimuth_deg       = panel_azimuth_deg,
        panel_efficiency        = panel_efficiency,
        horizon_source          = horizon_source,
        data_source_radiation   = str(meta.get('inputs', {})
                                      .get('meteo_data', {})
                                      .get('radiation_db', 'PVGIS')),
        data_source_precip      = 'ERA5 via Open-Meteo ~31km',
        spatial_warning_alpine  = alt_m > 800,
        startyear               = startyear,
        endyear                 = endyear,
        months                  = MONTHS,
    )

    return result


# ---------------------------------------------------------------------------
# Report leggibile
# ---------------------------------------------------------------------------

def report_site_climate(res, site_name=""):
    """Stampa tabellare dei risultati di get_site_climate."""
    w = 72
    M = res['months']

    def row(label, vals, fmt='.1f', unit=''):
        v_str = '  '.join(f'{v:{fmt}}' for v in vals)
        return f"  {label:<22} {v_str}  {unit}"

    def srow(label, val, fmt='.1f', unit=''):
        return f"  {label:<30} {val:{fmt}}  {unit}"

    lines = []
    s = lines.append

    s("=" * w)
    s(f"SITE CLIMATE SUMMARY  {site_name}")
    s("=" * w)
    s(f"  Period     : {res['startyear']}–{res['endyear']}")
    s(f"  Radiation  : {res['data_source_radiation']}")
    s(f"  Horizon    : {res['horizon_source']}")
    s(f"  Precip     : {res['data_source_precip']}")
    if res['spatial_warning_alpine']:
        s("  [!] Alpine site: precipitation uncertainty ±30-50%")
    s("")

    # Header mesi
    hdr = "  " + " " * 22 + "  ".join(f"{m:>5}" for m in M)
    s(hdr)
    s("  " + "-" * (w - 2))

    s("  RADIATION  [kWh/m²/month]")
    s(row("GHI",              res['GHI_monthly_kWh_m2'],      '.0f'))
    s(row("DNI",              res['DNI_monthly_kWh_m2'],      '.0f'))
    s(row("DHI",              res['DHI_monthly_kWh_m2'],      '.0f'))
    s(row("POA (panel)",      res['POA_monthly_kWh_m2'],      '.0f'))
    s(row("Sunshine hrs",     res['sunshine_hours_monthly'],  '.0f', 'h'))
    s("")

    s("  PHOTOVOLTAIC  [kWh/month, 1m², "
      f"η={res['panel_efficiency']:.0%}, tilt={res['panel_tilt_deg']}°]")
    s(row("Production",       res['energy_monthly_kWh'],      '.1f', 'kWh'))
    s("")

    s("  TEMPERATURE  [°C]")
    s(row("T mean",           res['T_mean_monthly_C'],        '.1f'))
    s(row("T min",            res['T_min_monthly_C'],         '.1f'))
    s(row("T max",            res['T_max_monthly_C'],         '.1f'))
    s(row("Frost days",       res['frost_days_monthly'],      '.1f', 'd'))
    s("")

    s("  RELATIVE HUMIDITY  [%]")
    s(row("RH mean",          res['RH_mean_monthly_pct'],     '.0f'))
    s(row("RH min",           res['RH_min_monthly_pct'],      '.0f'))
    s(row("RH max",           res['RH_max_monthly_pct'],      '.0f'))
    s("")

    s("  PRESSURE  [hPa]")
    s(row("SP mean",          res['SP_mean_monthly_hPa'],     '.1f'))
    s("")

    s("  WIND  [m/s]")
    s(row("WS mean",          res['WS_mean_monthly_ms'],      '.1f'))
    s(row("WS max",           res['WS_max_monthly_ms'],       '.1f'))
    s(row("WS p95",           res['WS_p95_monthly_ms'],       '.1f'))
    s("")

    s("  PRECIPITATION  [mm/month]")
    s(row("Precip",           res['precip_monthly_mm'],       '.0f', 'mm'))
    s(row("Rainy days",       res['rainy_days_monthly'],      '.1f', 'd'))
    s("")

    s("  " + "=" * (w - 2))
    s("  ANNUAL SUMMARY")
    s("  " + "-" * (w - 2))
    s(srow("GHI annual",          res['GHI_annual_kWh_m2'],   '.0f', 'kWh/m²'))
    s(srow("POA annual",          res['POA_annual_kWh_m2'],   '.0f', 'kWh/m²'))
    s(srow("FV production annual",res['energy_annual_kWh'],   '.1f', 'kWh'))
    s(srow("T mean annual",       res['T_mean_annual_C'],     '.1f', '°C'))
    s(srow("T min annual (abs)",  res['T_min_annual_C'],      '.1f', '°C'))
    s(srow("T max annual (abs)",  res['T_max_annual_C'],      '.1f', '°C'))
    s(srow("Frost days annual",   res['frost_days_annual'],   '.0f', 'd'))
    s(srow("RH mean annual",      res['RH_mean_annual_pct'],  '.0f', '%'))
    s(srow("Pressure mean annual",res['SP_mean_annual_hPa'],  '.1f', 'hPa'))
    s(srow("Wind mean annual",    res['WS_mean_annual_ms'],   '.1f', 'm/s'))
    s(srow("Wind max annual",     res['WS_max_annual_ms'],    '.1f', 'm/s'))
    s(srow("Wind p95 annual",     res['WS_p95_annual_ms'],    '.1f', 'm/s'))
    s(srow("Precipitation annual",res['precip_annual_mm'],    '.0f', 'mm'))
    s(srow("Rainy days annual",   res['rainy_days_annual'],   '.0f', 'd'))
    s(f"  {'Dry months':<30} {', '.join(res['dry_months']) or 'none'}")
    s(f"  {'Wet months':<30} {', '.join(res['wet_months']) or 'none'}")
    s("=" * w)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test  (richiede connessione internet)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sites = [
        ("LIMENA",         45.467, 11.851,   25.0),
        ("Malga Fadner",   46.925, 11.861, 1100.0),
    ]

    for name, lat, lon, alt in sites:
        print(f"\nFetching climate data for {name} ...")
        try:
            res = get_site_climate(lat, lon, alt,
                                   startyear=2010, endyear=2020)
            print(report_site_climate(res, site_name=name))
        except Exception as e:
            print(f"  ERRORE: {e}")
            sys.exit(1)
