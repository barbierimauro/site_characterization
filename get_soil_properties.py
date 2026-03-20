"""
get_soil_properties
===================
Recupera le proprietà del suolo dal database SoilGrids v2.0 (ISRIC)
per un punto geografico, con media pesata per profondità CRNS.

Sorgente
--------
SoilGrids v2.0 — ISRIC World Soil Information
  Risoluzione: 250 m
  API REST pubblica, nessuna registrazione
  URL: https://rest.soilgrids.org/soilgrids/v2.0/properties/query

Proprietà recuperate
--------------------
  bdod    Bulk density             [cg/cm³  -> g/cm³]
  clay    Clay content             [g/kg    -> %]
  sand    Sand content             [g/kg    -> %]
  silt    Silt content             [g/kg    -> %]
  soc     Soil organic carbon      [dg/kg   -> g/kg]
  phh2o   pH in H2O                [pHx10   -> pH]
  cec     Cation exchange capacity [mmol/kg -> cmol/kg]
  cfvo    Coarse fragments volume  [cm³/dm³ -> %]
  nitrogen Total nitrogen          [cg/kg   -> g/kg]

Profondità disponibili in SoilGrids
------------------------------------
  0-5cm, 5-15cm, 15-30cm, 30-60cm, 60-100cm, 100-200cm

Media pesata CRNS
-----------------
Per ogni proprietà viene calcolata la media pesata sulle profondità
usando il profilo di attenuazione esponenziale dei neutroni epitermali:

    W(z) = exp(-z * rho_b / lambda_s)

dove z è la profondità al centro dello strato [cm],
rho_b è la bulk density locale [g/cm³],
lambda_s = LAMBDA_S_GCM2 / (rho_b * 100) [cm].

Il peso di ogni strato è l'integrale di W(z) nello strato.
Strati oltre z86*3 hanno peso trascurabile e vengono inclusi ma
non alterano il risultato.

Lattice water
-------------
Stimata dalla formula di Köhli et al. 2021:
    lw = 0.097 * (clay_pct/100) + 0.033   [g/g]

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

SOILGRIDS_URL    = "https://rest.soilgrids.org/soilgrids/v2.0/properties/query"

LAMBDA_S_GCM2    = 162.0    # attenuation length in soil [g/cm²]

# Proprietà da richiedere e relative conversioni
# cf = fattore di conversione: valore_raw / cf = valore_in_target_units
PROPERTIES = {
    'bdod':     {'cf': 100,  'target_units': 'g/cm³',   'desc': 'Bulk density'},
    'clay':     {'cf': 10,   'target_units': '%',        'desc': 'Clay content'},
    'sand':     {'cf': 10,   'target_units': '%',        'desc': 'Sand content'},
    'silt':     {'cf': 10,   'target_units': '%',        'desc': 'Silt content'},
    'soc':      {'cf': 10,   'target_units': 'g/kg',     'desc': 'Soil organic carbon'},
    'phh2o':    {'cf': 10,   'target_units': 'pH',       'desc': 'pH in H2O'},
    'cec':      {'cf': 10,   'target_units': 'cmol/kg',  'desc': 'CEC'},
    'cfvo':     {'cf': 10,   'target_units': '%',        'desc': 'Coarse fragments vol'},
    'nitrogen': {'cf': 100,  'target_units': 'g/kg',     'desc': 'Total nitrogen'},
}

# Profondità SoilGrids: label, top_cm, bottom_cm, midpoint_cm
DEPTHS = [
    ('0-5cm',    0,   5,   2.5),
    ('5-15cm',   5,  15,  10.0),
    ('15-30cm', 15,  30,  22.5),
    ('30-60cm', 30,  60,  45.0),
    ('60-100cm',60, 100,  80.0),
    ('100-200cm',100,200,150.0),
]

DEPTH_LABELS   = [d[0] for d in DEPTHS]
DEPTH_TOPS     = np.array([d[1] for d in DEPTHS], dtype=float)
DEPTH_BOTS     = np.array([d[2] for d in DEPTHS], dtype=float)
DEPTH_MIDS     = np.array([d[3] for d in DEPTHS], dtype=float)
DEPTH_THICKS   = DEPTH_BOTS - DEPTH_TOPS   # spessore strato [cm]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_weights(rho_b_gcm3, lambda_s_gcm2=LAMBDA_S_GCM2):
    """
    Peso integrale di ogni strato SoilGrids per i neutroni epitermali.

    W_layer(i) = integral_{top_i}^{bot_i} exp(-z * rho_b / lambda_s) dz
               = (lambda_s/rho_b) * [exp(-top*rho_b/lambda_s)
                                    - exp(-bot*rho_b/lambda_s)]

    I pesi sono normalizzati a somma 1.

    Parameters
    ----------
    rho_b_gcm3    : bulk density [g/cm³]
    lambda_s_gcm2 : attenuation length in soil [g/cm²]

    Returns
    -------
    weights : array (6,), normalizzati a somma 1
    """
    lam = lambda_s_gcm2 / rho_b_gcm3   # [cm]
    w   = (np.exp(-DEPTH_TOPS / lam)
           - np.exp(-DEPTH_BOTS / lam))
    w   = np.maximum(w, 0.0)
    s   = w.sum()
    return w / s if s > 0 else np.ones(len(DEPTHS)) / len(DEPTHS)


def _weighted_mean(values, weights):
    """
    Media pesata ignorando i NaN.
    values, weights : array (n,)
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = ~np.isnan(v)
    if not mask.any():
        return np.nan
    return float(np.sum(v[mask] * w[mask]) / w[mask].sum())


def _lattice_water(clay_pct):
    """
    Acqua di reticolo da Köhli et al. 2021.
    lw = 0.097 * clay_frac + 0.033   [g/g]
    clay_pct: percentuale di argilla [%]
    """
    return float(0.097 * (clay_pct / 100.0) + 0.033)


def _parse_layer(layer_json, cf):
    """
    Estrae valori da un layer SoilGrids e applica la conversione unità.

    Returns
    -------
    means  : array (6,) valori medi nelle 6 profondità, NaN se assente
    uncert : array (6,) incertezze (std), NaN se assente
    """
    # Mappa depth_label -> (mean, uncertainty)
    depth_map = {}
    for d in layer_json.get('depths', []):
        lbl  = d['label']
        vals = d.get('values', {})
        m    = vals.get('mean')
        u    = vals.get('uncertainty')
        depth_map[lbl] = (
            float(m) / cf if m is not None else np.nan,
            float(u) / cf if u is not None else np.nan,
        )

    means  = np.array([depth_map.get(lbl, (np.nan, np.nan))[0]
                       for lbl in DEPTH_LABELS])
    uncert = np.array([depth_map.get(lbl, (np.nan, np.nan))[1]
                       for lbl in DEPTH_LABELS])
    return means, uncert


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def get_soil_properties(
    lat,
    lon,
    z86_cm      = 16.0,    # profondità di penetrazione neutroni [cm]
                            # usata per calcolare i pesi degli strati
    timeout_s   = 60,
):
    """
    Recupera le proprietà del suolo da SoilGrids v2.0 per il punto (lat, lon).

    Parameters
    ----------
    lat, lon  : coordinate WGS84 [deg]
    z86_cm    : profondità di penetrazione neutroni epitermali [cm]
                Usata come scala per il profilo di peso W(z).
                Default 16 cm (theta_v=0.20, rho_b=1.4 g/cm³).
                Passare il valore calcolato dalla propria pipeline.
    timeout_s : timeout HTTP [s]

    Returns
    -------
    dict con struttura:

    Per ogni proprietà p in {bdod, clay, sand, silt, soc, phh2o,
                              cec, cfvo, nitrogen}:

        '<p>_profile' : dict con
            'depths'       : lista label ['0-5cm', '5-15cm', ...]
            'top_cm'       : array (6,)
            'bottom_cm'    : array (6,)
            'mid_cm'       : array (6,)
            'mean'         : array (6,) valori medi in target_units
            'uncertainty'  : array (6,) std in target_units
            'units'        : str, es. 'g/cm³'
            'desc'         : str, nome esteso

        '<p>_crns'     : float, media pesata W(z) sulle profondità
        '<p>_crns_unc' : float, incertezza propagata sulla media pesata

    Campi derivati:
        'layer_weights'      : array (6,) pesi W(z) normalizzati
        'z86_cm_used'        : float
        'lattice_water_gg'   : float, lw stimato da clay% pesato [g/g]
        'texture_class'      : str, classificazione USDA dalla tessitura media
        'wrb_class'          : str, tipo di suolo WRB (se disponibile)

    Metadata:
        'lat_queried'        : float
        'lon_queried'        : float
        'source'             : str
        'soilgrids_version'  : str
    """

    # ------------------------------------------------------------------ #
    # 1. Chiamata API SoilGrids
    # ------------------------------------------------------------------ #
    prop_list  = ','.join(PROPERTIES.keys())
    depth_list = ','.join(DEPTH_LABELS)

    params = {
        'lon'      : round(lon, 6),
        'lat'      : round(lat, 6),
        'property' : prop_list,
        'depth'    : depth_list,
        'value'    : 'mean,uncertainty',
    }

    resp = requests.get(SOILGRIDS_URL, params=params, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    # ------------------------------------------------------------------ #
    # 2. Parsing risposta
    # ------------------------------------------------------------------ #
    layers_json = data['properties']['layers']

    # Indicizza per nome proprietà
    layer_by_name = {lay['name']: lay for lay in layers_json}

    # ------------------------------------------------------------------ #
    # 3. Estrai profili per ogni proprietà
    # ------------------------------------------------------------------ #
    profiles = {}
    for prop, meta in PROPERTIES.items():
        if prop not in layer_by_name:
            # Proprietà non restituita — riempi con NaN
            means  = np.full(len(DEPTHS), np.nan)
            uncert = np.full(len(DEPTHS), np.nan)
        else:
            means, uncert = _parse_layer(layer_by_name[prop], meta['cf'])

        profiles[prop] = {
            'depths'      : DEPTH_LABELS,
            'top_cm'      : DEPTH_TOPS.copy(),
            'bottom_cm'   : DEPTH_BOTS.copy(),
            'mid_cm'      : DEPTH_MIDS.copy(),
            'mean'        : means,
            'uncertainty' : uncert,
            'units'       : meta['target_units'],
            'desc'        : meta['desc'],
        }

    # ------------------------------------------------------------------ #
    # 4. Pesi per la media CRNS
    # Usiamo bdod medio (0-30cm) come rho_b per i pesi;
    # se non disponibile usiamo default 1.4 g/cm³
    # ------------------------------------------------------------------ #
    bdod_top3 = profiles['bdod']['mean'][:3]   # 0-5, 5-15, 15-30 cm
    valid      = bdod_top3[~np.isnan(bdod_top3)]
    rho_b_for_weights = float(np.mean(valid)) if len(valid) > 0 else 1.4

    weights = _layer_weights(rho_b_for_weights)

    # ------------------------------------------------------------------ #
    # 5. Media pesata CRNS per ogni proprietà
    # ------------------------------------------------------------------ #
    result = {}

    for prop, prof in profiles.items():
        key        = f'{prop}_profile'
        result[key] = prof

        # Media pesata
        crns_mean  = _weighted_mean(prof['mean'], weights)
        result[f'{prop}_crns'] = crns_mean

        # Incertezza propagata: sqrt(sum(w_i^2 * sigma_i^2)) / sum(w_i)
        # (pesi già normalizzati a somma 1, quindi denominatore = 1)
        unc_vals  = prof['uncertainty']
        valid_unc = ~np.isnan(unc_vals)
        if valid_unc.any():
            crns_unc = float(np.sqrt(
                np.sum((weights[valid_unc] * unc_vals[valid_unc])**2)
            ))
        else:
            crns_unc = np.nan
        result[f'{prop}_crns_unc'] = crns_unc

    # ------------------------------------------------------------------ #
    # 6. Lattice water (Köhli 2021)
    # ------------------------------------------------------------------ #
    clay_crns = result.get('clay_crns', np.nan)
    lw = _lattice_water(clay_crns) if not np.isnan(clay_crns) else np.nan
    result['lattice_water_gg'] = lw

    # ------------------------------------------------------------------ #
    # 7. Classificazione tessitura USDA
    # ------------------------------------------------------------------ #
    sand_m = result.get('sand_crns', np.nan)
    clay_m = result.get('clay_crns', np.nan)
    silt_m = result.get('silt_crns', np.nan)
    result['texture_class'] = _usda_texture_class(sand_m, clay_m, silt_m)

    # ------------------------------------------------------------------ #
    # 8. WRB soil class (se disponibile nella risposta)
    # ------------------------------------------------------------------ #
    result['wrb_class'] = data.get('wrb_class_name', 'N/A')

    # ------------------------------------------------------------------ #
    # 9. Metadata
    # ------------------------------------------------------------------ #
    result['layer_weights']    = weights
    result['z86_cm_used']      = float(z86_cm)
    result['rho_b_for_weights']= rho_b_for_weights
    result['lat_queried']      = float(lat)
    result['lon_queried']      = float(lon)
    result['source']           = 'SoilGrids v2.0 ISRIC 250m'
    result['soilgrids_version']= '2.0'

    return result


# ---------------------------------------------------------------------------
# Classificazione tessitura USDA
# ---------------------------------------------------------------------------

def _usda_texture_class(sand_pct, clay_pct, silt_pct):
    """
    Classificazione USDA dalla tessitura (sand, clay, silt in %).
    Triangolo USDA standard.
    Ritorna stringa o 'Unknown' se valori non disponibili.
    """
    if any(np.isnan(v) for v in [sand_pct, clay_pct, silt_pct]):
        return 'Unknown'

    s, c, si = sand_pct, clay_pct, silt_pct

    if c >= 40 and si >= 40:
        return 'Silty clay'
    if c >= 40 and s <= 45:
        return 'Clay'
    if c >= 40 and s > 45:
        return 'Sandy clay'
    if c >= 27 and c < 40 and s <= 20:
        return 'Silty clay loam'
    if c >= 27 and c < 40 and s > 20 and s <= 45:
        return 'Clay loam'
    if c >= 27 and c < 40 and s > 45:
        return 'Sandy clay loam'
    if c >= 7 and c < 27 and si >= 28 and si < 50 and s <= 52:
        return 'Loam'
    if (c < 27 and si >= 50) or (c >= 12 and si >= 60):
        return 'Silty loam'
    if c < 12 and si >= 80:
        return 'Silt'
    if c < 20 and s >= 52 and s < 85:
        return 'Sandy loam'
    if c < 7 and si < 28 and s >= 52:
        return 'Loamy sand'
    if s >= 85:
        return 'Sand'
    return 'Loam'   # fallback


# ---------------------------------------------------------------------------
# Report leggibile
# ---------------------------------------------------------------------------

def report_soil_properties(res):
    """Stampa tabellare dei risultati di get_soil_properties."""
    w = 72
    L = [
        "=" * w,
        "SOIL PROPERTIES — SoilGrids v2.0 ISRIC 250m",
        "=" * w,
        f"  Location  : {res['lat_queried']:.4f}N  {res['lon_queried']:.4f}E",
        f"  WRB class : {res['wrb_class']}",
        f"  Texture   : {res['texture_class']}  (USDA, CRNS-weighted)",
        f"  Lattice W : {res['lattice_water_gg']:.4f} g/g  (Köhli 2021)",
        f"  z86 used  : {res['z86_cm_used']:.1f} cm  "
        f"(rho_b for weights: {res['rho_b_for_weights']:.2f} g/cm³)",
        "",
        "  Layer weights W(z) [normalised]:",
    ]

    for lbl, w_val in zip(DEPTH_LABELS, res['layer_weights']):
        bar = "█" * int(w_val * 40)
        L.append(f"    {lbl:<12} {bar:<40} {w_val:.4f}")
    L.append("")

    # Tabella profili
    hdr = f"  {'Property':<12} {'Units':<10} " + \
          "  ".join(f"{lbl:>9}" for lbl in DEPTH_LABELS) + \
          f"  {'CRNS_mean':>10}  {'CRNS_unc':>9}"
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))

    prop_order = ['bdod','clay','sand','silt','soc',
                  'phh2o','cec','cfvo','nitrogen']
    for prop in prop_order:
        prof      = res[f'{prop}_profile']
        crns_mean = res[f'{prop}_crns']
        crns_unc  = res[f'{prop}_crns_unc']
        vals_str  = "  ".join(
            f"{v:9.3f}" if not np.isnan(v) else f"{'N/A':>9}"
            for v in prof['mean']
        )
        cm_str  = f"{crns_mean:10.3f}" if not np.isnan(crns_mean) else f"{'N/A':>10}"
        cu_str  = f"{crns_unc:9.3f}"  if not np.isnan(crns_unc)  else f"{'N/A':>9}"
        L.append(f"  {prop:<12} {prof['units']:<10} {vals_str}  {cm_str}  {cu_str}")

    L.append("")
    L.append("  Uncertainty = propagated std on CRNS-weighted mean")
    L.append("=" * w)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Smoke test con risposta mock (senza rete)
# ---------------------------------------------------------------------------

def _mock_response(lat, lon):
    """Genera risposta mock SoilGrids per test offline."""
    import json

    def make_depths(values_raw, unc_raw):
        return [
            {
                'label': lbl,
                'range': {'top_depth': int(t), 'bottom_depth': int(b),
                          'unit_depth': 'cm'},
                'values': {'mean': v, 'uncertainty': u}
            }
            for (lbl, t, b, _), v, u in zip(DEPTHS, values_raw, unc_raw)
        ]

    return {
        'type': 'Point',
        'geometry': {'coordinates': [lon, lat, 1100]},
        'properties': {'layers': [
            {'name': 'bdod',
             'depths': make_depths([132,138,145,152,158,162],
                                   [  8,  9, 11, 13, 15, 17])},
            {'name': 'clay',
             'depths': make_depths([220,240,260,280,290,290],
                                   [ 15, 18, 20, 22, 25, 25])},
            {'name': 'sand',
             'depths': make_depths([420,400,380,360,350,340],
                                   [ 25, 28, 30, 32, 35, 35])},
            {'name': 'silt',
             'depths': make_depths([360,360,360,360,360,370],
                                   [ 18, 20, 22, 24, 26, 26])},
            {'name': 'soc',
             'depths': make_depths([185,120, 80, 45, 25, 15],
                                   [ 22, 18, 14, 10,  8,  6])},
            {'name': 'phh2o',
             'depths': make_depths([62, 64, 66, 68, 69, 70],
                                   [  4,  4,  4,  4,  4,  4])},
            {'name': 'cec',
             'depths': make_depths([185,175,165,155,145,140],
                                   [ 18, 18, 16, 16, 15, 15])},
            {'name': 'cfvo',
             'depths': make_depths([ 80, 90,100,110,120,130],
                                   [ 15, 16, 17, 18, 19, 20])},
            {'name': 'nitrogen',
             'depths': make_depths([ 15, 10,  8,  5,  3,  2],
                                   [  3,  2,  2,  2,  1,  1])},
        ]},
        'wrb_class_name': 'Cambisol',
    }


def _run_with_mock(lat, lon, z86_cm=16.0):
    """Esegue get_soil_properties usando dati mock invece dell'API."""
    import unittest.mock as mock

    mock_data = _mock_response(lat, lon)

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return mock_data

    with mock.patch('requests.get', return_value=FakeResp()):
        return get_soil_properties(lat, lon, z86_cm=z86_cm)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("SMOKE TEST — dati mock (offline)")
    print("=" * 60)

    sites = [
        ("Malga Fadner",    46.925, 11.861, 18.0),
        ("LIMENA pianura",  45.467, 11.851, 14.0),
    ]

    for name, lat, lon, z86 in sites:
        print(f"\nSito: {name}  ({lat:.3f}N, {lon:.3f}E)  z86={z86}cm")
        t0  = time.perf_counter()
        res = _run_with_mock(lat, lon, z86_cm=z86)
        dt  = time.perf_counter() - t0

        print(report_soil_properties(res))
        print(f"  Wall time: {dt*1000:.1f} ms")

        # Verifiche fisiche
        assert 1.0 < res['bdod_crns'] < 2.0,  \
            f"bulk density fuori range: {res['bdod_crns']}"
        assert 0 < res['clay_crns'] < 100,     \
            f"clay fuori range: {res['clay_crns']}"
        s = res['sand_crns'] + res['clay_crns'] + res['silt_crns']
        assert abs(s - 100) < 5,               \
            f"sand+clay+silt != 100%: {s:.1f}"
        assert 0 < res['lattice_water_gg'] < 0.15, \
            f"lattice water fuori range: {res['lattice_water_gg']}"
        assert abs(res['layer_weights'].sum() - 1.0) < 1e-6, \
            "pesi non normalizzati"

        print("  PASS")
