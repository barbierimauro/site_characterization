"""
rf_core.py
==========
Core per analisi RF del sito CRNS:
  - Download DEM GLO-90 (90m) fino a 25 km per viewshed
  - Download antenne da OpenCelliD
  - Download infrastrutture RFI da OSM Overpass
  - Fisica: FSPL + knife-edge diffraction
  - Viewshed: line-of-sight dal DEM

Caching:
  rf_dem_{hash}.npz       DEM 90m raggio 25 km
  rf_cells_{hash}.json.gz celle OpenCelliD
  rf_osm_rfi_{hash}.json.gz infrastrutture OSM per RFI

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import os, json, gzip, hashlib, time
import numpy as np

OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
GLO90_BASE    = "https://copernicus-dem-90m.s3.amazonaws.com"
OPENCELLID_URL= "https://opencellid.org/cell/getInArea"

# Parametri antenna BTS tipici per Italia
BTS_PARAMS = {
    "GSM" : {"f_mhz": 900,  "ptx_dbm": 43, "gtx_dbi": 15},
    "UMTS": {"f_mhz": 2100, "ptx_dbm": 43, "gtx_dbi": 15},
    "LTE" : {"f_mhz": 800,  "ptx_dbm": 46, "gtx_dbi": 18},
    "NR"  : {"f_mhz": 3500, "ptx_dbm": 46, "gtx_dbi": 21},
}
SENSITIVITY_DBM   = -100.0  # soglia minima ricezione
GOOD_SIGNAL_DBM   = -85.0   # segnale buono
RF_RADIUS_M       = 25000.0 # raggio analisi RF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(lat, lon, tag=""):
    s = f"{lat:.5f}_{lon:.5f}_{tag}"
    return hashlib.sha256(s.encode()).hexdigest()[:14]


def _haversine_m(lat1, lon1, lat2, lon2):
    """Distanza in metri tra due punti WGS84."""
    R  = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a  = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return float(2*R*np.arcsin(np.sqrt(a)))


# ---------------------------------------------------------------------------
# DEM GLO-90
# ---------------------------------------------------------------------------

def _glo90_url(tile_lat, tile_lon):
    ns = "N" if tile_lat >= 0 else "S"
    ew = "E" if tile_lon >= 0 else "W"
    fn = (f"Copernicus_DSM_COG_30_{ns}{abs(tile_lat):02d}_00_"
          f"{ew}{abs(tile_lon):03d}_00_DEM")
    return f"{GLO90_BASE}/{fn}/{fn}.tif"


def load_dem_glo90(lat, lon, radius_m, cache_dir, verbose=True):
    """
    Scarica DEM GLO-90 (90m) centrato su (lat,lon) entro radius_m.
    Ritorna (elev, lats_1d, lons_1d).
    Cache: rf_dem_{hash}.npz
    """
    import requests, tempfile
    try:
        import rasterio
        from rasterio.merge import merge as rasterio_merge
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
        from rasterio.crs import CRS
    except ImportError:
        raise ImportError("pip install rasterio")

    os.makedirs(cache_dir, exist_ok=True)
    npz = os.path.join(cache_dir, f"rf_dem_{_hash(lat,lon,'dem')}.npz")

    if os.path.exists(npz):
        if verbose: print("   DEM GLO-90: from cache", flush=True)
        d = np.load(npz)
        return d["elev"], d["lats"], d["lons"]

    c           = np.cos(np.radians(lat))
    margin_lon  = radius_m / (111320.0 * c) * 1.05
    margin_lat  = radius_m / 111320.0 * 1.05
    bbox_wgs84  = (lon-margin_lon, lat-margin_lat,
                   lon+margin_lon, lat+margin_lat)

    # Tile(s) necessarie
    lon_tiles = range(int(np.floor(lon-margin_lon)),
                      int(np.ceil(lon+margin_lon)))
    lat_tiles = range(int(np.floor(lat-margin_lat)),
                      int(np.ceil(lat+margin_lat)))
    tiles = [(lt, ln) for lt in lat_tiles for ln in lon_tiles]

    all_data, tmp_files, datasets = [], [], []
    for tl, tn in tiles:
        url  = _glo90_url(tl, tn)
        if verbose: print(f"   DEM GLO-90: tile {tl}N{tn}E ...",
                          end=" ", flush=True)
        try:
            t0   = time.perf_counter()
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200:
                if verbose: print(f"HTTP {resp.status_code} skip")
                continue
            tmp = tempfile.NamedTemporaryFile(
                suffix=".tif", delete=False)
            tmp.write(resp.content); tmp.close()
            tmp_files.append(tmp.name)
            datasets.append(rasterio.open(tmp.name))
            if verbose:
                print(f"OK {len(resp.content)//1024}KB "
                      f"{time.perf_counter()-t0:.1f}s", flush=True)
        except Exception as e:
            if verbose: print(f"FAILED: {e}")

    if not datasets:
        raise RuntimeError("Nessuna tile GLO-90 scaricata")

    mosaic, tf = rasterio_merge(datasets)
    for ds in datasets: ds.close()
    for f in tmp_files: os.unlink(f)

    elev = mosaic[0].astype(float)
    elev[elev < -9000] = np.nan

    nr, nc  = elev.shape
    lons_1d = np.array([tf * (j+0.5, 0) for j in range(nc)])[:,0]
    lats_1d = np.array([tf * (0, i+0.5) for i in range(nr)])[:,1]

    # Ritaglia al bbox esatto
    col_mask = (lons_1d >= bbox_wgs84[0]) & (lons_1d <= bbox_wgs84[2])
    row_mask = (lats_1d >= bbox_wgs84[1]) & (lats_1d <= bbox_wgs84[3])
    elev     = elev[np.ix_(row_mask, col_mask)]
    lats_1d  = lats_1d[row_mask]
    lons_1d  = lons_1d[col_mask]

    np.savez_compressed(npz, elev=elev.astype(np.float32),
                         lats=lats_1d.astype(np.float32),
                         lons=lons_1d.astype(np.float32))
    if verbose:
        print(f"   DEM GLO-90: {elev.shape} cached", flush=True)

    return elev, lats_1d, lons_1d


# ---------------------------------------------------------------------------
# Viewshed: line-of-sight sul DEM
# ---------------------------------------------------------------------------

def line_of_sight(elev, lats_1d, lons_1d,
                   site_lat, site_lon, site_elev_m,
                   tgt_lat, tgt_lon, tgt_elev_m,
                   antenna_height_m=30.0):
    """
    Verifica line-of-sight tra sito e antenna tramite ray-marching sul DEM.
    Ritorna (is_los, max_obstruction_m, obstruction_frac).
      max_obstruction_m: quanto l'ostacolo supera la linea di vista [m]
      obstruction_frac:  posizione relativa ostacolo sul percorso [0-1]
    """
    c       = np.cos(np.radians(site_lat))
    dx_deg  = tgt_lon - site_lon
    dy_deg  = tgt_lat - site_lat
    d_total = _haversine_m(site_lat, site_lon, tgt_lat, tgt_lon)

    if d_total < 100:
        return True, 0.0, 0.0

    # Campiona 200 punti lungo il percorso
    n_steps = 200
    ts      = np.linspace(0, 1, n_steps)
    lats_ray = site_lat + ts * dy_deg
    lons_ray = site_lon + ts * dx_deg

    # Quota linea di vista (interpolazione lineare)
    z_src = site_elev_m + antenna_height_m
    z_tgt = tgt_elev_m  + antenna_height_m
    z_los = z_src + ts * (z_tgt - z_src)

    # Interpola DEM lungo il raggio
    from scipy.interpolate import RegularGridInterpolator
    lats_sorted = np.sort(lats_1d)
    lons_sorted = np.sort(lons_1d)
    elev_sorted = elev.copy()
    if lats_1d[0] > lats_1d[-1]:
        elev_sorted = elev_sorted[::-1, :]
    if lons_1d[0] > lons_1d[-1]:
        elev_sorted = elev_sorted[:, ::-1]
    elev_sorted = np.where(np.isnan(elev_sorted), 0, elev_sorted)

    interp = RegularGridInterpolator(
        (lats_sorted, lons_sorted), elev_sorted,
        method="linear", bounds_error=False,
        fill_value=float(np.nanmean(elev_sorted)))

    pts      = np.column_stack([lats_ray, lons_ray])
    z_terrain = interp(pts)

    # Ostacolo = terreno sopra la linea di vista
    obstruction = z_terrain - z_los
    max_obs     = float(obstruction.max())
    idx_max     = int(obstruction.argmax())

    return max_obs <= 0, max(max_obs, 0.0), float(ts[idx_max])


# ---------------------------------------------------------------------------
# FSPL + diffrazione knife-edge
# ---------------------------------------------------------------------------

def fspl_db(d_km, f_mhz):
    """Free Space Path Loss [dB]. d in km, f in MHz."""
    if d_km < 0.01: d_km = 0.01
    return 20*np.log10(d_km) + 20*np.log10(f_mhz) + 32.44


def knife_edge_db(h_m, d1_km, d2_km, f_mhz):
    """
    Attenuazione da diffrazione knife-edge ITU-R P.526.
    h_m: altezza ostacolo sopra la linea di vista [m] (>0 = ostruito)
    """
    if h_m <= 0:
        return 0.0
    lam = 300.0 / f_mhz        # lunghezza d'onda [m]
    d1  = max(d1_km, 0.01)*1e3
    d2  = max(d2_km, 0.01)*1e3
    r   = np.sqrt(2*d1*d2 / (lam*(d1+d2)))
    v   = h_m * r
    if   v < -1:   J = 0.0
    elif v <  0:   J = 20*np.log10(0.5 - 0.62*v)
    elif v <  1:   J = 20*np.log10(0.5*np.exp(-0.95*v))
    elif v <  2.4: J = 20*np.log10(0.4 - np.sqrt(0.1184-(0.38-0.1*v)**2))
    else:          J = 20*np.log10(0.225/v)
    return float(-J)  # positivo = perdita


def rx_level_dbm(d_km, f_mhz, ptx_dbm, gtx_dbi,
                  h_obs_m=0.0, d1_km=None, d2_km=None):
    """Livello segnale ricevuto [dBm] con FSPL + knife-edge."""
    grx   = 0.0   # antenna omnidirezionale ricevente
    L     = fspl_db(d_km, f_mhz)
    if h_obs_m > 0 and d1_km and d2_km:
        L += knife_edge_db(h_obs_m, d1_km, d2_km, f_mhz)
    return float(ptx_dbm + gtx_dbi + grx - L)


# ---------------------------------------------------------------------------
# OpenCelliD download
# ---------------------------------------------------------------------------

def load_cells(lat, lon, radius_m, token, cache_dir, verbose=True):
    """
    Scarica antenne da OpenCelliD entro radius_m.
    Cache: rf_cells_{hash}.json.gz
    Ritorna lista di dict con lat, lon, radio, range, ecc.
    """
    import requests

    os.makedirs(cache_dir, exist_ok=True)
    gz = os.path.join(cache_dir,
                      f"rf_cells_{_hash(lat,lon,'cells')}.json.gz")

    if os.path.exists(gz):
        if verbose: print("   Cells: from cache", flush=True)
        with gzip.open(gz, "rt") as f:
            return json.load(f)

    c      = np.cos(np.radians(lat))
    margin_lon = radius_m / (111320.0 * c)
    margin_lat = radius_m / 111320.0

    params = {
        "key"   : token,
        "BBOX"  : (f"{lat-margin_lat},{lon-margin_lon},"
                   f"{lat+margin_lat},{lon+margin_lon}"),
        "format": "json",
        "limit" : 10000,
    }
    if verbose:
        print("   Cells: querying OpenCelliD ...", flush=True)

    resp = requests.get(OPENCELLID_URL, params=params, timeout=30)
    if resp.status_code == 403:
        raise ValueError("Token OpenCelliD non valido (HTTP 403)")
    resp.raise_for_status()

    cells = resp.json().get("cells", [])
    if verbose:
        radios = {}
        for c in cells:
            r = c.get("radio","?")
            radios[r] = radios.get(r,0)+1
        print(f"   Cells: {len(cells)} found  {radios}", flush=True)

    with gzip.open(gz, "wt") as f:
        json.dump(cells, f)

    return cells


# ---------------------------------------------------------------------------
# OSM: infrastrutture RFI
# ---------------------------------------------------------------------------

def load_osm_rfi(lat, lon, radius_m, cache_dir, verbose=True):
    """
    Scarica da OSM le infrastrutture rilevanti per RFI:
    linee elettriche, ferrovie, torri radio, edifici industriali.
    Cache: rf_osm_rfi_{hash}.json.gz
    """
    import requests

    os.makedirs(cache_dir, exist_ok=True)
    gz = os.path.join(cache_dir,
                      f"rf_osm_rfi_{_hash(lat,lon,'rfi')}.json.gz")

    if os.path.exists(gz):
        if verbose: print("   OSM RFI: from cache", flush=True)
        with gzip.open(gz, "rt") as f:
            return json.load(f)

    c          = np.cos(np.radians(lat))
    margin_lat = radius_m / 111320.0 * 1.1
    margin_lon = radius_m / (111320.0*c) * 1.1
    bb = (f"{lat-margin_lat},{lon-margin_lon},"
          f"{lat+margin_lat},{lon+margin_lon}")

    query = f"""
[out:json][timeout:60];
(
  way["power"="line"]({bb});
  way["power"="minor_line"]({bb});
  node["power"="tower"]({bb});
  node["power"="pole"]({bb});
  way["power"="substation"]({bb});
  node["power"="substation"]({bb});
  way["railway"="rail"]({bb});
  way["railway"="tram"]({bb});
  node["man_made"="mast"]({bb});
  node["man_made"="tower"]({bb});
  node["communication"="antenna"]({bb});
  node["communication"="radio"]({bb});
);
out geom;
"""
    if verbose:
        print("   OSM RFI: querying Overpass ...", flush=True)

    resp = requests.post(OVERPASS_URL,
                         data={"data": query}, timeout=60)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    if verbose:
        tags = {}
        for e in elements:
            k = (e.get("tags",{}).get("power") or
                 e.get("tags",{}).get("railway") or
                 e.get("tags",{}).get("man_made") or
                 e.get("tags",{}).get("communication","?"))
            tags[k] = tags.get(k,0)+1
        print(f"   OSM RFI: {len(elements)} elements  {tags}",
              flush=True)

    with gzip.open(gz, "wt") as f:
        json.dump(elements, f)

    return elements
    
    
"""
rf_analysis.py
==============
Calcola i due risultati principali:
  1. Connettività cellulare (2G/3G/4G/5G)
  2. Indice disturbi RF da infrastrutture OSM

Usa le funzioni di rf_core.py per DEM, celle e OSM.

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import os, json, gzip, hashlib, time
import numpy as np

# Soglie segnale [dBm]
SENSITIVITY_DBM = -100.0
GOOD_SIGNAL_DBM =  -85.0
FAIR_SIGNAL_DBM =  -95.0

# Parametri BTS tipici Italia
BTS_PARAMS = {
    "GSM" : {"f_mhz":  900, "ptx_dbm": 43, "gtx_dbi": 15},
    "UMTS": {"f_mhz": 2100, "ptx_dbm": 43, "gtx_dbi": 15},
    "LTE" : {"f_mhz":  800, "ptx_dbm": 46, "gtx_dbi": 18},
    "NR"  : {"f_mhz": 3500, "ptx_dbm": 46, "gtx_dbi": 21},
}

# Fattori disturbo RFI per tipo OSM [0-10 scala relativa]
RFI_WEIGHTS = {
    "power_line_HV"    : 8.0,   # >100 kV
    "power_line_MV"    : 4.0,   # <100 kV
    "power_line_minor" : 2.0,
    "railway"          : 6.0,
    "substation"       : 7.0,
    "mast_tower"       : 3.0,
    "antenna"          : 5.0,
}


# ---------------------------------------------------------------------------
# Helper geometria
# ---------------------------------------------------------------------------

def _haversine_m(lat1, lon1, lat2, lon2):
    R  = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dp = np.radians(lat2-lat1)
    dl = np.radians(lon2-lon1)
    a  = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return float(2*R*np.arcsin(np.sqrt(a)))


def _element_position(el):
    """Estrae (lat, lon) dal centro di un elemento OSM."""
    if el.get("type") == "node":
        return el.get("lat"), el.get("lon")
    geom = el.get("geometry", [])
    if not geom:
        return None, None
    lats = [p["lat"] for p in geom if "lat" in p]
    lons = [p["lon"] for p in geom if "lon" in p]
    if not lats:
        return None, None
    return float(np.mean(lats)), float(np.mean(lons))


# ---------------------------------------------------------------------------
# 1. Analisi connettività cellulare
# ---------------------------------------------------------------------------

def analyze_connectivity(cells, site_lat, site_lon, site_elev_m,
                          elev, lats_1d, lons_1d,
                          verbose=True):
    """
    Per ogni cella OpenCelliD:
      - Calcola distanza dal sito
      - Viewshed sul DEM GLO-90
      - Rx level con FSPL + knife-edge
      - Classifica: good / fair / weak / none

    Parameters
    ----------
    cells       : lista da load_cells()
    site_*      : coordinate e quota sensore
    elev/lats/lons : DEM GLO-90 da load_dem_glo90()

    Returns
    -------
    dict con risultati per tecnologia + lista antenne analizzate
    """
    from scipy.interpolate import RegularGridInterpolator

    if not cells:
        return _empty_connectivity_result()

    # Prepara interpolatore DEM per viewshed
    lats_s = np.sort(lats_1d)
    lons_s = np.sort(lons_1d)
    elev_s = elev.copy()
    if lats_1d[0] > lats_1d[-1]: elev_s = elev_s[::-1, :]
    if lons_1d[0] > lons_1d[-1]: elev_s = elev_s[:, ::-1]
    elev_s = np.where(np.isnan(elev_s), float(np.nanmean(elev_s)), elev_s)

    dem_interp = RegularGridInterpolator(
        (lats_s, lons_s), elev_s,
        method="linear", bounds_error=False,
        fill_value=float(np.nanmean(elev_s)))

    results  = []
    by_radio = {}

    for cell in cells:
        c_lat = float(cell.get("lat", 0))
        c_lon = float(cell.get("lon", 0))
        radio = cell.get("radio", "LTE")

        if c_lat == 0 and c_lon == 0:
            continue

        d_m  = _haversine_m(site_lat, site_lon, c_lat, c_lon)
        if d_m < 10 or d_m > 30000:
            continue

        d_km = d_m / 1000.0
        p    = BTS_PARAMS.get(radio, BTS_PARAMS["LTE"])
        f    = p["f_mhz"]

        # Quota antenna BTS (stimata: quota DEM + 30m)
        try:
            c_elev = float(dem_interp([[c_lat, c_lon]])[0]) + 30.0
        except Exception:
            c_elev = site_elev_m + 30.0

        # Viewshed: ray-marching 200 punti
        n_s    = 200
        ts     = np.linspace(0, 1, n_s)
        lats_r = site_lat + ts*(c_lat - site_lat)
        lons_r = site_lon + ts*(c_lon - site_lon)
        z_los  = (site_elev_m+30) + ts*((c_elev)-(site_elev_m+30))

        pts_r    = np.column_stack([lats_r, lons_r])
        z_ter    = dem_interp(pts_r)
        obs_arr  = z_ter - z_los
        max_obs  = float(obs_arr.max())
        idx_obs  = int(obs_arr.argmax())
        t_obs    = float(ts[idx_obs])
        is_los   = max_obs <= 0

        # Path loss
        if is_los:
            L_ke = 0.0
        else:
            d1 = d_km * t_obs
            d2 = d_km * (1 - t_obs)
            L_ke = _knife_edge_db(max(max_obs, 0), d1, d2, f)

        L_fs  = _fspl_db(d_km, f)
        rx    = p["ptx_dbm"] + p["gtx_dbi"] - L_fs - L_ke

        quality = ("none" if rx < SENSITIVITY_DBM else
                   "weak" if rx < FAIR_SIGNAL_DBM else
                   "fair" if rx < GOOD_SIGNAL_DBM else "good")

        rec = {
            "radio"    : radio,
            "d_km"     : round(d_km, 2),
            "rx_dbm"   : round(rx, 1),
            "is_los"   : is_los,
            "L_fs_db"  : round(L_fs, 1),
            "L_ke_db"  : round(L_ke, 1),
            "max_obs_m": round(max(max_obs, 0), 1),
            "quality"  : quality,
            "lat"      : c_lat,
            "lon"      : c_lon,
            "mcc"      : cell.get("mcc","?"),
            "net"      : cell.get("net","?"),
        }
        results.append(rec)

        # Tieni il miglior segnale per tecnologia
        if radio not in by_radio or rx > by_radio[radio]["rx_dbm"]:
            by_radio[radio] = rec

    # Sommario per tecnologia
    tech_summary = {}
    for radio, best in by_radio.items():
        rx = best["rx_dbm"]
        tech_summary[radio] = {
            "best_rx_dbm" : rx,
            "quality"     : best["quality"],
            "coverage"    : rx >= SENSITIVITY_DBM,
            "best_antenna": best,
            "n_cells"     : sum(1 for r in results
                                if r["radio"]==radio),
        }

    # Tecnologie disponibili ordinate per segnale
    coverage = {t: v["coverage"]
                for t, v in tech_summary.items()}

    return {
        "antennas"      : results,
        "by_radio"      : tech_summary,
        "coverage"      : coverage,
        "has_any"       : any(coverage.values()),
        "n_cells_total" : len(results),
        "best_overall"  : max(results, key=lambda x: x["rx_dbm"],
                              default=None) if results else None,
    }


def _empty_connectivity_result():
    return {
        "antennas": [], "by_radio": {},
        "coverage": {}, "has_any": False,
        "n_cells_total": 0, "best_overall": None,
        "warning": "Nessuna cella OpenCelliD trovata nel raggio "
                   "di 25 km. Zona probabilmente senza copertura "
                   "o non rilevata dal crowd-sourcing."
    }


def _fspl_db(d_km, f_mhz):
    if d_km < 0.01: d_km = 0.01
    return 20*np.log10(d_km) + 20*np.log10(f_mhz) + 32.44


def _knife_edge_db(h_m, d1_km, d2_km, f_mhz):
    if h_m <= 0: return 0.0
    lam = 300.0 / f_mhz
    d1  = max(d1_km, 0.01)*1e3
    d2  = max(d2_km, 0.01)*1e3
    r   = np.sqrt(2*d1*d2 / (lam*(d1+d2)))
    v   = h_m * r
    if   v < -1:   J = 0.0
    elif v <  0:   J = 20*np.log10(0.5 - 0.62*v)
    elif v <  1:   J = 20*np.log10(0.5*np.exp(-0.95*v))
    elif v <  2.4: J = 20*np.log10(0.4 - np.sqrt(0.1184-(0.38-0.1*v)**2))
    else:          J = 20*np.log10(0.225/v)
    return float(-J)


# ---------------------------------------------------------------------------
# 2. Analisi disturbi RF da OSM
# ---------------------------------------------------------------------------

def analyze_rfi(osm_elements, site_lat, site_lon, verbose=True):
    """
    Calcola indice RFI dal sito basato su distanza e tipo
    delle infrastrutture OSM.

    Modello: contributo_i = RFI_weight_i / (d_i_km^2)
    normalizzato a 0-10.

    Returns
    -------
    dict con rfi_index, contributi per categoria, lista sorgenti
    """
    sources = []

    for el in osm_elements:
        tags = el.get("tags", {})
        e_lat, e_lon = _element_position(el)
        if e_lat is None:
            continue

        d_m = _haversine_m(site_lat, site_lon, e_lat, e_lon)
        if d_m < 1:
            d_m = 1.0
        d_km = d_m / 1000.0

        # Classifica
        power  = tags.get("power","")
        railway= tags.get("railway","")
        man    = tags.get("man_made","")
        comm   = tags.get("communication","")
        volt   = tags.get("voltage","0").replace(" kV","")

        try:
            kv = float(str(volt).split(";")[0].replace("kV","")
                       .replace("V","").strip())
            if kv > 1000: kv /= 1000   # V -> kV
        except Exception:
            kv = 0

        if power in ("line","cable"):
            rfi_type = ("power_line_HV" if kv >= 100
                        else "power_line_MV" if kv >= 1
                        else "power_line_minor")
        elif power in ("minor_line",):
            rfi_type = "power_line_minor"
        elif power in ("substation","transformer"):
            rfi_type = "substation"
        elif railway in ("rail","tram","light_rail"):
            rfi_type = "railway"
        elif man in ("mast","tower"):
            rfi_type = "mast_tower"
        elif comm in ("antenna","radio","television"):
            rfi_type = "antenna"
        else:
            continue

        weight = RFI_WEIGHTS.get(rfi_type, 1.0)
        # Contributo: peso / d^1.5 (intermedio tra 1/d e 1/d^2)
        contrib = weight / (d_km ** 1.5)

        sources.append({
            "type"   : rfi_type,
            "d_km"   : round(d_km, 2),
            "weight" : weight,
            "contrib": round(contrib, 4),
            "lat"    : e_lat,
            "lon"    : e_lon,
            "tags"   : {k: tags[k] for k in
                        ["power","railway","man_made",
                         "communication","voltage","name"]
                        if k in tags},
        })

    if not sources:
        return {
            "rfi_index"  : 0.0,
            "rfi_level"  : "very_low",
            "sources"    : [],
            "by_type"    : {},
            "n_sources"  : 0,
            "warning"    : "Nessuna sorgente RFI trovata — "
                           "sito radiofregquenzialmente pulito.",
        }

    # Indice normalizzato 0-10
    total = sum(s["contrib"] for s in sources)
    # Calibrazione empirica: totale > 50 = sito molto disturbato
    rfi_index = float(np.clip(total / 5.0, 0, 10))

    level = ("very_low"  if rfi_index < 1 else
             "low"       if rfi_index < 3 else
             "moderate"  if rfi_index < 5 else
             "high"      if rfi_index < 7 else "very_high")

    # Aggregato per tipo
    by_type = {}
    for s in sources:
        t = s["type"]
        by_type.setdefault(t, {"n": 0, "contrib": 0.0,
                                 "min_d_km": 999})
        by_type[t]["n"]       += 1
        by_type[t]["contrib"] += s["contrib"]
        by_type[t]["min_d_km"] = min(by_type[t]["min_d_km"],
                                      s["d_km"])

    sources.sort(key=lambda x: -x["contrib"])

    return {
        "rfi_index"  : round(rfi_index, 2),
        "rfi_level"  : level,
        "sources"    : sources,
        "by_type"    : by_type,
        "n_sources"  : len(sources),
        "total_contrib": round(total, 3),
    }


# ---------------------------------------------------------------------------
# Entry point: analisi completa
# ---------------------------------------------------------------------------

def run_rf_analysis(lat, lon, site_elev_m,
                     token, cache_dir,
                     radius_m=25000, verbose=True):
    """
    Esegue l'analisi RF completa per un sito CRNS.

    Parameters
    ----------
    lat, lon      : coordinate WGS84 sito
    site_elev_m   : quota sito [m a.s.l.]
    token         : token OpenCelliD
    cache_dir     : directory cache
    radius_m      : raggio analisi [m] (default 25000)
    verbose       : stampa progressi

    Returns
    -------
    dict con 'connectivity', 'rfi', 'dem_shape', metadata
    """
    if verbose:
        print("[RF] Loading DEM GLO-90 ...", flush=True)
    elev, lats_1d, lons_1d = load_dem_glo90(
        lat, lon, radius_m, cache_dir, verbose)

    if verbose:
        print("[RF] Loading OpenCelliD ...", flush=True)
    cells = load_cells(lat, lon, radius_m, token,
                        cache_dir, verbose)

    if verbose:
        print("[RF] Loading OSM RFI ...", flush=True)
    osm_rfi = load_osm_rfi(lat, lon, radius_m,
                             cache_dir, verbose)

    if verbose:
        print("[RF] Computing connectivity ...", flush=True)
    conn = analyze_connectivity(
        cells, lat, lon, site_elev_m,
        elev, lats_1d, lons_1d, verbose)

    if verbose:
        print("[RF] Computing RFI index ...", flush=True)
    rfi = analyze_rfi(osm_rfi, lat, lon, verbose)

    return {
        "connectivity" : conn,
        "rfi"          : rfi,
        "dem_shape"    : elev.shape,
        "lat"          : lat,
        "lon"          : lon,
        "site_elev_m"  : site_elev_m,
        "radius_m"     : radius_m,
    }


# ---------------------------------------------------------------------------
# Report testuale
# ---------------------------------------------------------------------------

def report_rf(rf):
    """
    Genera il blocco testuale per la sezione RF del report.
    rf: dict ritornato da run_rf_analysis()
    """
    if rf is None:
        return "  [non disponibile — RF analysis non eseguita]"

    L = []
    def s(x=""): L.append(x)

    conn = rf.get("connectivity", {})
    rfi  = rf.get("rfi", {})

    # --- Connettività cellulare ---
    s(f"  Raggio analisi : {rf.get('radius_m', 25000)/1000:.0f} km")
    s(f"  DEM GLO-90     : {rf.get('dem_shape', ('?','?'))}")
    s()

    s("  A) CONNETTIVITA' CELLULARE (OpenCelliD)")
    n_cells = conn.get("n_cells_total", 0)
    s(f"     Celle analizzate : {n_cells}")
    if n_cells == 0 or not conn.get("has_any", False):
        w = conn.get("warning", "Nessuna cella trovata.")
        s(f"     [WARN] {w}")
    else:
        s(f"     {'Tecnologia':<8}  {'Copertura':<10}  "
          f"{'Best Rx [dBm]':<14}  {'Qualità':<8}  Celle")
        s(f"     {'-'*8}  {'-'*10}  {'-'*14}  {'-'*8}  -----")
        for tech, info in sorted(conn.get("by_radio", {}).items()):
            cov  = "SI" if info.get("coverage") else "NO"
            rx   = info.get("best_rx_dbm", 0.0)
            qual = info.get("quality", "?")
            nc   = info.get("n_cells", 0)
            s(f"     {tech:<8}  {cov:<10}  {rx:>+8.1f} dBm    {qual:<8}  {nc}")
        best = conn.get("best_overall")
        if best:
            s()
            s(f"     Miglior segnale: {best['radio']}  "
              f"Rx={best['rx_dbm']:+.1f} dBm  "
              f"d={best['d_km']:.1f} km  "
              f"LOS={'si' if best['is_los'] else 'no'}")

    s()
    s("  B) DISTURBI RF (OSM RFI)")
    s(f"     Sorgenti trovate : {rfi.get('n_sources', 0)}")
    s(f"     RFI index        : {rfi.get('rfi_index', 0.0):.1f}/10  "
      f"livello={rfi.get('rfi_level', '?')}")
    if rfi.get("warning"):
        s(f"     [NOTA] {rfi['warning']}")

    by_type = rfi.get("by_type", {})
    if by_type:
        s()
        s(f"     {'Tipo':<20}  {'N':>4}  {'Contrib':>8}  {'Min dist':>9}")
        s(f"     {'-'*20}  {'-'*4}  {'-'*8}  {'-'*9}")
        for t, v in sorted(by_type.items(),
                           key=lambda x: -x[1]["contrib"]):
            s(f"     {t:<20}  {v['n']:>4}  "
              f"{v['contrib']:>8.2f}  "
              f"{v['min_d_km']:>6.1f} km")

    top_src = rfi.get("sources", [])[:5]
    if top_src:
        s()
        s("     Top 5 sorgenti per contributo:")
        for src in top_src:
            tags_str = ", ".join(f"{k}={v}"
                                 for k, v in src.get("tags", {}).items())
            s(f"       {src['type']:<20}  "
              f"d={src['d_km']:.2f} km  "
              f"contrib={src['contrib']:.3f}  "
              f"{tags_str}")

    return "\n".join(L)
