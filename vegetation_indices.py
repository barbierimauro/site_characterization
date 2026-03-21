"""
vegetation_indices.py
=====================
Pipeline per il calcolo degli indici di vegetazione per un sito CRNS.

Sorgenti
--------
Landsat Collection 2 Level-2 (Planetary Computer, pubblico):
  - NDVI  = (NIR - Red) / (NIR + Red)
  - EVI   = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
  - NDWI  = (Green - NIR) / (Green + NIR)   [acqua/neve]
  - FCOVER = (NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil)  [Gutman 1998]
  Risoluzione: 30m | Dal 2013 in poi | Revisit ~8gg (L8+L9)
  Footprint intero (r86), media pesata W(r)

MODIS MCD15A3H (Planetary Computer, pubblico):
  - LAI   = Leaf Area Index  [m2/m2]
  Risoluzione: 500m | Dal 2002 in poi | Ogni 4 giorni
  Array 5x5 pixel centrato sulle coordinate

Cache
-----
Per ogni scena: hash(item_id + bbox) -> .npz con valori scalati e mascherati.
La cache è condivisa tra run successivi — un sito non viene mai riscaricato.

Output
------
- Medie mensili climatologiche (12 valori per variabile)
- Std interannuale mensile
- Numero di osservazioni valide per mese
- Serie temporale completa (tutte le scene valide)
- Valore più recente disponibile

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import os
import json
import hashlib
import warnings
import numpy as np
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Costanti fisiche
# ---------------------------------------------------------------------------

# Landsat C2 L2 scale/offset per riflettanza superficiale
L2_SCALE  = 2.75e-5
L2_OFFSET = -0.2
L2_NODATA = 0

# QA_PIXEL bitmask Landsat Collection 2
QA_FILL       = 1 << 0
QA_CLOUD      = 1 << 3
QA_CLOUD_SHAD = 1 << 4
QA_SNOW       = 1 << 5
QA_CLEAR      = 1 << 6
QA_WATER      = 1 << 7

# FCOVER parametri Gutman & Ignatov 1998
FCOVER_NDVI_SOIL = 0.05
FCOVER_NDVI_VEG  = 0.95

# MODIS LAI scale
MODIS_LAI_SCALE  = 0.1
MODIS_LAI_NODATA = 255

# Landsat: anno di inizio (Landsat 8 lancio aprile 2013)
LANDSAT_START_YEAR = 2013

# MODIS MCD15A3H: inizio dataset
MODIS_START_DATE = "2002-07-04"

# Planetary Computer STAC URL
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(item_id, bbox_str):
    tag = f"{item_id}_{bbox_str}"
    return hashlib.sha256(tag.encode()).hexdigest()[:20]


def _cache_save(cache_dir, key, arrays_dict, meta_dict):
    os.makedirs(cache_dir, exist_ok=True)
    npz  = os.path.join(cache_dir, f"veg_{key}.npz")
    meta = os.path.join(cache_dir, f"veg_{key}.json")
    np.savez_compressed(npz, **arrays_dict)
    with open(meta, "w") as f:
        json.dump(meta_dict, f)


def _cache_load(cache_dir, key):
    npz  = os.path.join(cache_dir, f"veg_{key}.npz")
    meta = os.path.join(cache_dir, f"veg_{key}.json")
    if not (os.path.exists(npz) and os.path.exists(meta)):
        return None, None
    with open(meta) as f:
        m = json.load(f)
    d = np.load(npz)
    return dict(d), m


# ---------------------------------------------------------------------------
# Radial weight
# ---------------------------------------------------------------------------

def _weight_radial(dist, r86):
    lam = r86 / 3.0
    return np.where(dist < 1e-3, 0.0, np.exp(-dist / lam))


# ---------------------------------------------------------------------------
# Landsat helpers
# ---------------------------------------------------------------------------

def _l2_reflectance(data, nodata=L2_NODATA,
                    scale=L2_SCALE, offset=L2_OFFSET):
    """Converte uint16 DN in riflettanza superficiale [0-1]."""
    mask  = (data != nodata) & (data > 0)
    refl  = np.where(mask,
                     data.astype(float) * scale + offset,
                     np.nan)
    return np.clip(refl, 0.0, 1.0)


def _qa_clear_mask(qa_data):
    """
    Maschera pixel validi da QA_PIXEL Landsat C2.
    Valido = bit CLEAR attivo E nessun cloud/shadow/snow/fill.
    """
    qa = qa_data.astype(np.uint16)
    clear  = (qa & QA_CLEAR) > 0
    no_cld = (qa & QA_CLOUD) == 0
    no_shd = (qa & QA_CLOUD_SHAD) == 0
    no_snw = (qa & QA_SNOW) == 0
    no_fill= (qa & QA_FILL) == 0
    return clear & no_cld & no_shd & no_snw & no_fill


def _compute_indices(red, nir, green, blue, clear_mask):
    """
    Calcola NDVI, EVI, NDWI, FCOVER su array 2D già in riflettanza.
    Pixel non validi -> NaN.
    """
    # NDVI
    denom_ndvi = nir + red
    ndvi = np.where(
        clear_mask & (denom_ndvi > 0.01),
        (nir - red) / denom_ndvi,
        np.nan)

    # EVI (Huete 2002)
    denom_evi = nir + 6.0*red - 7.5*blue + 1.0
    evi = np.where(
        clear_mask & (denom_evi > 0.01),
        2.5 * (nir - red) / denom_evi,
        np.nan)
    evi = np.clip(evi, -1.0, 1.0)

    # NDWI (Gao 1996 — acqua in vegetazione)
    denom_ndwi = green + nir
    ndwi = np.where(
        clear_mask & (denom_ndwi > 0.01),
        (green - nir) / denom_ndwi,
        np.nan)

    # FCOVER (Gutman & Ignatov 1998)
    fcover = np.where(
        ~np.isnan(ndvi),
        np.clip(
            (ndvi - FCOVER_NDVI_SOIL) /
            (FCOVER_NDVI_VEG - FCOVER_NDVI_SOIL),
            0.0, 1.0),
        np.nan)

    return ndvi, evi, ndwi, fcover


def _read_landsat_scene(signed_item, lat, lon, r86, dx_grid, dy_grid):
    """
    Legge le bande Red, NIR, Green, Blue, QA di una scena Landsat
    e ritorna gli indici NDVI, EVI, NDWI, FCOVER sulla griglia DEM.

    Returns
    -------
    dict con arrays 2D (stessa shape di dx_grid) per ogni indice,
    oppure None se la scena non ha pixel validi nel footprint.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from rasterio.crs import CRS
    from scipy.spatial import cKDTree

    wgs84   = CRS.from_epsg(4326)
    bands   = {"red": None, "nir08": None,
               "green": None, "blue": None, "qa_pixel": None}

    # Bounding box footprint in WGS84
    c           = np.cos(np.radians(lat))
    margin_deg  = (r86 * 1.2) / (111320.0 * c)
    margin_deg_lat = (r86 * 1.2) / 111320.0
    bbox_wgs84  = (lon - margin_deg, lat - margin_deg_lat,
                   lon + margin_deg, lat + margin_deg_lat)

    raw_data = {}
    crs_raster = None
    transform_raster = None
    shape_raster = None

    for band_name in bands:
        if band_name not in signed_item.assets:
            continue
        href = signed_item.assets[band_name].href
        try:
            with rasterio.open(href) as src:
                if crs_raster is None:
                    crs_raster = src.crs
                # Trasforma bbox in CRS del raster
                l, b, r, t = transform_bounds(
                    wgs84, src.crs, *bbox_wgs84)
                win  = from_bounds(l, b, r, t,
                                   transform=src.transform)
                data = src.read(1, window=win)
                win_tf = src.window_transform(win)
                if transform_raster is None:
                    transform_raster = win_tf
                    shape_raster     = data.shape
                raw_data[band_name] = data
        except Exception:
            return None

    if not all(b in raw_data for b in
               ["red", "nir08", "green", "blue", "qa_pixel"]):
        return None

    # Converte in riflettanza
    red   = _l2_reflectance(raw_data["red"])
    nir   = _l2_reflectance(raw_data["nir08"])
    green = _l2_reflectance(raw_data["green"])
    blue  = _l2_reflectance(raw_data["blue"])
    qa    = raw_data["qa_pixel"]
    clear = _qa_clear_mask(qa)

    # Calcola indici sulla griglia Landsat
    ndvi, evi, ndwi, fcover = _compute_indices(
        red, nir, green, blue, clear)

    # Coordinate pixel Landsat nel crop
    nr, nc = shape_raster
    col_idx = np.arange(nc)
    row_idx = np.arange(nr)

    # Pixel centers in CRS raster
    px_x = transform_raster.c + (col_idx + 0.5) * transform_raster.a
    px_y = transform_raster.f + (row_idx + 0.5) * transform_raster.e
    PX, PY = np.meshgrid(px_x, px_y)

    # Trasforma in coordinate metriche centrate sul sensore (WGS84 -> m)
    # Usiamo trasformazione approssimata: da CRS UTM a WGS84 a offset metrico
    from rasterio.warp import transform as rasterio_transform
    lons_flat, lats_flat = rasterio_transform(
        crs_raster, wgs84,
        PX.ravel(), PY.ravel())
    lons_flat = np.array(lons_flat)
    lats_flat = np.array(lats_flat)

    dx_ls = (lons_flat - lon) * 111320.0 * c
    dy_ls = (lats_flat - lat) * 111320.0
    dist_ls = np.sqrt(dx_ls**2 + dy_ls**2)

    # Maschera: solo pixel dentro r86
    fp_mask_ls = dist_ls <= r86

    if fp_mask_ls.sum() == 0:
        return None

    # Ricampiona sulla griglia DEM tramite nearest neighbor
    pts_ls  = np.column_stack([dx_ls[fp_mask_ls],
                                dy_ls[fp_mask_ls]])
    tree    = cKDTree(pts_ls)
    pts_dem = np.column_stack([dx_grid.ravel(), dy_grid.ravel()])
    _, idx  = tree.query(pts_dem)

    def _resample(arr):
        flat = arr.ravel()[fp_mask_ls]
        return flat[idx].reshape(dx_grid.shape)

    return {
        "ndvi"  : _resample(ndvi),
        "evi"   : _resample(evi),
        "ndwi"  : _resample(ndwi),
        "fcover": _resample(fcover),
        "clear" : _resample(clear.astype(float)),
    }


# ---------------------------------------------------------------------------
# MODIS LAI helpers
# ---------------------------------------------------------------------------

def _read_modis_lai_scene(signed_item, lat, lon, n_pixels=5):
    """
    Legge un array n_pixels x n_pixels centrato su (lat,lon) dal
    prodotto LAI MODIS MCD15A3H.

    Returns
    -------
    lai_arr : array (n_pixels, n_pixels) valori LAI [m2/m2], NaN=nodata
    oppure None se lettura fallisce
    """
    import rasterio
    from rasterio.warp import transform_bounds, transform as rio_transform
    from rasterio.windows import from_bounds
    from rasterio.crs import CRS

    asset_name = "Lai_500m"
    if asset_name not in signed_item.assets:
        return None

    href  = signed_item.assets[asset_name].href
    wgs84 = CRS.from_epsg(4326)

    # Margine: n_pixels/2 pixel MODIS da 500m
    margin_m   = (n_pixels / 2 + 0.5) * 500.0
    c          = np.cos(np.radians(lat))
    margin_lon = margin_m / (111320.0 * c)
    margin_lat = margin_m / 111320.0

    try:
        with rasterio.open(href) as src:
            l, b, r, t = transform_bounds(
                wgs84, src.crs,
                lon - margin_lon, lat - margin_lat,
                lon + margin_lon, lat + margin_lat)
            win  = from_bounds(l, b, r, t, transform=src.transform)
            data = src.read(1, window=win)

        # Maschera nodata e scala
        valid = data != MODIS_LAI_NODATA
        lai   = np.where(valid,
                         data.astype(float) * MODIS_LAI_SCALE,
                         np.nan)

        # Ritaglia / resampla a n_pixels x n_pixels
        # Se il crop è più grande del necessario, prendi il centro
        nr, nc = lai.shape
        r0 = max(0, (nr - n_pixels) // 2)
        c0 = max(0, (nc - n_pixels) // 2)
        lai_crop = lai[r0:r0+n_pixels, c0:c0+n_pixels]

        # Pad se troppo piccolo
        if lai_crop.shape != (n_pixels, n_pixels):
            out = np.full((n_pixels, n_pixels), np.nan)
            h   = min(lai_crop.shape[0], n_pixels)
            w   = min(lai_crop.shape[1], n_pixels)
            out[:h, :w] = lai_crop[:h, :w]
            lai_crop = out

        return lai_crop

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def get_vegetation_indices(
    lat,
    lon,
    dx_grid,
    dy_grid,
    dist_grid,
    r86,
    cache_dir,
    landsat_start_year = LANDSAT_START_YEAR,
    modis_start_date   = MODIS_START_DATE,
    modis_n_pixels     = 5,
    min_clear_fraction = 0.1,   # frazione minima pixel clear per usare scena
    cloud_cover_max    = 80.0,  # max cloud% per includere scena nella ricerca
    verbose            = True,
):
    """
    Calcola gli indici di vegetazione per un sito CRNS.

    Landsat (30m, footprint intero):
        NDVI, EVI, NDWI, FCOVER
        Medie mensili climatologiche + std + serie temporale

    MODIS MCD15A3H (500m, 5x5 pixel):
        LAI
        Medie mensili climatologiche + std + serie temporale

    Parameters
    ----------
    lat, lon        : coordinate sensore WGS84
    dx_grid         : 2D array offset easting [m]  (da clip_dem_to_radius)
    dy_grid         : 2D array offset northing [m]
    dist_grid       : 2D array distanza dal sensore [m]
    r86             : raggio footprint [m]
    cache_dir       : directory cache locale
    landsat_start_year : primo anno Landsat (default 2013)
    modis_start_date   : prima data MODIS (default '2002-07-04')
    modis_n_pixels     : dimensione array MODIS (default 5x5)
    min_clear_fraction : frazione minima pixel clear per usare la scena
    cloud_cover_max    : soglia cloud cover per search STAC
    verbose         : stampa progressi

    Returns
    -------
    dict con:

    === LANDSAT ===
    Per ogni indice idx in {ndvi, evi, ndwi, fcover}:

        landsat_{idx}_monthly_mean  : array (12,) media mensile pesata W(r)
        landsat_{idx}_monthly_std   : array (12,) std interannuale
        landsat_{idx}_monthly_nobs  : array (12,) n osservazioni valide
        landsat_{idx}_current       : float, valore più recente
        landsat_{idx}_current_date  : str, data scena più recente

    landsat_timeseries : list di dict
        [{'date': str, 'ndvi': float, 'evi': float,
          'ndwi': float, 'fcover': float, 'n_clear': int}, ...]
    landsat_scene_map  : dict mese->list di mappe 2D (per plotting)

    === MODIS LAI ===
        modis_lai_monthly_mean  : array (12,) media mensile dell'array 5x5
        modis_lai_monthly_std   : array (12,) std
        modis_lai_monthly_nobs  : array (12,) n osservazioni
        modis_lai_current       : float, valore più recente (media 5x5)
        modis_lai_current_date  : str
        modis_lai_timeseries    : list di dict
            [{'date': str, 'lai_mean': float,
              'lai_std': float, 'lai_map': array(5,5)}, ...]

    === CORRREZIONE CRNS ===
        f_veg_monthly  : array (12,) fattore correzione Baatz 2015
                         f_veg = exp(-0.257 * LAI / cos(sza_mean))
                         sza_mean = 30° (angolo zenitale medio tipico)
        f_veg_current  : float, valore attuale

    === METADATA ===
        lat, lon, r86
        landsat_n_scenes_total  : int
        modis_n_scenes_total    : int
        cache_dir               : str
    """

    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        raise ImportError(
            "Installa: pip install pystac-client planetary-computer")

    os.makedirs(cache_dir, exist_ok=True)

    now      = datetime.now(timezone.utc)
    date_end = now.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------ #
    # Connessione catalogo
    # ------------------------------------------------------------------ #
    if verbose:
        print("Connecting to Planetary Computer ...", flush=True)

    catalog = pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    # ------------------------------------------------------------------ #
    # Bbox per Landsat (footprint) e MODIS (punto)
    # ------------------------------------------------------------------ #
    c           = np.cos(np.radians(lat))
    margin_lon  = (r86 * 1.3) / (111320.0 * c)
    margin_lat  = (r86 * 1.3) / 111320.0
    bbox_landsat = [lon - margin_lon, lat - margin_lat,
                    lon + margin_lon, lat + margin_lat]
    bbox_modis   = [lon - 0.01, lat - 0.01,
                    lon + 0.01, lat + 0.01]

    # ------------------------------------------------------------------ #
    # Search Landsat
    # ------------------------------------------------------------------ #
    landsat_start = f"{landsat_start_year}-01-01"
    if verbose:
        print(f"Searching Landsat C2 L2  {landsat_start} -> {date_end} ...",
              flush=True)

    search_ls = catalog.search(
        collections = ["landsat-c2-l2"],
        bbox        = bbox_landsat,
        datetime    = f"{landsat_start}/{date_end}",
        query       = {"eo:cloud_cover": {"lt": cloud_cover_max}},
        max_items   = 2000,
    )
    items_ls = list(search_ls.items())
    if verbose:
        print(f"  Found {len(items_ls)} Landsat scenes", flush=True)

    # ------------------------------------------------------------------ #
    # Search MODIS
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"Searching MODIS MCD15A3H  {modis_start_date} -> {date_end} ...",
              flush=True)

    search_mod = catalog.search(
        collections = ["modis-15A3H-061"],
        bbox        = bbox_modis,
        datetime    = f"{modis_start_date}/{date_end}",
        max_items   = 5000,
    )
    items_mod = list(search_mod.items())
    if verbose:
        print(f"  Found {len(items_mod)} MODIS scenes", flush=True)

    # ------------------------------------------------------------------ #
    # Strutture accumulo dati
    # ------------------------------------------------------------------ #
    # Landsat: per mese, lista di valori scalari (media W(r) nel footprint)
    ls_monthly = {idx: {m: [] for m in range(1, 13)}
                  for idx in ["ndvi", "evi", "ndwi", "fcover"]}
    ls_timeseries = []
    ls_latest     = {idx: (None, None) for idx in ["ndvi","evi","ndwi","fcover"]}
    # Mappa 2D più recente per plotting
    ls_latest_map = {idx: None for idx in ["ndvi","evi","ndwi","fcover"]}

    # MODIS: per mese, lista di valori scalari (media 5x5)
    mod_monthly = {m: [] for m in range(1, 13)}
    mod_timeseries = []
    mod_latest     = (None, None)   # (valore, data)
    mod_latest_map = None

    # ------------------------------------------------------------------ #
    # Processa scene Landsat
    # ------------------------------------------------------------------ #
    W_fp = _weight_radial(dist_grid, r86)
    W_fp_sum = W_fp.sum()

    def _weighted_mean_fp(arr):
        """Media pesata W(r) nel footprint, ignora NaN."""
        valid = ~np.isnan(arr)
        if not valid.any():
            return np.nan
        w = W_fp[valid]
        return float(np.sum(w * arr[valid]) / w.sum()) if w.sum() > 0 else np.nan

    if verbose:
        print("Processing Landsat scenes ...", flush=True)

    for i, item in enumerate(items_ls):
        # Data della scena
        dt_str = (item.properties.get("datetime")
                  or item.properties.get("start_datetime"))
        if not dt_str:
            continue
        try:
            dt_obj = datetime.fromisoformat(
                dt_str.replace("Z", "+00:00"))
        except Exception:
            continue

        month = dt_obj.month

        # Cache key
        bbox_str = f"{bbox_landsat[0]:.4f}_{bbox_landsat[2]:.4f}"
        ckey     = _cache_key(item.id, bbox_str)
        cached, cached_meta = _cache_load(cache_dir, ckey)

        if cached is not None:
            indices = cached
            n_clear = cached_meta.get("n_clear", 0)
        else:
            # Leggi scena
            signed  = planetary_computer.sign(item)
            indices = _read_landsat_scene(
                signed, lat, lon, r86, dx_grid, dy_grid)

            if indices is None:
                continue

            # Conta pixel clear
            n_clear = int(np.nansum(indices["clear"] > 0.5))
            fp_pixels = int((dist_grid <= r86).sum())

            # Salva cache
            _cache_save(cache_dir, ckey, indices,
                        {"item_id": item.id,
                         "date": dt_str,
                         "n_clear": n_clear,
                         "fp_pixels": fp_pixels})

        # Scarta scena se troppo pochi pixel clear
        fp_pixels = int((dist_grid <= r86).sum())
        if fp_pixels > 0 and n_clear / fp_pixels < min_clear_fraction:
            continue

        # Calcola medie pesate
        scene_vals = {}
        for idx in ["ndvi", "evi", "ndwi", "fcover"]:
            if idx in indices:
                scene_vals[idx] = _weighted_mean_fp(indices[idx])
                if not np.isnan(scene_vals[idx]):
                    ls_monthly[idx][month].append(scene_vals[idx])

        if any(not np.isnan(v) for v in scene_vals.values()):
            ls_timeseries.append({
                "date"   : dt_obj.date().isoformat(),
                "year"   : dt_obj.year,
                "month"  : month,
                **{k: float(v) if not np.isnan(v) else None
                   for k, v in scene_vals.items()},
                "n_clear": n_clear,
            })
            # Aggiorna latest (items sono in ordine STAC, non cronologico)
            for idx in ["ndvi", "evi", "ndwi", "fcover"]:
                prev_date = ls_latest[idx][1]
                if (scene_vals.get(idx) is not None and
                        not np.isnan(scene_vals.get(idx, np.nan)) and
                        (prev_date is None or dt_obj.date().isoformat() > prev_date)):
                    ls_latest[idx] = (float(scene_vals[idx]),
                                      dt_obj.date().isoformat())
                    ls_latest_map[idx] = indices.get(idx)

        if verbose and (i + 1) % 50 == 0:
            print(f"  Landsat: {i+1}/{len(items_ls)} processed", flush=True)

    # ------------------------------------------------------------------ #
    # Processa scene MODIS
    # ------------------------------------------------------------------ #
    if verbose:
        print("Processing MODIS LAI scenes ...", flush=True)

    for i, item in enumerate(items_mod):
        dt_str = (item.properties.get("datetime")
                  or item.properties.get("start_datetime"))
        if not dt_str:
            continue
        try:
            dt_obj = datetime.fromisoformat(
                dt_str.replace("Z", "+00:00"))
        except Exception:
            continue

        month = dt_obj.month

        ckey         = _cache_key(item.id, f"{lat:.4f}_{lon:.4f}")
        cached, cmeta = _cache_load(cache_dir, ckey)

        if cached is not None:
            lai_map = cached.get("lai_map")
            if lai_map is None:
                continue
        else:
            signed  = planetary_computer.sign(item)
            lai_map = _read_modis_lai_scene(
                signed, lat, lon, modis_n_pixels)
            if lai_map is None:
                continue
            _cache_save(cache_dir, ckey, {"lai_map": lai_map},
                        {"item_id": item.id, "date": dt_str})

        lai_mean = float(np.nanmean(lai_map))
        lai_std  = float(np.nanstd(lai_map))

        if not np.isnan(lai_mean):
            mod_monthly[month].append(lai_mean)
            mod_timeseries.append({
                "date"    : dt_obj.date().isoformat(),
                "year"    : dt_obj.year,
                "month"   : month,
                "lai_mean": lai_mean,
                "lai_std" : lai_std,
            })
            prev = mod_latest[1]
            if prev is None or dt_obj.date().isoformat() > prev:
                mod_latest     = (lai_mean, dt_obj.date().isoformat())
                mod_latest_map = lai_map.copy()

        if verbose and (i + 1) % 200 == 0:
            print(f"  MODIS: {i+1}/{len(items_mod)} processed", flush=True)

    # ------------------------------------------------------------------ #
    # Aggregazione mensile
    # ------------------------------------------------------------------ #
    def _monthly_stats(monthly_dict):
        means = np.full(12, np.nan)
        stds  = np.full(12, np.nan)
        nobs  = np.zeros(12, dtype=int)
        for m in range(1, 13):
            vals = monthly_dict[m]
            if len(vals) >= 1:
                means[m-1] = float(np.nanmean(vals))
                stds[m-1]  = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
                nobs[m-1]  = len(vals)
        return means, stds, nobs

    # ------------------------------------------------------------------ #
    # Correzione f_veg Baatz 2015
    # LAI -> f_veg = exp(-0.257 * LAI / cos(SZA))
    # SZA medio ~30° (tipico latitudine 46N, mezzogiorno)
    # ------------------------------------------------------------------ #
    SZA_MEAN_RAD = np.radians(30.0)
    lai_means, _, _ = _monthly_stats(mod_monthly)
    f_veg_monthly = np.where(
        ~np.isnan(lai_means),
        np.exp(-0.257 * lai_means / np.cos(SZA_MEAN_RAD)),
        np.nan)

    f_veg_current = np.nan
    if mod_latest[0] is not None:
        f_veg_current = float(np.exp(
            -0.257 * mod_latest[0] / np.cos(SZA_MEAN_RAD)))

    # ------------------------------------------------------------------ #
    # Ordina serie temporali per data
    # ------------------------------------------------------------------ #
    ls_timeseries.sort(key=lambda x: x["date"])
    mod_timeseries.sort(key=lambda x: x["date"])

    # ------------------------------------------------------------------ #
    # Assembla risultato
    # ------------------------------------------------------------------ #
    result = {}

    for idx in ["ndvi", "evi", "ndwi", "fcover"]:
        mn, sd, nb = _monthly_stats(ls_monthly[idx])
        result[f"landsat_{idx}_monthly_mean"] = mn
        result[f"landsat_{idx}_monthly_std"]  = sd
        result[f"landsat_{idx}_monthly_nobs"] = nb
        result[f"landsat_{idx}_current"]      = ls_latest[idx][0]
        result[f"landsat_{idx}_current_date"] = ls_latest[idx][1]
        result[f"landsat_{idx}_latest_map"]   = ls_latest_map[idx]

    lai_means, lai_stds, lai_nobs = _monthly_stats(mod_monthly)
    result["modis_lai_monthly_mean"]  = lai_means
    result["modis_lai_monthly_std"]   = lai_stds
    result["modis_lai_monthly_nobs"]  = lai_nobs
    result["modis_lai_current"]       = mod_latest[0]
    result["modis_lai_current_date"]  = mod_latest[1]
    result["modis_lai_latest_map"]    = mod_latest_map

    result["f_veg_monthly"]  = f_veg_monthly
    result["f_veg_current"]  = f_veg_current

    result["landsat_timeseries"]     = ls_timeseries
    result["modis_lai_timeseries"]   = mod_timeseries

    result["lat"]                    = lat
    result["lon"]                    = lon
    result["r86"]                    = r86
    result["landsat_n_scenes_total"] = len(items_ls)
    result["modis_n_scenes_total"]   = len(items_mod)
    result["cache_dir"]              = cache_dir
    result["months"]                 = ["Jan","Feb","Mar","Apr","May","Jun",
                                        "Jul","Aug","Sep","Oct","Nov","Dec"]

    return result


def report_vegetation(res):
    """Stampa testuale dei risultati."""
    M  = res["months"]
    w  = 72
    L  = ["="*w,
          "VEGETATION INDICES — Monthly Climatology",
          "="*w,
          f"  Landsat scenes : {res['landsat_n_scenes_total']}  "
          f"(from {LANDSAT_START_YEAR})",
          f"  MODIS scenes   : {res['modis_n_scenes_total']}  "
          f"(from {MODIS_START_DATE})",
          ""]

    hdr = "  " + " "*14 + "  ".join(f"{m:>5}" for m in M)
    L.append(hdr)
    L.append("  " + "-"*(w-2))

    for idx in ["ndvi","evi","ndwi","fcover"]:
        mn = res[f"landsat_{idx}_monthly_mean"]
        sd = res[f"landsat_{idx}_monthly_std"]
        nb = res[f"landsat_{idx}_monthly_nobs"]
        mn_s = "  ".join(f"{v:5.3f}" if not np.isnan(v) else "  N/A" for v in mn)
        sd_s = "  ".join(f"{v:5.3f}" if not np.isnan(v) else "  N/A" for v in sd)
        nb_s = "  ".join(f"{v:5d}" for v in nb)
        cur  = res[f"landsat_{idx}_current"]
        L.append(f"  {idx.upper():<8} mean  {mn_s}")
        L.append(f"  {'':<8} std   {sd_s}")
        L.append(f"  {'':<8} nobs  {nb_s}")
        L.append(f"  {'':<8} current = "
                 f"{cur:.3f}  ({res[f'landsat_{idx}_current_date']})"
                 if cur is not None else f"  {'':<8} current = N/A")
        L.append("")

    mn = res["modis_lai_monthly_mean"]
    sd = res["modis_lai_monthly_std"]
    nb = res["modis_lai_monthly_nobs"]
    mn_s = "  ".join(f"{v:5.2f}" if not np.isnan(v) else "  N/A" for v in mn)
    cur  = res["modis_lai_current"]
    L.append(f"  {'LAI':<8} mean  {mn_s}")
    L.append(f"  {'':<8} current = "
             f"{cur:.2f}  ({res['modis_lai_current_date']})"
             if cur is not None else f"  {'':<8} current = N/A")
    L.append("")

    fv = res["f_veg_monthly"]
    fv_s = "  ".join(f"{v:5.3f}" if not np.isnan(v) else "  N/A" for v in fv)
    L.append(f"  {'f_veg':<8} Baatz {fv_s}")
    L.append(f"  (f_veg = exp(-0.257*LAI/cos(30°)), "
             f"current={res['f_veg_current']:.3f})"
             if res['f_veg_current'] is not None else "")
    L.append("="*w)
    return "\n".join(L)
