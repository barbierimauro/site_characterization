"""
vegetation_indices.py
=====================
Pipeline per il calcolo degli indici di vegetazione per un sito CRNS.

Sorgenti
--------
Landsat Collection 2 Level-2 (Planetary Computer, pubblico):
  NDVI, EVI, NDWI, FCOVER — 30m, footprint intero, media pesata W(r)
  Dal 2013 in poi, Landsat 8+9

MODIS MCD15A3H (Planetary Computer, pubblico):
  LAI — 500m, array 5×5 pixel centrati sulle coordinate
  Dal 2013 in poi

Cache
-----
UN solo file per sorgente per sito:
  landsat_cache_{hash16}.npz   — serie temporale completa Landsat
  modis_cache_{hash16}.npz     — serie temporale completa MODIS

La cache è aggiornata incrementalmente: alla seconda run scarica
solo le scene successive all'ultima data già in cache.

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import os
import json
import hashlib
import itertools
import time
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=RuntimeWarning)

# GDAL / network tuning – must be set before any rasterio / GDAL import.
# GDAL_CACHEMAX: 128 MB cap to avoid OOM under WSL with parallel COG reads.
# CPL_VSIL_CURL_CHUNK_SIZE: 2 MB chunks → fewer round-trips per windowed read.
# GDAL_HTTP_MULTIPLEX: HTTP/2 multiplexing – all 5 band requests per scene
#   share one TCP connection, cutting latency significantly.
# GDAL_HTTP_MERGE_CONSECUTIVE_RANGES: coalesces adjacent range requests.
os.environ.setdefault("GDAL_CACHEMAX", "128")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "2097152")   # 2 MB chunks
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")

# ---------------------------------------------------------------------------
# Costanti fisiche
# ---------------------------------------------------------------------------

L2_SCALE  = 2.75e-5
L2_OFFSET = -0.2
L2_NODATA = 0

QA_FILL       = 1 << 0
QA_CLOUD      = 1 << 3
QA_CLOUD_SHAD = 1 << 4
QA_SNOW       = 1 << 5
QA_CLEAR      = 1 << 6
QA_WATER      = 1 << 7

FCOVER_NDVI_SOIL = 0.05
FCOVER_NDVI_VEG  = 0.95

MODIS_LAI_SCALE  = 0.1
MODIS_LAI_NODATA = 255

LANDSAT_START_YEAR = 2020
MODIS_START_YEAR   = 2020

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

N_WORKERS = 3   # thread paralleli per download; sliding-window limita i future in volo
PARALLEL_WINDOW = N_WORKERS * 3  # max future in volo contemporaneamente (evita OOM)


# ---------------------------------------------------------------------------
# Cache: UN file per sorgente per sito
# ---------------------------------------------------------------------------

def _site_hash(lat, lon, r86, start_year):
    tag = f"{lat:.5f}_{lon:.5f}_{r86:.1f}_{start_year}"
    return hashlib.sha256(tag.encode()).hexdigest()[:16]


def _landsat_cache_path(cache_dir, lat, lon, r86, start_year):
    h = _site_hash(lat, lon, r86, start_year)
    return os.path.join(cache_dir, f"landsat_cache_{h}.npz"), \
           os.path.join(cache_dir, f"landsat_cache_{h}.json")


def _modis_cache_path(cache_dir, lat, lon, start_year):
    h = _site_hash(lat, lon, 0.0, start_year)
    return os.path.join(cache_dir, f"modis_cache_{h}.npz"), \
           os.path.join(cache_dir, f"modis_cache_{h}.json")


def _load_landsat_cache(cache_dir, lat, lon, r86, start_year):
    """
    Carica la cache Landsat.
    Ritorna (timeseries_list, last_date_str) oppure ([], None).
    timeseries_list: lista di dict {date, ndvi, evi, ndwi, fcover, n_clear}
    """
    npz, meta = _landsat_cache_path(cache_dir, lat, lon, r86, start_year)
    if not (os.path.exists(npz) and os.path.exists(meta)):
        return [], None
    try:
        d  = np.load(npz, allow_pickle=True)
        ts = d["timeseries"].tolist()   # lista di dict
        with open(meta) as f:
            m = json.load(f)
        last_date = m.get("last_date")
        return ts, last_date
    except Exception:
        return [], None


def _save_landsat_cache(cache_dir, lat, lon, r86, start_year,
                         timeseries, last_date):
    os.makedirs(cache_dir, exist_ok=True)
    npz, meta = _landsat_cache_path(cache_dir, lat, lon, r86, start_year)
    np.savez_compressed(npz, timeseries=np.array(timeseries, dtype=object))
    with open(meta, "w") as f:
        json.dump({"last_date": last_date,
                   "n_scenes": len(timeseries),
                   "lat": lat, "lon": lon, "r86": r86,
                   "start_year": start_year}, f, indent=2)


def _snow_cache_path(cache_dir, lat, lon, start_year, product="10A2"):
    h = _site_hash(lat, lon, 0.0, start_year)
    return (os.path.join(cache_dir, f"snow_{product}_cache_{h}.npz"),
            os.path.join(cache_dir, f"snow_{product}_cache_{h}.json"))


def _load_snow_cache(cache_dir, lat, lon, start_year, product="10A2"):
    """
    Carica la cache snow cover.
    Ritorna (timeseries_list, last_date_str).
    timeseries_list: lista di dict {date, year, month, snow_cover_pct}
    """
    npz, meta = _snow_cache_path(cache_dir, lat, lon, start_year, product)
    if not (os.path.exists(npz) and os.path.exists(meta)):
        return [], None
    try:
        d  = np.load(npz, allow_pickle=True)
        ts = d["timeseries"].tolist()
        with open(meta) as f:
            m = json.load(f)
        return ts, m.get("last_date")
    except Exception:
        return [], None


def _save_snow_cache(cache_dir, lat, lon, start_year, timeseries, last_date,
                     product="10A2"):
    os.makedirs(cache_dir, exist_ok=True)
    npz, meta = _snow_cache_path(cache_dir, lat, lon, start_year, product)
    np.savez_compressed(npz, timeseries=np.array(timeseries, dtype=object))
    with open(meta, "w") as f:
        json.dump({"last_date": last_date,
                   "n_scenes": len(timeseries),
                   "lat": lat, "lon": lon,
                   "start_year": start_year,
                   "product": product}, f, indent=2)


def _load_modis_cache(cache_dir, lat, lon, start_year):
    """
    Carica la cache MODIS.
    Ritorna (timeseries_list, last_date_str).
    timeseries_list: lista di dict {date, lai_mean, lai_std}
    """
    npz, meta = _modis_cache_path(cache_dir, lat, lon, start_year)
    if not (os.path.exists(npz) and os.path.exists(meta)):
        return [], None
    try:
        d  = np.load(npz, allow_pickle=True)
        ts = d["timeseries"].tolist()
        with open(meta) as f:
            m = json.load(f)
        return ts, m.get("last_date")
    except Exception:
        return [], None


def _save_modis_cache(cache_dir, lat, lon, start_year,
                       timeseries, last_date):
    os.makedirs(cache_dir, exist_ok=True)
    npz, meta = _modis_cache_path(cache_dir, lat, lon, start_year)
    np.savez_compressed(npz, timeseries=np.array(timeseries, dtype=object))
    with open(meta, "w") as f:
        json.dump({"last_date": last_date,
                   "n_scenes": len(timeseries),
                   "lat": lat, "lon": lon,
                   "start_year": start_year}, f, indent=2)


# ---------------------------------------------------------------------------
# Peso radiale
# ---------------------------------------------------------------------------

def _weight_radial(dist, r86):
    lam = r86 / 3.0
    return np.where(dist < 1e-3, 0.0, np.exp(-dist / lam))


# ---------------------------------------------------------------------------
# Landsat helpers
# ---------------------------------------------------------------------------

def _l2_reflectance(data):
    mask = (data != L2_NODATA) & (data > 0)
    refl = np.where(mask,
                    data.astype(float) * L2_SCALE + L2_OFFSET,
                    np.nan)
    return np.clip(refl, 0.0, 1.0)


def _qa_clear_mask(qa_data):
    qa     = qa_data.astype(np.uint16)
    clear  = (qa & QA_CLEAR)      > 0
    no_cld = (qa & QA_CLOUD)      == 0
    no_shd = (qa & QA_CLOUD_SHAD) == 0
    no_snw = (qa & QA_SNOW)       == 0
    no_fill= (qa & QA_FILL)       == 0
    return clear & no_cld & no_shd & no_snw & no_fill


def _compute_indices(red, nir, green, blue, clear_mask):
    denom_ndvi = nir + red
    ndvi = np.where(clear_mask & (denom_ndvi > 0.01),
                    (nir - red) / denom_ndvi, np.nan)

    denom_evi = nir + 6.0*red - 7.5*blue + 1.0
    evi = np.where(clear_mask & (denom_evi > 0.01),
                   2.5 * (nir - red) / denom_evi, np.nan)
    evi = np.clip(evi, -1.0, 1.0)

    denom_ndwi = green + nir
    ndwi = np.where(clear_mask & (denom_ndwi > 0.01),
                    (green - nir) / denom_ndwi, np.nan)

    fcover = np.where(
        ~np.isnan(ndvi),
        np.clip((ndvi - FCOVER_NDVI_SOIL) /
                (FCOVER_NDVI_VEG - FCOVER_NDVI_SOIL), 0.0, 1.0),
        np.nan)

    return ndvi, evi, ndwi, fcover


def _read_landsat_scene(signed_item, lat, lon, r86, dx_grid, dy_grid,
                         min_clear_frac_pre=0.05):
    """
    Windowed read di una scena Landsat + calcolo indici.

    Strategia fast-reject a 2 fasi per minimizzare le connessioni HTTP:
    1. Legge qa_pixel (1 conn): se il sito è troppo nuvoloso → None subito.
    2. Legge le 4 bande spettrali IN PARALLELO (4 conn concorrenti).
    Risparmio tipico: ~60% delle scene scartate dopo 1 sola connessione.

    Ritorna dict con array 2D (shape = dx_grid.shape) per ogni indice,
    oppure None se fallisce o nessun pixel valido.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.warp import transform as rio_transform
    from rasterio.windows import from_bounds
    from rasterio.crs import CRS
    from scipy.spatial import cKDTree

    wgs84      = CRS.from_epsg(4326)
    c          = np.cos(np.radians(lat))
    margin_lon = (r86 * 1.2) / (111320.0 * c)
    margin_lat = (r86 * 1.2) / 111320.0
    bbox_wgs84 = (lon - margin_lon, lat - margin_lat,
                  lon + margin_lon, lat + margin_lat)

    for band in ["red", "nir08", "green", "blue", "qa_pixel"]:
        if band not in signed_item.assets:
            return None

    # ------------------------------------------------------------------ #
    # Fase 1: qa_pixel → fast-reject se il footprint è nuvoloso
    # ------------------------------------------------------------------ #
    try:
        with rasterio.open(signed_item.assets["qa_pixel"].href) as src:
            crs_raster = src.crs
            l, b, r, t = transform_bounds(wgs84, src.crs, *bbox_wgs84)
            win  = from_bounds(l, b, r, t, transform=src.transform)
            qa_data = src.read(1, window=win)
            transform_raster = src.window_transform(win)
            shape_raster     = qa_data.shape
    except Exception:
        return None

    clear = _qa_clear_mask(qa_data)

    # Calcola fp_mask qui per il pre-screening, riusata dopo
    nr, nc  = shape_raster
    col_idx = np.arange(nc)
    row_idx = np.arange(nr)
    px_x    = transform_raster.c + (col_idx + 0.5) * transform_raster.a
    px_y    = transform_raster.f + (row_idx + 0.5) * transform_raster.e
    PX, PY  = np.meshgrid(px_x, px_y)
    lons_flat, lats_flat = rio_transform(
        crs_raster, wgs84, PX.ravel(), PY.ravel())
    lons_flat = np.array(lons_flat)
    lats_flat = np.array(lats_flat)
    dx_ls   = (lons_flat - lon) * 111320.0 * c
    dy_ls   = (lats_flat - lat) * 111320.0
    dist_ls = np.sqrt(dx_ls**2 + dy_ls**2)
    fp_mask = dist_ls <= r86

    if fp_mask.sum() == 0:
        return None

    # Pre-screening cloud: se troppi pixel nuvolosi nel footprint → salta
    clear_flat = clear.ravel()[fp_mask]
    if clear_flat.mean() < min_clear_frac_pre:
        return None

    # ------------------------------------------------------------------ #
    # Fase 2: legge le 4 bande spettrali SEQUENZIALMENTE
    # (GDAL non è thread-safe su Windows: multi-thread causa crash silenzioso)
    # ------------------------------------------------------------------ #
    spectral_bands = ["red", "nir08", "green", "blue"]
    band_results = {}
    for _band in spectral_bands:
        href = signed_item.assets[_band].href
        try:
            with rasterio.open(href) as src:
                l2, b2, r2, t2 = transform_bounds(wgs84, src.crs, *bbox_wgs84)
                win2 = from_bounds(l2, b2, r2, t2, transform=src.transform)
                band_results[_band] = src.read(1, window=win2)
        except Exception:
            band_results[_band] = None

    if any(v is None for v in band_results.values()):
        return None

    red   = _l2_reflectance(band_results["red"])
    nir   = _l2_reflectance(band_results["nir08"])
    green = _l2_reflectance(band_results["green"])
    blue  = _l2_reflectance(band_results["blue"])

    ndvi, evi, ndwi, fcover = _compute_indices(
        red, nir, green, blue, clear)

    pts_ls  = np.column_stack([dx_ls[fp_mask], dy_ls[fp_mask]])
    tree    = cKDTree(pts_ls)
    pts_dem = np.column_stack([dx_grid.ravel(), dy_grid.ravel()])
    _, idx  = tree.query(pts_dem)

    def _resample(arr):
        return arr.ravel()[fp_mask][idx].reshape(dx_grid.shape)

    return {
        "ndvi"  : _resample(ndvi),
        "evi"   : _resample(evi),
        "ndwi"  : _resample(ndwi),
        "fcover": _resample(fcover),
        "clear" : _resample(clear.astype(float)),
    }


# ---------------------------------------------------------------------------
# MODIS helpers
# ---------------------------------------------------------------------------

def _read_modis_lai_scene(signed_item, lat, lon, n_pixels=5):
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from rasterio.crs import CRS

    if "Lai_500m" not in signed_item.assets:
        return None

    href       = signed_item.assets["Lai_500m"].href
    wgs84      = CRS.from_epsg(4326)
    c          = np.cos(np.radians(lat))
    margin_m   = (n_pixels / 2 + 0.5) * 500.0
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

        valid    = data != MODIS_LAI_NODATA
        lai      = np.where(valid,
                            data.astype(float) * MODIS_LAI_SCALE,
                            np.nan)
        nr, nc   = lai.shape
        r0 = max(0, (nr - n_pixels) // 2)
        c0 = max(0, (nc - n_pixels) // 2)
        crop     = lai[r0:r0+n_pixels, c0:c0+n_pixels]
        out      = np.full((n_pixels, n_pixels), np.nan)
        h, w     = min(crop.shape[0], n_pixels), min(crop.shape[1], n_pixels)
        out[:h, :w] = crop[:h, :w]
        return out
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
    modis_start_year   = MODIS_START_YEAR,
    modis_n_pixels     = 5,
    min_clear_fraction = 0.1,
    cloud_cover_max    = 80.0,
    verbose            = True,
):
    """
    Calcola indici di vegetazione per un sito CRNS.

    Cache: 1 file Landsat + 1 file MODIS per sito.
    Aggiornamento incrementale: solo le scene nuove vengono scaricate.

    Parameters
    ----------
    lat, lon            : coordinate WGS84
    dx_grid, dy_grid    : offset metrici dal sensore (da clip_dem_to_radius)
    dist_grid           : distanza dal sensore [m]
    r86                 : raggio footprint [m]
    cache_dir           : directory cache
    landsat_start_year  : default 2013
    modis_start_year    : default 2013
    modis_n_pixels      : array MODIS NxN, default 5
    min_clear_fraction  : frazione min pixel clear per usare scena
    cloud_cover_max     : soglia cloud% STAC search
    verbose             : stampa progressi
    """
    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        raise ImportError(
            "pip install pystac-client planetary-computer")

    os.makedirs(cache_dir, exist_ok=True)

    now      = datetime.now(timezone.utc)
    date_end = now.strftime("%Y-%m-%d")

    # Pesi radiali (costanti per il sito)
    W_fp     = _weight_radial(dist_grid, r86)
    fp_mask  = dist_grid <= r86
    fp_pixels = int(fp_mask.sum())

    def _weighted_mean_fp(arr):
        valid = ~np.isnan(arr)
        if not valid.any():
            return np.nan
        w = W_fp[valid]
        return float(np.sum(w * arr[valid]) / w.sum()) if w.sum() > 0 else np.nan

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
    # Bbox
    # ------------------------------------------------------------------ #
    c            = np.cos(np.radians(lat))
    margin_lon   = (r86 * 1.3) / (111320.0 * c)
    margin_lat   = (r86 * 1.3) / 111320.0
    bbox_landsat = [lon - margin_lon, lat - margin_lat,
                    lon + margin_lon, lat + margin_lat]
    bbox_modis   = [lon - 0.01, lat - 0.01,
                    lon + 0.01, lat + 0.01]

    # ================================================================== #
    # LANDSAT
    # ================================================================== #

    # --- Carica cache esistente ---
    ls_timeseries, ls_last_date = _load_landsat_cache(
        cache_dir, lat, lon, r86, landsat_start_year)

    if ls_last_date:
        # Aggiornamento incrementale: cerca solo scene più recenti
        ls_search_start = ls_last_date
        if verbose:
            print(f"Landsat cache: {len(ls_timeseries)} scenes, "
                  f"last={ls_last_date}. Fetching updates ...", flush=True)
    else:
        ls_search_start = f"{landsat_start_year}-01-01"
        if verbose:
            print(f"Landsat: no cache, full download from "
                  f"{ls_search_start} ...", flush=True)

    search_ls = catalog.search(
        collections = ["landsat-c2-l2"],
        bbox        = bbox_landsat,
        datetime    = f"{ls_search_start}/{date_end}",
        query       = {"eo:cloud_cover": {"lt": cloud_cover_max}},
        max_items   = 2000,
    )
    items_ls = [it for it in search_ls.items()
                if it.properties.get("platform", "") != "landsat-7"]

    # Escludi item già in cache (stessa data o precedente)
    if ls_last_date:
        items_ls = [it for it in items_ls
                    if (it.properties.get("datetime","") or
                        it.properties.get("start_datetime","")) > ls_last_date]

    if verbose:
        print(f"  Landsat search range : {ls_search_start} → {date_end}",
              flush=True)
        print(f"  {len(items_ls)} new Landsat scenes to download",
              flush=True)

    # --- Download parallelo con sliding-window per evitare OOM ---
    # Non sottomettiamo tutti gli N items contemporaneamente (OOM).
    # Teniamo al massimo PARALLEL_WINDOW future in volo; appena uno completa
    # ne sottomettiamo un altro dall'iteratore, così la memoria è bounded.

    def _proc_ls(item):
        import planetary_computer as _pc
        dt_str = (item.properties.get("datetime")
                  or item.properties.get("start_datetime"))
        if not dt_str:
            return None
        try:
            dt_obj = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            return None
        signed  = _pc.sign(item)
        indices = _read_landsat_scene(signed, lat, lon, r86, dx_grid, dy_grid)
        if indices is None:
            return None
        n_clear = int(np.nansum(indices["clear"] > 0.5))
        if fp_pixels > 0 and n_clear / fp_pixels < min_clear_fraction:
            return None
        scene_vals = {
            k: _weighted_mean_fp(indices[k])
            for k in ["ndvi","evi","ndwi","fcover"] if k in indices
        }
        return dt_obj, scene_vals, n_clear, indices

    LANDSAT_BATCH = 25
    new_ls_scenes = []
    latest_ls = {}   # idx -> (val, date_str, map_arr) — solo l'ultima scena
    n_ls = len(items_ls)
    done = 0
    _ls_t0 = time.perf_counter()        # inizio download Landsat
    _ls_batch_t0 = _ls_t0               # inizio batch corrente

    def _handle_ls_result(res):
        nonlocal done, _ls_batch_t0
        done += 1
        if res is None:
            return
        dt_obj, scene_vals, n_clear, indices = res
        date_str = dt_obj.date().isoformat()
        row = {"date": date_str, "year": dt_obj.year,
               "month": dt_obj.month, "n_clear": n_clear}
        for idx in ["ndvi","evi","ndwi","fcover"]:
            v = scene_vals.get(idx, np.nan)
            row[idx] = float(v) if (v is not None and not np.isnan(v)) else None
        new_ls_scenes.append(row)
        for idx in ["ndvi","evi","ndwi","fcover"]:
            v = row.get(idx)
            if v is not None:
                prev = latest_ls.get(idx)
                if prev is None or date_str > prev[1]:
                    latest_ls[idx] = (v, date_str, indices.get(idx))
        del indices   # libera subito le mappe numpy (grandi) dopo l'uso

        # Checkpoint: salva cache ogni LANDSAT_BATCH scene valide
        if len(new_ls_scenes) % LANDSAT_BATCH == 0:
            partial = sorted(ls_timeseries + new_ls_scenes,
                             key=lambda x: x["date"])
            _save_landsat_cache(cache_dir, lat, lon, r86, landsat_start_year,
                                partial, partial[-1]["date"])
            if verbose:
                now        = time.perf_counter()
                batch_s    = now - _ls_batch_t0
                elapsed_s  = now - _ls_t0
                rate       = done / elapsed_s if elapsed_s > 0 else 0
                remaining  = (n_ls - done) / rate if rate > 0 else 0
                print(
                    f"  Landsat checkpoint: {done}/{n_ls} scanned, "
                    f"{len(new_ls_scenes)} valid  |  "
                    f"batch {batch_s:.0f}s  elapsed {elapsed_s:.0f}s  "
                    f"rate {rate*60:.1f} scene/min  ETA {remaining:.0f}s",
                    flush=True)
                _ls_batch_t0 = now

    with ThreadPoolExecutor(max_workers=N_WORKERS) as _pool:
        _iter = iter(items_ls)
        _pending = set()
        # Riempi la finestra iniziale
        for _item in itertools.islice(_iter, PARALLEL_WINDOW):
            _pending.add(_pool.submit(_proc_ls, _item))
        while _pending:
            _done_set, _pending = wait(_pending, return_when=FIRST_COMPLETED)
            # Rifornisci la finestra
            for _ in _done_set:
                try:
                    _pending.add(_pool.submit(_proc_ls, next(_iter)))
                except StopIteration:
                    pass
            # Processa i risultati completati
            for _f in _done_set:
                try:
                    _handle_ls_result(_f.result())
                except Exception as _e:
                    print(f"  [WARN] Landsat scene error: {_e}", flush=True)

    # Merge con cache esistente e salva
    ls_timeseries = ls_timeseries + new_ls_scenes
    ls_timeseries.sort(key=lambda x: x["date"])
    new_last = ls_timeseries[-1]["date"] if ls_timeseries else ls_last_date
    if new_ls_scenes:
        _save_landsat_cache(cache_dir, lat, lon, r86, landsat_start_year,
                             ls_timeseries, new_last)
        if verbose:
            print(f"  Landsat cache updated: {len(ls_timeseries)} total scenes",
                  flush=True)

    # latest maps: prende le più recenti tra cache + nuove
    ls_latest_map = {idx: latest_ls.get(idx, (None,None,None))[2]
                     for idx in ["ndvi","evi","ndwi","fcover"]}
    ls_latest_val = {}
    ls_latest_date= {}
    for idx in ["ndvi","evi","ndwi","fcover"]:
        best = None
        for row in reversed(ls_timeseries):
            v = row.get(idx)
            if v is not None:
                best = (v, row["date"])
                break
        ls_latest_val[idx]  = best[0] if best else None
        ls_latest_date[idx] = best[1] if best else None

    # ================================================================== #
    # MODIS
    # ================================================================== #

    mod_timeseries, mod_last_date = _load_modis_cache(
        cache_dir, lat, lon, modis_start_year)

    if mod_last_date:
        mod_search_start = mod_last_date
        if verbose:
            print(f"MODIS cache: {len(mod_timeseries)} scenes, "
                  f"last={mod_last_date}. Fetching updates ...", flush=True)
    else:
        mod_search_start = f"{modis_start_year}-01-01"
        if verbose:
            print(f"MODIS: no cache, full download from "
                  f"{mod_search_start} ...", flush=True)

    search_mod = catalog.search(
        collections = ["modis-15A3H-061"],
        bbox        = bbox_modis,
        datetime    = f"{mod_search_start}/{date_end}",
        max_items   = 5000,
    )
    items_mod = list(search_mod.items())
    if mod_last_date:
        items_mod = [it for it in items_mod
                     if (it.properties.get("datetime","") or
                         it.properties.get("start_datetime",""))
                        > mod_last_date]

    if verbose:
        print(f"  MODIS search range   : {mod_search_start} → {date_end}",
              flush=True)
        print(f"  {len(items_mod)} new MODIS scenes to download", flush=True)

    new_mod_scenes = []
    n_mod = len(items_mod)
    done_mod = 0
    mod_latest_val  = None
    mod_latest_date_ = None
    mod_latest_map  = None
    MODIS_BATCH = 50

    def _proc_mod(item):
        import planetary_computer as _pc
        dt_str = (item.properties.get("datetime")
                  or item.properties.get("start_datetime"))
        if not dt_str:
            return None
        try:
            dt_obj = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            return None
        signed  = _pc.sign(item)
        lai_map = _read_modis_lai_scene(signed, lat, lon, modis_n_pixels)
        if lai_map is None:
            return None
        lai_mean = float(np.nanmean(lai_map))
        lai_std  = float(np.nanstd(lai_map))
        if np.isnan(lai_mean):
            return None
        return dt_obj, lai_mean, lai_std, lai_map

    _mod_t0 = time.perf_counter()
    _mod_batch_t0 = _mod_t0

    def _handle_mod_result(res):
        nonlocal done_mod, _mod_batch_t0, mod_latest_val, mod_latest_date_, mod_latest_map
        done_mod += 1
        if res is None:
            return
        dt_obj, lai_mean, lai_std, lai_map = res
        date_str = dt_obj.date().isoformat()
        new_mod_scenes.append({
            "date"    : date_str,
            "year"    : dt_obj.year,
            "month"   : dt_obj.month,
            "lai_mean": lai_mean,
            "lai_std" : lai_std,
        })
        if mod_latest_date_ is None or date_str > mod_latest_date_:
            mod_latest_val   = lai_mean
            mod_latest_date_ = date_str
            mod_latest_map   = lai_map.copy()
        del lai_map   # libera la mappa dopo l'uso

        # Checkpoint ogni MODIS_BATCH scene valide
        if len(new_mod_scenes) % MODIS_BATCH == 0:
            partial = sorted(mod_timeseries + new_mod_scenes,
                             key=lambda x: x["date"])
            _save_modis_cache(cache_dir, lat, lon, modis_start_year,
                              partial, partial[-1]["date"])
            if verbose:
                now       = time.perf_counter()
                batch_s   = now - _mod_batch_t0
                elapsed_s = now - _mod_t0
                rate      = done_mod / elapsed_s if elapsed_s > 0 else 0
                remaining = (n_mod - done_mod) / rate if rate > 0 else 0
                print(
                    f"  MODIS checkpoint: {done_mod}/{n_mod} scanned, "
                    f"{len(new_mod_scenes)} valid  |  "
                    f"batch {batch_s:.0f}s  elapsed {elapsed_s:.0f}s  "
                    f"rate {rate*60:.1f} scene/min  ETA {remaining:.0f}s",
                    flush=True)
                _mod_batch_t0 = now

    with ThreadPoolExecutor(max_workers=N_WORKERS) as _pool:
        _iter = iter(items_mod)
        _pending = set()
        for _item in itertools.islice(_iter, PARALLEL_WINDOW):
            _pending.add(_pool.submit(_proc_mod, _item))
        while _pending:
            _done_set, _pending = wait(_pending, return_when=FIRST_COMPLETED)
            for _ in _done_set:
                try:
                    _pending.add(_pool.submit(_proc_mod, next(_iter)))
                except StopIteration:
                    pass
            for _f in _done_set:
                try:
                    _handle_mod_result(_f.result())
                except Exception as _e:
                    print(f"  [WARN] MODIS scene error: {_e}", flush=True)

    mod_timeseries = mod_timeseries + new_mod_scenes
    mod_timeseries.sort(key=lambda x: x["date"])
    mod_new_last = (mod_timeseries[-1]["date"]
                    if mod_timeseries else mod_last_date)
    if new_mod_scenes:
        _save_modis_cache(cache_dir, lat, lon, modis_start_year,
                           mod_timeseries, mod_new_last)
        if verbose:
            print(f"  MODIS cache updated: {len(mod_timeseries)} total scenes",
                  flush=True)

    # Latest MODIS dalla cache completa
    if mod_latest_val is None and mod_timeseries:
        last = mod_timeseries[-1]
        mod_latest_val  = last["lai_mean"]
        mod_latest_date_= last["date"]

    # ================================================================== #
    # Aggregazione mensile da timeseries complete
    # ================================================================== #

    def _monthly_stats_from_ts(ts, key):
        by_month = {m: [] for m in range(1, 13)}
        for row in ts:
            v = row.get(key)
            if v is not None and not np.isnan(float(v)):
                by_month[row["month"]].append(float(v))
        means = np.array([np.nanmean(by_month[m]) if by_month[m]
                          else np.nan for m in range(1,13)])
        stds  = np.array([np.nanstd(by_month[m])  if len(by_month[m])>1
                          else 0.0 for m in range(1,13)])
        nobs  = np.array([len(by_month[m]) for m in range(1,13)])
        return means, stds, nobs

    # Correzione f_veg Baatz 2015
    SZA_RAD = np.radians(30.0)
    lai_means, _, _ = _monthly_stats_from_ts(mod_timeseries, "lai_mean")
    f_veg_monthly   = np.where(
        ~np.isnan(lai_means),
        np.exp(-0.257 * lai_means / np.cos(SZA_RAD)),
        np.nan)
    f_veg_current   = (float(np.exp(-0.257 * mod_latest_val
                                    / np.cos(SZA_RAD)))
                       if mod_latest_val is not None else None)

    # ================================================================== #
    # Assembla risultato
    # ================================================================== #
    result = {}

    for idx in ["ndvi","evi","ndwi","fcover"]:
        mn, sd, nb = _monthly_stats_from_ts(ls_timeseries, idx)
        result[f"landsat_{idx}_monthly_mean"] = mn
        result[f"landsat_{idx}_monthly_std"]  = sd
        result[f"landsat_{idx}_monthly_nobs"] = nb
        result[f"landsat_{idx}_current"]      = ls_latest_val.get(idx)
        result[f"landsat_{idx}_current_date"] = ls_latest_date.get(idx)
        result[f"landsat_{idx}_latest_map"]   = ls_latest_map.get(idx)

    lai_mn, lai_sd, lai_nb = _monthly_stats_from_ts(
        mod_timeseries, "lai_mean")
    result["modis_lai_monthly_mean"]  = lai_mn
    result["modis_lai_monthly_std"]   = lai_sd
    result["modis_lai_monthly_nobs"]  = lai_nb
    result["modis_lai_current"]       = mod_latest_val
    result["modis_lai_current_date"]  = mod_latest_date_
    result["modis_lai_latest_map"]    = mod_latest_map

    result["f_veg_monthly"]           = f_veg_monthly
    result["f_veg_current"]           = f_veg_current

    result["landsat_timeseries"]      = ls_timeseries
    result["modis_lai_timeseries"]    = mod_timeseries

    result["lat"]                     = lat
    result["lon"]                     = lon
    result["r86"]                     = r86
    result["landsat_n_scenes_total"]  = len(ls_timeseries)
    result["modis_n_scenes_total"]    = len(mod_timeseries)
    result["cache_dir"]               = cache_dir
    result["months"]                  = ["Jan","Feb","Mar","Apr","May","Jun",
                                         "Jul","Aug","Sep","Oct","Nov","Dec"]
    return result


def _read_mod10cm_value(signed_item, lat, lon):
    """
    Legge il valore Snow_Cover_Monthly_CMG dal prodotto MOD10CM (0.05°/pixel).
    Restituisce float (0-100 %) oppure None se il pixel è invalido.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from rasterio.crs import CRS

    band_name = None
    for candidate in ["Snow_Cover_Monthly_CMG", "Snow_Cover_Daily_Tile",
                      "Snow_Cover_8-Day_Tile"]:
        if candidate in signed_item.assets:
            band_name = candidate
            break
    if band_name is None:
        for aname in signed_item.assets:
            if "snow" in aname.lower() or "Snow" in aname:
                band_name = aname
                break
    if band_name is None:
        return None

    href  = signed_item.assets[band_name].href
    wgs84 = CRS.from_epsg(4326)
    margin = 0.1  # ~2 pixel CMG

    try:
        with rasterio.open(href) as src:
            l, b, r, t = transform_bounds(
                wgs84, src.crs,
                lon - margin, lat - margin,
                lon + margin, lat + margin)
            win  = from_bounds(l, b, r, t, transform=src.transform)
            data = src.read(1, window=win)

        if data.size == 0:
            return None
        r0, c0 = data.shape[0] // 2, data.shape[1] // 2
        val = int(data[r0, c0])
        if 0 <= val <= 100:
            return float(val)
        if val == 200:          # lake ice → neve
            return 100.0
        return None
    except Exception:
        return None


def _get_snow_active_months(catalog, lat, lon, threshold_pct=5.0, verbose=True):
    """
    Usa MOD10CM (Terra Snow Cover Monthly CMG, 0.05°) per determinare quali mesi
    hanno snow_cover medio >= threshold_pct %.

    Restituisce un set di interi 1-12.  Se i dati non sono disponibili o nessun
    mese supera la soglia (sito tropicale/desertico), restituisce tutti i mesi.
    """
    import planetary_computer as _pc

    bbox = [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1]

    # Prova più nomi di collection per il prodotto mensile CMG.
    # "modis-10CM-061" non è un collection ID valido su Planetary Computer;
    # il prodotto reale è MOD10C2 (8-day CMG) → "modis-10C2-061"
    # oppure MOD10C3 (monthly CMG) → "modis-10C3-061".
    _CMG_COLLECTIONS = ["modis-10C2-061", "modis-10C3-061", "modis-10CM-061"]
    items = []
    for _coll in _CMG_COLLECTIONS:
        try:
            search = catalog.search(
                collections=[_coll],
                bbox=bbox,
                max_items=84,   # ~7 anni di mensili
            )
            items = list(search.items())
            if items:
                if verbose:
                    print(f"  MOD10CM usando collection: {_coll}", flush=True)
                break
        except Exception:
            continue

    if not items:
        if verbose:
            print("  MOD10CM: 0 scene (collection non trovata o nessun dato) "
                  "→ nessun filtro stagionale.", flush=True)
        return set(range(1, 13))

    monthly_vals = {m: [] for m in range(1, 13)}
    for item in items:
        dt_str = (item.properties.get("datetime")
                  or item.properties.get("start_datetime"))
        if not dt_str:
            continue
        try:
            dt_obj = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            continue
        try:
            signed = _pc.sign(item)
            val = _read_mod10cm_value(signed, lat, lon)
            if val is not None and not np.isnan(val):
                monthly_vals[dt_obj.month].append(val)
        except Exception:
            continue

    active = {m for m in range(1, 13)
              if monthly_vals[m] and float(np.mean(monthly_vals[m])) >= threshold_pct}

    if not active:
        if verbose:
            print(f"  MOD10CM: nessun mese ≥ {threshold_pct}% → sito senza neve, "
                  f"nessun filtro.", flush=True)
        return set(range(1, 13))

    if verbose:
        mn = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
        names = [mn[m - 1] for m in sorted(active)]
        print(f"  MOD10CM: mesi con neve ≥ {threshold_pct}%: {', '.join(names)}",
              flush=True)

    return active


def _pick_primary_tile(items, lat, lon, verbose=True):
    """
    Filtra una lista di STAC items tenendo solo quelli della tile MODIS
    il cui centro geografico è più vicino al sito (lat, lon).
    Scarta le tile sovrapposte ai bordi, dimezzando tipicamente il numero di scene.

    Strategia di raggruppamento (in ordine di priorità):
      1. proprietà STAC modis:horizontal-tile / modis:vertical-tile  ← stabile
      2. tile H/V estratta dall'item ID via regex  (es. "...h18v04...")
      3. arrotondamento SW bbox a 2° (fallback robusto vs variazioni fp)

    Il vecchio arrotondamento a 1° causava falsi split: se il catalogo PC
    riportava variazioni fp nel bbox SW (es. -5.14 vs -5.05), lo stesso
    tile veniva diviso in due gruppi → quello con 2 scene "vinceva" su
    quello con 506 scene valide perché il suo centro era leggermente più
    vicino al sito.
    """
    import re
    if not items:
        return items

    from collections import defaultdict

    def _tile_key(item):
        # 1. proprietà MODIS native
        h = item.properties.get("modis:horizontal-tile")
        v = item.properties.get("modis:vertical-tile")
        if h is not None and v is not None:
            return (int(h), int(v))
        # 2. regex sull'item ID  (es. "MOD10A2.A2020001.h18v04.061...")
        m = re.search(r'[._](h\d{2}v\d{2})[._]', item.id or "")
        if m:
            return m.group(1)   # stringa tipo "h18v04"
        # 3. bbox SW arrotondato a 2° (fallback)
        bbox = item.bbox
        if bbox is not None:
            return (round(bbox[0] / 2) * 2, round(bbox[1] / 2) * 2)
        return None

    tile_groups = defaultdict(list)
    for item in items:
        key = _tile_key(item)
        if key is None:
            continue
        tile_groups[key].append(item)

    if not tile_groups:
        return items

    # Per ogni gruppo calcola il centro medio del bbox (media di tutti gli items
    # → stabile anche se singoli bbox hanno lievi variazioni fp)
    best_key, best_dist = None, float("inf")
    for key, group in tile_groups.items():
        bboxes = [it.bbox for it in group if it.bbox is not None]
        if not bboxes:
            continue
        c_lon = sum((b[0] + b[2]) / 2 for b in bboxes) / len(bboxes)
        c_lat = sum((b[1] + b[3]) / 2 for b in bboxes) / len(bboxes)
        dist  = ((c_lon - lon) ** 2 + (c_lat - lat) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best_key = dist, key

    filtered = tile_groups[best_key]
    if verbose and len(tile_groups) > 1:
        bboxes = [it.bbox for it in filtered if it.bbox is not None]
        if bboxes:
            c_lon = sum((b[0] + b[2]) / 2 for b in bboxes) / len(bboxes)
            c_lat = sum((b[1] + b[3]) / 2 for b in bboxes) / len(bboxes)
        else:
            c_lon = c_lat = float("nan")
        n_out = len(items) - len(filtered)
        print(f"  Tile primaria: key={best_key}  centro ({c_lat:.1f}°N, {c_lon:.1f}°E)  "
              f"[{n_out} scene da tile secondarie scartate]", flush=True)

    return filtered


def _item_month(item):
    """Estrae il numero di mese (1-12) da un STAC item, oppure None."""
    dt_str = (item.properties.get("datetime")
              or item.properties.get("start_datetime"))
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).month
    except Exception:
        return None


def _read_modis_snow10a2_scene(signed_item, lat, lon, n_pixels=5):
    """
    Legge la copertura nevosa da una scena MODIS MOD10A2 (500m, 8-day composito).

    Il band 'Maximum_Snow_Extent' è categorico:
      25  = no snow
      50  = cloud  (invalido per il calcolo)
      200 = snow
      (altri valori = fill/oceano/lago/notte → invalidi)

    Restituisce snow_frac (0-100 %) calcolata sui soli pixel validi
    (snow + no-snow), oppure None se tutti i pixel sono nuvolosi/invalidi.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from rasterio.crs import CRS

    band_name = None
    for candidate in ["Maximum_Snow_Extent", "Eight_Day_Snow_Cover",
                      "NDSI_Snow_Cover"]:
        if candidate in signed_item.assets:
            band_name = candidate
            break
    if band_name is None:
        for aname in signed_item.assets:
            if "snow" in aname.lower() or "Snow" in aname:
                band_name = aname
                break
    if band_name is None:
        return None

    href  = signed_item.assets[band_name].href
    wgs84 = CRS.from_epsg(4326)
    c     = np.cos(np.radians(lat))
    margin_m   = (n_pixels / 2 + 0.5) * 500.0
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

        is_snow   = (data == 200)
        is_nosnow = (data == 25)
        is_valid  = is_snow | is_nosnow

        nr, nc = data.shape
        r0 = max(0, (nr - n_pixels) // 2)
        c0 = max(0, (nc - n_pixels) // 2)
        snow_crop  = is_snow [r0:r0 + n_pixels, c0:c0 + n_pixels]
        valid_crop = is_valid[r0:r0 + n_pixels, c0:c0 + n_pixels]

        n_valid = int(np.sum(valid_crop))
        if n_valid == 0:
            return None
        snow_frac = float(np.sum(snow_crop)) / n_valid * 100.0

        out = np.full((n_pixels, n_pixels), snow_frac)
        return out
    except Exception:
        return None


def _read_modis_snow_scene(signed_item, lat, lon, n_pixels=5):
    """
    Legge la copertura nevosa da una scena MODIS MOD10A1 (500m, giornaliero).

    Tenta i band names: 'NDSI_Snow_Cover' e 'CGF_NDSI_Snow_Cover'.
    Valori validi: 0–100 (% di copertura nevosa NDSI).
    Valori speciali (> 100) ignorati (notte, oceano, cloud, fill, etc.).

    Ritorna array (n_pixels × n_pixels) oppure None se fallisce.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from rasterio.crs import CRS

    band_name = None
    for candidate in ["NDSI_Snow_Cover", "CGF_NDSI_Snow_Cover",
                      "NDSI_Snow_Cover_Basic_QA"]:
        if candidate in signed_item.assets:
            band_name = "NDSI_Snow_Cover" if "NDSI_Snow_Cover" in signed_item.assets \
                        else candidate
            break
    # Se nessuno dei candidati è trovato, usa il primo asset disponibile
    # che contenga "Snow" nel nome
    if band_name is None:
        for aname in signed_item.assets:
            if "snow" in aname.lower() or "Snow" in aname:
                band_name = aname
                break
    if band_name is None:
        return None

    href  = signed_item.assets[band_name].href
    wgs84 = CRS.from_epsg(4326)
    c     = np.cos(np.radians(lat))
    margin_m   = (n_pixels / 2 + 0.5) * 500.0
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

        # Valori 0–100 = % copertura neve; tutto il resto = dato invalido
        valid = (data >= 0) & (data <= 100)
        snow  = np.where(valid, data.astype(float), np.nan)

        nr, nc = snow.shape
        r0     = max(0, (nr - n_pixels) // 2)
        c0     = max(0, (nc - n_pixels) // 2)
        crop   = snow[r0:r0+n_pixels, c0:c0+n_pixels]
        out    = np.full((n_pixels, n_pixels), np.nan)
        h_, w_ = min(crop.shape[0], n_pixels), min(crop.shape[1], n_pixels)
        out[:h_, :w_] = crop[:h_, :w_]
        return out
    except Exception:
        return None


def get_snow_cover(
    lat,
    lon,
    cache_dir,
    start_year        = MODIS_START_YEAR,
    n_pixels          = 5,
    snow_threshold_pct = 5.0,
    verbose           = True,
):
    """
    Scarica la copertura nevosa stagionale da MODIS MOD10A2.061
    (Terra Snow Cover 8-Day Global 500m) via Planetary Computer.

    Ottimizzazioni rispetto a MOD10A1:
      1. MOD10A2 (8-day composite) → ~8× meno scene rispetto al giornaliero
      2. Filtro stagionale via MOD10CM: scarica solo i mesi con snow_cover
         medio >= snow_threshold_pct % (climatologia mensile CMG 0.05°)
      3. Tile primaria: mantiene solo la tile MODIS il cui centro è più
         vicino al sito, scartando le tile sovrapposte ai bordi

    Cache separata da MOD10A1 (nome file contiene "10A2").

    Parameters
    ----------
    lat, lon           : coordinate WGS84
    cache_dir          : directory per cache
    start_year         : primo anno di download (default MODIS_START_YEAR)
    n_pixels           : array NxN di pixel MODIS centrati sul sito (default 5)
    snow_threshold_pct : soglia % per il filtro MOD10CM (default 5.0)
    verbose            : stampa progressi

    Returns
    -------
    dict con:
        snow_cover_monthly_pct  array (12,) copertura neve media mensile [%]
        snow_cover_monthly_std  array (12,) deviazione std mensile [%]
        snow_days_monthly       array (12,) periodi 8-day medi con snow > 50%
        snow_cover_annual_pct   float       media annuale [%]
        snow_days_annual        float       totale periodi/anno con neve
        snow_months             list        mesi (nome) con snow_days > 2
        n_scenes_total          int         scene in cache
        months                  list        nomi mesi
    """
    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        raise ImportError("pip install pystac-client planetary-computer")

    PRODUCT = "10A2"

    os.makedirs(cache_dir, exist_ok=True)
    now      = datetime.now(timezone.utc)
    date_end = now.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    snow_timeseries, snow_last_date = _load_snow_cache(
        cache_dir, lat, lon, start_year, PRODUCT)

    if snow_last_date:
        search_start = snow_last_date
        if verbose:
            print(f"Snow cache (MOD10A2): {len(snow_timeseries)} scenes, "
                  f"last={snow_last_date}. Fetching updates ...", flush=True)
    else:
        search_start = f"{start_year}-01-01"
        if verbose:
            print(f"Snow (MOD10A2): no cache, full download from {search_start} ...",
                  flush=True)

    # ------------------------------------------------------------------ #
    # Catalogo Planetary Computer
    # ------------------------------------------------------------------ #
    if verbose:
        print("Connecting to Planetary Computer (snow) ...", flush=True)
    catalog = pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    # ------------------------------------------------------------------ #
    # Filtro stagionale via MOD10CM
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"  Fetching MOD10CM climatology (threshold {snow_threshold_pct}%) ...",
              flush=True)
    active_months = _get_snow_active_months(
        catalog, lat, lon,
        threshold_pct=snow_threshold_pct,
        verbose=verbose,
    )

    # ------------------------------------------------------------------ #
    # Ricerca scene MOD10A2
    # ------------------------------------------------------------------ #
    # 0.05° (~5 km) garantisce intersezione affidabile con il tile MODIS 500m
    # senza restituire tile lontanissime; il vecchio 0.015° era borderline.
    bbox_snow = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05]

    search_snow = catalog.search(
        collections = ["modis-10A2-061"],
        bbox        = bbox_snow,
        datetime    = f"{search_start}/{date_end}",
        max_items   = 5000,
    )
    items_snow = list(search_snow.items())

    # Rimuovi scene già in cache
    if snow_last_date:
        items_snow = [it for it in items_snow
                      if (it.properties.get("datetime", "") or
                          it.properties.get("start_datetime", ""))
                         > snow_last_date]

    # Filtro tile primaria
    items_snow = _pick_primary_tile(items_snow, lat, lon, verbose=verbose)

    # Filtro stagionale (mesi non attivi → skip)
    if active_months != set(range(1, 13)):
        n_before   = len(items_snow)
        items_snow = [it for it in items_snow
                      if _item_month(it) in active_months]
        if verbose:
            print(f"  Filtro stagionale: {n_before} → {len(items_snow)} scene",
                  flush=True)

    if verbose:
        print(f"  Snow search range    : {search_start} → {date_end}",
              flush=True)
        print(f"  {len(items_snow)} new snow scenes to download", flush=True)

    # ------------------------------------------------------------------ #
    # Download sequenziale (evita OOM)
    # ------------------------------------------------------------------ #
    new_scenes     = []
    done           = 0
    n_items        = len(items_snow)
    SNOW_BATCH     = 50
    _snow_t0       = time.perf_counter()
    _snow_batch_t0 = _snow_t0

    def _proc_snow(item):
        import planetary_computer as _pc
        dt_str = (item.properties.get("datetime")
                  or item.properties.get("start_datetime"))
        if not dt_str:
            return None
        try:
            dt_obj = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            return None
        signed   = _pc.sign(item)
        snow_map = _read_modis_snow10a2_scene(signed, lat, lon, n_pixels)
        if snow_map is None:
            return None
        cover_mean = float(np.nanmean(snow_map))
        if np.isnan(cover_mean):
            return None
        return dt_obj, cover_mean

    for item in items_snow:
        res = _proc_snow(item)
        done += 1
        if res is None:
            continue
        dt_obj, cover_mean = res
        date_str = dt_obj.date().isoformat()
        new_scenes.append({
            "date"           : date_str,
            "year"           : dt_obj.year,
            "month"          : dt_obj.month,
            "snow_cover_pct" : cover_mean,
        })

        # Checkpoint ogni SNOW_BATCH scene valide
        if len(new_scenes) % SNOW_BATCH == 0:
            partial = sorted(snow_timeseries + new_scenes,
                             key=lambda x: x["date"])
            _save_snow_cache(cache_dir, lat, lon, start_year,
                             partial, partial[-1]["date"], PRODUCT)
            if verbose:
                _now       = time.perf_counter()
                batch_s    = _now - _snow_batch_t0
                elapsed_s  = _now - _snow_t0
                rate       = done / elapsed_s if elapsed_s > 0 else 0
                remaining  = (n_items - done) / rate if rate > 0 else 0
                print(
                    f"  Snow checkpoint: {done}/{n_items} scanned, "
                    f"{len(new_scenes)} valid  |  "
                    f"batch {batch_s:.0f}s  elapsed {elapsed_s:.0f}s  "
                    f"rate {rate*60:.1f} scene/min  ETA {remaining:.0f}s",
                    flush=True)
                _snow_batch_t0 = _now

    # ------------------------------------------------------------------ #
    # Merge e salva cache
    # ------------------------------------------------------------------ #
    snow_timeseries = snow_timeseries + new_scenes
    snow_timeseries.sort(key=lambda x: x["date"])
    new_last = (snow_timeseries[-1]["date"]
                if snow_timeseries else snow_last_date)
    if new_scenes:
        _save_snow_cache(cache_dir, lat, lon, start_year,
                         snow_timeseries, new_last, PRODUCT)
        if verbose:
            print(f"  Snow cache updated: {len(snow_timeseries)} total scenes",
                  flush=True)

    # ------------------------------------------------------------------ #
    # Aggregazione mensile
    # ------------------------------------------------------------------ #
    by_month = {m: [] for m in range(1, 13)}
    for row in snow_timeseries:
        v = row.get("snow_cover_pct")
        if v is not None and not np.isnan(float(v)):
            by_month[row["month"]].append(float(v))

    snow_cover_monthly_pct = np.array(
        [np.nanmean(by_month[m]) if by_month[m] else np.nan
         for m in range(1, 13)])
    snow_cover_monthly_std = np.array(
        [np.nanstd(by_month[m]) if len(by_month[m]) > 1 else 0.0
         for m in range(1, 13)])

    # Giorni medi al mese con snow_cover > 50%
    by_month_days = {m: [] for m in range(1, 13)}
    for row in snow_timeseries:
        v = row.get("snow_cover_pct")
        if v is not None:
            # Raggruppa per anno+mese e conta giorni con > 50%
            key = (row["year"], row["month"])
            by_month_days[row["month"]].append(float(v) > 50.0)

    # Media giorni di neve per mese
    by_month_ndays = {m: [] for m in range(1, 13)}
    for row in snow_timeseries:
        v = row.get("snow_cover_pct")
        if v is not None:
            by_month_ndays[row["month"]].append(float(v) > 50.0)

    # Per ogni mese: aggrega per anno poi media tra anni
    from collections import defaultdict
    per_year_month = defaultdict(list)
    for row in snow_timeseries:
        v = row.get("snow_cover_pct")
        if v is not None:
            per_year_month[(row["year"], row["month"])].append(float(v) > 50.0)

    snow_days_per_year_month = {}
    for (yr, mo), flags in per_year_month.items():
        snow_days_per_year_month[(yr, mo)] = sum(flags)

    monthly_snow_days = np.zeros(12)
    for m in range(1, 13):
        days_list = [v for (yr, mo), v in snow_days_per_year_month.items()
                     if mo == m]
        monthly_snow_days[m - 1] = float(np.mean(days_list)) if days_list else 0.0

    months_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    return dict(
        snow_cover_monthly_pct = snow_cover_monthly_pct,
        snow_cover_monthly_std = snow_cover_monthly_std,
        snow_days_monthly      = monthly_snow_days,
        snow_cover_annual_pct  = float(np.nanmean(snow_cover_monthly_pct)),
        snow_days_annual       = float(np.sum(monthly_snow_days)),
        snow_months            = [months_names[i] for i in range(12)
                                  if monthly_snow_days[i] > 2],
        n_scenes_total         = len(snow_timeseries),
        months                 = months_names,
    )


def report_snow_cover(snow):
    """Stampa tabellare della copertura nevosa stagionale."""
    M = snow["months"]
    w = 72
    L = [
        "=" * w,
        "SNOW COVER — MODIS MOD10A2.061 (Terra 500m 8-Day)",
        "=" * w,
        f"  Total scenes     : {snow['n_scenes_total']}",
        f"  Annual snow cover: {snow['snow_cover_annual_pct']:.1f} %",
        f"  Snow periods/yr  : {snow['snow_days_annual']:.0f} 8-day periods/yr  "
        f"(periods with snow_cover > 50%)",
    ]
    if snow['snow_months']:
        L.append(f"  Snowy months     : {', '.join(snow['snow_months'])}"
                 f"  (>2 snowy 8-day periods/month)")
    else:
        L.append("  Snowy months     : none  (<15 snow-days per month)")
    L.append("")
    hdr = "  " + " " * 10 + "  ".join(f"{m:>5}" for m in M)
    L.append(hdr)
    L.append("  " + "-" * (w - 2))
    mn = snow["snow_cover_monthly_pct"]
    sd = snow["snow_cover_monthly_std"]
    nd = snow["snow_days_monthly"]
    mn_s = "  ".join(f"{v:5.1f}" if not np.isnan(v) else "  N/A" for v in mn)
    sd_s = "  ".join(f"{v:5.1f}" if not np.isnan(v) else "  N/A" for v in sd)
    nd_s = "  ".join(f"{v:5.1f}" for v in nd)
    L.append(f"  {'Cover [%]':<10} {mn_s}")
    L.append(f"  {'Std [%]':<10} {sd_s}")
    L.append(f"  {'Days>50%':<10} {nd_s}")
    L.append("=" * w)
    return "\n".join(L)


def report_vegetation(res):
    M  = res["months"]
    w  = 72
    L  = ["="*w,
          "VEGETATION INDICES — Monthly Climatology",
          "="*w,
          f"  Landsat scenes : {res['landsat_n_scenes_total']}",
          f"  MODIS scenes   : {res['modis_n_scenes_total']}",
          ""]
    hdr = "  " + " "*10 + "  ".join(f"{m:>5}" for m in M)
    L.append(hdr)
    L.append("  " + "-"*(w-2))
    for idx in ["ndvi","evi","ndwi","fcover"]:
        mn = res[f"landsat_{idx}_monthly_mean"]
        sd = res[f"landsat_{idx}_monthly_std"]
        nb = res[f"landsat_{idx}_monthly_nobs"]
        mn_s = "  ".join(f"{v:5.3f}" if not np.isnan(v) else "  N/A" for v in mn)
        sd_s = "  ".join(f"{v:5.3f}" if not np.isnan(v) else "  N/A" for v in sd)
        cur  = res[f"landsat_{idx}_current"]
        cur_d= res[f"landsat_{idx}_current_date"]
        L.append(f"  {idx.upper():<8} mean  {mn_s}")
        L.append(f"           std   {sd_s}")
        L.append(f"           nobs  {'  '.join(f'{v:5d}' for v in nb)}")
        L.append(f"           curr  {cur:.3f} ({cur_d})"
                 if cur is not None else "           curr  N/A")
        L.append("")
    mn = res["modis_lai_monthly_mean"]
    mn_s = "  ".join(f"{v:5.2f}" if not np.isnan(v) else "  N/A" for v in mn)
    cur  = res["modis_lai_current"]
    cur_d= res["modis_lai_current_date"]
    L.append(f"  {'LAI':<8} mean  {mn_s}")
    L.append(f"           curr  {cur:.2f} ({cur_d})"
             if cur is not None else "           curr  N/A")
    L.append("")
    fv   = res["f_veg_monthly"]
    fv_s = "  ".join(f"{v:5.3f}" if not np.isnan(v) else "  N/A" for v in fv)
    L.append(f"  {'f_veg':<8}       {fv_s}")
    L.append("="*w)
    return "\n".join(L)


def plot_snow_cover(snow, path, site_name=""):
    """
    Due pannelli:
      Sinistra : copertura neve mensile [%] con barre di errore (std)
      Destra   : periodi 8-day con neve > 50% per mese
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STYLE = {
        "figure.dpi": 100, "figure.facecolor": "white",
        "axes.facecolor": "#f8f8f6", "axes.grid": True,
        "grid.color": "white", "grid.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    }

    M  = snow.get("months", ["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"])
    x  = np.arange(1, 13)
    mn = np.asarray(snow["snow_cover_monthly_pct"], dtype=float)
    sd = np.asarray(snow["snow_cover_monthly_std"], dtype=float)
    nd = np.asarray(snow["snow_days_monthly"],      dtype=float)

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                        facecolor="white")

        # --- copertura neve % ---
        bar_colors = ["#92c5de" if v < 30 else
                      ("#4393c3" if v < 70 else "#2166ac")
                      for v in mn]
        bars = ax1.bar(x, mn, color=bar_colors, alpha=0.85,
                       edgecolor="white", width=0.7)
        # error bars
        valid = ~np.isnan(mn) & ~np.isnan(sd)
        ax1.errorbar(x[valid], mn[valid], yerr=sd[valid],
                     fmt="none", color="#555", capsize=4, lw=1.2, zorder=5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(M, fontsize=9)
        ax1.set_ylabel("Snow cover [%]")
        ax1.set_ylim(0, min(100, np.nanmax(mn + sd) * 1.15 + 5))
        ax1.set_xlim(0.4, 12.6)
        ax1.set_title(
            f"Copertura neve mensile  ({snow['n_scenes_total']} scene)\n"
            f"Annuale: {snow['snow_cover_annual_pct']:.1f} %",
            fontsize=11,
        )
        # annotazione mesi nevosi
        snowy = snow.get("snow_months", [])
        if snowy:
            ax1.text(0.02, 0.97,
                     "Mesi nevosi: " + ", ".join(snowy),
                     transform=ax1.transAxes, va="top", fontsize=9,
                     color="#1a6faf")

        # --- periodi 8-day ---
        bar_c2 = ["#d1e5f0" if v < 1 else
                  ("#4dac26" if v < 3 else "#1a6faf")
                  for v in nd]
        ax2.bar(x, nd, color=bar_c2, alpha=0.85,
                edgecolor="white", width=0.7)
        ax2.axhline(2, color="#b2182b", ls="--", lw=1.3,
                    label="Soglia 2 periodi/mese")
        ax2.set_xticks(x)
        ax2.set_xticklabels(M, fontsize=9)
        ax2.set_ylabel("Periodi 8-day con neve > 50%")
        ax2.set_xlim(0.4, 12.6)
        ax2.set_ylim(0, None)
        ax2.legend(fontsize=9)
        ax2.set_title(
            f"Periodi nevosi mensili\n"
            f"Totale/anno: {snow['snow_days_annual']:.1f} periodi",
            fontsize=11,
        )

        fig.suptitle(
            f"Snow Cover — MODIS MOD10A2.061  |  {site_name}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(path, dpi=100)
        plt.close(fig)

    print(f"  Saved: {path}")
