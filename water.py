"""
compute_water_eta
=================
Calcola il coefficiente η di alterazione del conteggio neutroni CRNS
dovuto alla presenza di acqua superficiale nel footprint del sensore.

Fisica
------
L'acqua libera modera i neutroni epitermali molto più efficacemente
del suolo umido. Su acqua libera il conteggio scende a circa f_water
(~3%) rispetto al suolo asciutto (Zreda et al. 2012).

Per ogni pixel i nel footprint, il fattore di scala del conteggio è:

    scale_i = f_water * occ_i + 1.0 * (1 - occ_i)
            = 1 - occ_i * (1 - f_water)

dove occ_i è la frazione temporale di copertura idrica [0-1]
(occurrence JRC / 100).

Il coefficiente η (riduzione relativa del conteggio) è:

    η = 1 - Σ W(r_i) * scale_i / Σ W(r_i)
      = (1 - f_water) * Σ W(r_i) * occ_i / Σ W(r_i)

La correzione del conteggio osservato è:

    N_corrected = N_observed / (1 - η)

η = 0  → nessuna acqua nel footprint, nessuna correzione
η > 0  → acqua presente, N_corrected > N_observed

Sorgente dati
-------------
JRC Global Surface Water v1.4 (Pekel et al. 2016, Nature)
- Layer: occurrence [0-100%] = frequenza copertura 1984-2021
- Risoluzione: 1/3600° ≈ 30m
- Formato: Cloud-Optimized GeoTIFF (COG)
- Accesso: Google Cloud Storage (pubblico, no auth)
  URL: https://storage.googleapis.com/global-surface-water/
       downloads2021/occurrence/occurrence_{tile}v1_4_2021.tif

Download
--------
Windowed read via rasterio su COG: scarica solo i blocchi
corrispondenti alla finestra spaziale richiesta (~10-100 KB),
non l'intera tile (~500 MB).

Cache locale: il crop viene salvato come .npz con hash delle
coordinate e del raggio — identica alla strategia DEM già usata
nella pipeline.

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import os
import json
import hashlib
import numpy as np


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

JRC_BASE_URL   = ("https://storage.googleapis.com/global-surface-water"
                  "/downloads2021/occurrence")
JRC_NODATA     = 255     # valore nodata nel GeoTIFF JRC
JRC_PIXEL_DEG  = 1/3600  # risoluzione angolare [deg]

# Margine spaziale oltre r86 per il crop [m]
# Garantisce che i pixel ai bordi del footprint siano inclusi
CROP_MARGIN_M  = 50.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jrc_tile_name(lon, lat):
    """
    Nome della tile JRC 10°×10° che contiene il punto (lon, lat).
    Naming convention: corner SW della tile.
    Es. lon=11.86, lat=46.92 -> 'occurrence_10E_40N'
    """
    lon_tile = int(np.floor(lon / 10) * 10)
    lat_tile = int(np.floor(lat / 10) * 10)
    ew = 'E' if lon_tile >= 0 else 'W'
    ns = 'N' if lat_tile >= 0 else 'S'
    return f"occurrence_{abs(lon_tile)}{ew}_{abs(lat_tile)}{ns}"


def _jrc_url(tile_name):
    return f"{JRC_BASE_URL}/{tile_name}v1_4_2021.tif"


def _cache_key(lat, lon, radius_m):
    tag = f"jrc_occ_{lat:.6f}_{lon:.6f}_{radius_m:.1f}"
    return hashlib.sha256(tag.encode()).hexdigest()[:16]


def _cache_paths(cache_dir, lat, lon, radius_m):
    key  = _cache_key(lat, lon, radius_m)
    base = os.path.join(cache_dir, f"jrc_cache_{key}")
    return base + ".npz", base + ".json"


def _save_cache(cache_dir, lat, lon, radius_m,
                occ_crop, lons_crop, lats_crop, tile_name):
    npz, meta = _cache_paths(cache_dir, lat, lon, radius_m)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(npz,
                        occurrence=occ_crop,
                        lons=lons_crop,
                        lats=lats_crop)
    with open(meta, 'w') as f:
        json.dump(dict(lat=lat, lon=lon, radius_m=radius_m,
                       tile=tile_name,
                       shape=list(occ_crop.shape)), f, indent=2)
    print(f"   JRC cache saved -> {npz}", flush=True)


def _load_cache(cache_dir, lat, lon, radius_m):
    npz, meta = _cache_paths(cache_dir, lat, lon, radius_m)
    if not (os.path.exists(npz) and os.path.exists(meta)):
        return None, None, None, None
    with open(meta) as f:
        m = json.load(f)
    if (abs(m['lat'] - lat) > 1e-7 or abs(m['lon'] - lon) > 1e-7
            or abs(m['radius_m'] - radius_m) > 1.0):
        return None, None, None, None
    d = np.load(npz)
    print(f"   JRC loaded from cache: {npz}", flush=True)
    return d['occurrence'], d['lons'], d['lats'], m['tile']


def _weight_radial(r, r86):
    """W(r) = exp(-r / lambda),  lambda = r86 / 3."""
    lam = r86 / 3.0
    return np.where(r < 1e-3, 0.0, np.exp(-r / lam))


# ---------------------------------------------------------------------------
# Download COG con windowed read
# ---------------------------------------------------------------------------

def _download_jrc_crop(lat, lon, radius_m):
    """
    Scarica il crop JRC occurrence attorno a (lat, lon) entro radius_m.

    Usa rasterio windowed read su COG — scarica solo i blocchi necessari
    (tipicamente 10-100 KB, non la tile intera da ~500 MB).

    Returns
    -------
    occ_crop  : 2D uint8 array, occurrence [0-100], 255=nodata->0
    lons_crop : 1D array, longitudine centro pixel [deg]
    lats_crop : 1D array, latitudine centro pixel [deg]
    tile_name : str
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        raise ImportError("rasterio non installato. "
                          "pip install rasterio")

    tile_name = _jrc_tile_name(lon, lat)
    url       = _jrc_url(tile_name)

    # Bounding box in gradi geografici
    c         = np.cos(np.radians(lat))
    margin_deg_lon = (radius_m + CROP_MARGIN_M) / (111320.0 * c)
    margin_deg_lat = (radius_m + CROP_MARGIN_M) / 111320.0

    west  = lon - margin_deg_lon
    east  = lon + margin_deg_lon
    south = lat - margin_deg_lat
    north = lat + margin_deg_lat

    print(f"   JRC: tile={tile_name}  crop={west:.4f}–{east:.4f}E "
          f"{south:.4f}–{north:.4f}N", flush=True)

    with rasterio.open(url) as src:
        window = from_bounds(west, south, east, north,
                             transform=src.transform)
        # Legge solo la finestra — rasterio gestisce i range request HTTP
        data   = src.read(1, window=window)
        # Calcola coordinate centro pixel per la finestra letta
        win_tf = src.window_transform(window)

    # Finestra vuota: tile non copre l'area richiesta -> nessuna acqua
    if data.size == 0:
        print(f"   JRC: empty window (tile {tile_name} does not cover area) "
              f"-> assuming no surface water", flush=True)
        lons_crop = np.array([lon])
        lats_crop = np.array([lat])
        return np.zeros((1, 1), dtype=np.uint8), lons_crop, lats_crop, tile_name

    nr, nc = data.shape

    # Coordinate centro pixel
    col_idx  = np.arange(nc)
    row_idx  = np.arange(nr)
    # win_tf * (col+0.5, row+0.5) = centro pixel
    lons_crop = win_tf.c + (col_idx + 0.5) * win_tf.a   # lon
    lats_crop = win_tf.f + (row_idx + 0.5) * win_tf.e   # lat (negativo -> N->S)

    # Sostituisce nodata con 0 (nessuna acqua osservata)
    occ_crop = np.where(data == JRC_NODATA, 0, data).astype(np.uint8)

    print(f"   JRC: crop shape={data.shape}  "
          f"size={data.nbytes/1024:.1f} KB  "
          f"max_occ={occ_crop.max()}%", flush=True)

    return occ_crop, lons_crop, lats_crop, tile_name


# ---------------------------------------------------------------------------
# Ricampionamento sulla griglia DEM
# ---------------------------------------------------------------------------

def _resample_to_dem_grid(occ_crop, lons_crop, lats_crop,
                           dx_grid, dy_grid, lat0, lon0):
    """
    Ricampiona il crop JRC (griglia lat/lon) sulla griglia DEM
    (griglia metrica centrata sul sensore).

    Usa nearest-neighbor — sufficiente per dati categorici/occurrence
    a risoluzione simile (~30m).

    Parameters
    ----------
    occ_crop           : 2D array (nr_jrc, nc_jrc), occurrence [0-100]
    lons_crop, lats_crop : 1D arrays, coordinate centro pixel JRC
    dx_grid, dy_grid   : 2D arrays, offset metrico della griglia DEM
    lat0, lon0         : coordinate sensore

    Returns
    -------
    occ_dem : 2D array, stessa shape di dx_grid, occurrence ricampionata
    """
    from scipy.spatial import cKDTree

    c = np.cos(np.radians(lat0))

    # Coordinate metriche dei centri pixel JRC (offset dal sensore)
    LONS, LATS = np.meshgrid(lons_crop, lats_crop)
    dx_jrc = (LONS - lon0) * 111320.0 * c
    dy_jrc = (LATS - lat0) * 111320.0

    pts_jrc = np.column_stack([dx_jrc.ravel(), dy_jrc.ravel()])
    tree    = cKDTree(pts_jrc)

    pts_dem = np.column_stack([dx_grid.ravel(), dy_grid.ravel()])
    _, idx  = tree.query(pts_dem)

    occ_dem = occ_crop.ravel()[idx].reshape(dx_grid.shape).astype(float)
    return occ_dem


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def compute_water_eta(
    lat,
    lon,
    dx_grid,
    dy_grid,
    dist_grid,
    r86,
    cache_dir,
    f_water   = 0.03,    # rapporto conteggio acqua/suolo secco [-]
):
    """
    Calcola η — coefficiente di alterazione del conteggio CRNS
    dovuto all'acqua superficiale nel footprint.

    η = (1 - f_water) * Σ W(r_i) * occ_i / Σ W(r_i)

    Dove occ_i è la fraction temporale di copertura idrica [0-1]
    dal layer JRC occurrence (1984-2021).

    N_corrected = N_observed / (1 - η)

    Parameters
    ----------
    lat, lon    : coordinate sensore WGS84 [deg]
    dx_grid     : 2D array, offset easting dalla clip_dem_to_radius [m]
    dy_grid     : 2D array, offset northing [m]
    dist_grid   : 2D array, distanza dal sensore [m]
    r86         : raggio footprint [m]
    cache_dir   : directory per cache locale (es. OUTPUT_DIR)
    f_water     : rapporto conteggio su acqua libera vs suolo secco
                  default 0.03 (Zreda et al. 2012)

    Returns
    -------
    dict con:
        eta                 : float [0-1], coefficiente di alterazione
        N_correction_factor : float, fattore moltiplicativo per N_corrected
                              = 1/(1-eta).  Se eta=0 -> 1.0 (nessuna correzione)
        occ_weighted_mean   : float [0-1], media pesata W(r) di occurrence
                              = eta / (1 - f_water)
        occ_map_dem         : 2D array, occurrence ricampionata sulla griglia DEM
        water_mask_fp       : 2D bool array, pixel con occurrence > 0 nel footprint
        pixels_water_fp     : int, pixel con acqua nel footprint
        pixels_total_fp     : int, pixel totali nel footprint
        water_area_m2       : float, area stimata con acqua nel footprint [m²]
        tile_name           : str, nome tile JRC usata
        from_cache          : bool
        f_water_used        : float
    """

    # ------------------------------------------------------------------ #
    # 1. Cache
    # ------------------------------------------------------------------ #
    occ_crop, lons_crop, lats_crop, tile_name = \
        _load_cache(cache_dir, lat, lon, r86)
    from_cache = occ_crop is not None

    if not from_cache:
        print("   JRC: no cache, downloading ...", flush=True)
        occ_crop, lons_crop, lats_crop, tile_name = \
            _download_jrc_crop(lat, lon, r86)
        _save_cache(cache_dir, lat, lon, r86,
                    occ_crop, lons_crop, lats_crop, tile_name)

    # ------------------------------------------------------------------ #
    # 2. Ricampiona sulla griglia DEM
    # ------------------------------------------------------------------ #
    occ_dem = _resample_to_dem_grid(
        occ_crop, lons_crop, lats_crop,
        dx_grid, dy_grid, lat, lon)

    # occurrence in [0,1]
    occ_frac = occ_dem / 100.0

    # ------------------------------------------------------------------ #
    # 3. Calcolo η dentro il footprint
    # ------------------------------------------------------------------ #
    fp_mask = (dist_grid <= r86)

    r_fp    = dist_grid[fp_mask]
    occ_fp  = occ_frac[fp_mask]
    W_fp    = _weight_radial(r_fp, r86)
    W_sum   = W_fp.sum()

    if W_sum > 0:
        occ_weighted = float(np.sum(W_fp * occ_fp) / W_sum)
    else:
        occ_weighted = 0.0

    eta                = (1.0 - f_water) * occ_weighted
    N_correction_factor = 1.0 / (1.0 - eta) if eta < 1.0 else np.inf

    # ------------------------------------------------------------------ #
    # 4. Statistiche descrittive
    # ------------------------------------------------------------------ #
    water_mask_fp  = fp_mask & (occ_dem > 0)
    pixels_water   = int(water_mask_fp.sum())
    pixels_total   = int(fp_mask.sum())

    # Area pixel in m² dalla griglia DEM
    nr, nc = dx_grid.shape
    dpx    = abs(float(np.nanmedian(np.diff(dx_grid[nr // 2, :]))))
    dpy    = abs(float(np.nanmedian(np.diff(dy_grid[:, nc // 2]))))
    if dpx < 1: dpx = 30.0
    if dpy < 1: dpy = 30.0
    pixel_area_m2  = dpx * dpy
    water_area_m2  = float(pixels_water * pixel_area_m2)

    return dict(
        eta                  = float(eta),
        N_correction_factor  = float(N_correction_factor),
        occ_weighted_mean    = float(occ_weighted),
        occ_map_dem          = occ_dem,
        water_mask_fp        = water_mask_fp,
        pixels_water_fp      = pixels_water,
        pixels_total_fp      = pixels_total,
        water_area_m2        = water_area_m2,
        tile_name            = tile_name,
        from_cache           = from_cache,
        f_water_used         = float(f_water),
    )


def report_water_eta(res):
    """Stampa leggibile dei risultati di compute_water_eta."""
    w = 62
    eta = res['eta']
    L = [
        "=" * w,
        "JRC SURFACE WATER — Neutron Count Correction",
        "=" * w,
        f"  Tile            : {res['tile_name']}",
        f"  From cache      : {res['from_cache']}",
        f"  f_water used    : {res['f_water_used']:.3f}  "
        f"(Zreda 2012)",
        "",
        f"  Occurrence weighted mean : {res['occ_weighted_mean']:.4f}  "
        f"({res['occ_weighted_mean']*100:.2f}%)",
        f"  Water pixels / total     : "
        f"{res['pixels_water_fp']} / {res['pixels_total_fp']}",
        f"  Water area in footprint  : "
        f"{res['water_area_m2']:.0f} m²",
        "",
        f"  eta             : {eta:.4f}",
        f"  Interpretation  : {'negligible (<0.1%)' if eta < 0.001 else 'minor (<1%)' if eta < 0.01 else 'significant (>1%)' if eta < 0.05 else 'LARGE (>5%) — correction required'}",
        "",
        f"  N_correction_factor : {res['N_correction_factor']:.4f}",
        f"  N_corrected = N_observed x {res['N_correction_factor']:.4f}",
        "=" * w,
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Smoke test con dati mock
# ---------------------------------------------------------------------------

def _mock_download(lat, lon, radius_m):
    """Genera crop JRC sintetico per test offline."""
    c          = np.cos(np.radians(lat))
    n_lon      = int(2 * (radius_m + CROP_MARGIN_M) / (111320.0 * c * JRC_PIXEL_DEG)) + 10
    n_lat      = int(2 * (radius_m + CROP_MARGIN_M) / (111320.0 * JRC_PIXEL_DEG)) + 10

    margin_lon = (radius_m + CROP_MARGIN_M) / (111320.0 * c)
    margin_lat = (radius_m + CROP_MARGIN_M) / 111320.0

    lons_crop  = np.linspace(lon - margin_lon, lon + margin_lon, n_lon)
    lats_crop  = np.linspace(lat + margin_lat, lat - margin_lat, n_lat)

    LONS, LATS = np.meshgrid(lons_crop, lats_crop)
    dx_jrc     = (LONS - lon) * 111320.0 * c
    dy_jrc     = (LATS - lat) * 111320.0
    dist_jrc   = np.sqrt(dx_jrc**2 + dy_jrc**2)

    occ = np.zeros((n_lat, n_lon), dtype=np.uint8)
    # Fiume a est
    river = (dx_jrc > 80) & (dx_jrc < 130) & (np.abs(dy_jrc) < 40)
    occ[river] = 92
    # Laghetto a nord-ovest
    lake  = np.sqrt((dx_jrc + 60)**2 + (dy_jrc - 80)**2) < 35
    occ[lake]  = 100

    return occ, lons_crop, lats_crop, _jrc_tile_name(lon, lat)


if __name__ == "__main__":
    import tempfile, time

    print("SMOKE TEST — dati mock (offline)")
    print("=" * 62)

    # Griglia DEM sintetica
    n      = 60
    r_max  = 300.0
    x1     = np.linspace(-r_max, r_max, n)
    XX, YY = np.meshgrid(x1, x1)
    dist   = np.sqrt(XX**2 + YY**2)
    lat0, lon0 = 46.925, 11.861
    r86    = 130.0

    # Patch _download_jrc_crop con mock
    import unittest.mock as mock

    with mock.patch(
        f'{__name__}._download_jrc_crop',
        side_effect=lambda la, lo, r: _mock_download(la, lo, r)
    ):
        with tempfile.TemporaryDirectory() as tmpdir:

            # --- Prima chiamata: scarica (mock) e salva cache ---
            print("\nPrima chiamata (download mock):")
            t0  = time.perf_counter()
            res = compute_water_eta(lat0, lon0, XX, YY, dist, r86,
                                    cache_dir=tmpdir)
            dt  = time.perf_counter() - t0
            print(report_water_eta(res))
            print(f"  Wall time: {dt*1000:.1f} ms")
            assert not res['from_cache'], "Prima chiamata non deve usare cache"

            # --- Seconda chiamata: dalla cache ---
            print("\nSeconda chiamata (da cache):")
            t0  = time.perf_counter()
            res2 = compute_water_eta(lat0, lon0, XX, YY, dist, r86,
                                     cache_dir=tmpdir)
            dt2  = time.perf_counter() - t0
            print(f"  eta={res2['eta']:.4f}  from_cache={res2['from_cache']}")
            print(f"  Wall time: {dt2*1000:.1f} ms")
            assert res2['from_cache'],        "Seconda chiamata deve usare cache"
            assert abs(res2['eta'] - res['eta']) < 1e-9, "Cache diversa!"

            # --- Verifica fisica ---
            assert 0.0 <= res['eta'] <= 1.0,  "eta fuori range [0,1]"
            assert res['N_correction_factor'] >= 1.0, \
                "N_correction deve essere >= 1"
            assert res['pixels_water_fp'] <= res['pixels_total_fp'], \
                "pixel acqua > pixel totali"

            print("\n  PASS — tutti i test superati")
