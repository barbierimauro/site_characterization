"""
lulc_constants.py
=================
Costanti fisiche, lookup tables e helpers condivisi
per la pipeline LULC CRNS.

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import os
import hashlib
import json
import gzip
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# WorldCover — classi e parametri fisici
# f_H: fattore idrogeno equivalente relativo a suolo baseline (= 1.0)
# Fonti: Schrön 2017, Iwema 2017, Franz 2012
# ---------------------------------------------------------------------------

WC_CLASSES = {
    10 : {"name": "Tree cover",       "f_H": 1.50, "color": "#1a9641"},
    20 : {"name": "Shrubland",        "f_H": 1.15, "color": "#a6d96a"},
    30 : {"name": "Grassland",        "f_H": 0.95, "color": "#ffffb2"},
    40 : {"name": "Cropland",         "f_H": 1.00, "color": "#fdae61"},
    50 : {"name": "Built-up",         "f_H": 0.05, "color": "#d7191c"},
    60 : {"name": "Bare/sparse veg.", "f_H": 0.03, "color": "#b2b2b2"},
    70 : {"name": "Snow and ice",     "f_H": 3.00, "color": "#e0f3f8"},
    80 : {"name": "Water",            "f_H": 4.00, "color": "#2c7bb6"},
    90 : {"name": "Wetland",          "f_H": 2.50, "color": "#abd9e9"},
    95 : {"name": "Mangrove",         "f_H": 1.80, "color": "#004529"},
    100: {"name": "Moss/lichen",      "f_H": 1.20, "color": "#d9f0a3"},
    0  : {"name": "No data",          "f_H": 1.00, "color": "#ffffff"},
}
WC_NODATA       = 0
WC_BASELINE_FH  = 1.0

# ---------------------------------------------------------------------------
# OSM — f_H per tipo di oggetto
# ---------------------------------------------------------------------------

OSM_FH = {
    # highway surface
    ("highway", "asphalt")       : 0.03,
    ("highway", "concrete")      : 0.06,
    ("highway", "paved")         : 0.04,
    ("highway", "gravel")        : 0.40,
    ("highway", "unpaved")       : 0.60,
    ("highway", "dirt")          : 0.70,
    ("highway", "grass")         : 0.90,
    # highway tipo (senza surface)
    ("highway_type", "motorway")    : 0.03,
    ("highway_type", "trunk")       : 0.03,
    ("highway_type", "primary")     : 0.04,
    ("highway_type", "secondary")   : 0.04,
    ("highway_type", "tertiary")    : 0.05,
    ("highway_type", "residential") : 0.10,
    ("highway_type", "track")       : 0.55,
    ("highway_type", "path")        : 0.80,
    ("highway_type", "footway")     : 0.75,
    ("highway_type", "cycleway")    : 0.30,
    ("highway_type", "default")     : 0.50,
    # building
    ("building", "*")            : 0.00,
    # railway
    ("railway", "*")             : 0.15,
    # natural
    ("natural", "bare_rock")     : 0.01,
    ("natural", "scree")         : 0.05,
    ("natural", "glacier")       : 3.00,
    ("natural", "water")         : 4.00,
    ("natural", "wetland")       : 2.50,
    ("natural", "wood")          : 1.50,
    ("natural", "scrub")         : 1.15,
    ("natural", "heath")         : 1.10,
    ("natural", "grassland")     : 0.95,
    # landuse
    ("landuse", "forest")        : 1.50,
    ("landuse", "meadow")        : 0.95,
    ("landuse", "grass")         : 0.90,
    ("landuse", "farmland")      : 1.00,
    ("landuse", "residential")   : 0.30,
    ("landuse", "industrial")    : 0.10,
    ("landuse", "commercial")    : 0.10,
    ("landuse", "railway")       : 0.15,
    ("landuse", "quarry")        : 0.05,
}

# Larghezza stimata per highway [m] — usata per LineString -> area
HIGHWAY_WIDTH = {
    "motorway": 14.0, "trunk": 12.0, "primary": 9.0,
    "secondary": 7.0, "tertiary": 6.0, "residential": 5.0,
    "unclassified": 4.0, "track": 3.0, "path": 1.5,
    "footway": 1.5, "cycleway": 2.0, "service": 4.0,
    "steps": 2.0, "default": 3.0,
}

# Colori per la mappa OSM
OSM_COLORS = {
    "highway" : "#e8b87a",
    "building": "#888888",
    "railway" : "#555555",
    "natural" : "#7ec850",
    "landuse" : "#b5d29e",
    "water"   : "#6baed6",
}

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ---------------------------------------------------------------------------
# Helpers condivisi
# ---------------------------------------------------------------------------

def site_hash(lat, lon, r86=0.0):
    tag = f"{lat:.5f}_{lon:.5f}_{r86:.1f}"
    return hashlib.sha256(tag.encode()).hexdigest()[:16]


def weight_radial(r, r86):
    """W(r) = exp(-r / (r86/3))."""
    lam = r86 / 3.0
    return np.where(r < 1e-3, 0.0, np.exp(-r / lam))


def pixel_size(dx_grid, dy_grid):
    """Stima risoluzione pixel [m] dalla griglia metrica."""
    nr, nc = dx_grid.shape
    dpx = abs(float(np.nanmedian(np.diff(dx_grid[nr // 2, :]))))
    dpy = abs(float(np.nanmedian(np.diff(dy_grid[:, nc // 2]))))
    if dpx < 1: dpx = 30.0
    if dpy < 1: dpy = 30.0
    return dpx, dpy


"""
lulc_worldcover.py
==================
Download, cache e calcolo kappa_lulc da ESA WorldCover 10m.

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""



# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _wc_cache_path(cache_dir, lat, lon, r86):
    h = site_hash(lat, lon, r86)
    return os.path.join(cache_dir, f"worldcover_{h}.npz")


def load_wc_cache(cache_dir, lat, lon, r86):
    p = _wc_cache_path(cache_dir, lat, lon, r86)
    if not os.path.exists(p):
        return None, None, None
    d = np.load(p)
    return d["wc_map"], d["dx_1d"], d["dy_1d"]


def save_wc_cache(cache_dir, lat, lon, r86, wc_map, dx_1d, dy_1d):
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        _wc_cache_path(cache_dir, lat, lon, r86),
        wc_map=wc_map, dx_1d=dx_1d, dy_1d=dy_1d)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_worldcover(lat, lon, r86, verbose=True):
    """
    Windowed read COG ESA WorldCover 2021 da Planetary Computer.

    Returns
    -------
    wc_map : 2D uint8, codici classe WC
    dx_1d  : 1D float, easting offset centri pixel [m]
    dy_1d  : 1D float, northing offset centri pixel [m]
    """
    import pystac_client
    import planetary_computer
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from rasterio.crs import CRS

    wgs84      = CRS.from_epsg(4326)
    c          = np.cos(np.radians(lat))
    margin_lon = (r86 * 1.3) / (111320.0 * c)
    margin_lat = (r86 * 1.3) / 111320.0
    bbox       = [lon - margin_lon, lat - margin_lat,
                  lon + margin_lon, lat + margin_lat]

    catalog = pystac_client.Client.open(
        PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    items   = list(catalog.search(
        collections=["esa-worldcover"], bbox=bbox,
        max_items=4).items())

    if not items:
        raise RuntimeError(
            "Nessun item WorldCover trovato per questo sito")

    signed = planetary_computer.sign(items[0])
    href   = signed.assets["map"].href

    if verbose:
        print(f"   WorldCover: {href[:60]}...", flush=True)

    with rasterio.open(href) as src:
        if src.crs.to_epsg() == 4326:
            l, b, r, t = bbox
        else:
            l, b, r, t = transform_bounds(wgs84, src.crs, *bbox)
        win     = from_bounds(l, b, r, t, transform=src.transform)
        wc_data = src.read(1, window=win)
        win_tf  = src.window_transform(win)

    nr, nc  = wc_data.shape
    col_idx = np.arange(nc)
    row_idx = np.arange(nr)
    lons_wc = win_tf.c + (col_idx + 0.5) * win_tf.a
    lats_wc = win_tf.f + (row_idx + 0.5) * win_tf.e
    c       = np.cos(np.radians(lat))
    dx_1d   = (lons_wc - lon) * 111320.0 * c
    dy_1d   = (lats_wc - lat) * 111320.0

    if verbose:
        print(f"   WorldCover: shape={wc_data.shape}  "
              f"classes={sorted(np.unique(wc_data).tolist())}",
              flush=True)

    return (wc_data.astype(np.uint8),
            dx_1d.astype(np.float32),
            dy_1d.astype(np.float32))


# ---------------------------------------------------------------------------
# Resample su griglia DEM
# ---------------------------------------------------------------------------

def resample_wc_to_dem(wc_map, dx_1d, dy_1d, dx_grid, dy_grid):
    """Nearest-neighbor resample WorldCover -> griglia DEM."""
    from scipy.spatial import cKDTree
    WC_X, WC_Y = np.meshgrid(dx_1d, dy_1d)
    pts_wc  = np.column_stack([WC_X.ravel(), WC_Y.ravel()])
    tree    = cKDTree(pts_wc)
    pts_dem = np.column_stack([dx_grid.ravel(), dy_grid.ravel()])
    _, idx  = tree.query(pts_dem)
    return wc_map.ravel()[idx].reshape(dx_grid.shape)


# ---------------------------------------------------------------------------
# Calcolo kappa_lulc WorldCover
# ---------------------------------------------------------------------------

def compute_wc_kappa(wc_dem, dx_grid, dy_grid, dist_grid, r86):
    """
    kappa_lulc da mappa WorldCover sulla griglia DEM.

    Returns
    -------
    kappa  : float
    fractions : dict {code: {name, f_H, fraction, area_m2,
                              kappa_contribution}}
    """
    fp_mask    = dist_grid <= r86
    W_fp       = weight_radial(dist_grid, r86)
    dpx, dpy   = pixel_size(dx_grid, dy_grid)
    pixel_area = dpx * dpy

    codes      = wc_dem[fp_mask]
    W_vals     = W_fp[fp_mask]
    fH_arr     = np.array([WC_CLASSES.get(int(c),
                            WC_CLASSES[0])["f_H"] for c in codes])

    num   = float(np.sum(W_vals * fH_arr * pixel_area))
    denom = float(np.sum(W_vals * pixel_area))
    kappa = num / denom if denom > 0 else 1.0

    fractions = {}
    for code, meta in WC_CLASSES.items():
        mask_c = codes == code
        if not mask_c.any():
            continue
        w_c = float(np.sum(W_vals[mask_c] * pixel_area))
        fractions[code] = {
            "name"    : meta["name"],
            "f_H"     : meta["f_H"],
            "fraction": w_c / denom if denom > 0 else 0.0,
            "area_m2" : float(mask_c.sum()) * pixel_area,
            "kappa_contribution": float(
                np.sum(W_vals[mask_c] * fH_arr[mask_c] * pixel_area)
            ) / denom if denom > 0 else 0.0,
        }

    return kappa, fractions


"""
lulc_osm.py
===========
Download, cache, geometria e calcolo kappa_lulc da OSM Overpass.

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _osm_cache_path(cache_dir, lat, lon):
    h = hashlib.sha256(f"{lat:.5f}_{lon:.5f}".encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"osm_{h}.json.gz")


def load_osm_cache(cache_dir, lat, lon):
    p = _osm_cache_path(cache_dir, lat, lon)
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_osm_cache(cache_dir, lat, lon, elements):
    os.makedirs(cache_dir, exist_ok=True)
    with gzip.open(_osm_cache_path(cache_dir, lat, lon),
                   "wt", encoding="utf-8") as f:
        json.dump(elements, f)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_osm(lat, lon, radius_m=600, verbose=True):
    """
    Scarica elementi OSM entro radius_m via Overpass API.
    Usa 'out geom' per includere le coordinate di ogni nodo.
    """
    import requests

    c          = np.cos(np.radians(lat))
    margin_lat = radius_m / 111320.0 * 1.1
    margin_lon = radius_m / (111320.0 * c) * 1.1
    bb = (f"{lat-margin_lat},{lon-margin_lon},"
          f"{lat+margin_lat},{lon+margin_lon}")

    query = f"""
[out:json][timeout:60];
(
  way["highway"]({bb});
  way["building"]({bb});
  way["landuse"]({bb});
  way["natural"]({bb});
  way["railway"]({bb});
  way["waterway"]({bb});
  relation["landuse"]({bb});
  relation["natural"]({bb});
);
out geom;
"""
    if verbose:
        print("   OSM: querying Overpass ...", flush=True)

    resp = requests.post(OVERPASS_URL,
                         data={"data": query}, timeout=60)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    if verbose:
        print(f"   OSM: {len(elements)} elements", flush=True)

    return elements


# ---------------------------------------------------------------------------
# Lookup f_H
# ---------------------------------------------------------------------------

def lookup_fH(tags):
    """
    Ritorna (f_H, category_name, color) dai tag OSM di un elemento.
    Gerarchia: building > railway > highway > natural > landuse.
    """
    if "building" in tags:
        return 0.0, "building", OSM_COLORS["building"]

    if "railway" in tags and tags["railway"] in ("rail","tram","subway"):
        return OSM_FH.get(("railway","*"), 0.15), \
               "railway", OSM_COLORS["railway"]

    if "highway" in tags:
        surface = tags.get("surface","").lower()
        fh = (OSM_FH.get(("highway", surface)) or
              OSM_FH.get(("highway_type", tags["highway"])) or
              OSM_FH.get(("highway_type", "default"), 0.50))
        return fh, "highway", OSM_COLORS["highway"]

    if "natural" in tags:
        nat = tags["natural"]
        if nat in ("water","river","lake","pond"):
            return OSM_FH.get(("natural","water"), 4.0), \
                   "water", OSM_COLORS["water"]
        return (OSM_FH.get(("natural", nat), WC_BASELINE_FH),
                "natural", OSM_COLORS["natural"])

    if "landuse" in tags:
        lu = tags["landuse"]
        return (OSM_FH.get(("landuse", lu), WC_BASELINE_FH),
                "landuse", OSM_COLORS["landuse"])

    if "waterway" in tags:
        return OSM_FH.get(("natural","water"), 4.0), \
               "water", OSM_COLORS["water"]

    return WC_BASELINE_FH, "unknown", "#cccccc"


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------

def osm_to_shapely(element, lat0, lon0):
    """
    Converte elemento OSM in geometria shapely in coordinate
    metriche centrate sul sensore.
    Ritorna (geom, gtype) oppure (None, None).
    """
    from shapely.geometry import Polygon, LineString
    from shapely.validation import make_valid
    from shapely.ops import unary_union

    c = np.cos(np.radians(lat0))

    def _m(pts):
        return [((p["lon"] - lon0) * 111320.0 * c,
                 (p["lat"] - lat0) * 111320.0)
                for p in pts]

    etype = element.get("type","")

    if etype == "way":
        pts = element.get("geometry", [])
        if len(pts) < 2:
            return None, None
        coords = _m(pts)
        closed = (len(coords) >= 3 and
                  abs(coords[0][0]-coords[-1][0]) < 1e-6 and
                  abs(coords[0][1]-coords[-1][1]) < 1e-6)
        try:
            if closed:
                return make_valid(Polygon(coords)), "polygon"
            return LineString(coords), "line"
        except Exception:
            return None, None

    if etype == "relation":
        outer = []
        for m in element.get("members", []):
            if m.get("role") == "outer" and "geometry" in m:
                coords = _m(m["geometry"])
                if len(coords) >= 3:
                    try:
                        outer.append(Polygon(coords))
                    except Exception:
                        pass
        if not outer:
            return None, None
        try:
            return make_valid(unary_union(outer)), "polygon"
        except Exception:
            return None, None

    return None, None


def mean_distance(geom, n_samples=30):
    """Distanza media dal punto (0,0) integrata sulla geometria."""
    try:
        if geom.geom_type in ("Polygon","MultiPolygon"):
            minx, miny, maxx, maxy = geom.bounds
            step = max((maxx-minx)/n_samples,
                       (maxy-miny)/n_samples, 0.1)
            from shapely.geometry import Point
            pts = [(x, y)
                   for x in np.arange(minx, maxx, step)
                   for y in np.arange(miny, maxy, step)
                   if geom.contains(Point(x, y))]
            if pts:
                a = np.array(pts)
                return float(np.mean(np.sqrt(a[:,0]**2 + a[:,1]**2)))
        length = geom.length
        step   = max(length / n_samples, 0.5)
        dists, d = [], 0.0
        while d <= length:
            pt = geom.interpolate(d)
            dists.append(np.sqrt(pt.x**2 + pt.y**2))
            d += step
        return float(np.mean(dists)) if dists else 0.0
    except Exception:
        p = geom.centroid
        return float(np.sqrt(p.x**2 + p.y**2))


# ---------------------------------------------------------------------------
# Calcolo kappa_lulc OSM
# ---------------------------------------------------------------------------

def compute_osm_kappa(elements, lat, lon, r86,
                       dx_grid, dy_grid, dist_grid):
    """
    kappa_lulc da elementi OSM vettoriali.

    Per ogni elemento:
      - interseca con cerchio r86
      - area (poligono) o length*width_stimata (linea)
      - W(r_medio) * area * f_H

    Il denominatore integra W(r)*dA sulla griglia DEM intera dentro r86.
    Area non coperta da OSM -> f_H = WC_BASELINE_FH = 1.0.

    Returns
    -------
    kappa_lulc, contributions, kappa_by_cat, details
    """
    from shapely.geometry import Point

    footprint  = Point(0, 0).buffer(r86)
    fp_mask    = dist_grid <= r86
    W_grid     = weight_radial(dist_grid, r86)
    dpx, dpy   = pixel_size(dx_grid, dy_grid)
    pixel_area = dpx * dpy
    denom      = float(np.sum(W_grid[fp_mask]) * pixel_area)

    contributions    = []
    osm_covered_area = 0.0

    for el in elements:
        tags = el.get("tags", {})
        geom, gtype = osm_to_shapely(el, lat, lon)
        if geom is None or geom.is_empty:
            continue
        try:
            inter = geom.intersection(footprint)
        except Exception:
            continue
        if inter.is_empty:
            continue

        if gtype == "polygon":
            area = float(inter.area)
        else:
            hw   = tags.get("highway", "default")
            w    = HIGHWAY_WIDTH.get(hw, HIGHWAY_WIDTH["default"])
            area = float(inter.length) * w

        if area < 0.1:
            continue

        f_H, category, color = lookup_fH(tags)
        r_mean = mean_distance(inter)
        W_i    = float(weight_radial(np.array([r_mean]), r86)[0])
        contrib = W_i * area * f_H
        osm_covered_area += area

        contributions.append({
            "osm_id"  : el.get("id", -1),
            "category": category,
            "tags"    : {k: tags[k] for k in
                         ["highway","building","landuse","natural",
                          "surface","railway","waterway"]
                         if k in tags},
            "f_H"     : f_H,
            "area_m2" : area,
            "r_mean_m": r_mean,
            "W_r"     : W_i,
            "contrib" : contrib,
            "color"   : color,
            "geom"    : inter,
        })

    footprint_area   = float(footprint.area)

    # Contributo baseline corretto: denom - somma dei pesi W già assegnati a OSM
    # denom = integral W(r)*dA su TUTTI i pixel DEM nel footprint
    # osm_W_area = integral W(r)*dA sui pixel OSM (calcolato come W_i * area_i)
    # baseline_contrib = f_H_baseline * (denom - osm_W_area)
    # Questo usa la distribuzione spaziale reale di W(r) invece di W_mean.
    osm_W_area       = sum(c["W_r"] * c["area_m2"] for c in contributions)
    baseline_contrib = WC_BASELINE_FH * max(0.0, denom - osm_W_area)

    osm_num   = sum(c["contrib"] for c in contributions)
    kappa     = (osm_num + baseline_contrib) / denom \
                if denom > 0 else 1.0

    cats = {}
    for c in contributions:
        cat = c["category"]
        cats.setdefault(cat, {"area": 0.0, "contrib": 0.0, "n": 0})
        cats[cat]["area"]   += c["area_m2"]
        cats[cat]["contrib"]+= c["contrib"]
        cats[cat]["n"]      += 1

    kappa_by_cat = {cat: v["contrib"] / denom
                    for cat, v in cats.items() if denom > 0}
    kappa_by_cat["baseline"] = baseline_contrib / denom

    details = {
        "denom"            : denom,
        "osm_num"          : osm_num,
        "baseline_contrib" : baseline_contrib,
        "osm_covered_area" : osm_covered_area,
        "footprint_area"   : footprint_area,
        "n_elements"       : len(contributions),
        "pixel_area_m2"    : pixel_area,
    }

    return kappa, contributions, kappa_by_cat, details




"""
lulc_main.py
============
Funzione principale get_lulc(), report testuale e mappe.

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""



# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def get_lulc(
    lat, lon,
    dx_grid, dy_grid, dist_grid,
    r86,
    cache_dir,
    osm_radius_m  = 600,
    theta_v_init  = 0.20,   # SM di input [m³/m³] — usata per documentare
                             # la condizione di riferimento di kappa_lulc
    verbose       = True,
):
    """
    Calcola LULC e kappa_lulc per un sito CRNS da due sorgenti:
      - ESA WorldCover 10m (Planetary Computer)
      - OSM Overpass API (vettoriale)

    NOTA sui valori f_H e theta_v:
        I valori f_H nella lookup table (es. acqua=4.0, foresta=1.5) sono
        normalizzati rispetto al suolo di riferimento (f_H=1.0 = baseline soil).
        kappa_lulc è il rapporto H_footprint / H_baseline_soil e dipende
        implicitamente da theta_v_init: al variare di theta_v, il rapporto tra
        le varie coperture rimane approssimativamente costante solo se le
        variazioni di SM sono uniformi nel footprint. theta_v_init è quindi
        la condizione di riferimento per cui i kappa_lulc calcolati sono validi.

    Parameters
    ----------
    lat, lon            : coordinate WGS84
    dx_grid, dy_grid    : offset metrici dalla griglia DEM
    dist_grid           : distanza dal sensore [m]
    r86                 : raggio footprint [m]
    cache_dir           : directory cache locale
    osm_radius_m        : raggio download OSM [m]  (default 600)
    theta_v_init        : SM di riferimento [m³/m³] (da main.py THETA_V_INIT)
    verbose             : stampa progressi

    Returns
    -------
    dict con wc_*, osm_*, kappa_*, mappe e metadata
    """
    os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # WorldCover
    # ------------------------------------------------------------------ #
    if verbose:
        print("  WorldCover ...", flush=True)

    wc_map, dx_1d, dy_1d = load_wc_cache(cache_dir, lat, lon, r86)
    wc_from_cache = wc_map is not None

    if not wc_from_cache:
        wc_map, dx_1d, dy_1d = download_worldcover(lat, lon, r86,
                                                     verbose)
        save_wc_cache(cache_dir, lat, lon, r86, wc_map, dx_1d, dy_1d)
        if verbose:
            print("   WorldCover cached", flush=True)
    else:
        if verbose:
            print("   WorldCover: from cache", flush=True)

    wc_dem = resample_wc_to_dem(wc_map, dx_1d, dy_1d,
                                  dx_grid, dy_grid)
    wc_kappa, wc_fractions = compute_wc_kappa(
        wc_dem, dx_grid, dy_grid, dist_grid, r86)

    if verbose:
        print(f"   kappa_lulc (WorldCover) = {wc_kappa:.4f}",
              flush=True)

    # ------------------------------------------------------------------ #
    # OSM
    # ------------------------------------------------------------------ #
    if verbose:
        print("  OSM ...", flush=True)

    elements = load_osm_cache(cache_dir, lat, lon)
    osm_from_cache = elements is not None

    if not osm_from_cache:
        elements = download_osm(lat, lon, osm_radius_m, verbose)
        save_osm_cache(cache_dir, lat, lon, elements)
        if verbose:
            print("   OSM cached", flush=True)
    else:
        if verbose:
            print(f"   OSM: from cache ({len(elements)} elements)",
                  flush=True)

    osm_kappa, contributions, kappa_by_cat, osm_details = \
        compute_osm_kappa(elements, lat, lon, r86,
                           dx_grid, dy_grid, dist_grid)

    if verbose:
        print(f"   kappa_lulc (OSM) = {osm_kappa:.4f}", flush=True)

    return dict(
        wc_map_dem         = wc_dem,
        wc_kappa           = wc_kappa,
        wc_class_fractions = wc_fractions,
        osm_elements       = contributions,
        osm_kappa          = osm_kappa,
        osm_kappa_by_cat   = kappa_by_cat,
        osm_details        = osm_details,
        wc_from_cache      = wc_from_cache,
        osm_from_cache     = osm_from_cache,
        lat=lat, lon=lon, r86=r86,
        osm_radius_m       = osm_radius_m,
        theta_v_ref        = float(theta_v_init),
    )


# ---------------------------------------------------------------------------
# Report testuale
# ---------------------------------------------------------------------------

def report_lulc(res):
    w = 72
    L = ["="*w,
         "LULC — Land Use / Land Cover  (CRNS footprint)",
         "="*w,
         f"  r86 = {res['r86']:.0f} m  |  "
         f"{res['lat']:.4f}N {res['lon']:.4f}E",
         ""]

    # WorldCover
    L += ["  WorldCover 10m (ESA 2021)",
          f"  kappa_lulc = {res['wc_kappa']:.4f}"
          f"  ({'cache' if res['wc_from_cache'] else 'download'})",
          f"  {'Class':<22} {'f_H':>5}  {'Fraction':>8}  "
          f"{'Area m²':>9}  {'Kappa contrib':>13}",
          "  " + "-"*(w-2)]
    for code, v in sorted(res["wc_class_fractions"].items()):
        L.append(f"  {v['name']:<22} {v['f_H']:5.2f}  "
                 f"{v['fraction']:8.3f}  {v['area_m2']:9.0f}  "
                 f"{v['kappa_contribution']:13.4f}")

    # OSM
    d = res["osm_details"]
    L += ["",
          "  OSM Overpass (vettoriale)",
          f"  kappa_lulc = {res['osm_kappa']:.4f}"
          f"  ({'cache' if res['osm_from_cache'] else 'download'})",
          f"  Elementi nel footprint: {d['n_elements']}",
          f"  Area OSM / totale: {d['osm_covered_area']:.0f} m²"
          f" / {d['footprint_area']:.0f} m²",
          "",
          f"  {'Categoria':<12} {'kappa contrib':>13}",
          "  " + "-"*30]
    for cat, kv in sorted(res["osm_kappa_by_cat"].items(),
                           key=lambda x: -x[1]):
        L.append(f"  {cat:<12} {kv:13.4f}")

    L.append("")
    top_els = sorted(res["osm_elements"],
                      key=lambda x: -x["contrib"])[:15]
    if top_els:
        L.append("  Top oggetti OSM per contributo:")
        for el in top_els:
            ts = " ".join(f"{k}={v}"
                          for k,v in el["tags"].items())[:35]
            L.append(f"    {el['category']:<10} "
                     f"f_H={el['f_H']:.2f}  "
                     f"A={el['area_m2']:6.0f}m²  "
                     f"r={el['r_mean_m']:5.0f}m  {ts}")
    L.append("="*w)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Mappe
# ---------------------------------------------------------------------------

def plot_lulc_worldcover(res, dx_grid, dy_grid, dist_grid,
                          path, site_name=""):
    """Mappa 2D WorldCover nel footprint + grafico a torta frazioni."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap, BoundaryNorm

    r86    = res["r86"]
    wc_dem = res["wc_map_dem"].astype(float)
    wc_dem[dist_grid > r86] = np.nan

    codes_present = sorted(set(
        int(c) for c in wc_dem[~np.isnan(wc_dem)].ravel()))
    cmap   = ListedColormap([WC_CLASSES.get(c, WC_CLASSES[0])["color"]
                              for c in codes_present])
    bounds = [c - 0.5 for c in codes_present] + \
             [codes_present[-1] + 0.5]
    norm   = BoundaryNorm(bounds, len(codes_present))

    theta  = np.linspace(0, 2*np.pi, 360)
    cx     = r86*np.sin(theta)
    cy     = r86*np.cos(theta)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              facecolor="white")

    ax = axes[0]
    clip = r86 * 1.3
    ax.pcolormesh(dx_grid, dy_grid, wc_dem,
                  cmap=cmap, norm=norm, shading="auto")
    ax.plot(cx, cy, "k--", lw=2, label=f"r86={r86:.0f}m")
    ax.plot(0, 0, "r^", ms=12, zorder=5, label="Sensor")
    ax.set_xlim(-clip, clip)
    ax.set_ylim(-clip, clip)
    ax.set_xlabel("Easting offset (m)")
    ax.set_ylabel("Northing offset (m)")
    ax.set_title(f"WorldCover 10m — footprint\n"
                 f"κ_lulc = {res['wc_kappa']:.4f}", fontsize=12)
    patches = [mpatches.Patch(
        color=WC_CLASSES.get(c, WC_CLASSES[0])["color"],
        label=(f"{c}: {WC_CLASSES.get(c,WC_CLASSES[0])['name']} "
               f"(f_H={WC_CLASSES.get(c,WC_CLASSES[0])['f_H']:.2f})"))
               for c in codes_present]
    ax.legend(handles=patches, fontsize=8, loc="upper right")

    ax2 = axes[1]
    fracs = res["wc_class_fractions"]
    codes_pie  = [c for c in codes_present if c in fracs]
    sizes      = [fracs[c]["fraction"] for c in codes_pie]
    colors_pie = [WC_CLASSES.get(c, WC_CLASSES[0])["color"]
                  for c in codes_pie]
    labels_pie = [f"{fracs[c]['name']}\n"
                  f"f_H={fracs[c]['f_H']:.2f}\n"
                  f"{fracs[c]['fraction']*100:.1f}%"
                  for c in codes_pie]
    ax2.pie(sizes, labels=labels_pie, colors=colors_pie,
            startangle=90, textprops={"fontsize": 9})
    ax2.set_title(f"W(r)-weighted fractions\n"
                  f"κ_lulc = {res['wc_kappa']:.4f}", fontsize=12)

    fig.suptitle(f"WorldCover LULC  |  {site_name}  |  "
                 f"{res['lat']:.4f}N {res['lon']:.4f}E",
                 fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_lulc_osm(res, path, site_name="", map_radius_m=500):
    """Mappa vettoriale OSM a map_radius_m con cerchi r86 e 500m."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection, LineCollection
    import numpy as np

    r86   = res["r86"]
    theta = np.linspace(0, 2*np.pi, 360)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor="white")
    ax.set_facecolor("#f0ede8")

    poly_patches, poly_colors = [], []
    line_segs,    line_colors = [], []
    legend_items = {}

    for el in res["osm_elements"]:
        geom  = el.get("geom")
        color = el["color"]
        cat   = el["category"]
        if geom is None or geom.is_empty:
            continue
        legend_items[cat] = color
        geoms = (list(geom.geoms)
                 if geom.geom_type in
                    ("GeometryCollection","MultiPolygon",
                     "MultiLineString")
                 else [geom])
        for g in geoms:
            if g.is_empty:
                continue
            if g.geom_type == "Polygon":
                poly_patches.append(
                    MplPolygon(np.array(g.exterior.coords),
                               closed=True))
                poly_colors.append(color)
            elif g.geom_type == "LineString":
                line_segs.append(np.array(g.coords))
                line_colors.append(color)

    if poly_patches:
        ax.add_collection(PatchCollection(
            poly_patches, facecolors=poly_colors,
            edgecolors="white", linewidths=0.5,
            alpha=0.85, zorder=2))
    if line_segs:
        ax.add_collection(LineCollection(
            line_segs, colors=line_colors,
            linewidths=2.5, alpha=0.9, zorder=3))

    ax.plot(r86*np.sin(theta), r86*np.cos(theta),
            "r--", lw=2.5, zorder=5,
            label=f"r86 = {r86:.0f} m")
    ax.plot(map_radius_m*np.sin(theta), map_radius_m*np.cos(theta),
            "k:", lw=1.5, zorder=4,
            label=f"Map radius = {map_radius_m} m")
    ax.plot(0, 0, "r^", ms=14, zorder=6, label="Sensor")

    d = res["osm_details"]
    ax.text(0.02, 0.98,
            f"κ_lulc (OSM) = {res['osm_kappa']:.4f}\n"
            f"N elements: {d['n_elements']}\n"
            f"OSM area: {d['osm_covered_area']:.0f} m²",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    legend_patches = [
        mpatches.Patch(color=col, label=cat.capitalize())
        for cat, col in legend_items.items()
    ] + [
        plt.Line2D([0],[0], color="r", ls="--", lw=2,
                   label=f"r86={r86:.0f}m"),
        plt.Line2D([0],[0], color="k", ls=":", lw=1.5,
                   label=f"Map={map_radius_m}m"),
        plt.Line2D([0],[0], marker="^", color="r", ms=10,
                   linestyle="none", label="Sensor"),
    ]
    ax.legend(handles=legend_patches, fontsize=9,
              loc="lower right", framealpha=0.9)

    lim = map_radius_m * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting offset (m)", fontsize=12)
    ax.set_ylabel("Northing offset (m)", fontsize=12)
    ax.set_title(f"OSM Map  |  {site_name}  |  "
                 f"{res['lat']:.4f}N {res['lon']:.4f}E\n"
                 f"κ_lulc = {res['osm_kappa']:.4f}  "
                 f"(1.0 = baseline soil)",
                 fontsize=13, fontweight="bold")
    for d_val in range(-int(lim), int(lim)+1, 100):
        ax.axvline(d_val, color="white", lw=0.4, alpha=0.5, zorder=1)
        ax.axhline(d_val, color="white", lw=0.4, alpha=0.5, zorder=1)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
