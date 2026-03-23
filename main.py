"""
CRNS Topographic & FOV Correction Tool
=======================================
Computes kappa (topographic volume correction) and muon FOV correction
for a CRNS sensor given its geographic location.

kappa_topo: ratio of real soil volume inside the theoretical flat-reference
            slab (radius r86, depth z86) to the flat-reference volume.
            Computed by DEM-cell summation — no iteration, no divergence.

kappa_muon: fraction of the isotropic cosmic-muon flux actually reaching
            the sensor after topographic obstruction.  Always <= 1.

DEM source: Copernicus GLO-30 (30 m) via AWS S3, cached locally.

"""

import warnings
warnings.filterwarnings("ignore")   # suppress matplotlib multi-install noise

# =============================================================================
# CONFIGURATION
# =============================================================================

# RA VALLES
LON             = 12.077010   # decimal degrees WGS84
LAT             = 46.548503   # decimal degrees WGS84

#Altopiano Pale
LON = 11.885655         # decimal degrees, WGS84
LAT = 46.279864          # decimal degrees, WGS84

#Cima Pradazzo
LON = 11.822478        # decimal degrees, WGS84
LAT = 46.355945          # decimal degrees, WGS84

#Malga Fadner
LON = 11.861398
LAT = 46.925545

# LIMENA
LON = 11.851084
LAT = 45.467279

# FINAPP
LON = 11.763470
LAT = 45.338058

#================

NAME = "pradazzo"
#Cima Pradazzo
LON = 11.822478        # decimal degrees, WGS84
LAT = 46.355945          # decimal degrees, WGS84

NAME = "ELHAMAN"
LON = 30.727636
LAT = 29.234043

SENSOR_HEIGHT_M = 2.0         # sensor height above ground (m)
RHO_BULK        = 1.4         # soil bulk density (g/cm3)
THETA_V_INIT    = 0.20        # soil moisture estimate for r86/z86 (m3/m3)
DEM_RADIUS_M    = 2000.0      # DEM download radius (m)
AZIMUTH_STEP_DEG= 2.0         # azimuth resolution for horizon scan (deg)
OUTPUT_DIR      = NAME
OPENTOPO_API_KEY= ""
DEM_SOURCE      = "copernicus_aws"
N_CORES         = 4

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import requests, os, json, hashlib, time, math
import multiprocessing as mp
import gc
from kappa_topo_3d import compute_kappa_topo_3d, report_kappa_3d
from site_fluxes        import (compute_site_fluxes, report_site_fluxes,
                                compute_desilets_curve, report_desilets_curve)
from site_climate       import (get_site_climate, report_site_climate,
                                compute_power_budget, report_power_budget)
from terrain_indices    import (compute_twi, report_twi,
                                compute_thermal_index, report_thermal_index)
from get_soil_properties import get_soil_properties, report_soil_properties
from water               import compute_water_eta, report_water_eta
from vegetation_indices  import (get_vegetation_indices, report_vegetation,
                                 get_snow_cover, report_snow_cover)
from plots import (plot_main, plot_footprint, plot_horizon, plot_fov_detail,
                   plot_climate, plot_soil, plot_thermal, plot_twi,
                   plot_kappa_budget, plot_water)
from vegetation_plots    import plot_seasonal_cycles, plot_timeseries, plot_maps
from lulc import get_lulc, report_lulc, plot_lulc_worldcover, plot_lulc_osm
from reports import write_report
from crns_corrections import get_crns_corrections, report_crns_corrections
from geology import get_geology, report_geology
from sampling_plan import (compute_sampling_plan, report_sampling_plan,
                            plot_sampling_plan)
from era5sm import (get_era5_soil_moisture, report_era5_sm, plot_era5_sm)
from smphysics import (fuse_soil_moisture, report_sm_fusion, plot_sm_fusion)
from config_parser import load_config, get as cfg_get
from radiofreq import run_rf_analysis

try:
    import rasterio
    from rasterio.merge import merge as rasterio_merge
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# =============================================================================
# PATHS
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT        = os.path.join(_SCRIPT_DIR, OUTPUT_DIR)

def _outpath(fname):
    os.makedirs(_OUT, exist_ok=True)
    return os.path.join(_OUT, fname)

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

P0_HPA     = 1013.25    # hPa
RHO_AIR_0  = 1.225e-3   # g/cm3
H_SCALE_M  = 8500.0     # m

# Kohli 2015: r86 = (A1/(theta_v+A2))*(P0/P) -> ~113 m at theta_v=0.20, P=P0
KOHLI_A1   = 29.13
KOHLI_A2   = 0.0578

# Kohli/Bogena: z86 = NUM/(rho_b*(ALPHA+theta_v)) -> ~16 cm at theta_v=0.20
DESILETS_ALPHA = 0.0564
DESILETS_NUM   = 8.3

MUON_COS_POWER = 2.0   # dN/dOmega ~ cos^n(theta_z)

# =============================================================================
# PHYSICS
# =============================================================================

def pressure_at_altitude(alt_m):
    return P0_HPA * np.exp(-alt_m / H_SCALE_M)

def air_density_at_altitude(alt_m):
    return RHO_AIR_0 * np.exp(-alt_m / H_SCALE_M)

def r86_kohli(theta_v, pressure_hpa):
    """Footprint radius (86% sensitivity, m) — Kohli 2015."""
    return KOHLI_A1 / (theta_v + KOHLI_A2) * (P0_HPA / pressure_hpa)

def z86_desilets(theta_v, rho_b, lw=0.0):
    """
    Penetration depth (86% sensitivity, cm).

    Formula Köhli/Bogena con correzione lattice water (lw):
        z86 = 8.3 / (rho_b * (0.0564 + theta_v + lw))

    lw [g/g] è l'acqua strutturalmente legata nei minerali argillosi
    (Köhli 2021). Anche a suolo "secco" (theta_v_liquida=0), lw è sempre
    presente e riduce z86 sistematicamente.
    Con lw=0 si recupera la formula originale.
    """
    return DESILETS_NUM / (rho_b * (DESILETS_ALPHA + theta_v + max(0.0, lw)))

def weight_radial(r, r86):
    """Radial sensitivity weight W(r) — exponential decay, Kohli 2015."""
    lam = r86 / 3.0
    return np.where(r < 1e-3, 0.0, np.exp(-r / lam))

# =============================================================================
# DEM CACHE
# =============================================================================

def _cache_paths(lat, lon, radius_m, source):
    tag  = f"{lat:.6f}_{lon:.6f}_{radius_m:.1f}_{source}"
    key  = hashlib.sha256(tag.encode()).hexdigest()[:16]
    base = _outpath(f"dem_cache_{key}")
    return base + ".npz", base + ".json"

def _save_dem_cache(elev, lats_grid, lons_grid, lat, lon, radius_m, source):
    npz, meta = _cache_paths(lat, lon, radius_m, source)
    np.savez_compressed(npz, elev=elev, lats=lats_grid, lons=lons_grid)
    with open(meta, "w") as f:
        json.dump(dict(lat=lat, lon=lon, radius_m=radius_m, source=source,
                       shape=list(elev.shape)), f, indent=2)
    print(f"   DEM cached -> {npz}", flush=True)

def _load_dem_cache(lat, lon, radius_m, source):
    npz, meta = _cache_paths(lat, lon, radius_m, source)
    if not (os.path.exists(npz) and os.path.exists(meta)):
        return None, None, None
    with open(meta) as f:
        m = json.load(f)
    if (abs(m["lat"]-lat)>1e-7 or abs(m["lon"]-lon)>1e-7 or
            abs(m["radius_m"]-radius_m)>1.0 or m["source"]!=source):
        return None, None, None
    d = np.load(npz)
    print(f"   DEM loaded from cache: {npz}", flush=True)
    return d["elev"], d["lats"], d["lons"]

# =============================================================================
# DEM DOWNLOAD
# =============================================================================

def download_dem_copernicus_aws(lat, lon, radius_m):
    deg   = (radius_m / 111320.0) * 1.3
    c     = max(np.cos(np.radians(lat)), 0.01)
    tiles = [(tlat, tlon)
             for tlat in range(int(np.floor(lat-deg)), int(np.ceil(lat+deg)))
             for tlon in range(int(np.floor(lon-deg/c)), int(np.ceil(lon+deg/c)))]
    all_data = []
    for tlat, tlon in tiles:
        ns  = "N" if tlat >= 0 else "S"
        ew  = "E" if tlon >= 0 else "W"
        fn  = (f"Copernicus_DSM_COG_10_{ns}{abs(tlat):02d}_00_"
               f"{ew}{abs(tlon):03d}_00_DEM")
        url = f"https://copernicus-dem-30m.s3.amazonaws.com/{fn}/{fn}.tif"
        print(f"   Fetching tile: {fn}", flush=True)
        try:
            resp = requests.get(url, timeout=90)
            if resp.status_code == 200:
                all_data.append(resp.content)
                print(f"   OK ({len(resp.content)//1024} kB)", flush=True)
            else:
                print(f"   HTTP {resp.status_code} -- skipped", flush=True)
        except Exception as e:
            print(f"   Error: {e} -- skipped", flush=True)

    if not all_data or not HAS_RASTERIO:
        print("  [WARN] Using synthetic flat DEM.", flush=True)
        return _synthetic_dem(lat, lon, radius_m)

    import tempfile
    tmp_files, datasets = [], []
    for data in all_data:
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.write(data); tmp.close()
        tmp_files.append(tmp.name)
        datasets.append(rasterio.open(tmp.name))
    mosaic, tf = rasterio_merge(datasets)
    for ds in datasets: ds.close()
    for f in tmp_files: os.unlink(f)
    elev = mosaic[0].astype(float); elev[elev < -9000] = np.nan
    nr, nc = elev.shape
    lo  = np.array([tf * (c2, 0) for c2 in range(nc)])[:, 0]
    la  = np.array([tf * (0,  r) for r  in range(nr)])[:, 1]
    lons_grid, lats_grid = np.meshgrid(lo, la)
    return elev, lats_grid, lons_grid, tf

def _synthetic_dem(lat, lon, radius_m, res=30.0):
    n  = int(2*radius_m/res)+10
    c  = max(np.cos(np.radians(lat)), 0.01)
    la = np.linspace(lat-radius_m/111320, lat+radius_m/111320, n)
    lo = np.linspace(lon-radius_m/(111320*c), lon+radius_m/(111320*c), n)
    lg, latg = np.meshgrid(lo, la)
    return np.zeros((n,n)), latg, lg, None

def clip_dem_to_radius(elev, lats_grid, lons_grid, lat0, lon0, radius_m):
    c    = np.cos(np.radians(lat0))
    dx   = (lons_grid - lon0) * 111320.0 * c
    dy   = (lats_grid - lat0) * 111320.0
    dist = np.sqrt(dx**2 + dy**2)
    return elev, lats_grid, lons_grid, dx, dy, dist, dist <= radius_m

# =============================================================================
# KAPPA_TOPO — DEM cell summation (no iteration, no divergence)
#
# For each DEM pixel inside r86:
#   overlap = max(0, min(z_sensor, z_DEM_pixel) - (z_sensor - z86_m))
#           = how much of the reference slab column is actually filled by soil
#   kappa = sum(W(r) * overlap * pixel_area) / sum(W(r) * z86_m * pixel_area)
#
# This is a purely geometric integral over the DEM.
# r86 and z86 are fixed from THETA_V_INIT and local pressure — no iteration.
# =============================================================================

def compute_kappa_topo(elev, dx_grid, dy_grid, dist_grid, sz, r86, z86_cm,
                       s_elev=0.0):
    """
    .. deprecated::
        Use :func:`kappa_topo_3d.compute_kappa_topo_3d` instead.
        This function uses a simplified DEM-cell summation approach;
        ``compute_kappa_topo_3d`` implements a physically correct 3-D
        ray-casting algorithm that supersedes it.

    Compute kappa_topo by summing DEM pixel contributions.

    Reference slab: vertical cylinder of radius r86 centred on sensor,
    extending from the ground surface (s_elev) down to depth z86.
    The sensor itself is ABOVE the ground at height SENSOR_HEIGHT_M —
    this height is irrelevant for the soil volume calculation.

    For each DEM pixel within r86:
      z_top_soil = min(z_DEM_pixel, s_elev)   # soil surface capped at ref level
      overlap    = max(0, z_top_soil - (s_elev - z86_m))
                 = max(0, z_top_soil - z_bot_ref)

    If z_DEM > s_elev: terrain is higher than sensor reference level
      → overlap = z86_m (full column, overestimate condition)
    If z_DEM = s_elev: flat terrain → overlap = z86_m (reference case, kappa=1)
    If z_DEM < s_elev - z86_m: terrain drops below slab bottom → overlap = 0

    Returns kappa and a 2D weight map for plotting.
    """
    warnings.warn(
        "compute_kappa_topo is deprecated and will be removed in a future version. "
        "Use compute_kappa_topo_3d from kappa_topo_3d instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    z86_m     = z86_cm / 100.0
    
    MAX_DEPTH = 2.0 # meters
    
    z_bot_ref = s_elev - MAX_DEPTH      # bottom of reference slab (below ground)

    # Pixel area (m2) — approximate, uniform over footprint
    nr, nc = elev.shape
    dpx = abs(np.nanmedian(np.diff(dx_grid[nr//2, :])))
    dpy = abs(np.nanmedian(np.diff(dy_grid[:, nc//2])))
    if dpx < 1: dpx = 30.0
    if dpy < 1: dpy = 30.0
    pixel_area = dpx * dpy

    # Mask: only pixels within r86
    mask = (dist_grid <= r86) & ~np.isnan(elev)

    r_pix   = dist_grid[mask]
    e_pix   = elev[mask]

    # Radial weight W(r)
    W = weight_radial(r_pix, r86)

    # Overlap: soil column inside slab [z_bot_ref, s_elev]
    # z_top_soil = min(z_DEM, s_elev): soil surface capped at reference level
    # If z_DEM > s_elev (terrain above ref): full column contributes
    # If z_DEM < z_bot_ref: no contribution (terrain below slab)
    z_top_soil = np.minimum(e_pix, s_elev)    # soil surface inside slab
    overlap    = np.maximum(0.0, z_top_soil - z_bot_ref)
    overlap    = np.minimum(overlap, z86_m)   # cap at z86

    num = np.sum(W * overlap * pixel_area)
    den = np.sum(W * z86_m  * pixel_area)
    kappa = num / den if den > 0 else 1.0

    # 2D weight map for plotting
    wmap = np.full_like(elev, np.nan)
    wmap[mask] = W * overlap / z86_m   # normalised effective weight

    return kappa, wmap

# =============================================================================
# HORIZON ANGLES — parallel
# =============================================================================

def _horizon_cache_path(lat, lon, sz, azimuth_step_deg, max_radius_m, dem_shape):
    tag = (f"{lat:.6f}_{lon:.6f}_{sz:.2f}_"
           f"{azimuth_step_deg:.3f}_{max_radius_m:.1f}_"
           f"{dem_shape[0]}x{dem_shape[1]}")
    key = hashlib.sha256(tag.encode()).hexdigest()[:16]
    return _outpath(f"horizon_cache_{key}.npz")


def compute_horizon_angles(elev, dx_grid, dy_grid, sx, sy, sz,
                            azimuth_step_deg, max_radius_m, n_cores=N_CORES,
                            cache_dir=None, lat=None, lon=None):
    """
    Compute horizon elevation angle (deg) for every azimuth.

    Vectorised implementation using RegularGridInterpolator — no
    multiprocessing overhead.  Results are cached to a .npz file keyed
    on (lat, lon, sensor_z, az_step, radius, dem_shape).
    """
    from scipy.interpolate import RegularGridInterpolator

    azimuths = np.arange(0, 360, azimuth_step_deg)
    n_az     = len(azimuths)
    r_vals   = np.arange(10, max_radius_m, 20.0)
    n_r      = len(r_vals)

    # Cache check
    cache_file = None
    if lat is not None and lon is not None:
        cache_file = _horizon_cache_path(
            lat, lon, sz, azimuth_step_deg, max_radius_m, elev.shape)
        if os.path.exists(cache_file):
            d = np.load(cache_file)
            print(f"   Horizon: from cache  ({cache_file})", flush=True)
            return d["azimuths"], d["horizon"]

    print(f"   Horizon: {n_az} az x {n_r} r  [vectorised]", flush=True)
    t0 = time.perf_counter()

    # Build RegularGridInterpolator on the regular (dy, dx) meter grid.
    # dx_grid varies only along columns; dy_grid only along rows.
    dx_1d = dx_grid[0, :]
    dy_1d = dy_grid[:, 0]
    elev_clean = np.where(np.isnan(elev), float(np.nanmean(elev)), elev)

    interp = RegularGridInterpolator(
        (dy_1d, dx_1d), elev_clean,
        method="linear", bounds_error=False,
        fill_value=float(np.nanmean(elev_clean)))

    # All ray sample points: shape (n_az, n_r)
    az_rad = np.radians(azimuths)
    px = sx + r_vals[None, :] * np.sin(az_rad[:, None])   # (n_az, n_r)
    py = sy + r_vals[None, :] * np.cos(az_rad[:, None])

    pts   = np.column_stack([py.ravel(), px.ravel()])
    elevs = interp(pts).reshape(n_az, n_r)

    dz   = elevs - sz
    angs = np.degrees(np.arctan2(dz, r_vals[None, :]))
    horizon = np.maximum(0.0, angs.max(axis=1))

    print(f"   Horizon done  total={time.perf_counter()-t0:.1f}s", flush=True)

    if cache_file is not None:
        np.savez_compressed(cache_file, azimuths=azimuths, horizon=horizon)
        print(f"   Horizon cached -> {cache_file}", flush=True)

    return azimuths, horizon

# =============================================================================
# KAPPA_MUON
#
# The cosmic-muon flux at sea level follows dN/dOmega ~ cos^2(theta_z),
# where theta_z is the zenith angle (0 = vertical, 90 = horizontal).
#
# An obstacle subtending elevation angle psi(phi) in azimuth phi blocks
# all muons with zenith angle theta_z > (90 - psi), i.e. those arriving
# at shallow angles from that direction.
#
# kappa_muon = integral over visible sky of cos^2(theta_z) sin(theta_z) dtheta daz
#            / integral over full upper hemisphere
#
# kappa_muon < 1 always (obstruction reduces flux).
# kappa_muon = 1 if horizon is flat (open sky in all directions).
# kappa_muon decreases as surrounding topography gets higher.
#
# Per-azimuth contribution:
#   f(phi) = [1 - cos^3(theta_z_max(phi))] / 3   (for n=2)
#   with theta_z_max(phi) = pi/2 - psi(phi)
# =============================================================================

def compute_kappa_muon(azimuths_deg, horizon_deg, azimuth_step_deg,
                       cos_power=MUON_COS_POWER):
    n     = cos_power
    daz   = np.radians(azimuth_step_deg)
    N_ref = 2 * np.pi / (n + 1)   # full upper hemisphere
    N_obs = 0.0
    per_az = np.zeros(len(azimuths_deg))
    for i, psi_deg in enumerate(horizon_deg):
        psi      = np.radians(psi_deg)
        th_max   = np.pi / 2.0 - psi
        if th_max <= 0:
            per_az[i] = 0.0; continue
        contrib   = (1.0 - np.cos(th_max)**(n+1)) / (n+1)
        N_obs    += contrib * daz
        per_az[i] = contrib / (1.0/(n+1))   # normalised to unobstructed
    return N_obs / N_ref, per_az

# =============================================================================
# NEUTRON — per-azimuth overlap profile
#
# For each azimuth, sample the DEM along a radial ray and compute the
# W(r)-weighted mean overlap fraction = actual soil / reference slab.
# =============================================================================

def compute_neutron_fov(elev, dx_grid, dy_grid, sx, sy, s_elev, r86, z86_cm,
                         azimuth_step_deg, max_radius_m):
    """
    Per-azimuth W(r)-weighted mean overlap fraction (0=no soil, 1=full slab).
    Also returns per-azimuth mean terrain deficit below slab.
    """
    from scipy.spatial import cKDTree
    z86_m     = z86_cm / 100.0
    z_bot_ref = s_elev - z86_m
    pts       = np.column_stack([dx_grid.ravel(), dy_grid.ravel()])
    tree      = cKDTree(pts)
    elev_flat = elev.ravel()
    azimuths  = np.arange(0, 360, azimuth_step_deg)
    r_vals    = np.arange(5.0, r86, 10.0)
    n_r       = len(r_vals)
    overlap_az = np.zeros(len(azimuths))
    deficit_az = np.zeros(len(azimuths))
    for i, az in enumerate(azimuths):
        ar = np.radians(az)
        px = sx + r_vals * np.sin(ar)
        py = sy + r_vals * np.cos(ar)
        _, idx = tree.query(np.column_stack([px, py]))
        ev  = elev_flat[idx]
        ok  = ~np.isnan(ev)
        if not np.any(ok) or n_r == 0:
            overlap_az[i] = 0.0
            deficit_az[i] = z86_m
            continue
        W     = weight_radial(r_vals[ok], r86)
        ev_ok = ev[ok]
        z_top = np.minimum(ev_ok, s_elev)
        ovl   = np.clip(z_top - z_bot_ref, 0.0, z86_m) / z86_m
        Wsum  = W.sum()
        overlap_az[i] = float(np.sum(W * ovl) / Wsum) if Wsum > 0 else 0.0
        deficit_az[i] = float(np.mean(np.maximum(0.0, z_bot_ref - ev_ok)))
    return azimuths, overlap_az, deficit_az


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Legge config.cfg (opzionale — il programma funziona anche senza)   #
    # ------------------------------------------------------------------ #
    _cfg_path = os.path.join(_SCRIPT_DIR, "config.cfg")
    _cfg = {}
    if os.path.exists(_cfg_path):
        try:
            _cfg = load_config(_cfg_path)
            print(f"[config] Loaded: {_cfg_path}")
        except Exception as _e:
            print(f"[config] WARNING: cannot parse {_cfg_path}: {_e}")
    else:
        print(f"[config] No config.cfg found — RF analysis will be skipped "
              f"(create {_cfg_path} with OPENCELLID_TOKEN to enable)")

    OPENCELLID_TOKEN = cfg_get(_cfg, "OPENCELLID_TOKEN", default="")

    print("=" * 62)
    print("CRNS TOPOGRAPHIC CORRECTION TOOL")
    print("=" * 62)
    print(f"  Sensor  : LAT={LAT:.6f}  LON={LON:.6f}")
    print(f"  Height  : {SENSOR_HEIGHT_M} m a.g.l.")
    print(f"  rho_b   : {RHO_BULK} g/cm3   theta_v_init : {THETA_V_INIT}")
    print(f"  DEM     : radius={DEM_RADIUS_M} m   source={DEM_SOURCE}")
    print(f"  Az step : {AZIMUTH_STEP_DEG} deg   Cores: {N_CORES}")
    print(f"  Output  : {_OUT}")
    print("=" * 62)

    # 1 — DEM
    print(f"\n[1] DEM acquisition ...")
    elev, lats_grid, lons_grid = _load_dem_cache(LAT, LON, DEM_RADIUS_M, DEM_SOURCE)
    if elev is None:
        print("   No cache -- downloading ...", flush=True)
        res = download_dem_copernicus_aws(LAT, LON, DEM_RADIUS_M)
        elev, lats_grid, lons_grid = res[0], res[1], res[2]
        _save_dem_cache(elev, lats_grid, lons_grid,
                        LAT, LON, DEM_RADIUS_M, DEM_SOURCE)

    # 2 — Geometry
    print("\n[2] Sensor geometry ...")
    elev, lats_grid, lons_grid, dx_grid, dy_grid, dist_grid, _ = \
        clip_dem_to_radius(elev, lats_grid, lons_grid, LAT, LON, DEM_RADIUS_M)
    sx, sy   = 0.0, 0.0
    s_elev   = float(np.nanmedian(elev[dist_grid < 100])) \
               if np.any(dist_grid < 100) else 0.0
    sz       = s_elev + SENSOR_HEIGHT_M
    pressure = pressure_at_altitude(s_elev)
    rho_air  = air_density_at_altitude(s_elev)
    r86_sea  = r86_kohli(THETA_V_INIT, P0_HPA)
    r86      = r86_kohli(THETA_V_INIT, pressure)
    z86      = z86_desilets(THETA_V_INIT, RHO_BULK)
    print(f"   Sensor altitude : {s_elev:.1f} m")
    print(f"   Pressure        : {pressure:.2f} hPa")
    print(f"   r86 (sea level) : {r86_sea:.1f} m")
    print(f"   r86 (at site)   : {r86:.1f} m")
    print(f"   z86 (at site)   : {z86:.2f} cm")

    # 3 — kappa_topo by 3-D ray-casting
    print("\n[3] Computing kappa_topo (3-D ray-casting) ...")
    t1 = time.perf_counter()
    kappa_topo, kappa_pieno, kappa_sopra, kappa_vuoto, wmap, kappa_info = \
        compute_kappa_topo_3d(
            elev, dx_grid, dy_grid, dist_grid,
            sz, s_elev, r86, z86, RHO_BULK, SENSOR_HEIGHT_M)
    kappa_topo_label = kappa_topo
    print(report_kappa_3d(kappa_topo, kappa_pieno, kappa_sopra, kappa_vuoto, kappa_info))
    print(f"   elapsed={time.perf_counter()-t1:.1f}s")
    V0   = np.pi * r86**2 * (z86/100.0)
    Veff = V0 * kappa_topo

    # 4 — Horizon angles
    print("\n[4] Computing horizon angles ...")
    azimuths, horizon = compute_horizon_angles(
        elev, dx_grid, dy_grid, sx, sy, sz,
        AZIMUTH_STEP_DEG, DEM_RADIUS_M, N_CORES,
        cache_dir=_OUT, lat=LAT, lon=LON)

    # 5 — kappa_muon
    print("\n[5] Computing kappa_muon ...")
    kappa_muon, per_az_muon = compute_kappa_muon(
        azimuths, horizon, AZIMUTH_STEP_DEG)
    print(f"   kappa_muon = {kappa_muon:.4f}  "
          f"max_horizon={horizon.max():.1f}deg  "
          f"mean_horizon={horizon.mean():.1f}deg")

    # 6 — Site fluxes & N0 (provvisori, senza lw — verrà ricalcolato dopo soil)
    print("\n[6] Computing site fluxes (preliminary, lw=0 — updated after soil) ...")
    site_fluxes = compute_site_fluxes(LAT, LON, s_elev, kappa_topo, kappa_muon)
    print(report_site_fluxes(site_fluxes))

    # 7 — Site climate
    print("\n[7] Fetching site climate (PVGIS + Open-Meteo) ...")
    site_climate = get_site_climate(
        LAT, LON, s_elev,
        horizon_deg=horizon, azimuths_deg=azimuths,
        cache_dir=_OUT)
    print(report_site_climate(site_climate))

    # 7b — Power budget (pannello solare + batteria)
    power_budget = compute_power_budget(site_climate['energy_monthly_kWh'])
    print(report_power_budget(power_budget))

    # 8 — Soil properties (SoilGrids)
    print("\n[8] Fetching soil properties (SoilGrids) ...")
    soil = get_soil_properties(LAT, LON, z86_cm=z86, cache_dir=_OUT)
    print(report_soil_properties(soil))

    # 8b — Aggiorna z86 con lattice water dal suolo reale
    lw       = soil.get('lattice_water_gg', 0.0)
    soc_gkg  = soil.get('soc_crns', 0.0)
    if not np.isnan(float(lw)) and lw > 0:
        z86_lw = z86_desilets(THETA_V_INIT, RHO_BULK, lw=lw)
        print(f"   z86 senza lw : {z86:.2f} cm")
        print(f"   z86 con lw   : {z86_lw:.2f} cm  (lw={lw:.4f} g/g)")
        z86 = z86_lw
    lw  = float(lw) if not np.isnan(float(lw)) else 0.0
    soc = float(soc_gkg) if soc_gkg and not np.isnan(float(soc_gkg)) else 0.0

    # 8c — Ricalcola site_fluxes con lw aggiornato (N0 ora a theta_v=0, lw presente)
    site_fluxes = compute_site_fluxes(
        LAT, LON, s_elev, kappa_topo, kappa_muon, lw=lw,
        soc_gkg=soc, rho_b=RHO_BULK)
    print(report_site_fluxes(site_fluxes))

    # 8d — Desilets curve N(theta_v) con lw e SOC supplementare
    theta_wp_sr = soil.get('theta_wp', 0.05)
    theta_fc_sr = soil.get('theta_fc', 0.35)
    soc_equiv   = site_fluxes.get('theta_v_soc', 0.0)
    desilets_curve = compute_desilets_curve(
        site_fluxes['N0_theoretical'],
        theta_wp  = theta_wp_sr if not np.isnan(theta_wp_sr) else 0.05,
        theta_fc  = theta_fc_sr if not np.isnan(theta_fc_sr) else 0.35,
        lw        = lw,
        soc_equiv = soc_equiv,
    )
    print(report_desilets_curve(desilets_curve))

    # 9 — JRC Surface Water (eta correction)
    print("\n[9] Fetching JRC surface water occurrence (eta) ...")
    water = compute_water_eta(
        LAT, LON, dx_grid, dy_grid, dist_grid, r86,
        cache_dir=_OUT)
    print(report_water_eta(water))

    # 10 — Topographic Wetness Index (con cache)
    print("\n[10] Computing TWI ...")
    twi = compute_twi(elev, dx_grid, dy_grid, dist_grid, r86,
                      n_cores=N_CORES, cache_dir=_OUT)
    print(report_twi(twi))

    # 11 — Thermal index
    print("\n[11] Computing thermal index ...")
    # era5_elevation_m non è ancora esposto da get_site_climate:
    # si usa s_elev come fallback (delta_elevation = 0, nessuna correzione lapse)
    era5_elev = site_climate.get('era5_elevation_m', s_elev)
    thermal = compute_thermal_index(
        elev, dist_grid, s_elev,
        horizon_deg       = horizon,
        azimuths_deg      = azimuths,
        T_mean_monthly_era5 = site_climate['T_mean_monthly_C'],
        T_min_monthly_era5  = site_climate['T_min_monthly_C'],
        T_max_monthly_era5  = site_climate['T_max_monthly_C'],
        POA_monthly_kWh_m2  = site_climate['POA_monthly_kWh_m2'],
        era5_elevation_m    = era5_elev,
    )
    print(report_thermal_index(thermal,
                               site_climate['T_mean_monthly_C'],
                               site_climate['T_min_monthly_C'],
                               site_climate['T_max_monthly_C']))

    # 12 — Vegetation indices (Landsat + MODIS)
    print("\n[12] Fetching vegetation indices (Landsat + MODIS) ...")
    try:
        veg = get_vegetation_indices(
            LAT, LON, dx_grid, dy_grid, dist_grid, r86,
            cache_dir=_OUT)
        print(report_vegetation(veg))
    except Exception as _veg_e:
        print(f"   [WARN] Vegetation indices failed: {_veg_e}")
        import traceback; traceback.print_exc()
        veg = {"landsat": {}, "modis": {}, "warning": str(_veg_e)}

    # 13 — Snow cover (MODIS MOD10A1)
    print("\n[13] Fetching snow cover (MODIS MOD10A1) ...")
    snow = get_snow_cover(LAT, LON, cache_dir=_OUT)
    print(report_snow_cover(snow))

    # 14 — LULC (Land Use / Land Cover)
    print("\n[14] Computing LULC kappa (WorldCover + OSM) ...")
    lulc_res = get_lulc(
        LAT, LON,
        dx_grid, dy_grid, dist_grid,
        r86,
        cache_dir=_OUT,
        osm_radius_m=int(r86 * 1.5),
        theta_v_init=THETA_V_INIT,
        verbose=True)
    print(report_lulc(lulc_res))

    # 15 — Neutron FOV per-azimuth
    print("\n[15] Computing neutron per-azimuth r_eff ...")
    az_neutron, overlap_az, deficit_az = compute_neutron_fov(
        elev, dx_grid, dy_grid, sx, sy, s_elev, r86, z86,
        AZIMUTH_STEP_DEG, DEM_RADIUS_M)

    # 15b — Mean slope
    nr, nc = elev.shape
    dpx = abs(np.nanmedian(np.diff(dx_grid[nr//2,:])))
    dpy = abs(np.nanmedian(np.diff(dy_grid[:,nc//2])))
    if dpx < 1: dpx = 30.0
    if dpy < 1: dpy = 30.0
    ag = np.arctan(np.sqrt(np.gradient(elev,dpx,axis=1)**2 +
                           np.gradient(elev,dpy,axis=0)**2))
    fpm        = dist_grid <= r86
    mean_slope = float(np.nanmean(ag[fpm])) if np.any(fpm) else 0.0

    # kappa_lulc: usa WorldCover come riferimento (più affidabile globalmente)
    kappa_lulc    = lulc_res.get('wc_kappa', 1.0)
    kappa_tot     = kappa_topo * kappa_muon * kappa_lulc
    theta_v_corr  = THETA_V_INIT / kappa_topo if kappa_topo > 0 else float("inf")

    # 16 — Correzioni supplementari (WV, AGBH, SWE)
    print("\n[16] Computing supplementary CRNS corrections (WV, AGBH, SWE) ...")
    lai_mean = 0.0
    try:
        lai_data = veg.get('lai_monthly_mean')
        if lai_data is not None and len(lai_data) > 0:
            lai_mean = float(np.nanmean(lai_data))
    except Exception:
        pass
    crns_corr = get_crns_corrections(
        LAT, LON,
        lai_annual_m2m2=lai_mean,
        cache_dir=_OUT)
    print(report_crns_corrections(crns_corr, z86_cm=z86))

    # 17 — Geology (Macrostrat API)
    print("\n[17] Fetching geology (Macrostrat) ...")
    geology = get_geology(LAT, LON, cache_dir=_OUT)
    print(report_geology(geology))

    # 18 — ERA5-Land hourly soil moisture
    print("\n[18] Downloading ERA5-Land soil moisture ...")
    era5_sm = get_era5_soil_moisture(
        LAT, LON,
        cache_dir  = _OUT,
        start_year = 2015,
        verbose    = True,
    )
    print(report_era5_sm(era5_sm))

    # 19 — Downscaled soil moisture (data fusion)
    print("\n[19] Downscaling soil moisture (ERA5 + local data) ...")
    sm_fused = fuse_soil_moisture(
        era5_res    = era5_sm,
        soil_res    = soil,
        twi_res     = twi,
        climate_res = crns_corr,
        lulc_res    = lulc_res,
        verbose     = True,
    )
    print(report_sm_fusion(sm_fused))

    # 20 — RF analysis (celle + RFI da OSM)
    rf_result = None
    print("\n[20] RF analysis (OpenCelliD + OSM RFI) ...")
    if OPENCELLID_TOKEN:
        try:
            rf_result = run_rf_analysis(
                lat         = LAT,
                lon         = LON,
                site_elev_m = s_elev,
                token       = OPENCELLID_TOKEN,
                cache_dir   = _OUT,
                verbose     = True,
            )
            conn = rf_result["connectivity"]
            rfi  = rf_result["rfi"]
            print(f"   Connectivity: {conn['n_cells_total']} celle analizzate")
            for tech, info in conn.get("by_radio", {}).items():
                print(f"     {tech:<5}  best={info['best_rx_dbm']:+.0f} dBm  "
                      f"quality={info['quality']}")
            print(f"   RFI index: {rfi['rfi_index']:.1f}/10  "
                  f"level={rfi['rfi_level']}  "
                  f"n_sources={rfi['n_sources']}")
        except Exception as _rf_e:
            print(f"   [WARN] RF analysis failed: {_rf_e}")
    else:
        print("   Skipped (OPENCELLID_TOKEN not set in config.cfg)")

    sampling = compute_sampling_plan(
        r86,
        theta_v_init = THETA_V_INIT,
        theta_wp     = soil.get('theta_wp', 0.05),
        theta_fc     = soil.get('theta_fc', 0.35),
        pressure_hpa = pressure,
    )
    print(report_sampling_plan(sampling))

    results = dict(
        sensor_alt=s_elev, pressure=pressure, rho_air=rho_air,
        mean_slope_rad=mean_slope,
        max_horizon=float(horizon.max()), mean_horizon=float(horizon.mean()),
        r86_sealevel=r86_sea, r86=r86, z86=z86,
        lw=lw, soc_gkg=soc,
        V0=V0, Veff=Veff,
        kappa_topo=kappa_topo, kappa_muon=kappa_muon,
        kappa_lulc=kappa_lulc, kappa_total=kappa_tot,
        kappa_pieno=kappa_pieno, kappa_sopra=kappa_sopra,
        kappa_vuoto=kappa_vuoto, kappa_info=kappa_info,
        theta_v_corrected=theta_v_corr,
        site_fluxes=site_fluxes,
        site_climate=site_climate,
        soil=soil,
        water=water,
        twi=twi,
        thermal=thermal,
        veg=veg,
        snow=snow,
        lulc=lulc_res,
        power_budget=power_budget,
        desilets_curve=desilets_curve,
        crns_corrections=crns_corr,
        geology=geology,
        era5_sm=era5_sm,
        sm_fused=sm_fused,
        sampling=sampling,
        rf=rf_result,
        history=[],   # no iteration history with cell-summation method
    )
    params = dict(
        lat=LAT, lon=LON, h=SENSOR_HEIGHT_M,
        rho_b=RHO_BULK, theta_v_init=THETA_V_INIT,
        dem_radius=DEM_RADIUS_M, dem_source=DEM_SOURCE,
        az_step=AZIMUTH_STEP_DEG, n_cores=N_CORES,
        omega="n/a", tol="n/a",
    )

    # 21 — Report
    rpt = _outpath("crns_report.txt")
    print(f"\n[21] Writing report -> {rpt}")
    print(write_report(rpt, params, results))

    # 22 — Plots
    print("[22] Generating plots ...")

    def _plot(fn, func, *args, **kwargs):
        func(*args, **kwargs)
        gc.collect()
        print(f"  Saved: {fn}", flush=True)

    _plot(_outpath("crns_topo_main.png"),
          plot_main, elev, dx_grid, dy_grid, r86, kappa_topo, kappa_muon,
          results, _outpath("crns_topo_main.png"),
          lat=LAT, lon=LON, dem_radius_m=DEM_RADIUS_M)
    _plot(_outpath("crns_footprint.png"),
          plot_footprint, elev, dx_grid, dy_grid, dist_grid, s_elev, r86, z86,
          kappa_topo, wmap, az_neutron, overlap_az, deficit_az,
          _outpath("crns_footprint.png"))
    _plot(_outpath("crns_horizon.png"),
          plot_horizon, azimuths, horizon, kappa_muon, per_az_muon,
          _outpath("crns_horizon.png"))
    _plot(_outpath("crns_fov_detail.png"),
          plot_fov_detail, azimuths, horizon, per_az_muon, kappa_muon,
          az_neutron, overlap_az, r86, z86, kappa_topo,
          _outpath("crns_fov_detail.png"),
          lat=LAT, lon=LON, sensor_alt=s_elev)
    _plot(_outpath("crns_climate.png"),
          plot_climate, site_climate, thermal,
          _outpath("crns_climate.png"),
          lat=LAT, lon=LON, sensor_alt=s_elev)
    _plot(_outpath("crns_soil.png"),
          plot_soil, soil, _outpath("crns_soil.png"), lat=LAT, lon=LON)
    _plot(_outpath("crns_thermal.png"),
          plot_thermal, site_climate, thermal,
          _outpath("crns_thermal.png"),
          lat=LAT, lon=LON, sensor_alt=s_elev)
    _plot(_outpath("crns_twi.png"),
          plot_twi, twi, elev, dx_grid, dy_grid, dist_grid, r86,
          _outpath("crns_twi.png"), lat=LAT, lon=LON)
    _plot(_outpath("crns_kappa_budget.png"),
          plot_kappa_budget, results, _outpath("crns_kappa_budget.png"),
          lat=LAT, lon=LON)
    _plot(_outpath("crns_water.png"),
          plot_water, water, dx_grid, dy_grid, dist_grid, r86,
          _outpath("crns_water.png"), lat=LAT, lon=LON)
    _plot(_outpath("crns_veg_seasonal.png"),
          plot_seasonal_cycles, veg, _outpath("crns_veg_seasonal.png"),
          site_name=NAME)
    _plot(_outpath("crns_veg_timeseries.png"),
          plot_timeseries, veg, _outpath("crns_veg_timeseries.png"),
          site_name=NAME)
    _plot(_outpath("crns_veg_maps.png"),
          plot_maps, veg, dx_grid, dy_grid, dist_grid, r86,
          _outpath("crns_veg_maps.png"), site_name=NAME)
    _plot(_outpath("crns_lulc_worldcover.png"),
          plot_lulc_worldcover, lulc_res, dx_grid, dy_grid, dist_grid,
          _outpath("crns_lulc_worldcover.png"), site_name=NAME)
    _plot(_outpath("crns_lulc_osm.png"),
          plot_lulc_osm, lulc_res, _outpath("crns_lulc_osm.png"),
          site_name=NAME, map_radius_m=int(r86 * 1.5))
    _plot(_outpath("crns_sampling_plan.png"),
          plot_sampling_plan, sampling, elev, dx_grid, dy_grid, dist_grid,
          _outpath("crns_sampling_plan.png"), site_name=NAME)
    _plot(_outpath("crns_era5_sm.png"),
          plot_era5_sm, era5_sm, _outpath("crns_era5_sm.png"), site_name=NAME)
    _plot(_outpath("crns_sm_fusion.png"),
          plot_sm_fusion, sm_fused, _outpath("crns_sm_fusion.png"), site_name=NAME)

    elapsed = time.perf_counter() - t0
    print(f"\n[DONE]  wall time = {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Output dir : {_OUT}")
    print(f"  Files:")
    for fn in ["crns_report.txt", "crns_topo_main.png",
               "crns_footprint.png", "crns_horizon.png", "crns_fov_detail.png",
               "crns_climate.png", "crns_soil.png", "crns_thermal.png",
               "crns_twi.png", "crns_kappa_budget.png", "crns_water.png",
               "crns_veg_seasonal.png", "crns_veg_timeseries.png",
               "crns_veg_maps.png"]:
        p = _outpath(fn)
        sz_kb = os.path.getsize(p)//1024 if os.path.exists(p) else 0
        print(f"    {fn}  ({sz_kb} kB)")


if __name__ == "__main__":
    mp.freeze_support()
    main()
