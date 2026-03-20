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
LON = 11.885655         # decimal degrees, WGS84
LAT = 46.279864          # decimal degrees, WGS84




SENSOR_HEIGHT_M = 2.0         # sensor height above ground (m)
RHO_BULK        = 1.4         # soil bulk density (g/cm3)
THETA_V_INIT    = 0.20        # soil moisture estimate for r86/z86 (m3/m3)
DEM_RADIUS_M    = 2000.0      # DEM download radius (m)
AZIMUTH_STEP_DEG= 2.0         # azimuth resolution for horizon scan (deg)
OUTPUT_DIR      = "pale"
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
from kappa_topo_3d import compute_kappa_topo_3d, report_kappa_3d
from site_fluxes        import compute_site_fluxes, report_site_fluxes
from site_climate       import get_site_climate, report_site_climate
from terrain_indices    import (compute_twi, report_twi,
                                compute_thermal_index, report_thermal_index)
from get_soil_properties import get_soil_properties, report_soil_properties
from plots import (plot_main, plot_footprint, plot_horizon, plot_fov_detail)

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

def z86_desilets(theta_v, rho_b):
    """Penetration depth (86% sensitivity, cm) — Kohli/Bogena."""
    return DESILETS_NUM / (rho_b * (DESILETS_ALPHA + theta_v))

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

_G_TREE      = None
_G_ELEV_FLAT = None
_G_SENSOR_Z  = None

def _init_horizon(pts, elev_flat, sensor_z):
    global _G_TREE, _G_ELEV_FLAT, _G_SENSOR_Z
    from scipy.spatial import cKDTree
    _G_TREE      = cKDTree(pts)
    _G_ELEV_FLAT = elev_flat
    _G_SENSOR_Z  = sensor_z

def _work_horizon(args):
    az_batch, r_vals, sx, sy = args
    out = []
    for az in az_batch:
        ar  = np.radians(az)
        px  = sx + r_vals * np.sin(ar)
        py  = sy + r_vals * np.cos(ar)
        _, idx = _G_TREE.query(np.column_stack([px, py]))
        ev  = _G_ELEV_FLAT[idx]
        ok  = ~np.isnan(ev)
        if not np.any(ok):
            out.append(0.0); continue
        dz  = ev[ok] - _G_SENSOR_Z
        ang = np.degrees(np.arctan2(dz, r_vals[ok]))
        out.append(max(0.0, float(ang.max())))
    return out

def compute_horizon_angles(elev, dx_grid, dy_grid, sx, sy, sz,
                            azimuth_step_deg, max_radius_m, n_cores=N_CORES):
    """Compute horizon elevation angle (deg) for every azimuth."""
    azimuths  = np.arange(0, 360, azimuth_step_deg)
    n_az      = len(azimuths)
    r_vals    = np.arange(10, max_radius_m, 20.0)
    n_r       = len(r_vals)
    pts       = np.column_stack([dx_grid.ravel(), dy_grid.ravel()])
    elev_flat = elev.ravel()
    batches   = np.array_split(azimuths, n_cores)
    n_b       = len(batches)
    print(f"   Horizon: {n_az} az x {n_r} r  [{n_cores} cores]", flush=True)
    args   = [(b, r_vals, sx, sy) for b in batches]
    parts  = [None] * n_b
    t0     = time.perf_counter()
    ctx    = mp.get_context("spawn")
    with ctx.Pool(n_cores, initializer=_init_horizon,
                  initargs=(pts, elev_flat, sz)) as pool:
        for i, res in enumerate(pool.imap(_work_horizon, args)):
            parts[i] = res
            el  = time.perf_counter() - t0
            eta = el / (i+1) * (n_b - i - 1)
            print(f"   Horizon  {i+1}/{n_b}  elapsed={el:.1f}s  ETA={eta:.1f}s",
                  flush=True)
    horizon = np.array([v for p in parts for v in p])
    print(f"   Horizon done  total={time.perf_counter()-t0:.1f}s", flush=True)
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
        AZIMUTH_STEP_DEG, DEM_RADIUS_M, N_CORES)

    # 5 — kappa_muon
    print("\n[5] Computing kappa_muon ...")
    kappa_muon, per_az_muon = compute_kappa_muon(
        azimuths, horizon, AZIMUTH_STEP_DEG)
    print(f"   kappa_muon = {kappa_muon:.4f}  "
          f"max_horizon={horizon.max():.1f}deg  "
          f"mean_horizon={horizon.mean():.1f}deg")

    # 6 — Site fluxes & N0
    print("\n[6] Computing site fluxes ...")
    site_fluxes = compute_site_fluxes(LAT, LON, s_elev, kappa_topo, kappa_muon)
    print(report_site_fluxes(site_fluxes))

    # 7 — Site climate
    print("\n[7] Fetching site climate (PVGIS + Open-Meteo) ...")
    site_climate = get_site_climate(
        LAT, LON, s_elev,
        horizon_deg=horizon, azimuths_deg=azimuths,
        cache_dir=_OUT)
    print(report_site_climate(site_climate))

    # 8 — Soil properties (SoilGrids)
    print("\n[8] Fetching soil properties (SoilGrids) ...")
    soil = get_soil_properties(LAT, LON, z86_cm=z86, cache_dir=_OUT)
    print(report_soil_properties(soil))

    # 9 — Topographic Wetness Index
    print("\n[9] Computing TWI ...")
    twi = compute_twi(elev, dx_grid, dy_grid, dist_grid, r86)
    print(report_twi(twi))

    # 10 — Thermal index
    print("\n[10] Computing thermal index ...")
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

    # 11 — Neutron FOV per-azimuth
    print("\n[8] Computing neutron per-azimuth r_eff ...")
    az_neutron, overlap_az, deficit_az = compute_neutron_fov(
        elev, dx_grid, dy_grid, sx, sy, s_elev, r86, z86,
        AZIMUTH_STEP_DEG, DEM_RADIUS_M)

    # 12 — Mean slope
    nr, nc = elev.shape
    dpx = abs(np.nanmedian(np.diff(dx_grid[nr//2,:])))
    dpy = abs(np.nanmedian(np.diff(dy_grid[:,nc//2])))
    if dpx < 1: dpx = 30.0
    if dpy < 1: dpy = 30.0
    ag = np.arctan(np.sqrt(np.gradient(elev,dpx,axis=1)**2 +
                           np.gradient(elev,dpy,axis=0)**2))
    fpm        = dist_grid <= r86
    mean_slope = float(np.nanmean(ag[fpm])) if np.any(fpm) else 0.0

    kappa_tot     = kappa_topo * kappa_muon
    theta_v_corr  = THETA_V_INIT / kappa_topo if kappa_topo > 0 else float("inf")

    results = dict(
        sensor_alt=s_elev, pressure=pressure, rho_air=rho_air,
        mean_slope_rad=mean_slope,
        max_horizon=float(horizon.max()), mean_horizon=float(horizon.mean()),
        r86_sealevel=r86_sea, r86=r86, z86=z86,
        V0=V0, Veff=Veff,
        kappa_topo=kappa_topo, kappa_muon=kappa_muon, kappa_total=kappa_tot,
        kappa_pieno=kappa_pieno, kappa_sopra=kappa_sopra,
        kappa_vuoto=kappa_vuoto, kappa_info=kappa_info,
        theta_v_corrected=theta_v_corr,
        site_fluxes=site_fluxes,
        site_climate=site_climate,
        soil=soil,
        twi=twi,
        thermal=thermal,
        history=[],   # no iteration history with cell-summation method
    )
    params = dict(
        lat=LAT, lon=LON, h=SENSOR_HEIGHT_M,
        rho_b=RHO_BULK, theta_v_init=THETA_V_INIT,
        dem_radius=DEM_RADIUS_M, dem_source=DEM_SOURCE,
        az_step=AZIMUTH_STEP_DEG, n_cores=N_CORES,
        omega="n/a", tol="n/a",
    )

    # 13 — Report
    rpt = _outpath("crns_report.txt")
    print(f"\n[13] Writing report -> {rpt}")
    print(write_report(rpt, params, results))

    # 14 — Plots
    print("[14] Generating plots ...")
    plot_main(elev, dx_grid, dy_grid, r86, kappa_topo, kappa_muon,
              results, _outpath("crns_topo_main.png"),
              lat=LAT, lon=LON, dem_radius_m=DEM_RADIUS_M)
    plot_footprint(elev, dx_grid, dy_grid, dist_grid, s_elev, r86, z86,
                   kappa_topo, wmap, az_neutron, overlap_az, deficit_az,
                   _outpath("crns_footprint.png"))
    plot_horizon(azimuths, horizon, kappa_muon, per_az_muon,
                 _outpath("crns_horizon.png"))
    plot_fov_detail(azimuths, horizon, per_az_muon, kappa_muon,
                    az_neutron, overlap_az, r86, z86, kappa_topo,
                    _outpath("crns_fov_detail.png"),
                    lat=LAT, lon=LON, sensor_alt=s_elev)

    elapsed = time.perf_counter() - t0
    print(f"\n[DONE]  wall time = {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Output dir : {_OUT}")
    print(f"  Files:")
    for fn in ["crns_report.txt","crns_topo_main.png",
               "crns_footprint.png","crns_horizon.png","crns_fov_detail.png"]:
        p = _outpath(fn)
        sz_kb = os.path.getsize(p)//1024 if os.path.exists(p) else 0
        print(f"    {fn}  ({sz_kb} kB)")


if __name__ == "__main__":
    mp.freeze_support()
    main()
