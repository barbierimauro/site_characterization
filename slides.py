"""
landslide.py
============
Mappa suscettibilità frane su griglia DEM.

Input:
  elev        : 2D array DEM [m]
  lats_1d     : 1D array latitudini DEM
  lons_1d     : 1D array longitudini DEM
  macrostrat  : lista unità geologiche da Macrostrat API
  sm_res      : dict da get_era5_soil_moisture() o fuse_soil_moisture()
  soil_res    : dict da get_soil_properties()
  site_lat/lon: coordinate sensore

Output:
  fs_map      : 2D array fattore di sicurezza FS
  susc_map    : 2D array classi 1-5 (1=very low, 5=very high)
  + report e plot

Modello: infinite slope stability
  FS = [c' + (γs - m·γw)·z·cos²β·tanφ'] / [γs·z·sinβ·cosβ]

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import os, json, gzip, hashlib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Lookup geotecnico per litologia Macrostrat
# c' [kPa], phi' [°], gamma_s [kN/m³]
# (valore_medio, range_incertezza)
# ---------------------------------------------------------------------------

LITH_PARAMS = {
    # lith_type -> (c_kPa, dc, phi_deg, dphi, gamma_s)
    # sedimentary
    "limestone"   : (15.0, 8.0, 38.0, 5.0, 26.0),
    "dolostone"   : (15.0, 8.0, 38.0, 5.0, 26.5),
    "sandstone"   : ( 8.0, 4.0, 33.0, 4.0, 23.5),
    "shale"       : (10.0, 6.0, 20.0, 5.0, 20.0),
    "mudstone"    : ( 8.0, 5.0, 18.0, 5.0, 19.5),
    "conglomerate": ( 5.0, 3.0, 35.0, 4.0, 23.0),
    "flysch"      : ( 5.0, 3.0, 24.0, 5.0, 22.0),
    # metamorphic
    "schist"      : ( 5.0, 3.0, 26.0, 5.0, 25.0),
    "gneiss"      : (10.0, 5.0, 38.0, 4.0, 26.5),
    "phyllite"    : ( 4.0, 3.0, 24.0, 5.0, 24.0),
    "marble"      : (12.0, 6.0, 36.0, 4.0, 27.0),
    "quartzite"   : (10.0, 5.0, 40.0, 4.0, 26.5),
    # igneous
    "granite"     : (10.0, 5.0, 40.0, 4.0, 27.0),
    "basalt"      : ( 8.0, 4.0, 38.0, 4.0, 28.0),
    "rhyolite"    : ( 8.0, 4.0, 36.0, 4.0, 26.0),
    "andesite"    : ( 8.0, 4.0, 36.0, 4.0, 26.5),
    # unconsolidated
    "alluvium"    : ( 2.0, 2.0, 30.0, 5.0, 19.0),
    "glacial"     : ( 3.0, 2.0, 32.0, 5.0, 20.0),
    "colluvium"   : ( 2.0, 2.0, 28.0, 5.0, 18.5),
    "till"        : ( 4.0, 2.0, 32.0, 4.0, 20.5),
    # default per tipi non riconosciuti
    "default"     : ( 5.0, 5.0, 28.0, 8.0, 22.0),
}

# Classi suscettibilità
SUSC_THRESHOLDS = [1.0, 1.25, 1.5, 2.0]   # FS boundaries
SUSC_LABELS     = ["Very High", "High", "Moderate", "Low", "Very Low"]
SUSC_COLORS     = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9641"]

GAMMA_W = 9.81   # kN/m³


# ---------------------------------------------------------------------------
# Rasterizza unità geologiche Macrostrat sulla griglia DEM
# ---------------------------------------------------------------------------

def rasterize_geology(macrostrat_units, lats_1d, lons_1d, verbose=True):
    """
    Rasterizza le unità geologiche Macrostrat sulla griglia DEM.
    Ogni pixel riceve i parametri geotecnici (c', φ', γs).

    macrostrat_units: lista di dict da Macrostrat API
      Ogni dict deve avere 'geometry' (GeoJSON polygon) e 'lith' o
      'lith_type' con il tipo litologico.

    Ritorna dict con array 2D:
      c_map, dc_map, phi_map, dphi_map, gamma_map
    """
    from shapely.geometry import shape, Point
    from shapely.ops import unary_union

    nr = len(lats_1d)
    nc = len(lons_1d)

    c_map     = np.full((nr, nc), LITH_PARAMS["default"][0])
    dc_map    = np.full((nr, nc), LITH_PARAMS["default"][1])
    phi_map   = np.full((nr, nc), LITH_PARAMS["default"][2])
    dphi_map  = np.full((nr, nc), LITH_PARAMS["default"][3])
    gamma_map = np.full((nr, nc), LITH_PARAMS["default"][4])

    if not macrostrat_units:
        if verbose:
            print("   Geology: no units, using defaults", flush=True)
        return dict(c_map=c_map, dc_map=dc_map, phi_map=phi_map,
                    dphi_map=dphi_map, gamma_map=gamma_map)

    # Prepara geometrie shapely
    unit_geoms = []
    for u in macrostrat_units:
        geom_data = u.get("geometry") or u.get("geom")
        if not geom_data:
            continue
        try:
            geom  = shape(geom_data)
            lith  = _get_lith_key(u)
            params= LITH_PARAMS.get(lith, LITH_PARAMS["default"])
            unit_geoms.append((geom, params))
        except Exception:
            continue

    if verbose:
        print(f"   Geology: {len(unit_geoms)} units to rasterize",
              flush=True)

    # Per ogni pixel: trova l'unità che lo contiene
    # Ottimizzazione: usa bounding box prima del test esatto
    LONS, LATS = np.meshgrid(lons_1d, lats_1d)

    for i in range(nr):
        for j in range(nc):
            pt = Point(lons_1d[j], lats_1d[i])
            for geom, params in unit_geoms:
                try:
                    if geom.contains(pt):
                        c_map[i,j]     = params[0]
                        dc_map[i,j]    = params[1]
                        phi_map[i,j]   = params[2]
                        dphi_map[i,j]  = params[3]
                        gamma_map[i,j] = params[4]
                        break
                except Exception:
                    continue

    if verbose:
        unique_c = len(np.unique(c_map.round(1)))
        print(f"   Geology: rasterized, {unique_c} distinct c' values",
              flush=True)

    return dict(c_map=c_map, dc_map=dc_map,
                phi_map=phi_map, dphi_map=dphi_map,
                gamma_map=gamma_map)


def _get_lith_key(unit):
    """Estrae chiave litologica da unità Macrostrat."""
    # Prova diversi campi comuni dell'API Macrostrat
    for field in ["lith", "lith_type", "map_unit_name",
                  "name", "unit_name"]:
        val = unit.get(field, "")
        if isinstance(val, list) and val:
            val = val[0].get("lith","") if isinstance(val[0],dict) else str(val[0])
        val = str(val).lower().strip()
        for key in LITH_PARAMS:
            if key in val:
                return key
    return "default"


# ---------------------------------------------------------------------------
# Calcolo pendenza e curvatura dal DEM
# ---------------------------------------------------------------------------

def compute_slope(elev, lats_1d, lons_1d):
    """
    Calcola pendenza [rad] e curvatura dal DEM.
    Usa gradiente centrale con passo metrico.
    """
    c      = np.cos(np.radians(np.mean(lats_1d)))
    dlat   = abs(float(np.mean(np.diff(lats_1d)))) * 111320.0
    dlon   = abs(float(np.mean(np.diff(lons_1d)))) * 111320.0 * c

    elev_f = np.where(np.isnan(elev),
                      float(np.nanmean(elev)), elev)
    dz_dy, dz_dx = np.gradient(elev_f, dlat, dlon)
    slope  = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

    # Curvatura planare (2° derivata)
    d2z_dx2 = np.gradient(dz_dx, dlon, axis=1)
    d2z_dy2 = np.gradient(dz_dy, dlat, axis=0)
    curv    = -(d2z_dx2 + d2z_dy2)   # negativo = concavo = accumulo

    return slope.astype(np.float32), curv.astype(np.float32)


# ---------------------------------------------------------------------------
# Calcolo FS
# ---------------------------------------------------------------------------

def compute_fs(slope, geo, sm_monthly_mean, soil_res,
                z_m=1.5, dz_m=0.5, month=None):
    """
    Calcola FS su griglia con propagazione incertezza.

    FS = [c' + (γs - m·γw)·z·cos²β·tanφ'] / [γs·z·sinβ·cosβ]

    Parameters
    ----------
    slope         : 2D [rad]
    geo           : dict da rasterize_geology()
    sm_monthly_mean: array (12,) SM media mensile [m³/m³]
    soil_res      : dict da get_soil_properties()
    z_m           : profondità strato [m] default 1.5
    dz_m          : incertezza su z [m]
    month         : int 1-12 o None (usa media annua)

    Returns
    -------
    fs_mean, fs_low, fs_high : 2D array
    m_map : 2D array grado saturazione
    """
    # Grado saturazione m = SM / SAT
    sat   = float(soil_res.get("theta_sat", 0.45))
    if sat < 0.01: sat = 0.45

    if month is not None:
        sm_val = float(sm_monthly_mean[month-1])
    else:
        sm_val = float(np.nanmean(sm_monthly_mean))

    m_val = float(np.clip(sm_val / sat, 0.0, 1.0))
    m_map = np.full(slope.shape, m_val, dtype=np.float32)

    c     = geo["c_map"];    dc    = geo["dc_map"]
    phi   = np.radians(geo["phi_map"])
    dphi  = np.radians(geo["dphi_map"])
    gs    = geo["gamma_map"]

    beta  = slope
    cos2b = np.cos(beta)**2
    sinb  = np.sin(beta)
    cosb  = np.cos(beta)

    def _fs(c_v, phi_v, gs_v, z_v, m_v):
        num = c_v + (gs_v - m_v*GAMMA_W)*z_v*cos2b*np.tan(phi_v)
        den = gs_v * z_v * sinb * cosb
        den = np.where(np.abs(den) < 0.01, 0.01, den)
        return np.clip(num / den, 0.0, 10.0)

    fs_mean = _fs(c, phi, gs, z_m, m_map)
    fs_low  = _fs(np.maximum(c-dc, 0),
                   np.maximum(phi-dphi, np.radians(5)),
                   gs, z_m+dz_m, np.minimum(m_map+0.1, 1.0))
    fs_high = _fs(c+dc,
                   np.minimum(phi+dphi, np.radians(80)),
                   gs, np.maximum(z_m-dz_m, 0.3),
                   np.maximum(m_map-0.1, 0.0))

    # Pendenza nulla o piatta -> molto stabile
    flat  = sinb < np.sin(np.radians(2))
    for arr in (fs_mean, fs_low, fs_high):
        arr[flat] = 10.0

    return (fs_mean.astype(np.float32),
            fs_low.astype(np.float32),
            fs_high.astype(np.float32),
            m_map)


# ---------------------------------------------------------------------------
# Classificazione suscettibilità
# ---------------------------------------------------------------------------

def classify_susceptibility(fs_mean, fs_low):
    """
    Classe 5=very high ... 1=very low.
    Usa il pessimistico (fs_low) per non sottostimare il rischio.
    """
    fs_use = np.minimum(fs_mean, fs_low)
    susc   = np.ones(fs_use.shape, dtype=np.int8)
    for i, thr in enumerate(SUSC_THRESHOLDS):
        susc = np.where(fs_use < thr,
                        len(SUSC_THRESHOLDS)+1-i, susc)
    return susc.astype(np.int8)


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def compute_landslide(elev, lats_1d, lons_1d,
                       macrostrat_units,
                       sm_res, soil_res,
                       site_lat, site_lon,
                       z_m=1.5, dz_m=0.5,
                       month=None,
                       verbose=True):
    """
    Calcola mappa suscettibilità frane su tutto il DEM.

    Parameters
    ----------
    elev            : 2D float array DEM [m]
    lats_1d, lons_1d: 1D coordinate griglia
    macrostrat_units: lista unità da Macrostrat API
    sm_res          : da get_era5_soil_moisture() o fuse_soil_moisture()
    soil_res        : da get_soil_properties()
    site_lat, site_lon : coordinate sensore
    z_m, dz_m       : profondità e incertezza strato scivolante [m]
    month           : 1-12 per SM mensile, None per media annua
    verbose         : stampa progressi

    Returns
    -------
    dict con:
      fs_mean, fs_low, fs_high  : 2D array fattore di sicurezza
      susc_map                  : 2D array classi 1-5
      slope_map                 : 2D array pendenza [°]
      m_map                     : 2D array grado saturazione
      class_areas_m2            : dict {label: area_m2}
      class_fractions           : dict {label: fraction}
      fs_at_sensor              : (mean, low, high) FS nel pixel sensore
      susc_at_sensor            : int, classe suscettibilità sensore
      high_risk_zones           : list di dict con zone ad alto rischio
                                   vicino al sensore
      lats_1d, lons_1d          : griglia (passthrough)
      site_lat, site_lon
      month, z_m
    """
    if verbose:
        print("   Landslide: computing slope ...", flush=True)
    slope, curv = compute_slope(elev, lats_1d, lons_1d)

    if verbose:
        print("   Landslide: rasterizing geology ...", flush=True)
    geo = rasterize_geology(macrostrat_units, lats_1d, lons_1d,
                             verbose)

    # SM: prova sm_res da fuse o da era5
    sm_key = "theta_monthly"
    if sm_key in sm_res:
        sm_monthly = sm_res[sm_key]
    else:
        sm_monthly = sm_res.get("sm0_7_monthly_mean",
                                 np.full(12, 0.25))

    if verbose:
        print("   Landslide: computing FS ...", flush=True)
    fs_mean, fs_low, fs_high, m_map = compute_fs(
        slope, geo, sm_monthly, soil_res, z_m, dz_m, month)

    susc = classify_susceptibility(fs_mean, fs_low)

    # Pixel size
    c    = np.cos(np.radians(float(np.mean(lats_1d))))
    dlat = abs(float(np.mean(np.diff(lats_1d)))) * 111320.0
    dlon = abs(float(np.mean(np.diff(lons_1d)))) * 111320.0 * c
    px_area = dlat * dlon

    # Aree per classe
    valid = ~np.isnan(elev)
    total = float(valid.sum()) * px_area
    class_areas = {}
    class_fracs = {}
    for i, lbl in enumerate(SUSC_LABELS, start=1):
        n = int(np.sum(susc[valid] == (6-i)))
        class_areas[lbl] = n * px_area
        class_fracs[lbl] = n * px_area / total if total > 0 else 0.0

    # FS al sensore
    si = int(np.argmin(np.abs(lats_1d - site_lat)))
    sj = int(np.argmin(np.abs(lons_1d - site_lon)))
    fs_at_sensor = (float(fs_mean[si,sj]),
                    float(fs_low[si,sj]),
                    float(fs_high[si,sj]))
    susc_at_sensor = int(susc[si,sj])

    # Zone ad alto rischio (susc >= 4) entro 2 km dal sensore
    high_risk = []
    for i in range(susc.shape[0]):
        for j in range(susc.shape[1]):
            if susc[i,j] < 4:
                continue
            d = np.sqrt(
                ((lats_1d[i]-site_lat)*111320)**2 +
                ((lons_1d[j]-site_lon)*111320*c)**2)
            if d < 2000:
                high_risk.append({
                    "lat": float(lats_1d[i]),
                    "lon": float(lons_1d[j]),
                    "d_m": float(d),
                    "fs_mean": float(fs_mean[i,j]),
                    "susc": int(susc[i,j]),
                    "slope_deg": float(np.degrees(slope[i,j])),
                })
    high_risk.sort(key=lambda x: x["d_m"])

    if verbose:
        vh = int(np.sum(susc[valid] == 5))
        print(f"   Landslide: done  "
              f"very_high={vh} pixels  "
              f"FS_sensor={fs_at_sensor[0]:.2f}", flush=True)

    return dict(
        fs_mean         = fs_mean,
        fs_low          = fs_low,
        fs_high         = fs_high,
        susc_map        = susc,
        slope_map       = np.degrees(slope).astype(np.float32),
        m_map           = m_map,
        class_areas_m2  = class_areas,
        class_fractions = class_fracs,
        fs_at_sensor    = fs_at_sensor,
        susc_at_sensor  = susc_at_sensor,
        high_risk_zones = high_risk,
        lats_1d         = lats_1d,
        lons_1d         = lons_1d,
        site_lat        = site_lat,
        site_lon        = site_lon,
        month           = month,
        z_m             = z_m,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report_landslide(res):
    w   = 72
    fs  = res["fs_at_sensor"]
    sc  = res["susc_at_sensor"]
    lbl = SUSC_LABELS[5 - sc] if 1 <= sc <= 5 else "?"
    L   = ["="*w,
           "LANDSLIDE SUSCEPTIBILITY  (infinite slope model)",
           "="*w,
           f"  Site: {res['site_lat']:.4f}N  {res['site_lon']:.4f}E",
           f"  z = {res['z_m']:.1f}m  |  "
           f"month = {res['month'] or 'annual mean'}",
           "",
           f"  FS at sensor: {fs[0]:.2f}  "
           f"[{fs[1]:.2f}, {fs[2]:.2f}]  ->  {lbl}",
           ""]

    L.append(f"  {'Class':<12} {'Area km²':>10} {'Fraction':>10}")
    L.append("  " + "-"*(w-2))
    for lbl2 in SUSC_LABELS:
        a = res["class_areas_m2"].get(lbl2, 0) / 1e6
        f = res["class_fractions"].get(lbl2, 0)
        bar = "█" * int(f * 30)
        L.append(f"  {lbl2:<12} {a:>10.2f} {f:>10.1%}  {bar}")

    if res["high_risk_zones"]:
        L += ["",
              f"  High/Very High zones within 2 km of sensor "
              f"({len(res['high_risk_zones'])} pixels):"]
        for z in res["high_risk_zones"][:8]:
            L.append(f"    d={z['d_m']:6.0f}m  "
                     f"FS={z['fs_mean']:.2f}  "
                     f"slope={z['slope_deg']:.0f}°  "
                     f"class={SUSC_LABELS[5-z['susc']]}")
    else:
        L.append("\n  No high-risk zones within 2 km of sensor.")

    L.append("="*w)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_landslide(res, path, site_name="", r86_m=150.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    lats  = res["lats_1d"]
    lons  = res["lons_1d"]
    c     = np.cos(np.radians(float(np.mean(lats))))
    LONS, LATS = np.meshgrid(lons, lats)
    # Coordinate metriche centrate sul sensore
    DX = (LONS - res["site_lon"]) * 111320.0 * c / 1000
    DY = (LATS - res["site_lat"]) * 111320.0 / 1000

    susc = res["susc_map"].astype(float)
    susc[susc == 0] = np.nan
    fs   = res["fs_mean"]
    slp  = res["slope_map"]

    cmap = ListedColormap(SUSC_COLORS[::-1])
    norm = BoundaryNorm([0.5,1.5,2.5,3.5,4.5,5.5], 5)

    theta  = np.linspace(0, 2*np.pi, 360)
    r86_km = r86_m / 1000.0
    clip   = r86_km * 1.3   # axis limits [km]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                              facecolor="white")

    # ---- 1: mappa suscettibilità ----
    ax = axes[0]
    im = ax.pcolormesh(DX, DY, susc, cmap=cmap, norm=norm,
                        shading="auto")
    ax.plot(r86_km*np.sin(theta), r86_km*np.cos(theta),
            "w--", lw=1.5, label=f"r86={r86_m:.0f}m")
    ax.plot(0, 0, "w^", ms=10, zorder=5)
    patches = [mpatches.Patch(color=SUSC_COLORS[4-i],
               label=SUSC_LABELS[i]) for i in range(5)]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    ax.set_xlim(-clip, clip)
    ax.set_ylim(-clip, clip)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting [km]"); ax.set_ylabel("Northing [km]")
    ax.set_title("Susceptibility classes\n(infinite slope model)",
                 fontsize=11)

    # ---- 2: mappa FS ----
    ax2 = axes[1]
    fs_plot = np.clip(fs, 0, 4)
    fs_plot[np.isnan(res["slope_map"])] = np.nan
    im2 = ax2.pcolormesh(DX, DY, fs_plot,
                          cmap="RdYlGn", vmin=0, vmax=4,
                          shading="auto")
    ax2.contour(DX, DY, fs, levels=[1.0, 1.5],
                colors=["red","orange"], linewidths=1.0)
    ax2.plot(r86_km*np.sin(theta), r86_km*np.cos(theta),
             "k--", lw=1.5)
    ax2.plot(0, 0, "k^", ms=10, zorder=5)
    plt.colorbar(im2, ax=ax2, label="FS")
    ax2.set_xlim(-clip, clip)
    ax2.set_ylim(-clip, clip)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Easting [km]")
    ax2.set_title("Safety Factor (FS)\n"
                  "Red=FS<1  Orange=FS<1.5", fontsize=11)

    # ---- 3: mappa pendenza ----
    ax3 = axes[2]
    im3 = ax3.pcolormesh(DX, DY, slp, cmap="terrain_r",
                          vmin=0, vmax=60, shading="auto")
    ax3.plot(r86_km*np.sin(theta), r86_km*np.cos(theta),
             "w--", lw=1.5)
    ax3.plot(0, 0, "w^", ms=10, zorder=5)
    plt.colorbar(im3, ax=ax3, label="Slope [°]")
    ax3.set_xlim(-clip, clip)
    ax3.set_ylim(-clip, clip)
    ax3.set_aspect("equal")
    ax3.set_xlabel("Easting [km]")
    ax3.set_title("Slope [°]", fontsize=11)

    fig.suptitle(f"Landslide Susceptibility  |  {site_name}  |  "
                 f"{res['site_lat']:.4f}N {res['site_lon']:.4f}E",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
