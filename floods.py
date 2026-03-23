"""
flood_core.py
=============
Analisi suscettibilità alluvionale per siti CRNS.

Pipeline:
  1. Mosaico DEM ibrido: GLO-30 (zona interna) + GLO-90 (zona esterna)
     ricampionato a 30m per continuità idrologica
  2. Depression filling (Wang & Liu 2006, priority queue)
  3. Flow direction D8
  4. Flow accumulation
  5. Reticolo idrografico con soglia adattiva (interna/esterna)
  6. HAND — Height Above Nearest Drainage
  7. FRI — Flood Risk Index (HAND + flow_acc + JRC + precip)

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import os
import gzip
import hashlib
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Direzioni D8: (drow, dcol, distanza_relativa)
D8_DIRS = [
    (-1,-1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
    ( 0,-1, 1.0),                  ( 0, 1, 1.0),
    ( 1,-1, 1.414), ( 1, 0, 1.0), ( 1, 1, 1.414),
]
# Codici D8 (potenze di 2, compatibili con ArcGIS/GRASS)
D8_CODES = [32, 64, 128, 16, 1, 8, 4, 2]

# Soglie flow accumulation per reticolo idrografico
FA_THRESH_INNER = 50    # pixel, zona r < r_inner (GLO-30)
FA_THRESH_OUTER = 500   # pixel, zona r > r_inner (GLO-90)

# Pesi FRI
FRI_WEIGHTS = dict(hand=0.45, flow_acc=0.25, jrc=0.20, precip=0.10)

# Classi suscettibilità alluvionale
FLOOD_THRESHOLDS = [2.0, 5.0, 10.0, 20.0]   # HAND [m] boundaries
FLOOD_LABELS     = ["Very High", "High", "Moderate", "Low", "Very Low"]
FLOOD_COLORS     = ["#023858", "#0570b0", "#74a9cf", "#bdc9e1", "#f1eef6"]


# ---------------------------------------------------------------------------
# Cache helpers — compute_flood
# ---------------------------------------------------------------------------

def _flood_hash(site_lat, site_lon, r_inner_km):
    tag = f"{site_lat:.5f}_{site_lon:.5f}_{r_inner_km:.3f}"
    return hashlib.sha256(tag.encode()).hexdigest()[:16]


def _flood_cache_path(cache_dir, site_lat, site_lon, r_inner_km):
    return os.path.join(cache_dir,
                        f"flood_{_flood_hash(site_lat, site_lon, r_inner_km)}.pkl.gz")


def load_flood_cache(cache_dir, site_lat, site_lon, r_inner_km):
    """Ritorna il dict risultato o None se non in cache."""
    if cache_dir is None:
        return None
    p = _flood_cache_path(cache_dir, site_lat, site_lon, r_inner_km)
    if not os.path.exists(p):
        return None
    try:
        with gzip.open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_flood_cache(cache_dir, site_lat, site_lon, r_inner_km, result):
    """Salva il dict risultato in cache."""
    if cache_dir is None:
        return
    os.makedirs(cache_dir, exist_ok=True)
    p = _flood_cache_path(cache_dir, site_lat, site_lon, r_inner_km)
    with gzip.open(p, "wb") as f:
        pickle.dump(result, f, protocol=4)


# ---------------------------------------------------------------------------
# 1. Mosaico DEM ibrido
# ---------------------------------------------------------------------------

def build_mosaic(elev_30, lats_30, lons_30,
                  elev_90, lats_90, lons_90,
                  site_lat, site_lon,
                  r_inner_km=2.0, verbose=True):
    """
    Crea DEM mosaico a 30m su tutta l'area del GLO-90.
    Zona interna (r < r_inner_km): valori GLO-30 originali.
    Zona esterna: GLO-90 bicubic resample a 30m.

    Returns
    -------
    elev_m, lats_m, lons_m  : DEM mosaico e griglie 1D
    inner_mask              : bool 2D, True = zona GLO-30
    """
    from scipy.interpolate import RegularGridInterpolator

    # Griglia output: passo 30m sull'intera estensione GLO-90
    c        = np.cos(np.radians(site_lat))
    dlat_30  = abs(float(np.mean(np.diff(lats_30))))
    dlon_30  = abs(float(np.mean(np.diff(lons_30))))

    lat_min  = float(min(lats_90.min(), lats_30.min()))
    lat_max  = float(max(lats_90.max(), lats_30.max()))
    lon_min  = float(min(lons_90.min(), lons_30.min()))
    lon_max  = float(max(lons_90.max(), lons_30.max()))

    lats_m   = np.arange(lat_min, lat_max + dlat_30/2, dlat_30)
    lons_m   = np.arange(lon_min, lon_max + dlon_30/2, dlon_30)
    LONS_M, LATS_M = np.meshgrid(lons_m, lats_m)

    # Interpola GLO-90 sull'intera griglia a 30m
    lats_90s  = np.sort(lats_90)
    lons_90s  = np.sort(lons_90)
    e90       = elev_90.copy()
    if lats_90[0] > lats_90[-1]: e90 = e90[::-1, :]
    if lons_90[0] > lons_90[-1]: e90 = e90[:, ::-1]
    e90       = np.where(np.isnan(e90), float(np.nanmean(e90)), e90)

    f90 = RegularGridInterpolator(
        (lats_90s, lons_90s), e90, method="linear",
        bounds_error=False, fill_value=float(np.nanmean(e90)))
    pts      = np.column_stack([LATS_M.ravel(), LONS_M.ravel()])
    elev_m   = f90(pts).reshape(LATS_M.shape)

    # Sovrapponi GLO-30 nella zona interna
    lats_30s  = np.sort(lats_30)
    lons_30s  = np.sort(lons_30)
    e30       = elev_30.copy()
    if lats_30[0] > lats_30[-1]: e30 = e30[::-1, :]
    if lons_30[0] > lons_30[-1]: e30 = e30[:, ::-1]
    e30_valid = np.where(np.isnan(e30), float(np.nanmean(e30)), e30)

    f30 = RegularGridInterpolator(
        (lats_30s, lons_30s), e30_valid, method="linear",
        bounds_error=False, fill_value=np.nan)
    elev_30_on_m = f30(pts).reshape(LATS_M.shape)

    # Maschera zona interna
    dx_km = (LONS_M - site_lon) * 111320.0 * c / 1000
    dy_km = (LATS_M - site_lat) * 111320.0 / 1000
    dist_km    = np.sqrt(dx_km**2 + dy_km**2)
    inner_mask = (dist_km <= r_inner_km) & ~np.isnan(elev_30_on_m)

    elev_m[inner_mask] = elev_30_on_m[inner_mask]
    elev_m = elev_m.astype(np.float32)

    if verbose:
        print(f"   Mosaic: {elev_m.shape}  "
              f"inner={inner_mask.sum()} px  "
              f"res=30m", flush=True)

    return elev_m, lats_m, lons_m, inner_mask


# ---------------------------------------------------------------------------
# 2. Depression filling (Wang & Liu 2006)
# ---------------------------------------------------------------------------

def fill_depressions(elev, verbose=True):
    """
    Wang & Liu 2006 priority-queue depression filling.
    Ritorna DEM riempito.
    """
    import heapq

    nr, nc   = elev.shape
    filled   = np.where(np.isnan(elev), -9999.0, elev).astype(np.float64)
    in_queue = np.zeros((nr, nc), dtype=bool)
    heap     = []

    # Inizializza bordi
    for i in range(nr):
        for j in [0, nc-1]:
            if not in_queue[i,j]:
                heapq.heappush(heap, (filled[i,j], i, j))
                in_queue[i,j] = True
    for j in range(nc):
        for i in [0, nr-1]:
            if not in_queue[i,j]:
                heapq.heappush(heap, (filled[i,j], i, j))
                in_queue[i,j] = True

    n_filled = 0
    while heap:
        z, r, c_idx = heapq.heappop(heap)
        for dr, dc, _ in D8_DIRS:
            nr2 = r + dr; nc2 = c_idx + dc
            if nr2 < 0 or nr2 >= nr or nc2 < 0 or nc2 >= nc:
                continue
            if in_queue[nr2, nc2]:
                continue
            if filled[nr2, nc2] < z:
                filled[nr2, nc2] = z
                n_filled += 1
            heapq.heappush(heap, (filled[nr2, nc2], nr2, nc2))
            in_queue[nr2, nc2] = True

    if verbose:
        print(f"   Fill: {n_filled} pixels filled", flush=True)
    return filled.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Flow direction D8
# ---------------------------------------------------------------------------

def flow_direction_d8(filled):
    """
    Calcola flow direction D8 (codice potenza di 2).
    Cella piatta o bordo → 0.
    """
    nr, nc    = filled.shape
    flowdir   = np.zeros((nr, nc), dtype=np.uint8)
    max_slope = np.full((nr, nc), -np.inf, dtype=np.float64)
    max_code  = np.zeros((nr, nc), dtype=np.uint8)

    for idx, (dr, dc, dist) in enumerate(D8_DIRS):
        r0, r1 = max(0, -dr), min(nr, nr - dr)
        c0, c1 = max(0, -dc), min(nc, nc - dc)
        r_src  = np.arange(r0, r1)
        c_src  = np.arange(c0, c1)

        z_src  = filled[np.ix_(r_src, c_src)]
        z_nbr  = filled[np.ix_(r_src + dr, c_src + dc)]
        slope  = (z_src - z_nbr) / dist

        # Aggiorna max_slope / max_code sulla griglia full-size
        s_cur  = max_slope[np.ix_(r_src, c_src)]
        better = slope > s_cur
        max_slope[np.ix_(r_src, c_src)] = np.where(better, slope, s_cur)
        c_cur  = max_code[np.ix_(r_src, c_src)]
        max_code[np.ix_(r_src, c_src)]  = np.where(
            better, np.uint8(D8_CODES[idx]), c_cur).astype(np.uint8)

    # Scrivi flowdir dove c'è un drop positivo
    pos = max_slope > 0
    flowdir[pos] = max_code[pos]
    return flowdir


# ---------------------------------------------------------------------------
# 4. Flow accumulation
# ---------------------------------------------------------------------------

def flow_accumulation(flowdir, verbose=True):
    """Flow accumulation D8 — ordine topologico."""
    from collections import deque
    nr, nc = flowdir.shape
    acc    = np.ones((nr, nc), dtype=np.int32)
    n_in   = np.zeros((nr, nc), dtype=np.int32)
    c2d    = {D8_CODES[i]: D8_DIRS[i][:2] for i in range(len(D8_CODES))}
    for r in range(nr):
        for c in range(nc):
            code = int(flowdir[r,c])
            if code == 0: continue
            dr,dc = c2d.get(code,(0,0))
            rn,cn = r+dr, c+dc
            if 0<=rn<nr and 0<=cn<nc: n_in[rn,cn] += 1
    q = deque((r,c) for r in range(nr) for c in range(nc) if n_in[r,c]==0)
    while q:
        r,c  = q.popleft()
        code = int(flowdir[r,c])
        if code == 0: continue
        dr,dc = c2d.get(code,(0,0))
        rn,cn = r+dr, c+dc
        if 0<=rn<nr and 0<=cn<nc:
            acc[rn,cn] += acc[r,c]
            n_in[rn,cn] -= 1
            if n_in[rn,cn] == 0: q.append((rn,cn))
    if verbose: print(f"   FlowAcc: max={acc.max()}", flush=True)
    return acc.astype(np.int32)


# ---------------------------------------------------------------------------
# 5. Reticolo idrografico
# ---------------------------------------------------------------------------

def extract_network(flow_acc, inner_mask,
                     thresh_inner=FA_THRESH_INNER,
                     thresh_outer=FA_THRESH_OUTER):
    """
    Estrae reticolo idrografico con soglia adattiva.
    inner_mask: True = zona GLO-30 (soglia bassa)
    Ritorna maschera booleana 2D.
    """
    network = np.zeros(flow_acc.shape, dtype=bool)
    network[inner_mask]  = flow_acc[inner_mask]  >= thresh_inner
    network[~inner_mask] = flow_acc[~inner_mask] >= thresh_outer
    return network


def network_to_lines(network, lats_1d, lons_1d):
    """
    Converte la maschera reticolo in segmenti lineari (x,y)
    per il plot. Ritorna lista di array Nx2.
    """
    from scipy.ndimage import label

    labeled, n_feat = label(network)
    segments = []
    c = np.cos(np.radians(float(np.mean(lats_1d))))
    site_lat = float(np.mean(lats_1d))
    site_lon = float(np.mean(lons_1d))

    for i in range(1, min(n_feat+1, 2000)):
        mask  = labeled == i
        rows, cols = np.where(mask)
        if len(rows) < 2:
            continue
        # Ordina approssimativamente lungo la direzione principale
        order = np.argsort(rows)
        xs = (lons_1d[cols[order]] - site_lon) * 111320.0 * c / 1000
        ys = (lats_1d[rows[order]] - site_lat) * 111320.0 / 1000
        segments.append(np.column_stack([xs, ys]))

    return segments


# ---------------------------------------------------------------------------
# 6. HAND — Height Above Nearest Drainage
# ---------------------------------------------------------------------------

def compute_hand(filled, flowdir, network, verbose=True):
    """
    HAND = quota DEM - quota del canale più vicino a valle.
    Implementazione O(n) con BFS inverso dai canali upstream.
    Per ogni pixel la HAND è la sua quota meno la quota del
    canale a cui è idrologicamente connesso (seguendo D8 a valle).
    """
    from collections import deque
    nr, nc      = filled.shape
    hand        = np.full((nr, nc), np.nan, dtype=np.float32)
    chan_elev   = np.full((nr, nc), np.nan, dtype=np.float64)
    code_to_dir = {D8_CODES[i]: D8_DIRS[i][:2]
                   for i in range(len(D8_CODES))}

    # Costruisci adiacenza inversa: rev[r*nc+c] = lista pixel che
    # confluiscono in (r,c) secondo D8
    rev = [[] for _ in range(nr * nc)]
    for r in range(nr):
        for c in range(nc):
            code = int(flowdir[r, c])
            if code == 0:
                continue
            dr, dc = code_to_dir.get(code, (0, 0))
            rn, cn = r + dr, c + dc
            if 0 <= rn < nr and 0 <= cn < nc:
                rev[rn * nc + cn].append((r, c))

    # BFS partendo dai pixel-canale
    q = deque()
    for r in range(nr):
        for c in range(nc):
            if network[r, c]:
                hand[r, c]      = 0.0
                chan_elev[r, c] = float(filled[r, c])
                q.append((r, c))

    while q:
        r, c = q.popleft()
        ce   = chan_elev[r, c]
        for ru, cu in rev[r * nc + c]:
            if np.isnan(hand[ru, cu]):
                hand[ru, cu]      = max(0.0,
                                        float(filled[ru, cu]) - ce)
                chan_elev[ru, cu] = ce
                q.append((ru, cu))

    valid = int(np.sum(~np.isnan(hand)))
    if verbose:
        print(f"   HAND: valid={valid}  "
              f"median={float(np.nanmedian(hand)):.1f}m  "
              f"max={float(np.nanmax(hand)):.1f}m", flush=True)
    return hand


# ---------------------------------------------------------------------------
# 7. FRI — Flood Risk Index
# ---------------------------------------------------------------------------

def compute_fri(hand, flow_acc, jrc_res,
                 era5_res, lats_1d, lons_1d,
                 site_lat, site_lon, verbose=True):
    """
    FRI = 0.45·f(HAND) + 0.25·f(flow_acc) + 0.20·f(JRC) + 0.10·f(precip)
    Tutti i fattori normalizzati [0,1], FRI in [0,1].
    """
    nr, nc = hand.shape

    # f(HAND): sigmoide decrescente, 0m=1.0, 20m=0.0
    f_hand = np.where(np.isnan(hand), 0.0,
                       1.0 / (1.0 + np.exp((hand - 5.0) / 2.0)))

    # f(flow_acc): log-normalizzato
    fa_max  = float(flow_acc.max())
    if fa_max > 0:
        f_fa = np.log1p(flow_acc.astype(float)) / np.log1p(fa_max)
    else:
        f_fa = np.zeros_like(hand)

    # f(JRC): occurrence layer ricampionato sulla griglia
    f_jrc = _resample_jrc(jrc_res, lats_1d, lons_1d)

    # f(precip): scalare da percentile 95 precipitazione
    # Usa ERA5 precipitation se disponibile
    f_precip_val = _precip_extreme_index(era5_res)
    f_precip = np.full((nr, nc), f_precip_val, dtype=np.float32)

    w = FRI_WEIGHTS
    fri = (w["hand"]     * f_hand  +
           w["flow_acc"] * f_fa    +
           w["jrc"]      * f_jrc   +
           w["precip"]   * f_precip)
    fri = np.clip(fri, 0.0, 1.0).astype(np.float32)

    if verbose:
        print(f"   FRI: mean={float(np.nanmean(fri)):.3f}  "
              f"max={float(np.nanmax(fri)):.3f}", flush=True)
    return fri, f_hand, f_fa, f_jrc, f_precip


def _resample_jrc(jrc_res, lats_1d, lons_1d):
    """Ricampiona JRC occurrence sulla griglia di output."""
    nr = len(lats_1d); nc = len(lons_1d)
    if jrc_res is None:
        return np.zeros((nr, nc), dtype=np.float32)
    # jrc_res['eta'] è un float, non una griglia — usa come costante
    # Se jrc_res ha un array 'occurrence_map' usalo
    if "occurrence_map" in jrc_res:
        occ = jrc_res["occurrence_map"]
        if occ.shape == (nr, nc):
            return (occ / 100.0).astype(np.float32)
    # Fallback: usa eta come proxy uniforme
    eta = float(jrc_res.get("eta", 0.0))
    return np.full((nr, nc), eta, dtype=np.float32)


def _precip_extreme_index(era5_res):
    """
    Stima indice [0,1] da precipitazioni estreme ERA5.
    Usa il massimo mensile come proxy per eventi estremi.
    """
    if era5_res is None:
        return 0.5
    # Prova diverse chiavi possibili
    for key in ["precip_monthly_max", "precipitation_monthly_mean",
                "sm0_7_monthly_max"]:
        arr = era5_res.get(key)
        if arr is not None:
            mx = float(np.nanmax(arr))
            # Normalizza: > 200mm/mese = 1.0
            return float(np.clip(mx / 200.0, 0.0, 1.0))
    return 0.5


# ---------------------------------------------------------------------------
# Classificazione suscettibilità
# ---------------------------------------------------------------------------

def classify_flood(hand, fri):
    """
    Combina HAND e FRI per classificazione finale.
    Priorità a HAND per valori molto bassi (< 2m = sempre very high).
    """
    susc = np.ones(hand.shape, dtype=np.int8)
    for i, thr in enumerate(FLOOD_THRESHOLDS):
        susc = np.where(
            (~np.isnan(hand)) & (hand < thr),
            len(FLOOD_THRESHOLDS)+1-i, susc)
    # Integra FRI: aumenta di 1 classe se FRI > 0.7 e susc < 5
    susc = np.where(
        (fri > 0.7) & (susc < 5) & ~np.isnan(hand),
        susc + 1, susc)
    return np.clip(susc, 1, 5).astype(np.int8)


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def compute_flood(elev_30, lats_30, lons_30,
                   elev_90, lats_90, lons_90,
                   jrc_res, era5_res, osm_elements,
                   site_lat, site_lon,
                   r_inner_km=2.0,
                   verbose=True,
                   cache_dir=None):
    """
    Analisi suscettibilità alluvionale completa.

    Parameters
    ----------
    elev_30/lats_30/lons_30 : DEM GLO-30 zona interna
    elev_90/lats_90/lons_90 : DEM GLO-90 zona esterna
    jrc_res         : da compute_water_eta() o None
    era5_res        : da get_era5_soil_moisture() o None
    osm_elements    : da download_osm() per waterway
    site_lat, site_lon : coordinate sensore
    r_inner_km      : raggio zona GLO-30 [km]
    verbose         : stampa progressi
    cache_dir       : directory cache (None = no cache)

    Returns
    -------
    dict con: elev_m, lats_m, lons_m, inner_mask,
              network, network_segments, hand, fri,
              susc_map, class_areas_m2, class_fractions,
              hand_at_sensor, fri_at_sensor, susc_at_sensor,
              high_risk_zones, osm_waterways
    """
    cached = load_flood_cache(cache_dir, site_lat, site_lon, r_inner_km)
    if cached is not None:
        if verbose:
            print("[Flood] Loaded from cache.", flush=True)
        return cached

    if verbose:
        print("[Flood] Building DEM mosaic ...", flush=True)
    elev_m, lats_m, lons_m, inner = build_mosaic(
        elev_30, lats_30, lons_30,
        elev_90, lats_90, lons_90,
        site_lat, site_lon, r_inner_km, verbose)

    if verbose:
        print("[Flood] Filling depressions ...", flush=True)
    filled = fill_depressions(elev_m, verbose)

    if verbose:
        print("[Flood] Flow direction D8 ...", flush=True)
    flowdir = flow_direction_d8(filled)

    if verbose:
        print("[Flood] Flow accumulation ...", flush=True)
    flow_acc = flow_accumulation(flowdir, verbose)

    if verbose:
        print("[Flood] Extracting network ...", flush=True)
    network  = extract_network(flow_acc, inner)
    segments = network_to_lines(network, lats_m, lons_m)

    if verbose:
        print(f"[Flood] Network: {network.sum()} channel pixels  "
              f"{len(segments)} segments", flush=True)

    if verbose:
        print("[Flood] Computing HAND ...", flush=True)
    hand = compute_hand(filled, flowdir, network, verbose)

    if verbose:
        print("[Flood] Computing FRI ...", flush=True)
    fri, f_hand, f_fa, f_jrc, f_precip = compute_fri(
        hand, flow_acc, jrc_res, era5_res,
        lats_m, lons_m, site_lat, site_lon, verbose)

    susc = classify_flood(hand, fri)

    # Pixel size
    c     = np.cos(np.radians(site_lat))
    dlat  = abs(float(np.mean(np.diff(lats_m)))) * 111320.0
    dlon  = abs(float(np.mean(np.diff(lons_m)))) * 111320.0 * c
    px_a  = dlat * dlon
    valid = ~np.isnan(hand)
    total = float(valid.sum()) * px_a

    class_areas = {}; class_fracs = {}
    for i, lbl in enumerate(FLOOD_LABELS, start=1):
        n = int(np.sum(susc[valid] == (6-i)))
        class_areas[lbl] = n * px_a
        class_fracs[lbl] = n * px_a / total if total > 0 else 0.0

    # Valori al sensore
    si = int(np.argmin(np.abs(lats_m - site_lat)))
    sj = int(np.argmin(np.abs(lons_m - site_lon)))
    hand_s = float(hand[si, sj]) if not np.isnan(hand[si,sj]) else -1.0
    fri_s  = float(fri[si, sj])
    susc_s = int(susc[si, sj])

    # Zone ad alto rischio entro 3 km
    LONS_M, LATS_M = np.meshgrid(lons_m, lats_m)
    dx_m = (LONS_M - site_lon) * 111320.0 * c
    dy_m = (LATS_M - site_lat) * 111320.0
    dist = np.sqrt(dx_m**2 + dy_m**2)

    high_mask = (susc >= 4) & valid & (dist < 3000)
    rows_h, cols_h = np.where(high_mask)
    high_risk = []
    for r, cc in zip(rows_h[:200], cols_h[:200]):
        high_risk.append({
            "lat"     : float(lats_m[r]),
            "lon"     : float(lons_m[cc]),
            "d_m"     : float(dist[r, cc]),
            "hand_m"  : float(hand[r, cc]),
            "fri"     : float(fri[r, cc]),
            "susc"    : int(susc[r, cc]),
        })
    high_risk.sort(key=lambda x: x["d_m"])

    # Waterway OSM
    osm_ww = _extract_osm_waterways(
        osm_elements, site_lat, site_lon, lons_m, lats_m)

    if verbose:
        print(f"[Flood] Done.  HAND_sensor={hand_s:.1f}m  "
              f"susc={FLOOD_LABELS[5-susc_s]}", flush=True)

    result = dict(
        elev_m          = elev_m,
        lats_m          = lats_m,
        lons_m          = lons_m,
        inner_mask      = inner,
        filled          = filled,
        flow_acc        = flow_acc,
        network         = network,
        network_segments= segments,
        hand            = hand,
        fri             = fri,
        f_hand          = f_hand,
        f_fa            = f_fa.astype(np.float32),
        f_jrc           = f_jrc,
        susc_map        = susc,
        class_areas_m2  = class_areas,
        class_fractions = class_fracs,
        hand_at_sensor  = hand_s,
        fri_at_sensor   = fri_s,
        susc_at_sensor  = susc_s,
        high_risk_zones = high_risk,
        osm_waterways   = osm_ww,
        site_lat        = site_lat,
        site_lon        = site_lon,
        r_inner_km      = r_inner_km,
        px_area_m2      = px_a,
    )
    save_flood_cache(cache_dir, site_lat, site_lon, r_inner_km, result)
    if verbose and cache_dir is not None:
        print("[Flood] Result saved to cache.", flush=True)
    return result


def _extract_osm_waterways(elements, site_lat, site_lon,
                             lons_m, lons_m2):
    """Estrae segmenti waterway da OSM per overlay sulla mappa."""
    if not elements:
        return []
    c    = np.cos(np.radians(site_lat))
    segs = []
    for el in elements:
        tags = el.get("tags", {})
        if "waterway" not in tags:
            continue
        geom = el.get("geometry", [])
        if len(geom) < 2:
            continue
        xs = [(p["lon"]-site_lon)*111320.0*c/1000 for p in geom]
        ys = [(p["lat"]-site_lat)*111320.0/1000 for p in geom]
        wtype = tags.get("waterway","stream")
        segs.append({
            "xy"  : np.column_stack([xs, ys]),
            "type": wtype,
            "name": tags.get("name",""),
        })
    return segs


WATERWAY_STYLE = {
    "river"        : {"color": "#2166ac", "lw": 2.0},
    "stream"       : {"color": "#4393c3", "lw": 1.2},
    "canal"        : {"color": "#1a6faf", "lw": 1.5},
    "ditch"        : {"color": "#74a9cf", "lw": 0.8},
    "default"      : {"color": "#6baed6", "lw": 0.8},
}


# ---------------------------------------------------------------------------
# Helpers coordinate
# ---------------------------------------------------------------------------

def _metric_grids(res):
    """Ritorna DX, DY in km centrati sul sensore."""
    lats  = res["lats_m"]
    lons  = res["lons_m"]
    c     = np.cos(np.radians(res["site_lat"]))
    LONS, LATS = np.meshgrid(lons, lats)
    DX = (LONS - res["site_lon"]) * 111320.0 * c / 1000
    DY = (LATS - res["site_lat"]) * 111320.0 / 1000
    return DX, DY


def _circle(r_km, n=360):
    t = np.linspace(0, 2*np.pi, n)
    return r_km*np.sin(t), r_km*np.cos(t)


def _overlay_network(ax, res, color="#1f78b4",
                      lw_scale=1.0, alpha=0.7):
    """Disegna reticolo idrografico estratto dal DEM."""
    for seg in res.get("network_segments", []):
        if len(seg) < 2: continue
        ax.plot(seg[:,0], seg[:,1],
                color=color, lw=0.6*lw_scale,
                alpha=alpha, zorder=3)


def _overlay_osm_waterways(ax, res, alpha=0.9):
    """Overlay waterway OSM come linee."""
    for ww in res.get("osm_waterways", []):
        xy   = ww["xy"]
        wtyp = ww.get("type","default")
        sty  = WATERWAY_STYLE.get(wtyp,
               WATERWAY_STYLE["default"])
        if len(xy) < 2: continue
        ax.plot(xy[:,0], xy[:,1],
                color=sty["color"], lw=sty["lw"],
                alpha=alpha, zorder=4,
                solid_capstyle="round")


# ---------------------------------------------------------------------------
# Report testuale
# ---------------------------------------------------------------------------

def report_flood(res, site_name=""):
    w    = 72
    hs   = res["hand_at_sensor"]
    fs   = res["fri_at_sensor"]
    sc   = res["susc_at_sensor"]
    lbl  = FLOOD_LABELS[5-sc] if 1 <= sc <= 5 else "?"
    L    = ["="*w,
            f"FLOOD SUSCEPTIBILITY  |  {site_name}",
            "="*w,
            f"  Site: {res['site_lat']:.4f}N  {res['site_lon']:.4f}E",
            f"  DEM mosaic: {res['elev_m'].shape}  "
            f"inner r={res['r_inner_km']:.1f}km (GLO-30)",
            f"  Network: {res['network'].sum()} channel pixels  "
            f"{len(res['network_segments'])} segments",
            "",
            f"  At sensor:",
            f"    HAND  = {hs:.1f} m  "
            f"{'(on channel!)' if hs < 1 else ''}",
            f"    FRI   = {fs:.3f}",
            f"    Class = {lbl}",
            ""]

    L.append(f"  {'Class':<12} {'Area km²':>10} "
             f"{'Fraction':>10}  Bar")
    L.append("  " + "-"*(w-2))
    for lbl2 in FLOOD_LABELS:
        a = res["class_areas_m2"].get(lbl2, 0) / 1e6
        f = res["class_fractions"].get(lbl2, 0)
        bar = "█" * int(f * 30)
        L.append(f"  {lbl2:<12} {a:>10.2f} "
                 f"{f:>10.1%}  {bar}")

    osm_ww = res.get("osm_waterways", [])
    L += ["",
          f"  OSM waterways in area: {len(osm_ww)}"]
    types = {}
    for ww in osm_ww:
        t = ww.get("type","?")
        types[t] = types.get(t, 0) + 1
    for t, n in sorted(types.items(), key=lambda x: -x[1]):
        L.append(f"    {t:<15} {n}")

    if res["high_risk_zones"]:
        L += ["",
              f"  High/Very High zones within 3 km "
              f"({len(res['high_risk_zones'])} pixels):"]
        for z in res["high_risk_zones"][:8]:
            L.append(f"    d={z['d_m']:6.0f}m  "
                     f"HAND={z['hand_m']:.1f}m  "
                     f"FRI={z['fri']:.3f}  "
                     f"{FLOOD_LABELS[5-z['susc']]}")
    else:
        L.append("\n  No high-risk zones within 3 km.")

    L.append("="*w)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Mappa 1: Topografica con reticolo idrografico
# ---------------------------------------------------------------------------

def plot_topo_network(res, path, site_name="",
                       hillshade=True, r86_m=150.0,
                       jrc_occ=None, jrc_dx=None, jrc_dy=None):
    """
    Mappa topografica (hillshade + contour) con:
      - Reticolo idrografico estratto dal DEM (blu tenue)
      - Waterway OSM (blu intenso, più spesso)
      - Overlay JRC Global Surface Water occurrence (opzionale)
      - Cerchio r86 e zona interna
      - Sensore

    jrc_occ : 2D array occurrence [0-100], stessa griglia di jrc_dx/jrc_dy
    jrc_dx  : 2D array Easting [km] centrato sul sensore
    jrc_dy  : 2D array Northing [km] centrato sul sensore
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    import matplotlib.patches as mpatches

    DX, DY = _metric_grids(res)
    elev   = res["elev_m"].astype(float)
    elev[np.isnan(elev)] = float(np.nanmean(elev))

    fig, ax = plt.subplots(1, 1, figsize=(14, 12),
                            facecolor="white")

    # Hillshade
    if hillshade:
        ls  = LightSource(azdeg=315, altdeg=45)
        hs  = ls.hillshade(elev, vert_exag=2.0,
                            dx=30, dy=30)
        ax.pcolormesh(DX, DY, hs, cmap="gray",
                      vmin=0, vmax=1, shading="auto",
                      alpha=0.6, zorder=1)

    # Quota con colore (terrain)
    im = ax.pcolormesh(DX, DY, elev, cmap="terrain",
                        alpha=0.5, shading="auto", zorder=2)
    plt.colorbar(im, ax=ax, label="Elevation [m a.s.l.]",
                 shrink=0.6)

    # Contour ogni 100m
    try:
        cs = ax.contour(DX, DY, elev,
                         levels=np.arange(
                             int(np.nanmin(elev)//100*100),
                             int(np.nanmax(elev)//100*100)+100, 100),
                         colors="gray", linewidths=0.4,
                         alpha=0.5, zorder=3)
        ax.clabel(cs, inline=True, fontsize=6,
                  fmt="%d m", inline_spacing=3)
    except Exception:
        pass

    # JRC Global Surface Water occurrence
    if (jrc_occ is not None and
            jrc_dx is not None and jrc_dy is not None):
        occ_norm = np.clip(jrc_occ / 100.0, 0.0, 1.0)
        occ_disp = np.where(occ_norm > 0.05, occ_norm, np.nan)
        ax.pcolormesh(jrc_dx, jrc_dy, occ_disp,
                      cmap="Blues", vmin=0.0, vmax=1.0,
                      shading="auto", alpha=0.75, zorder=4)

    # Reticolo DEM
    _overlay_network(ax, res, color="#4393c3",
                     lw_scale=1.0, alpha=0.6)

    # Waterway OSM
    _overlay_osm_waterways(ax, res, alpha=0.95)

    # Bordo zona GLO-30
    r_in = res["r_inner_km"]
    cx30, cy30 = _circle(r_in)
    ax.plot(cx30, cy30, "k-.", lw=1.2, alpha=0.5,
            label=f"GLO-30 inner (r={r_in}km)")

    # Cerchio r86
    r86_km = r86_m / 1000.0
    cx86, cy86 = _circle(r86_km)
    ax.plot(cx86, cy86, "r--", lw=1.8, label=f"r86={r86_m:.0f}m")

    # Sensore
    ax.plot(0, 0, "r^", ms=12, zorder=8, label="Sensor")

    # Legenda waterway OSM
    seen_types = {ww["type"] for ww in res.get("osm_waterways",[])}
    ww_patches = []
    for wt in seen_types:
        sty = WATERWAY_STYLE.get(wt, WATERWAY_STYLE["default"])
        ww_patches.append(
            plt.Line2D([0],[0], color=sty["color"],
                       lw=sty["lw"]+0.5, label=f"OSM {wt}"))
    ww_patches.append(
        plt.Line2D([0],[0], color="#4393c3", lw=1,
                   label="DEM network"))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles+ww_patches,
              fontsize=8, loc="upper right",
              framealpha=0.85)

    clip_km = res["r_inner_km"] * 1.5
    ax.set_xlim(-clip_km, clip_km)
    ax.set_ylim(-clip_km, clip_km)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting [km]", fontsize=11)
    ax.set_ylabel("Northing [km]", fontsize=11)
    jrc_note = "  |  Blue fill = JRC water occurrence" \
               if jrc_occ is not None else ""
    ax.set_title(
        f"Topography + Drainage Network  |  {site_name}\n"
        f"Thin blue = DEM network  |  Thick blue = OSM{jrc_note}",
        fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, lw=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Mappa 2: HAND
# ---------------------------------------------------------------------------

def plot_hand(res, path, site_name="", r86_m=150.0):
    """
    Mappa HAND con:
      - Colorscale logaritmica (0m=very high risk, 20m+=safe)
      - Reticolo idrografico sovrapposto
      - Waterway OSM
      - Classi HAND come contour
      - Pannello dx: istogramma HAND entro 5 km
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    DX, DY = _metric_grids(res)
    hand   = res["hand"].copy()
    hand[np.isnan(hand)] = 50.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                              facecolor="white",
                              gridspec_kw={"width_ratios":[2,1]})

    # ---- Left: mappa HAND ----
    ax = axes[0]
    hand_plot = np.clip(hand, 0.1, 50.0)
    im = ax.pcolormesh(DX, DY, hand_plot,
                        cmap="RdYlBu", norm=LogNorm(0.1, 50),
                        shading="auto", zorder=1)
    cb = plt.colorbar(im, ax=ax, label="HAND [m]",
                      shrink=0.7)
    cb.set_ticks([0.1, 0.5, 1, 2, 5, 10, 20, 50])
    cb.set_ticklabels(["0.1","0.5","1","2","5","10","20","50"])

    # Contour linee HAND
    for h_thr, col, lbl in [(2,"#d73027","HAND=2m"),
                              (5,"#fc8d59","HAND=5m"),
                              (10,"#fee08b","HAND=10m")]:
        try:
            ax.contour(DX, DY, hand, levels=[h_thr],
                       colors=[col], linewidths=1.2,
                       zorder=4)
            # Etichetta manuale
            mid_r = hand.shape[0]//2
            idxs  = np.where(
                np.abs(hand[mid_r,:] - h_thr) <
                h_thr * 0.2)[0]
            if len(idxs):
                ax.text(DX[mid_r, idxs[0]],
                         DY[mid_r, idxs[0]],
                         lbl, fontsize=7, color=col,
                         zorder=6)
        except Exception:
            pass

    # Reticolo e OSM
    _overlay_network(ax, res, color="white",
                     lw_scale=0.8, alpha=0.5)
    _overlay_osm_waterways(ax, res, alpha=0.9)

    # Sensore + r86
    r86_km = r86_m / 1000.0
    cx86, cy86 = _circle(r86_km)
    ax.plot(cx86, cy86, "k--", lw=1.5)
    ax.plot(0, 0, "k^", ms=12, zorder=8)
    hs  = res["hand_at_sensor"]
    ax.annotate(f"HAND={hs:.1f}m",
                xy=(0,0), xytext=(0.3, 0.5),
                fontsize=9, color="black",
                arrowprops=dict(arrowstyle="->",
                                color="black"))

    clip_km = res["r_inner_km"] * 1.5
    ax.set_xlim(-clip_km, clip_km)
    ax.set_ylim(-clip_km, clip_km)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting [km]")
    ax.set_ylabel("Northing [km]")
    ax.set_title("HAND — Height Above Nearest Drainage\n"
                 "Red=low HAND (flood risk)  Blue=safe",
                 fontsize=11)

    # ---- Right: istogramma HAND entro r_inner_km ----
    ax2 = axes[1]
    r_hist = res["r_inner_km"]
    mask5 = (np.sqrt(DX**2 + DY**2) <= r_hist) & (hand < 50)
    h_vals= hand[mask5].ravel()
    h_vals= h_vals[h_vals < 50]

    if len(h_vals) > 10:
        bins  = np.logspace(np.log10(0.1), np.log10(50), 40)
        ax2.hist(h_vals, bins=bins, color="#4393c3",
                  edgecolor="white", alpha=0.8)
        ax2.set_xscale("log")
        ax2.axvline(2,  color="#d73027", ls="--",
                    lw=1.5, label="2m")
        ax2.axvline(5,  color="#fc8d59", ls="--",
                    lw=1.5, label="5m")
        ax2.axvline(10, color="#fee08b", ls="--",
                    lw=1.5, label="10m")
        ax2.axvline(hs, color="black", ls="-",
                    lw=2, label=f"Sensor {hs:.1f}m")
        ax2.legend(fontsize=9)
        ax2.set_xlabel("HAND [m]")
        ax2.set_ylabel("Pixel count")
        pct_vh = float(np.sum(h_vals < 2)) / len(h_vals)
        ax2.set_title(f"HAND distribution (r<{r_hist:.1f}km)\n"
                      f"HAND<2m: {pct_vh:.1%} of area",
                      fontsize=11)

    fig.suptitle(f"HAND Map  |  {site_name}  |  "
                 f"{res['site_lat']:.4f}N "
                 f"{res['site_lon']:.4f}E",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Mappa 3: FRI + classi suscettibilità
# ---------------------------------------------------------------------------

def plot_fri(res, path, site_name="", r86_m=150.0):
    """
    Due pannelli:
      Left : mappa FRI [0-1] con classi suscettibilità come overlay
      Right: mappa classi suscettibilità + bar chart aree
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    DX, DY = _metric_grids(res)
    fri    = res["fri"]
    susc   = res["susc_map"].astype(float)
    susc[np.isnan(res["hand"])] = np.nan

    r86_km  = r86_m / 1000.0
    clip_km = res["r_inner_km"] * 1.5
    cx86, cy86 = _circle(r86_km)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8),
                              facecolor="white")

    # ---- Left: mappa FRI ----
    ax = axes[0]
    im = ax.pcolormesh(DX, DY, fri, cmap="YlOrRd",
                        vmin=0, vmax=1, shading="auto",
                        zorder=1)
    plt.colorbar(im, ax=ax, label="FRI [0–1]", shrink=0.7)
    _overlay_network(ax, res, color="white",
                     lw_scale=0.7, alpha=0.5)
    _overlay_osm_waterways(ax, res, alpha=0.85)
    ax.plot(cx86, cy86, "k--", lw=1.5)
    ax.plot(0, 0, "k^", ms=10, zorder=8)
    ax.set_xlim(-clip_km, clip_km)
    ax.set_ylim(-clip_km, clip_km)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting [km]")
    ax.set_ylabel("Northing [km]")
    ax.set_title("Flood Risk Index (FRI)\n"
                 "0=no risk  1=maximum", fontsize=11)

    # ---- Center: mappa classi ----
    ax2 = axes[1]
    cmap2 = ListedColormap(FLOOD_COLORS[::-1])
    norm2 = BoundaryNorm([0.5,1.5,2.5,3.5,4.5,5.5], 5)
    ax2.pcolormesh(DX, DY, susc, cmap=cmap2, norm=norm2,
                   shading="auto", zorder=1)
    _overlay_network(ax2, res, color="white",
                     lw_scale=0.7, alpha=0.5)
    _overlay_osm_waterways(ax2, res, alpha=0.85)
    ax2.plot(cx86, cy86, "k--", lw=1.5, label=f"r86={r86_m:.0f}m")
    ax2.plot(0, 0, "k^", ms=10, zorder=8, label="Sensor")
    patches = [mpatches.Patch(
        color=FLOOD_COLORS[4-i],
        label=FLOOD_LABELS[i]) for i in range(5)]
    ax2.legend(handles=patches, fontsize=8,
               loc="upper right", framealpha=0.85)
    ax2.set_xlim(-clip_km, clip_km)
    ax2.set_ylim(-clip_km, clip_km)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Easting [km]")
    sc  = res["susc_at_sensor"]
    lbl = FLOOD_LABELS[5-sc] if 1<=sc<=5 else "?"
    ax2.set_title(f"Susceptibility classes\n"
                  f"Sensor: {lbl}", fontsize=11)

    # ---- Right: FRI decomposition + bar chart aree ----
    ax3 = axes[2]
    # Bar chart aree
    areas   = [res["class_areas_m2"].get(l,0)/1e6
               for l in FLOOD_LABELS]
    colors3 = FLOOD_COLORS[::-1]
    bars    = ax3.barh(FLOOD_LABELS, areas,
                        color=colors3, edgecolor="white",
                        height=0.6)
    for bar, val in zip(bars, areas):
        if val > 0.01:
            ax3.text(val*1.02,
                     bar.get_y()+bar.get_height()/2,
                     f"{val:.2f} km²",
                     va="center", fontsize=9)
    ax3.set_xlabel("Area [km²]")
    ax3.set_title("Area by susceptibility class", fontsize=11)

    # Annotazione FRI componenti
    fri_s  = res["fri_at_sensor"]
    fh_s   = float(res["f_hand"][
        int(np.argmin(np.abs(res["lats_m"]-res["site_lat"]))),
        int(np.argmin(np.abs(res["lons_m"]-res["site_lon"])))])
    ax3.text(0.02, 0.12,
             f"FRI at sensor = {fri_s:.3f}\n"
             f"  f(HAND)     = {fh_s:.3f}\n"
             f"  HAND        = {res['hand_at_sensor']:.1f}m",
             transform=ax3.transAxes,
             fontsize=9, va="bottom",
             bbox=dict(boxstyle="round", fc="white",
                       alpha=0.8))

    fig.suptitle(
        f"Flood Risk Index  |  {site_name}  |  "
        f"{res['site_lat']:.4f}N {res['site_lon']:.4f}E",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
