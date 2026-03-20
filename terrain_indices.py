"""
terrain_indices.py
==================
Due funzioni per l'analisi terrain di un sito CRNS:

1. compute_twi()
   Topographic Wetness Index (Beven & Kirkby 1979) dal DEM.
   TWI = ln(a / tan(β))
   Algoritmo D8 con filling delle depressioni (Wang & Liu 2006).
   Output: mappa 2D + statistiche dentro il footprint pesate W(r).

2. compute_thermal_index()
   Stima dell'escursione termica locale rispetto a ERA5.
   Corregge il bias di ERA5 (~31 km) per:
     - lapse rate orografico  (γ = 6.5°C/1000m)
     - cold air pooling       (Lindkvist et al. 2000)
     - insolazione PISR       (da PVGIS già calcolato)
   Output: ΔT mensile rispetto a ERA5 + T corretta mensile.

Entrambe le funzioni sono compatibili con le strutture dati del
resto della pipeline (elev, dx_grid, dy_grid, dist_grid, wmap, horizon).

Dipendenze: numpy, scipy (solo per gaussian_filter in TWI)

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import numpy as np
import heapq
import multiprocessing as mp


# ===========================================================================
# COSTANTI FISICHE
# ===========================================================================

GAMMA_DRY_ADIABATIC  = 9.8e-3   # °C/m  (adiabatico secco)
GAMMA_STANDARD       = 6.5e-3   # °C/m  (lapse rate standard ICAO)
GAMMA_ALPINE_MOIST   = 5.5e-3   # °C/m  (alpino umido, più realistico)

# Coefficienti cold air pooling (Lindkvist et al. 2000, calibrati su Alpi)
K_COLD_POOL          = 3.0      # °C  (intensità massima cold pooling)

# Coefficiente correzione PISR -> temperatura
ALPHA_PISR           = 4.0      # °C  (sensitività T a variazione PISR relativa)


# ===========================================================================
# FUNZIONI INTERNE — TWI
# ===========================================================================

def _fill_depressions(elev):
    """
    Filling delle depressioni con priority queue (Wang & Liu 2006).
    Algoritmo: i pixel di bordo inizializzano il heap; ogni pixel
    viene visitato in ordine di quota crescente e alzato al minimo
    del suo vicino già visitato se necessario.
    Complessità: O(n log n).

    Parameters
    ----------
    elev : 2D array, quota [m]. NaN ignorati.

    Returns
    -------
    filled : 2D array, stesso shape di elev, depressioni rimosse.
    """
    nr, nc  = elev.shape
    filled  = elev.copy()
    visited = np.zeros((nr, nc), dtype=bool)
    heap    = []

    # Inizializza con i pixel di bordo
    for i in range(nr):
        for j in [0, nc - 1]:
            if not np.isnan(filled[i, j]):
                heapq.heappush(heap, (filled[i, j], i, j))
                visited[i, j] = True
    for j in range(nc):
        for i in [0, nr - 1]:
            if not np.isnan(filled[i, j]) and not visited[i, j]:
                heapq.heappush(heap, (filled[i, j], i, j))
                visited[i, j] = True

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while heap:
        z, i, j = heapq.heappop(heap)
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < nr and 0 <= nj < nc and not visited[ni, nj]:
                if np.isnan(filled[ni, nj]):
                    continue
                visited[ni, nj] = True
                filled[ni, nj]  = max(filled[ni, nj], z)
                heapq.heappush(heap, (filled[ni, nj], ni, nj))

    return filled


def _flow_accumulation_d8(elev, dx):
    """
    D8 flow direction + accumulo dell'area drenante.
    Vettorizzato: la direzione di flusso è calcolata con shift numpy,
    l'accumulo è propagato in ordine di quota decrescente.

    Parameters
    ----------
    elev : 2D array, quota [m], già riempita (no depressioni).
    dx   : risoluzione pixel [m].

    Returns
    -------
    acc : 2D array, numero di pixel che drenano in ogni pixel (incluso sé stesso).
    """
    nr, nc = elev.shape
    diag   = dx * np.sqrt(2)

    # 8 vicini tramite padding + slicing — tutto vettorizzato
    ep = np.pad(elev, 1, mode='edge')
    neighbors = np.stack([
        ep[0:nr,   0:nc],    ep[0:nr,   1:nc+1], ep[0:nr,   2:nc+2],
        ep[1:nr+1, 0:nc],                          ep[1:nr+1, 2:nc+2],
        ep[2:nr+2, 0:nc],    ep[2:nr+2, 1:nc+1], ep[2:nr+2, 2:nc+2],
    ])  # shape (8, nr, nc)

    dist8 = np.array([diag, dx, diag,
                      dx,        dx,
                      diag, dx, diag])[:, None, None]

    center = elev[None, :, :]
    slopes = (center - neighbors) / dist8        # positivo = discesa
    fdir   = np.argmax(slopes, axis=0)           # indice 0-7 del vicino più basso

    # Mappa indice -> (di, dj) per le 8 direzioni
    di_map = np.array([-1, -1, -1,  0,  0,  1,  1,  1])
    dj_map = np.array([-1,  0,  1, -1,  1, -1,  0,  1])

    di_grid = di_map[fdir]   # (nr, nc)
    dj_grid = dj_map[fdir]

    # Coordinate del pixel ricevente per ogni sorgente
    row_src = np.repeat(np.arange(nr)[:, None], nc, axis=1)
    col_src = np.repeat(np.arange(nc)[None, :], nr, axis=0)
    row_dst = np.clip(row_src + di_grid, 0, nr - 1)
    col_dst = np.clip(col_src + dj_grid, 0, nc - 1)

    dst_flat = row_dst.ravel() * nc + col_dst.ravel()  # (nr*nc,)
    src_flat = row_src.ravel() * nc + col_src.ravel()

    # Ordine di propagazione: dal pixel più alto al più basso
    order = np.argsort(elev.ravel())[::-1]

    acc = np.ones(nr * nc, dtype=float)
    for k in order:
        d = dst_flat[k]
        if d != src_flat[k]:       # non auto-loop (pixel di bordo)
            acc[d] += acc[k]

    return acc.reshape(nr, nc)


def _weight_radial(dist, r86):
    """Peso radiale Köhli: W(r) = exp(-r/lambda), lambda = r86/3."""
    lam = r86 / 3.0
    return np.where(dist < 1e-3, 0.0, np.exp(-dist / lam))


def _twi_classes(twi, n=5):
    """
    Classifica TWI in n classi da 'molto secco' a 'molto umido'
    usando i percentili come soglie.
    Ritorna array intero 1..n e soglie.
    """
    pct   = np.linspace(0, 100, n + 1)
    edges = np.nanpercentile(twi, pct)
    cls   = np.digitize(twi, edges[1:-1]) + 1   # 1..n
    return cls.astype(int), edges


# ===========================================================================
# WORKER MULTIPROCESSING — TWI
# ===========================================================================

def _twi_strip_worker(args):
    """
    Calcola fill-depressions + D8 + slope + TWI su una strip orizzontale
    del DEM già ripulita (nessun NaN).

    Parameters
    ----------
    args : tuple (strip_clean, dpx, dpy, slope_min_rad)
        strip_clean  : 2D array float, sottogriglia di elev_clean
        dpx, dpy     : risoluzione pixel [m] lungo x e y
        slope_min_rad: pendenza minima [rad]

    Returns
    -------
    twi_strip        : 2D array float
    slope_deg_strip  : 2D array float
    drain_area_strip : 2D array float  [m]
    """
    strip_clean, dpx, dpy, slope_min_rad = args
    dx_m       = 0.5 * (dpx + dpy)
    filled     = _fill_depressions(strip_clean)
    acc        = _flow_accumulation_d8(filled, dx_m)
    drain_area = acc * dx_m
    gy, gx     = np.gradient(strip_clean, dpy, dpx)
    slope_rad  = np.maximum(np.arctan(np.sqrt(gx**2 + gy**2)), slope_min_rad)
    twi        = np.log(drain_area / np.tan(slope_rad))
    return twi, np.degrees(slope_rad), drain_area


# ===========================================================================
# FUNZIONE PUBBLICA 1 — TWI
# ===========================================================================

def compute_twi(
    elev,
    dx_grid,
    dy_grid,
    dist_grid,
    r86,
    slope_min_rad = 0.001,    # pendenza minima per evitare TWI -> inf [rad]
    n_classes     = 5,        # classi TWI per la mappa
    n_cores       = 1,        # worker multiprocessing
):
    """
    Topographic Wetness Index dal DEM.

    TWI = ln(a / tan(β))

    dove a = area drenante specifica [m] (D8, filling depressioni Wang 2006)
         β = pendenza locale [rad]

    Parameters
    ----------
    elev          : 2D array, quota [m a.s.l.], stessa shape di dx_grid
    dx_grid       : 2D array, offset easting [m] da clip_dem_to_radius
    dy_grid       : 2D array, offset northing [m]
    dist_grid     : 2D array, distanza dal sensore [m]
    r86           : raggio footprint [m]
    slope_min_rad : pendenza minima [rad], default 0.001 (~0.06°)
    n_classes     : numero di classi per la mappa classificata

    Returns
    -------
    dict con:
        twi_map         : 2D array, TWI per ogni pixel DEM
        twi_class_map   : 2D array int, classi 1 (secco) .. n (umido)
        twi_class_edges : array, soglie tra le classi (percentili TWI)
        slope_map_deg   : 2D array, pendenza locale [gradi]
        drainage_area_m : 2D array, area drenante specifica [m]

        -- Statistiche dentro il footprint (dist <= r86) --
        twi_mean_fp     : media aritmetica TWI nel footprint
        twi_std_fp      : deviazione standard TWI nel footprint
        twi_min_fp      : minimo TWI nel footprint
        twi_max_fp      : massimo TWI nel footprint
        twi_weighted    : media pesata W(r) nel footprint  ← valore CRNS
        twi_class_fractions : array n_classes, frazione del footprint
                              in ogni classe [0..1]

        slope_mean_fp_deg : pendenza media nel footprint [gradi]
        dx_m              : risoluzione pixel usata [m]
    """

    # ------------------------------------------------------------------ #
    # Risoluzione pixel
    # ------------------------------------------------------------------ #
    nr, nc = elev.shape
    dpx    = abs(float(np.nanmedian(np.diff(dx_grid[nr // 2, :]))))
    dpy    = abs(float(np.nanmedian(np.diff(dy_grid[:, nc // 2]))))
    if dpx < 1: dpx = 30.0
    if dpy < 1: dpy = 30.0
    dx_m   = 0.5 * (dpx + dpy)

    # ------------------------------------------------------------------ #
    # Pulizia NaN globale (fill con nanmean del DEM intero)
    # ------------------------------------------------------------------ #
    elev_clean = np.where(np.isnan(elev), np.nanmean(elev), elev)

    # ------------------------------------------------------------------ #
    # Fill + D8 + slope + TWI  (single o parallel)
    # ------------------------------------------------------------------ #
    n_cores = max(1, min(int(n_cores), nr))

    if n_cores == 1:
        # ------- path single-process (originale) ----------------------- #
        gy, gx        = np.gradient(elev_clean, dpy, dpx)
        slope_rad     = np.maximum(np.arctan(np.sqrt(gx**2 + gy**2)), slope_min_rad)
        filled        = _fill_depressions(elev_clean)
        acc           = _flow_accumulation_d8(filled, dx_m)
        drain_area    = acc * dx_m
        twi           = np.log(drain_area / np.tan(slope_rad))
        slope_deg_map = np.degrees(slope_rad)
    else:
        # ------- path multiprocessing (strip orizzontali con overlap) -- #
        # L'overlap permette al D8 di ricevere il contributo di flusso
        # dai pixel a monte appartenenti alla strip adiacente.
        # Valore empirico: max(20 px, nr/(n_cores*4)).
        overlap    = min(max(20, nr // (n_cores * 4)), nr // 2)
        row_splits = np.array_split(np.arange(nr), n_cores)

        strip_args = []
        boundaries = []   # (full_r0, full_r1, v0_in_strip, v1_in_strip)
        for rows in row_splits:
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            s0, s1 = max(0, r0 - overlap), min(nr, r1 + overlap)
            strip_args.append((
                elev_clean[s0:s1, :].copy(),
                dpx, dpy, slope_min_rad,
            ))
            boundaries.append((r0, r1, r0 - s0, r1 - s0))

        print(f"   TWI: {n_cores} strips (overlap {overlap} px) ...", flush=True)
        ctx = mp.get_context('fork')
        with ctx.Pool(n_cores) as pool:
            results = pool.map(_twi_strip_worker, strip_args)

        twi           = np.full((nr, nc), np.nan)
        slope_deg_map = np.full((nr, nc), np.nan)
        drain_area    = np.full((nr, nc), np.nan)
        for (r0, r1, v0, v1), (twi_s, slope_s, drain_s) in zip(boundaries, results):
            twi[r0:r1, :]           = twi_s[v0:v1, :]
            slope_deg_map[r0:r1, :] = slope_s[v0:v1, :]
            drain_area[r0:r1, :]    = drain_s[v0:v1, :]

    twi[np.isnan(elev)] = np.nan

    # ------------------------------------------------------------------ #
    # Classificazione
    # ------------------------------------------------------------------ #
    twi_valid       = twi[~np.isnan(twi)]
    cls_map, edges  = _twi_classes(twi, n=n_classes)
    cls_map         = cls_map.astype(float)
    cls_map[np.isnan(twi)] = np.nan

    # ------------------------------------------------------------------ #
    # Statistiche nel footprint
    # ------------------------------------------------------------------ #
    fp_mask = (dist_grid <= r86) & ~np.isnan(twi)
    twi_fp  = twi[fp_mask]
    r_fp    = dist_grid[fp_mask]
    W_fp    = _weight_radial(r_fp, r86)

    twi_weighted = float(np.sum(W_fp * twi_fp) / np.sum(W_fp)) \
                   if W_fp.sum() > 0 else float(np.nanmean(twi_fp))

    # Frazioni per classe
    cls_fp = cls_map[fp_mask]
    fracs  = np.array([
        float(np.sum(cls_fp == c) / len(cls_fp)) if len(cls_fp) > 0 else 0.0
        for c in range(1, n_classes + 1)
    ])

    slope_mean_fp = float(np.nanmean(slope_deg_map[fp_mask]))

    return dict(
        twi_map              = twi,
        twi_class_map        = cls_map,
        twi_class_edges      = edges,
        slope_map_deg        = slope_deg_map,
        drainage_area_m      = drain_area,
        # Footprint stats
        twi_mean_fp          = float(np.nanmean(twi_fp)),
        twi_std_fp           = float(np.nanstd(twi_fp)),
        twi_min_fp           = float(np.nanmin(twi_fp)),
        twi_max_fp           = float(np.nanmax(twi_fp)),
        twi_weighted         = twi_weighted,
        twi_class_fractions  = fracs,
        slope_mean_fp_deg    = slope_mean_fp,
        dx_m                 = dx_m,
    )


def report_twi(res):
    """Stampa leggibile dei risultati di compute_twi."""
    w  = 62
    L  = [
        "=" * w,
        "TOPOGRAPHIC WETNESS INDEX (TWI)",
        "=" * w,
        f"  Pixel size          : {res['dx_m']:.1f} m",
        f"  TWI footprint mean  : {res['twi_mean_fp']:.2f}",
        f"  TWI footprint std   : {res['twi_std_fp']:.2f}",
        f"  TWI footprint range : {res['twi_min_fp']:.2f} – "
        f"{res['twi_max_fp']:.2f}",
        f"  TWI weighted W(r)   : {res['twi_weighted']:.2f}  "
        f"← valore CRNS rappresentativo",
        f"  Mean slope (fp)     : {res['slope_mean_fp_deg']:.1f}°",
        "",
        "  Class fractions inside footprint:",
    ]
    labels = ["Very dry", "Dry", "Moderate", "Wet", "Very wet"]
    edges  = res['twi_class_edges']
    for i, (lbl, frac) in enumerate(
            zip(labels[:len(res['twi_class_fractions'])],
                res['twi_class_fractions'])):
        bar = "█" * int(frac * 30)
        L.append(f"    {i+1} {lbl:<10} "
                 f"TWI {edges[i]:5.1f}–{edges[i+1]:5.1f} "
                 f"{bar:<30} {frac*100:.1f}%")
    L.append("=" * w)
    return "\n".join(L)


# ===========================================================================
# FUNZIONI INTERNE — INDICE TERMICO
# ===========================================================================

def _sky_view_factor(horizon_deg):
    """
    SVF da profilo di orizzonte (formula isotropa diffusa).
    SVF = 1 - mean(sin²(ψ(φ)))
    Varia da 0 (completamente occluso) a 1 (cielo aperto).
    """
    psi_rad = np.radians(np.asarray(horizon_deg, dtype=float))
    return float(1.0 - np.mean(np.sin(psi_rad) ** 2))


def _concavity_index(elev, dist_grid, r_inner=50.0, r_outer=300.0):
    """
    Indice di concavità del terreno intorno al sensore.
    = quota media anello esterno - quota media cerchio interno
    Positivo: sensore in conca (raccoglie aria fredda).
    Negativo: sensore su cresta (disperde aria fredda).

    Parameters
    ----------
    r_inner : raggio del cerchio interno [m]  (zona sensore)
    r_outer : raggio dell'anello esterno [m]  (contorno)
    """
    mask_inner = (dist_grid <= r_inner) & ~np.isnan(elev)
    mask_ring  = ((dist_grid > r_inner) & (dist_grid <= r_outer)
                  & ~np.isnan(elev))
    z_center = float(np.nanmedian(elev[mask_inner])) \
               if mask_inner.any() else float(np.nanmean(elev))
    z_ring   = float(np.nanmean(elev[mask_ring])) \
               if mask_ring.any() else z_center
    return z_ring - z_center   # [m], positivo = conca


def _cold_pool_correction(concavity_m, svf,
                           k=K_COLD_POOL):
    """
    ΔT da cold air pooling (Lindkvist et al. 2000).
    ΔT_pool = -k * (1 - SVF) * tanh(concavity / 50)

    La tanh normalizza la concavità: satura a ±1 per concavità > ~150m.
    Il fattore (1-SVF) amplifica l'effetto per siti con orizzonte alto.
    Il segno è negativo: più concavo + più chiuso = più freddo di notte.

    Restituisce correzione della temperatura minima [°C].
    Applicata alla T_min perché il cold pooling è un fenomeno notturno.
    """
    pool = -k * (1.0 - svf) * np.tanh(concavity_m / 50.0)
    return float(pool)


def _pisr_correction(pisr_site_kWh_m2_annual,
                     pisr_era5_kWh_m2_annual,
                     alpha=ALPHA_PISR):
    """
    ΔT da differenza di insolazione potenziale PISR.
    ΔT_pisr = alpha * (PISR_site - PISR_ERA5) / PISR_ERA5

    alpha = 4°C: una variazione del 100% di PISR produce 4°C di bias.
    Applicata a T_mean e T_max (fenomeno diurno).
    """
    if pisr_era5_kWh_m2_annual <= 0:
        return 0.0
    rel = (pisr_site_kWh_m2_annual - pisr_era5_kWh_m2_annual) \
          / pisr_era5_kWh_m2_annual
    return float(alpha * rel)


# ===========================================================================
# FUNZIONE PUBBLICA 2 — INDICE TERMICO
# ===========================================================================

def compute_thermal_index(
    elev,
    dist_grid,
    s_elev,
    horizon_deg,
    azimuths_deg,
    T_mean_monthly_era5,       # array (12,) [°C] da get_site_climate
    T_min_monthly_era5,        # array (12,) [°C]
    T_max_monthly_era5,        # array (12,) [°C]
    POA_monthly_kWh_m2,        # array (12,) irradianza sul pannello [kWh/m²/mese]
    era5_elevation_m,          # quota media cella ERA5 al sito [m]
                               # (da Open-Meteo: parametro 'elevation' nella risposta)
    gamma          = GAMMA_STANDARD,   # lapse rate [°C/m]
    k_cold_pool    = K_COLD_POOL,      # intensità cold pooling [°C]
    alpha_pisr     = ALPHA_PISR,       # sensitività PISR->T [°C]
    r_inner_m      = 50.0,    # raggio interno per concavità [m]
    r_outer_m      = 300.0,   # raggio esterno per concavità [m]
):
    """
    Stima la temperatura locale al sito corretta per i bias di ERA5.

    ERA5 (~31 km) non risolve:
      1. Lapse rate orografico: sito a quota diversa dalla media ERA5
      2. Cold air pooling: valli strette più fredde di notte
      3. Insolazione locale: versanti diversi ricevono irradianza diversa

    Parameters
    ----------
    elev                    : 2D array DEM [m a.s.l.]
    dist_grid               : 2D array distanze dal sensore [m]
    s_elev                  : quota del sensore [m a.s.l.]
    horizon_deg             : array elevazione orizzonte [deg]
    azimuths_deg            : array azimuths [deg]
    T_mean_monthly_era5     : array (12,) temperatura media ERA5 [°C]
    T_min_monthly_era5      : array (12,) temperatura minima ERA5 [°C]
    T_max_monthly_era5      : array (12,) temperatura massima ERA5 [°C]
    POA_monthly_kWh_m2      : array (12,) irradianza pannello PVGIS [kWh/m²/mese]
                              Proxy per PISR al sito (con orizzonte tuo DEM).
    era5_elevation_m        : quota media della cella ERA5 [m]
                              Recuperabile da Open-Meteo con parametro
                              'elevation' nella risposta API.
    gamma                   : lapse rate [°C/m]  default 6.5e-3
    k_cold_pool             : intensità cold pooling [°C]  default 3.0
    alpha_pisr              : sensitività PISR [°C]  default 4.0
    r_inner_m, r_outer_m    : raggi per indice di concavità [m]

    Returns
    -------
    dict con:

        -- Indici geometrici del sito --
        svf                     : Sky View Factor [0-1]
        concavity_m             : indice di concavità [m] (+ = conca)
        cold_pool_index         : indice normalizzato [0-1]

        -- Correzioni (scalari, applicate uniformemente a tutti i mesi) --
        dT_lapse_C              : correzione lapse rate [°C]
                                  = -gamma * (s_elev - era5_elevation_m)
        dT_cold_pool_C          : correzione cold pooling [°C] (sempre <=0)
                                  applicata a T_min
        dT_pisr_C               : correzione PISR [°C]
                                  applicata a T_mean e T_max

        -- Temperature corrette mensili (array 12,) --
        T_mean_corrected_C      : T_mean_era5 + dT_lapse + dT_pisr
        T_min_corrected_C       : T_min_era5  + dT_lapse + dT_cold_pool
        T_max_corrected_C       : T_max_era5  + dT_lapse + dT_pisr

        -- Scalari annuali --
        T_mean_annual_corrected_C
        T_min_annual_corrected_C   (minima assoluta)
        T_max_annual_corrected_C   (massima assoluta)
        frost_days_monthly      : array (12,) stima giorni di gelo corretti
        frost_days_annual       : scalare

        -- Diagnostici --
        era5_elevation_m        : quota ERA5 usata
        site_elevation_m        : quota sito
        delta_elevation_m       : differenza (sito - ERA5)
        gamma_used              : lapse rate usato [°C/m]
        uncertainty_C           : incertezza stimata totale [°C]
                                  (somma quadratica delle incertezze
                                   dei tre contributi)
    """

    # ------------------------------------------------------------------ #
    # 1. SVF dal profilo di orizzonte
    # ------------------------------------------------------------------ #
    svf = _sky_view_factor(horizon_deg)

    # ------------------------------------------------------------------ #
    # 2. Indice di concavità
    # ------------------------------------------------------------------ #
    concavity = _concavity_index(elev, dist_grid, r_inner_m, r_outer_m)

    # Cold pool index normalizzato [0-1]:
    # 0 = nessun cold pooling (cresta aperta)
    # 1 = cold pooling massimo (conca chiusa)
    cold_pool_index = float(
        np.clip((1.0 - svf) * np.tanh(max(concavity, 0) / 50.0), 0, 1))

    # ------------------------------------------------------------------ #
    # 3. Correzione lapse rate
    # Positiva se sito più alto di ERA5 (sito più freddo di ERA5)
    # Negativa se sito più basso di ERA5
    # ------------------------------------------------------------------ #
    delta_elev  = s_elev - era5_elevation_m    # [m]
    dT_lapse    = -gamma * delta_elev           # [°C]

    # ------------------------------------------------------------------ #
    # 4. Correzione cold air pooling
    # Applicata solo a T_min (fenomeno notturno/stagnante)
    # ------------------------------------------------------------------ #
    dT_pool = _cold_pool_correction(concavity, svf, k=k_cold_pool)

    # ------------------------------------------------------------------ #
    # 5. Correzione PISR
    # POA_monthly è il proxy per PISR al sito (già corretto per orizzonte
    # tuo DEM). Il valore ERA5 equivalente è POA senza correzione orizzonte,
    # approssimato come POA_site / SVF (SVF=1 -> cielo aperto = ERA5).
    # Usiamo il totale annuo per la correzione (variazione stagionale
    # è già in ERA5).
    # ------------------------------------------------------------------ #
    pisr_site  = float(np.sum(POA_monthly_kWh_m2))
    pisr_era5  = pisr_site / svf if svf > 0.01 else pisr_site
    dT_pisr    = _pisr_correction(pisr_site, pisr_era5, alpha=alpha_pisr)

    # ------------------------------------------------------------------ #
    # 6. Temperature corrette mensili
    # ------------------------------------------------------------------ #
    T_mean_era5 = np.asarray(T_mean_monthly_era5, dtype=float)
    T_min_era5  = np.asarray(T_min_monthly_era5,  dtype=float)
    T_max_era5  = np.asarray(T_max_monthly_era5,  dtype=float)

    T_mean_corr = T_mean_era5 + dT_lapse + dT_pisr
    T_min_corr  = T_min_era5  + dT_lapse + dT_pool
    T_max_corr  = T_max_era5  + dT_lapse + dT_pisr

    # ------------------------------------------------------------------ #
    # 7. Giorni di gelo stimati
    # Stima dalla T_min corretta: se T_min < 0 il giorno è a rischio gelo.
    # Proporzione lineare tra T_min e 0°C — approssimazione.
    # ------------------------------------------------------------------ #
    # Giorni nel mese
    days_per_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31],
                               dtype=float)
    # Fraction of days with frost: logistic approximation
    # P(frost) ~ sigmoid(-T_min_corr / 3)  (3°C = scala di transizione)
    p_frost        = 1.0 / (1.0 + np.exp(T_min_corr / 3.0))
    frost_days     = p_frost * days_per_month

    # ------------------------------------------------------------------ #
    # 8. Scalari annuali
    # ------------------------------------------------------------------ #
    T_mean_annual = float(np.mean(T_mean_corr))
    T_min_annual  = float(np.min(T_min_corr))
    T_max_annual  = float(np.max(T_max_corr))
    frost_annual  = float(np.sum(frost_days))

    # ------------------------------------------------------------------ #
    # 9. Incertezza totale
    # ------------------------------------------------------------------ #
    sigma_lapse   = 1.0    # °C  (incertezza lapse rate standard)
    sigma_pool    = 2.0    # °C  (incertezza cold pooling)
    sigma_pisr    = 1.5    # °C  (incertezza correzione PISR)
    uncertainty   = float(np.sqrt(sigma_lapse**2 + sigma_pool**2
                                  + sigma_pisr**2))

    return dict(
        # Geometria
        svf                        = svf,
        concavity_m                = float(concavity),
        cold_pool_index            = cold_pool_index,
        # Correzioni scalari
        dT_lapse_C                 = float(dT_lapse),
        dT_cold_pool_C             = float(dT_pool),
        dT_pisr_C                  = float(dT_pisr),
        # Temperature mensili corrette
        T_mean_corrected_C         = T_mean_corr,
        T_min_corrected_C          = T_min_corr,
        T_max_corrected_C          = T_max_corr,
        # Scalari annuali
        T_mean_annual_corrected_C  = T_mean_annual,
        T_min_annual_corrected_C   = T_min_annual,
        T_max_annual_corrected_C   = T_max_annual,
        frost_days_monthly         = frost_days,
        frost_days_annual          = frost_annual,
        # Diagnostici
        era5_elevation_m           = float(era5_elevation_m),
        site_elevation_m           = float(s_elev),
        delta_elevation_m          = float(delta_elev),
        gamma_used                 = float(gamma),
        pisr_site_kWh              = float(pisr_site),
        pisr_era5_kWh              = float(pisr_era5),
        uncertainty_C              = uncertainty,
    )


def report_thermal_index(res, T_mean_era5, T_min_era5, T_max_era5):
    """Stampa leggibile dei risultati di compute_thermal_index."""
    MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    w = 72
    L = [
        "=" * w,
        "THERMAL INDEX — Local Temperature Correction vs ERA5",
        "=" * w,
        f"  Site elevation   : {res['site_elevation_m']:.0f} m",
        f"  ERA5 elevation   : {res['era5_elevation_m']:.0f} m",
        f"  Δ elevation      : {res['delta_elevation_m']:+.0f} m",
        f"  Lapse rate       : {res['gamma_used']*1000:.1f} °C/1000m",
        f"  SVF              : {res['svf']:.3f}  "
        f"({'open sky' if res['svf']>0.9 else 'partial' if res['svf']>0.7 else 'enclosed'})",
        f"  Concavity index  : {res['concavity_m']:+.1f} m  "
        f"({'basin/valley' if res['concavity_m']>0 else 'ridge/crest'})",
        f"  Cold pool index  : {res['cold_pool_index']:.3f}",
        "",
        "  Corrections applied:",
        f"    ΔT lapse rate  : {res['dT_lapse_C']:+.2f} °C  "
        f"(all T: mean, min, max)",
        f"    ΔT cold pool   : {res['dT_cold_pool_C']:+.2f} °C  "
        f"(T_min only — nocturnal effect)",
        f"    ΔT PISR        : {res['dT_pisr_C']:+.2f} °C  "
        f"(T_mean, T_max — diurnal effect)",
        f"    Total unc.     : ±{res['uncertainty_C']:.1f} °C",
        "",
    ]

    # Tabella mensile
    hdr = "       " + "  ".join(f"{m:>5}" for m in MONTHS)
    L.append(hdr)
    L.append("  " + "-" * (w - 2))

    def row(label, era5_arr, corr_arr, fmt='.1f'):
        era5_s = "  ".join(f"{v:{fmt}}" for v in era5_arr)
        corr_s = "  ".join(f"{v:{fmt}}" for v in corr_arr)
        delta  = corr_arr - np.asarray(era5_arr)
        delt_s = "  ".join(f"{v:+{fmt}}" for v in delta)
        return (f"  ERA5  {label:<6} {era5_s}\n"
                f"  Corr  {label:<6} {corr_s}\n"
                f"  Delta {label:<6} {delt_s}")

    L.append("  T_mean [°C]")
    L.append(row("mean", np.array(T_mean_era5),
                 res['T_mean_corrected_C']))
    L.append("")
    L.append("  T_min [°C]")
    L.append(row("min ", np.array(T_min_era5),
                 res['T_min_corrected_C']))
    L.append("")
    L.append("  T_max [°C]")
    L.append(row("max ", np.array(T_max_era5),
                 res['T_max_corrected_C']))
    L.append("")

    frost = res['frost_days_monthly']
    frost_s = "  ".join(f"{v:5.1f}" for v in frost)
    L.append(f"  Frost days (corr)  {frost_s}")
    L.append("")
    L.append("  ANNUAL SUMMARY (corrected)")
    L.append(f"    T_mean : {res['T_mean_annual_corrected_C']:.1f} °C")
    L.append(f"    T_min  : {res['T_min_annual_corrected_C']:.1f} °C  "
             f"(absolute minimum)")
    L.append(f"    T_max  : {res['T_max_annual_corrected_C']:.1f} °C  "
             f"(absolute maximum)")
    L.append(f"    Frost  : {res['frost_days_annual']:.0f} days/year")
    L.append("=" * w)
    return "\n".join(L)


# ===========================================================================
# SMOKE TESTS
# ===========================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("TEST 1 — TWI su DEM sintetico")
    print("=" * 60)

    n   = 80
    x1  = np.linspace(-1200, 1200, n)
    XX, YY = np.meshgrid(x1, x1)
    dist   = np.sqrt(XX**2 + YY**2)
    dx_g   = XX
    dy_g   = YY

    # DEM: valle a est, collina a ovest
    elev_t = (1500
              - 0.03 * dist
              + 100 * np.exp(-((XX - 300)**2 + YY**2) / 80000)
              - 80  * np.exp(-((XX + 200)**2 + YY**2) / 50000))

    t0 = time.perf_counter()
    res_twi = compute_twi(elev_t, dx_g, dy_g, dist, r86=130.0)
    dt = time.perf_counter() - t0

    print(report_twi(res_twi))
    print(f"  Wall time: {dt:.2f} s")

    assert 0 < res_twi['twi_weighted'] < 20, "TWI fuori range fisico"
    assert res_twi['twi_std_fp'] > 0, "TWI std deve essere > 0"
    print("  PASS\n")

    print("=" * 60)
    print("TEST 2 — Indice termico: valle stretta alpina")
    print("=" * 60)

    # Simula: sito a 1500m, ERA5 a 800m, conca stretta, SVF basso
    horizon_test  = np.full(180, 25.0)   # orizzonte a 25° ovunque
    azimuths_test = np.linspace(0, 360, 180, endpoint=False)

    T_mean_era5 = np.array([-2, 0, 4, 9, 14, 18, 20, 19, 15, 9, 3, -1],
                            dtype=float)
    T_min_era5  = np.array([-6,-4,-1, 3,  8, 12, 14, 13,  9, 4,-1,-5],
                            dtype=float)
    T_max_era5  = np.array([ 2, 4, 9,15, 20, 24, 26, 25, 21,14, 7, 3],
                            dtype=float)
    POA_monthly = np.array([40,70,110,150,180,200,210,190,140,90,50,30],
                            dtype=float)

    res_th = compute_thermal_index(
        elev        = elev_t,
        dist_grid   = dist,
        s_elev      = 1500.0,
        horizon_deg = horizon_test,
        azimuths_deg= azimuths_test,
        T_mean_monthly_era5 = T_mean_era5,
        T_min_monthly_era5  = T_min_era5,
        T_max_monthly_era5  = T_max_era5,
        POA_monthly_kWh_m2  = POA_monthly,
        era5_elevation_m    = 800.0,
    )

    print(report_thermal_index(res_th, T_mean_era5, T_min_era5, T_max_era5))

    # Verifica fisica: sito più alto -> più freddo di ERA5
    assert res_th['dT_lapse_C'] < 0, "Sito più alto deve essere più freddo"
    # Valle stretta con SVF basso -> cold pooling negativo
    assert res_th['dT_cold_pool_C'] < 0, "Cold pool deve raffreddare T_min"
    print("  PASS")
