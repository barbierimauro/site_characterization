"""
era5_soil_moisture.py
=====================
Scarica soil moisture ERA5-Land oraria via Open-Meteo, anno per anno.
Salva ogni anno come file separato + aggregati mensili in cache.

Core: download, cache, aggregazione mensile.
Report e plot: era5_sm_output.py

Profondita disponibili (ERA5-Land):
  soil_moisture_0_to_7cm      [m³/m³]
  soil_moisture_7_to_28cm     [m³/m³]
  soil_moisture_28_to_100cm   [m³/m³]

Cache (una directory per sito):
  era5_sm_{hash}/
    hourly_YYYY.npz      — dati orari anno YYYY (riusabili per analisi TS)
    monthly_agg.npz      — aggregati mensili (ricalcolati solo se mancano)
    meta.json            — coordinate, periodo, data ultimo update

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import os
import json
import hashlib
import time
import numpy as np
from datetime import datetime, timezone

OPEN_METEO_HISTORY_URL = "https://archive-api.open-meteo.com/v1/archive"

SM_VARIABLES = [
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
]

SM_LABELS = {
    "soil_moisture_0_to_7cm"   : "SM 0-7 cm",
    "soil_moisture_7_to_28cm"  : "SM 7-28 cm",
    "soil_moisture_28_to_100cm": "SM 28-100 cm",
}

SM_DEPTHS_MID = {
    "soil_moisture_0_to_7cm"   : 3.5,
    "soil_moisture_7_to_28cm"  : 17.5,
    "soil_moisture_28_to_100cm": 64.0,
}

DEFAULT_START_YEAR = 2015
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

# Versione formato cache.  Incrementare quando cambia il layout dei .npz.
# v2 corregge il bug timestamps: pandas>=2.0 datetime64[ns]//10**9 sbagliato.
_CACHE_FORMAT_VERSION = 2


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

def _site_dir(cache_dir, lat, lon):
    h = hashlib.sha256(f"{lat:.5f}_{lon:.5f}".encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"era5_sm_{h}")


def _hourly_path(site_dir, year):
    return os.path.join(site_dir, f"hourly_{year}.npz")


def _monthly_path(site_dir):
    return os.path.join(site_dir, "monthly_agg.npz")


def _meta_path(site_dir):
    return os.path.join(site_dir, "meta.json")


def _load_meta(site_dir):
    p = _meta_path(site_dir)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def _save_meta(site_dir, meta):
    with open(_meta_path(site_dir), "w") as f:
        json.dump(meta, f, indent=2)


def _var_key(var):
    """soil_moisture_0_to_7cm -> sm0_7"""
    return (var.replace("soil_moisture_", "sm")
               .replace("_to_", "_")
               .replace("cm", ""))


# ---------------------------------------------------------------------------
# Download anno singolo
# ---------------------------------------------------------------------------

def _download_year(lat, lon, year):
    """
    Scarica dati orari ERA5-Land per un anno.
    Ritorna dict {var: float32 array} + timestamps int64.
    """
    from net_utils import http_get
    import pandas as pd

    now        = datetime.now(timezone.utc)
    date_start = f"{year}-01-01"
    date_end   = min(f"{year}-12-31", now.strftime("%Y-%m-%d"))

    params = {
        "latitude"  : lat,
        "longitude" : lon,
        "start_date": date_start,
        "end_date"  : date_end,
        "hourly"    : ",".join(SM_VARIABLES),
        "models"    : "era5_land",
        "timezone"  : "UTC",
    }

    t0   = time.perf_counter()
    resp = http_get(OPEN_METEO_HISTORY_URL,
                    params=params, timeout=120)
    dt_req = time.perf_counter() - t0

    hourly     = resp.json().get("hourly", {})
    times      = pd.to_datetime(hourly["time"])
    # Pandas >= 2.0 può usare risoluzione "s" (non "ns"): astype(int64)
    # restituisce già secondi, // 10**9 porta tutto a ~0 (= 1970 = gennaio).
    # Conversione robusta: forza datetime64[s] poi int64 → secondi epoch.
    timestamps = np.array(times, dtype="datetime64[s]").astype(np.int64)

    result = {"timestamps": timestamps,
              "_fmt_ver": np.array([_CACHE_FORMAT_VERSION], dtype=np.int32)}
    for var in SM_VARIABLES:
        vals = np.array(hourly.get(var, [np.nan]*len(times)),
                        dtype=np.float32)
        result[var] = vals

    n_valid = int(np.sum(~np.isnan(result[SM_VARIABLES[0]])))
    return result, len(timestamps), n_valid, dt_req


# ---------------------------------------------------------------------------
# Download tutti gli anni mancanti
# ---------------------------------------------------------------------------

def _download_missing_years(lat, lon, site_dir,
                              start_year, verbose=True):
    """
    Scarica anno per anno, salta quelli già in cache.
    Mostra anno corrente, elapsed, ETA.
    """
    now          = datetime.now(timezone.utc)
    current_year = now.year
    all_years    = list(range(start_year, current_year + 1))
    cached       = [y for y in all_years
                    if os.path.exists(_hourly_path(site_dir, y))]
    todo         = [y for y in all_years if y not in cached]

    if not todo:
        if verbose:
            print(f"   All {len(cached)} years already cached",
                  flush=True)
        return False   # nessun download

    if verbose and cached:
        print(f"   Cached: {cached}", flush=True)
    if verbose:
        print(f"   To download: {todo}", flush=True)

    t_global = time.perf_counter()
    for i, year in enumerate(todo):
        elapsed = time.perf_counter() - t_global
        if i > 0:
            eta = elapsed / i * (len(todo) - i)
            eta_str = f"ETA {eta:.0f}s"
        else:
            eta_str = "ETA unknown"

        if verbose:
            print(f"   [{i+1}/{len(todo)}] year {year}  "
                  f"elapsed={elapsed:.0f}s  {eta_str} ...",
                  end="  ", flush=True)

        try:
            data, n_hrs, n_val, dt_req = _download_year(
                lat, lon, year)
            np.savez_compressed(_hourly_path(site_dir, year), **data)
            if verbose:
                print(f"OK  {n_hrs} hrs  "
                      f"{n_val} valid  "
                      f"req={dt_req:.1f}s", flush=True)
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}", flush=True)

    total = time.perf_counter() - t_global
    if verbose:
        print(f"   Download complete in {total:.0f}s", flush=True)

    return True   # qualcosa scaricato


# ---------------------------------------------------------------------------
# Aggregazione mensile da cache
# ---------------------------------------------------------------------------

def _compute_monthly_agg(site_dir, years, verbose=True):
    """
    Calcola mean/std/min/max/nobs mensili da tutti gli anni in cache.
    """
    import pandas as pd

    by_month = {var: {m: [] for m in range(1, 13)}
                for var in SM_VARIABLES}

    for year in years:
        p = _hourly_path(site_dir, year)
        if not os.path.exists(p):
            continue
        d  = np.load(p)
        ts = pd.to_datetime(d["timestamps"], unit="s", utc=True)
        mo = ts.month.values

        for var in SM_VARIABLES:
            if var not in d:
                continue
            vals = d[var].astype(float)
            for m in range(1, 13):
                mask = (mo == m) & ~np.isnan(vals)
                if mask.any():
                    by_month[var][m].extend(vals[mask].tolist())

    agg = {}
    for var in SM_VARIABLES:
        key   = _var_key(var)
        means = np.full(12, np.nan, dtype=np.float32)
        stds  = np.full(12, np.nan, dtype=np.float32)
        mins  = np.full(12, np.nan, dtype=np.float32)
        maxs  = np.full(12, np.nan, dtype=np.float32)
        nobs  = np.zeros(12, dtype=np.int32)
        for m in range(1, 13):
            v = np.array(by_month[var][m])
            if len(v) == 0:
                continue
            means[m-1] = float(np.mean(v))
            stds[m-1]  = float(np.std(v))
            mins[m-1]  = float(np.min(v))
            maxs[m-1]  = float(np.max(v))
            nobs[m-1]  = len(v)
        agg[f"{key}_mean"] = means
        agg[f"{key}_std"]  = stds
        agg[f"{key}_min"]  = mins
        agg[f"{key}_max"]  = maxs
        agg[f"{key}_nobs"] = nobs

    if verbose:
        print(f"   Monthly aggregates from {len(years)} years",
              flush=True)
    return agg


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def get_era5_soil_moisture(
    lat, lon,
    cache_dir,
    start_year              = DEFAULT_START_YEAR,
    force_monthly_recompute = False,
    verbose                 = True,
):
    """
    Scarica e aggrega soil moisture ERA5-Land oraria per un sito CRNS.

    Prima run: scarica tutti gli anni dal start_year, salva hourly_YYYY.npz,
    calcola aggregati mensili in monthly_agg.npz.

    Run successive: salta anni già in cache, ricalcola aggregati solo
    se ci sono anni nuovi o se force_monthly_recompute=True.

    Parameters
    ----------
    lat, lon                : coordinate WGS84
    cache_dir               : directory root per la cache
    start_year              : primo anno (default 2015)
    force_monthly_recompute : forza ricalcolo aggregati mensili
    verbose                 : stampa progressi con ETA

    Returns
    -------
    dict con:
      Per ogni var in {sm0_7, sm7_28, sm28_100}:
        {var}_monthly_mean/std/min/max/nobs : array (12,)
        {var}_current                       : float, media ultimo mese
        {var}_annual_mean                   : float
      sm_profile_mean   : array (3,)
      sm_profile_depths : array (3,) [cm]
      years_available   : list[int]
      site_dir          : str  (usare con load_hourly_timeseries())
      from_cache        : bool
      months, lat, lon, start_year
    """
    site_dir = _site_dir(cache_dir, lat, lon)
    os.makedirs(site_dir, exist_ok=True)

    if verbose:
        print(f"   ERA5 SM site dir: {site_dir}", flush=True)

    # --- Invalida cache con formato vecchio (bug timestamp pandas>=2.0) ---
    now      = datetime.now(timezone.utc)
    all_years = list(range(start_year, now.year + 1))
    stale = []
    for y in all_years:
        p = _hourly_path(site_dir, y)
        if not os.path.exists(p):
            continue
        try:
            d = np.load(p, allow_pickle=False)
            ver = int(d["_fmt_ver"][0]) if "_fmt_ver" in d else 1
            if ver < _CACHE_FORMAT_VERSION:
                stale.append(y)
        except Exception:
            stale.append(y)
    if stale:
        if verbose:
            print(f"   ERA5 cache format outdated for years {stale} "
                  f"— re-downloading ...", flush=True)
        for y in stale:
            os.remove(_hourly_path(site_dir, y))
        # Forza anche ricalcolo aggregati
        mp = _monthly_path(site_dir)
        if os.path.exists(mp):
            os.remove(mp)

    # --- Download anni mancanti ---
    downloaded = _download_missing_years(
        lat, lon, site_dir, start_year, verbose)

    cached_years = sorted(
        y for y in all_years
        if os.path.exists(_hourly_path(site_dir, y)))
    from_cache   = not downloaded and not stale

    # --- Aggregati mensili ---
    meta          = _load_meta(site_dir)
    cached_agg_y  = meta.get("agg_years", [])
    need_recompute = (
        force_monthly_recompute
        or not os.path.exists(_monthly_path(site_dir))
        or set(cached_years) != set(cached_agg_y)
    )

    if need_recompute:
        if verbose:
            print("   Computing monthly aggregates ...", flush=True)
        agg = _compute_monthly_agg(site_dir, cached_years, verbose)
        np.savez_compressed(_monthly_path(site_dir), **agg)
        meta.update({"agg_years": cached_years,
                     "agg_computed": now.isoformat(),
                     "lat": lat, "lon": lon,
                     "start_year": start_year})
        _save_meta(site_dir, meta)
    else:
        if verbose:
            print("   Monthly aggregates: from cache", flush=True)
        agg = dict(np.load(_monthly_path(site_dir)))

    # --- Valore corrente (media ultimo mese) ---
    current_vals = {}
    if cached_years:
        try:
            import pandas as pd
            last_yr = max(cached_years)
            d       = np.load(_hourly_path(site_dir, last_yr))
            ts      = pd.to_datetime(d["timestamps"], unit="s", utc=True)
            last_mo = ts.month.max()
            for var in SM_VARIABLES:
                key  = _var_key(var)
                mask = (ts.month == last_mo) & \
                       ~np.isnan(d[var].astype(float))
                arr  = d[var][mask]
                current_vals[key] = float(np.mean(arr)) \
                                    if len(arr) > 0 else np.nan
        except Exception:
            for var in SM_VARIABLES:
                current_vals[_var_key(var)] = np.nan

    # --- Assembla ---
    result = {
        "from_cache"       : from_cache,
        "months"           : MONTHS,
        "lat"              : lat,
        "lon"              : lon,
        "start_year"       : start_year,
        "years_available"  : cached_years,
        "site_dir"         : site_dir,
        "sm_profile_depths": np.array([3.5, 17.5, 64.0]),
    }

    annual_means = []
    for var in SM_VARIABLES:
        key = _var_key(var)
        for stat in ["mean","std","min","max","nobs"]:
            result[f"{key}_monthly_{stat}"] = agg.get(
                f"{key}_{stat}", np.full(12, np.nan))
        result[f"{key}_current"]     = current_vals.get(key, np.nan)
        ann = float(np.nanmean(result[f"{key}_monthly_mean"]))
        result[f"{key}_annual_mean"] = ann
        annual_means.append(ann)

    result["sm_profile_mean"] = np.array(annual_means)
    return result


# ---------------------------------------------------------------------------
# Carica serie oraria per analisi TS
# ---------------------------------------------------------------------------

def load_hourly_timeseries(site_dir, years=None):
    """
    Carica la serie oraria completa dalla cache.

    Parameters
    ----------
    site_dir : str, da result['site_dir']
    years    : list[int] o None (tutti)

    Returns
    -------
    dict con 'timestamps' (int64 Unix UTC) e una chiave per ogni
    SM_VARIABLE (float32).
    Converti timestamps con: pd.to_datetime(ts, unit='s', utc=True)
    """
    meta  = _load_meta(site_dir)
    avail = meta.get("agg_years", [])
    if years is None:
        years = avail
    years = sorted(y for y in years if y in avail)
    if not years:
        raise ValueError(f"No years available in {site_dir}")

    all_ts, all_data = [], {v: [] for v in SM_VARIABLES}
    for year in years:
        p = _hourly_path(site_dir, year)
        if not os.path.exists(p):
            continue
        d = np.load(p)
        all_ts.append(d["timestamps"])
        for var in SM_VARIABLES:
            all_data[var].append(
                d[var] if var in d
                else np.full(len(d["timestamps"]),
                             np.nan, dtype=np.float32))

    result = {"timestamps": np.concatenate(all_ts)}
    for var in SM_VARIABLES:
        result[var] = np.concatenate(all_data[var])
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

"""
era5_sm_output.py
=================
Report testuale e plot per i risultati di get_era5_soil_moisture().

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

def report_era5_sm(res):
    w = 72
    M = res["months"]
    yr_range = (f"{res['start_year']}-"
                f"{max(res['years_available'])}"
                if res['years_available'] else "N/A")
    L = ["="*w,
         "ERA5-Land Soil Moisture  (Open-Meteo, ~9 km)",
         "="*w,
         f"  {res['lat']:.4f}N  {res['lon']:.4f}E  |  {yr_range}",
         f"  {len(res['years_available'])} years in cache  |  "
         f"{'all cached' if res['from_cache'] else 'updated'}",
         f"  Cache: {res['site_dir']}",
         ""]

    hdr = "  " + " "*13 + "  ".join(f"{m:>5}" for m in M)
    L.append(hdr)
    L.append("  " + "-"*(w-2))

    for var in SM_VARIABLES:
        key = _var_key(var)
        lbl = SM_LABELS[var]
        mn  = res[f"{key}_monthly_mean"]
        sd  = res[f"{key}_monthly_std"]
        nb  = res[f"{key}_monthly_nobs"].astype(int)
        cur = res[f"{key}_current"]
        ann = res[f"{key}_annual_mean"]
        mn_s = "  ".join(f"{v:.3f}" if not np.isnan(v)
                          else "  N/A" for v in mn)
        sd_s = "  ".join(f"{v:.3f}" if not np.isnan(v)
                          else "  N/A" for v in sd)
        nb_s = "  ".join(f"{v:5d}" for v in nb)
        cur_s = f"{cur:.3f}" if not np.isnan(cur) else "N/A"
        L += [f"  {lbl}",
              f"    mean  {mn_s}",
              f"    std   {sd_s}",
              f"    nobs  {nb_s}",
              f"    annual={ann:.3f}  current={cur_s}",
              ""]

    L.append("  Vertical profile (annual mean):")
    for d, v in zip(res["sm_profile_depths"], res["sm_profile_mean"]):
        bar = "█" * int(v * 80)
        L.append(f"    {d:5.1f} cm  {bar:<40} {v:.3f} m3/m3")
    L.append("="*w)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_era5_sm(res, path, site_name=""):
    """
    Sinistra: cicli mensili mean ± std per i 3 strati
    Destra  : profilo verticale annuo
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STYLE = {"figure.facecolor": "white",
             "axes.facecolor": "#f8f8f6",
             "axes.grid": True, "grid.color": "white",
             "axes.spines.top": False, "axes.spines.right": False}

    colors = ["#2166ac", "#4dac26", "#d01c8b"]
    x      = np.arange(1, 13)
    M      = res["months"]
    yr_str = (f"{res['start_year']}-"
              f"{max(res['years_available'])}"
              if res["years_available"] else "")

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                                  facecolor="white")
        ax = axes[0]
        for var, col in zip(SM_VARIABLES, colors):
            key   = _var_key(var)
            mn    = res[f"{key}_monthly_mean"]
            sd    = res[f"{key}_monthly_std"]
            valid = ~np.isnan(mn)
            ax.fill_between(x[valid],
                            (mn-sd)[valid], (mn+sd)[valid],
                            alpha=0.18, color=col)
            ax.plot(x[valid], mn[valid], "o-",
                    color=col, lw=2, ms=5,
                    label=SM_LABELS[var])
            cur = res[f"{key}_current"]
            if not np.isnan(cur):
                ax.axhline(cur, color=col, ls="--",
                           lw=1, alpha=0.5)

        ax.set_xlim(0.5, 12.5)
        ax.set_xticks(x)
        ax.set_xticklabels(M, fontsize=9)
        ax.set_xlabel("Month")
        ax.set_ylabel("Soil moisture [m3/m3]")
        ax.set_title(f"Monthly climatology  ({yr_str})\n"
                     "Shaded = +/-1 std  |  Dashed = current",
                     fontsize=11)
        ax.legend(fontsize=10)
        ax.set_ylim(0, None)

        ax2 = axes[1]
        depths = res["sm_profile_depths"]
        vals   = res["sm_profile_mean"]
        valid  = ~np.isnan(vals)
        ax2.barh(-depths[valid], vals[valid],
                 height=6,
                 color=[c for c,v in zip(colors, valid) if v],
                 alpha=0.8, edgecolor="white")
        ax2.plot(vals[valid], -depths[valid], "ko-", ms=7, zorder=5)
        for d, v in zip(depths[valid], vals[valid]):
            ax2.text(v + 0.003, -d, f"{v:.3f}",
                     va="center", fontsize=10)
        ax2.set_xlabel("Annual mean SM [m3/m3]")
        ax2.set_ylabel("Depth [cm]")
        ax2.set_yticks(-depths)
        ax2.set_yticklabels([f"{d:.0f} cm" for d in depths])
        ax2.set_title("Vertical SM profile\n(annual mean)",
                      fontsize=11)
        ax2.set_xlim(0, None)

        fig.suptitle(
            f"ERA5-Land Soil Moisture  |  {site_name}  |  "
            f"{res['lat']:.4f}N {res['lon']:.4f}E",
            fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {path}")
