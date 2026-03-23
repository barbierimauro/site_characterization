"""
vegetation_plots.py
===================
Figure per la visualizzazione degli indici di vegetazione CRNS.

Tre funzioni principali:
    plot_seasonal_cycles()  — cicli stagionali mensili (media ± std)
    plot_timeseries()       — serie temporali complete
    plot_maps()             — mappe 2D degli indici nel footprint

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from datetime import datetime

MONTHS_SHORT = ["J","F","M","A","M","J","J","A","S","O","N","D"]
MONTHS_LONG  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor"  : "#f8f8f6",
    "axes.grid"       : True,
    "grid.color"      : "white",
    "grid.linewidth"  : 1.2,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
}

# Colori e range per ogni indice
INDEX_STYLE = {
    "ndvi"  : {"color": "#27ae60", "label": "NDVI",
               "vmin": -0.2, "vmax": 1.0,  "cmap": "RdYlGn"},
    "evi"   : {"color": "#2980b9", "label": "EVI",
               "vmin": -0.2, "vmax": 1.0,  "cmap": "RdYlGn"},
    "ndwi"  : {"color": "#1abc9c", "label": "NDWI",
               "vmin": -0.5, "vmax": 0.5,  "cmap": "RdBu"},
    "fcover": {"color": "#8e44ad", "label": "FCOVER",
               "vmin":  0.0, "vmax": 1.0,  "cmap": "YlGn"},
    "lai"   : {"color": "#e67e22", "label": "LAI [m²/m²]",
               "vmin":  0.0, "vmax": 6.0,  "cmap": "YlGn"},
}


# ---------------------------------------------------------------------------
# 1. Cicli stagionali
# ---------------------------------------------------------------------------

def plot_seasonal_cycles(res, path, site_name=""):
    """
    6 pannelli: NDVI, EVI, NDWI, FCOVER (Landsat 30m) +
                LAI (MODIS 500m) + f_veg Baatz 2015.
    Ogni pannello: media mensile ± std, con n_obs come taglia punti.
    """
    x   = np.arange(1, 13)
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.42, wspace=0.32)

    panels = [
        ("ndvi",   gs[0, 0], "Landsat 30m"),
        ("evi",    gs[0, 1], "Landsat 30m"),
        ("ndwi",   gs[0, 2], "Landsat 30m"),
        ("fcover", gs[1, 0], "Landsat 30m"),
        ("lai",    gs[1, 1], "MODIS 500m"),
        ("fveg",   gs[1, 2], "Baatz 2015"),
    ]

    with plt.rc_context(STYLE):
        for idx_name, gpos, source in panels:
            ax = fig.add_subplot(gpos)

            if idx_name in ["ndvi","evi","ndwi","fcover"]:
                mn   = res[f"landsat_{idx_name}_monthly_mean"]
                sd   = res[f"landsat_{idx_name}_monthly_std"]
                nobs = res[f"landsat_{idx_name}_monthly_nobs"]
                ist  = INDEX_STYLE[idx_name]
                cur  = res.get(f"landsat_{idx_name}_current")
            elif idx_name == "lai":
                mn   = res["modis_lai_monthly_mean"]
                sd   = res["modis_lai_monthly_std"]
                nobs = res["modis_lai_monthly_nobs"]
                ist  = INDEX_STYLE["lai"]
                cur  = res.get("modis_lai_current")
            else:  # f_veg
                mn   = res["f_veg_monthly"]
                sd   = np.zeros(12)
                nobs = res["modis_lai_monthly_nobs"]
                ist  = {"color": "#c0392b", "label": "f_veg (Baatz)"}
                cur  = res.get("f_veg_current")

            valid = ~np.isnan(mn)

            # Banda std
            if valid.any():
                ax.fill_between(
                    x[valid],
                    (mn - sd)[valid],
                    (mn + sd)[valid],
                    alpha=0.25, color=ist["color"],
                    label="±1 std")

            # Linea media
            ax.plot(x[valid], mn[valid],
                    "o-", color=ist["color"], lw=2.2, ms=5,
                    label="Monthly mean")

            # Dimensione punto proporzionale a n_obs
            if valid.any():
                sizes = np.clip(nobs[valid], 1, 100) * 3
                ax.scatter(x[valid], mn[valid],
                           s=sizes, color=ist["color"],
                           zorder=5, alpha=0.7)

            # Valore corrente
            if cur is not None and not np.isnan(cur):
                ax.axhline(cur, color=ist["color"],
                           ls="--", lw=1.5, alpha=0.7,
                           label=f"Current: {cur:.3f}")

            ax.set_xlim(0.5, 12.5)
            ax.set_xticks(x)
            ax.set_xticklabels(MONTHS_SHORT, fontsize=10)
            ax.set_xlabel("Month")
            ax.set_ylabel(ist["label"])
            ax.set_title(f"{ist['label']}  ({source})", fontsize=12)
            ax.legend(fontsize=9, loc="best")

            # Annotazione n_obs minimi
            if valid.any():
                min_nobs = nobs[valid].min()
                ax.annotate(f"min obs/month: {min_nobs}",
                            xy=(0.02, 0.02), xycoords="axes fraction",
                            fontsize=8, color="gray")

        title = f"Seasonal Vegetation Cycles  |  {site_name}  " \
                f"|  {res['lat']:.4f}N {res['lon']:.4f}E"
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 2. Serie temporali
# ---------------------------------------------------------------------------

def plot_timeseries(res, path, site_name=""):
    """
    Serie temporale completa di NDVI, EVI, NDWI, FCOVER (Landsat)
    e LAI (MODIS) dal 2013/2002 a oggi.
    Ogni panel mostra i punti delle singole scene + media annuale mobile.
    """
    ls_ts  = res["landsat_timeseries"]
    mod_ts = res["modis_lai_timeseries"]

    if not ls_ts and not mod_ts:
        print("  No timeseries data to plot")
        return

    def _to_decimal_year(date_str):
        try:
            d = datetime.fromisoformat(date_str)
            return d.year + (d.timetuple().tm_yday - 1) / 365.25
        except Exception:
            return np.nan

    # Converti Landsat timeseries
    ls_dates = np.array([_to_decimal_year(s["date"]) for s in ls_ts])

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(5, 1, figsize=(18, 22),
                                  sharex=False)
        fig.subplots_adjust(hspace=0.38)

        # --- Landsat indices ---
        for ax_i, (idx, ax) in enumerate(zip(
                ["ndvi","evi","ndwi","fcover"], axes[:4])):
            ist  = INDEX_STYLE[idx]
            vals = np.array([s.get(idx) for s in ls_ts], dtype=float)
            valid = ~np.isnan(vals) & ~np.isnan(ls_dates)

            if valid.any():
                ax.scatter(ls_dates[valid], vals[valid],
                           s=8, alpha=0.5, color=ist["color"],
                           label="Scene value")

                # Media annuale mobile (finestra 1 anno)
                sorted_idx = np.argsort(ls_dates[valid])
                xv = ls_dates[valid][sorted_idx]
                yv = vals[valid][sorted_idx]
                window = 12   # ~12 scene per anno
                if len(yv) >= window:
                    kernel = np.ones(window) / window
                    ym = np.convolve(yv, kernel, mode="same")
                    ax.plot(xv, ym, "-", color=ist["color"],
                            lw=2, alpha=0.9, label="12-scene moving avg")

                # Linea guida zero (per NDWI)
                if idx == "ndwi":
                    ax.axhline(0, color="gray", ls=":", lw=1)

                ax.set_ylim(ist["vmin"] - 0.05, ist["vmax"] + 0.05)

            ax.set_ylabel(ist["label"], fontsize=11)
            ax.set_title(f"{ist['label']}  (Landsat C2 L2, 30m)",
                         fontsize=11)
            ax.legend(fontsize=9, loc="upper left")

            # Griglia verticale anni
            if valid.any():
                for yr in range(int(ls_dates[valid].min()),
                                int(ls_dates[valid].max()) + 2):
                    ax.axvline(yr, color="white", lw=0.8, alpha=0.6)

            ax.set_xlabel("Year")

        # --- MODIS LAI ---
        ax = axes[4]
        if mod_ts:
            mod_dates = np.array([_to_decimal_year(s["date"])
                                   for s in mod_ts])
            mod_vals  = np.array([s["lai_mean"] for s in mod_ts],
                                  dtype=float)
            mod_std   = np.array([s["lai_std"]  for s in mod_ts],
                                  dtype=float)
            valid_m   = ~np.isnan(mod_vals)

            if valid_m.any():
                ax.fill_between(mod_dates[valid_m],
                                (mod_vals - mod_std)[valid_m],
                                (mod_vals + mod_std)[valid_m],
                                alpha=0.2, color=INDEX_STYLE["lai"]["color"])
                ax.scatter(mod_dates[valid_m], mod_vals[valid_m],
                           s=4, alpha=0.4,
                           color=INDEX_STYLE["lai"]["color"],
                           label="Scene mean (5x5 px)")

                sorted_idx = np.argsort(mod_dates[valid_m])
                xv = mod_dates[valid_m][sorted_idx]
                yv = mod_vals[valid_m][sorted_idx]
                window = 23   # ~23 scene per anno (4-day product)
                if len(yv) >= window:
                    kernel = np.ones(window) / window
                    ym = np.convolve(yv, kernel, mode="same")
                    ax.plot(xv, ym, "-",
                            color=INDEX_STYLE["lai"]["color"],
                            lw=2, alpha=0.9, label="~Annual moving avg")

                for yr in range(int(mod_dates[valid_m].min()),
                                int(mod_dates[valid_m].max()) + 2):
                    ax.axvline(yr, color="white", lw=0.8, alpha=0.6)

        ax.set_ylabel("LAI [m²/m²]", fontsize=11)
        ax.set_title("LAI  (MODIS MCD15A3H, 500m)", fontsize=11)
        ax.set_xlabel("Year")
        ax.legend(fontsize=9, loc="upper left")

        title = f"Vegetation Time Series  |  {site_name}  " \
                f"|  {res['lat']:.4f}N {res['lon']:.4f}E"
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.005)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 3. Mappe 2D
# ---------------------------------------------------------------------------

def plot_maps(res, dx_grid, dy_grid, dist_grid, r86, path,
              site_name=""):
    """
    Mappe 2D degli indici Landsat più recenti nel footprint +
    array 5x5 LAI MODIS.
    """
    indices_to_plot = [
        ("ndvi",   "landsat_ndvi_latest_map"),
        ("evi",    "landsat_evi_latest_map"),
        ("ndwi",   "landsat_ndwi_latest_map"),
        ("fcover", "landsat_fcover_latest_map"),
    ]
    lai_map = res.get("modis_lai_latest_map")

    # Controlla quante mappe sono disponibili
    available = [(n, k) for n, k in indices_to_plot
                 if res.get(k) is not None]

    n_panels = len(available) + (1 if lai_map is not None else 0)
    if n_panels == 0:
        print("  No map data available for plotting")
        return

    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(7 * ncols, 6 * nrows))
        if n_panels == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        ax_flat = axes.ravel()
        panel_i = 0

        # Cerchio r86
        theta  = np.linspace(0, 2*np.pi, 360)
        circle = r86 * np.array([np.sin(theta), np.cos(theta)])

        # --- Landsat maps ---
        for idx_name, key in available:
            ax  = ax_flat[panel_i]
            ist = INDEX_STYLE[idx_name]
            arr = res[key].copy().astype(float)

            # Maschera fuori footprint
            arr[dist_grid > r86] = np.nan

            # Colormap con diverging per NDWI
            if idx_name == "ndwi":
                norm = TwoSlopeNorm(
                    vmin=ist["vmin"], vcenter=0, vmax=ist["vmax"])
            else:
                norm = Normalize(vmin=ist["vmin"], vmax=ist["vmax"])

            im = ax.pcolormesh(dx_grid, dy_grid, arr,
                               cmap=ist["cmap"], norm=norm,
                               shading="auto")
            plt.colorbar(im, ax=ax, fraction=0.04,
                         label=ist["label"])

            # Cerchio r86
            ax.plot(circle[0], circle[1], "k--", lw=1.5, alpha=0.6,
                    label=f"r86={r86:.0f}m")
            ax.plot(0, 0, "r^", ms=10, zorder=5, label="Sensor")

            cur_date = res.get(f"landsat_{idx_name}_current_date", "?")
            ax.set_title(f"{ist['label']}  (Landsat 30m)\n"
                         f"Most recent: {cur_date}", fontsize=11)
            ax.set_xlabel("Easting offset (m)")
            ax.set_ylabel("Northing offset (m)")
            ax.set_aspect("equal")
            ax.legend(fontsize=8, loc="upper right")
            panel_i += 1

        # --- MODIS LAI 5x5 map ---
        if lai_map is not None and panel_i < len(ax_flat):
            ax  = ax_flat[panel_i]
            ist = INDEX_STYLE["lai"]
            n   = lai_map.shape[0]

            # Coordinate approssimate dei pixel MODIS (500m)
            px_size = 500.0
            x_coords = (np.arange(n) - n//2) * px_size
            y_coords = (np.arange(n) - n//2) * px_size
            XX, YY   = np.meshgrid(x_coords, y_coords)

            im = ax.pcolormesh(XX, YY, lai_map,
                               cmap=ist["cmap"],
                               vmin=ist["vmin"], vmax=ist["vmax"],
                               shading="auto")
            plt.colorbar(im, ax=ax, fraction=0.04,
                         label="LAI [m²/m²]")

            # Cerchio r86 sovrapposto
            ax.plot(circle[0], circle[1], "k--", lw=2, alpha=0.7,
                    label=f"r86={r86:.0f}m")
            ax.plot(0, 0, "r^", ms=12, zorder=5, label="Sensor")

            # Griglia pixel MODIS
            for xi in x_coords - px_size/2:
                ax.axvline(xi, color="gray", lw=0.5, alpha=0.4)
            for yi in y_coords - px_size/2:
                ax.axhline(yi, color="gray", lw=0.5, alpha=0.4)

            # Valori numerici in ogni pixel
            for ii in range(n):
                for jj in range(n):
                    v = lai_map[ii, jj]
                    if not np.isnan(v):
                        ax.text(x_coords[jj], y_coords[ii],
                                f"{v:.1f}", ha="center", va="center",
                                fontsize=10, fontweight="bold",
                                color="black")

            cur_date = res.get("modis_lai_current_date", "?")
            ax.set_title(f"LAI  (MODIS 500m, 5×5 pixels)\n"
                         f"Most recent: {cur_date}", fontsize=11)
            ax.set_xlabel("Easting offset (m)")
            ax.set_ylabel("Northing offset (m)")
            ax.set_aspect("equal")
            ax.legend(fontsize=8, loc="upper right")
            panel_i += 1

        # Nasconde pannelli vuoti
        for j in range(panel_i, len(ax_flat)):
            ax_flat[j].set_visible(False)

        title = f"Vegetation Maps  |  {site_name}  " \
                f"|  {res['lat']:.4f}N {res['lon']:.4f}E"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {path}")
