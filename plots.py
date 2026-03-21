import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =============================================================================
# PLOTS
# =============================================================================

STYLE = {
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f6", "axes.grid": True,
    "grid.color": "white", "grid.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
}


def plot_main(elev, dx_grid, dy_grid, r86, kappa_topo, kappa_muon,
              results, path, lat, lon, dem_radius_m):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(20, 14))
        gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

        # DEM + footprint circles
        ax1  = fig.add_subplot(gs[0, :2])
        dist = np.sqrt(dx_grid**2 + dy_grid**2)
        em   = np.where(dist <= dem_radius_m,
                        np.where(np.isnan(elev), np.nanmean(elev), elev), np.nan)
        ext  = [dx_grid.min(), dx_grid.max(), dy_grid.min(), dy_grid.max()]
        im   = ax1.imshow(em, extent=ext, origin="upper",
                          cmap="terrain", aspect="equal", interpolation="bilinear")
        plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.02).set_label(
            "Elevation (m a.s.l.)", fontsize=10)
        th = np.linspace(0, 2*np.pi, 360)
        for frac, ls, lw, lbl in [(0.5,"--",1.2,"0.5*r86"),
                                    (1.0,"-", 2.0,"r86"),
                                    (1.5,":", 1.2,"1.5*r86")]:
            ax1.plot(r86*frac*np.sin(th), r86*frac*np.cos(th),
                     color="red", ls=ls, lw=lw, label=lbl)
        ax1.plot(0, 0, "r^", ms=12, zorder=5, label="Sensor")
        ax1.set_xlabel("Easting offset (m)"); ax1.set_ylabel("Northing offset (m)")
        ax1.set_title(f"DEM & Neutron Footprint  ({lat:.4f}N, {lon:.4f}E)  "
                      f"alt={results['sensor_alt']:.0f} m")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.set_xlim(-dem_radius_m*0.6, dem_radius_m*0.6)
        ax1.set_ylim(-dem_radius_m*0.6, dem_radius_m*0.6)

        # Theoretical kappa vs slope
        ax2 = fig.add_subplot(gs[0, 2])
        al  = np.linspace(0, 50, 300)
        ax2.plot(al, 1/np.cos(np.radians(al))**3, color="#c0392b", lw=2.5,
                 label="Hilltop (overestimate)")
        ax2.plot(al, np.cos(np.radians(al))**3,   color="#2980b9", lw=2.5,
                 label="Dolina (underestimate)")
        ax2.axhline(1.0, color="gray", ls="--", lw=1)
        ax2.axhline(kappa_topo, color="#e67e22", lw=2.5, ls="-.",
                    label=f"This site: {kappa_topo:.3f}")
        ax2.set_xlabel("Uniform slope angle (deg)")
        ax2.set_ylabel("kappa_topo (analytical)")
        ax2.set_title("Theoretical kappa vs slope")
        ax2.legend(fontsize=9); ax2.set_ylim(0.2, 2.5)

        # E-W profile
        ax3 = fig.add_subplot(gs[1, 0])
        mr  = np.argmin(np.abs(dy_grid[:, dy_grid.shape[1]//2]))
        ax3.plot(dx_grid[mr,:], elev[mr,:], color="#2c3e50", lw=1.8)
        ax3.axvline(0, color="red", ls="--", lw=1.2, label="Sensor")
        for s_ in [-1,1]:
            ax3.axvline(s_*r86, color="gray", ls=":", lw=1)
        ax3.axhline(results['sensor_alt'], color="orange", ls="-.", lw=1,
                    label=f"Sensor alt {results['sensor_alt']:.0f} m")
        ax3.set_xlabel("Easting offset (m)"); ax3.set_ylabel("Elevation (m)")
        ax3.set_title(f"E-W cross-section  (r86={r86:.0f} m)")
        ax3.legend(fontsize=8)
        ax3.set_xlim(-dem_radius_m*0.5, dem_radius_m*0.5)

        # N-S profile
        ax4 = fig.add_subplot(gs[1, 1])
        mc = np.argmin(np.abs(dx_grid[mr, :]))

        ax4.plot(dy_grid[:,mc], elev[:,mc], color="#2c3e50", lw=1.8)
        ax4.axvline(0, color="red", ls="--", lw=1.2, label="Sensor")
        for s_ in [-1,1]:
            ax4.axvline(s_*r86, color="gray", ls=":", lw=1)
        ax4.axhline(results['sensor_alt'], color="orange", ls="-.", lw=1,
                    label=f"Sensor alt {results['sensor_alt']:.0f} m")
        ax4.set_xlabel("Northing offset (m)"); ax4.set_ylabel("Elevation (m)")
        ax4.set_title(f"N-S cross-section  (r86={r86:.0f} m)")
        ax4.legend(fontsize=8)
        ax4.set_xlim(-dem_radius_m*0.5, dem_radius_m*0.5)

        # Kappa summary bar
        ax5  = fig.add_subplot(gs[1, 2])
        lbls = ["kappa_topo\n(neutrons)", "kappa_muon\n(muons)", "kappa_total"]
        vals = [kappa_topo, kappa_muon, kappa_topo*kappa_muon]
        cols = ["#e74c3c" if v > 1 else "#3498db" for v in vals]
        bars = ax5.bar(lbls, vals, color=cols, edgecolor="white", lw=1.5, width=0.5)
        ax5.axhline(1.0, color="gray", ls="--", lw=1.5, label="Reference (flat)")
        for bar, val in zip(bars, vals):
            ax5.text(bar.get_x()+bar.get_width()/2, val+0.01,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")
        ax5.set_ylabel("Correction factor k")
        ax5.set_title("Correction summary\n(blue=underestimate, red=overestimate)")
        ax5.set_ylim(0, max(max(vals)*1.25, 1.3))
        ax5.legend(fontsize=9)

        fig.suptitle(
            f"CRNS Topographic Correction  |  "
            f"{lat:.4f}N {lon:.4f}E  |  "
            f"Alt={results['sensor_alt']:.0f}m  P={results['pressure']:.1f}hPa",
            fontsize=14, fontweight="bold", y=0.99)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {path}")


def plot_footprint(elev, dx_grid, dy_grid, dist_grid, s_elev, r86, z86_cm,
                   kappa_topo, wmap, az_neutron, overlap_az, deficit_az, path):
    """
    Left:   2D pixel classification (deficit/partial/full).
    Centre: radial overlap profile.
    Right:  polar per-azimuth overlap fraction.
    """
    z86_m     = z86_cm / 100.0
    z_bot_ref = s_elev - z86_m
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(22, 8))
        gs  = GridSpec(1, 3, figure=fig, wspace=0.38)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2], projection="polar")
        th  = np.linspace(0, 2*np.pi, 360)
        clip = min(1.5 * r86, 400.0)

        # ── LEFT: pixel classification map ───────────────────────────────
        mask_fp = dist_grid <= r86
        cat_map = np.full_like(elev, np.nan)
        e_in    = elev[mask_fp]
        cat     = np.where(e_in >= s_elev, 3.0,
                  np.where(e_in >= z_bot_ref, 2.0, 1.0))
        cat_map[mask_fp] = cat

        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap3 = ListedColormap(["#e74c3c", "#f39c12", "#27ae60"])
        norm3 = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap3.N)
        ax1.pcolormesh(dx_grid, dy_grid, cat_map,
                       cmap=cmap3, norm=norm3, shading="auto")
        # W(r) intensity overlay
        wm_norm = np.where(mask_fp, wmap / max(float(np.nanmax(wmap)), 1e-6), np.nan)
        ax1.contourf(dx_grid, dy_grid, wm_norm,
                     levels=np.linspace(0, 1, 15), cmap="binary", alpha=0.22)

        ax1.plot(r86*np.sin(th), r86*np.cos(th), "k--", lw=2,
                 label=f"r86={r86:.0f}m")
        ax1.plot(0, 0, "k^", ms=10, zorder=5, label="Sensor")
        from matplotlib.patches import Patch
        ax1.legend(handles=[
            Patch(color="#e74c3c", label="Deficit (z<slab bottom)"),
            Patch(color="#f39c12", label="Partial overlap"),
            Patch(color="#27ae60", label="Full (z≥ref level)"),
            plt.Line2D([0],[0], color="k", ls="--", label=f"r86={r86:.0f}m"),
        ], fontsize=8, loc="upper right")
        ax1.set_xlabel("Easting offset (m)"); ax1.set_ylabel("Northing offset (m)")
        ax1.set_title(f"Footprint pixel classification\n"
                      f"kappa_topo={kappa_topo:.4f}  r86={r86:.0f}m  z86={z86_cm:.1f}cm\n"
                      f"Darker = higher W(r) weight")
        ax1.set_aspect("equal")
        ax1.set_xlim(-clip, clip); ax1.set_ylim(-clip, clip)

        # ── CENTRE: radial overlap profile ───────────────────────────────
        r_bins = np.arange(0, r86 + 15, 15.0)
        r_mid  = 0.5*(r_bins[:-1]+r_bins[1:])
        ovl_r  = np.zeros(len(r_mid))
        wgt_r  = np.zeros(len(r_mid))
        for k, (r0, r1) in enumerate(zip(r_bins[:-1], r_bins[1:])):
            ring = (dist_grid >= r0) & (dist_grid < r1) & mask_fp & ~np.isnan(elev)
            if not np.any(ring): continue
            e_r = elev[ring]; r_r = dist_grid[ring]
            W   = np.where(r_r < 1e-3, 0.0, np.exp(-r_r / (r86 / 3.0)))
            z_t = np.minimum(e_r, s_elev)
            ovl = np.clip(z_t - z_bot_ref, 0.0, z86_m) / z86_m
            Ws  = W.sum()
            ovl_r[k] = float(np.sum(W * ovl) / Ws) if Ws > 0 else 0
            wgt_r[k] = float(Ws)

        bar_cols = [plt.get_cmap("RdYlGn")(v) for v in ovl_r]
        ax2.bar(r_mid, ovl_r, width=13, color=bar_cols, edgecolor="white",
                align="center")
        wn = wgt_r / max(wgt_r.max(), 1e-6)
        ax2.plot(r_mid, wn, "k--", lw=1.8, label="W(r) normalised")
        ax2.axhline(1.0, color="gray", ls=":", lw=1)
        ax2.axhline(kappa_topo, color="#e67e22", ls="-.", lw=2,
                    label=f"kappa_topo={kappa_topo:.4f}")
        ax2.set_xlabel("Distance from sensor (m)")
        ax2.set_ylabel("Mean overlap fraction  (0=deficit, 1=full)")
        ax2.set_title("Radial overlap profile\n"
                      "Colour=overlap fraction per ring (red→green)\n"
                      "Dashed=W(r) shape (where sensitivity lives)")
        ax2.set_xlim(0, r86); ax2.set_ylim(0, 1.15)
        ax2.legend(fontsize=9)

        # ── RIGHT: polar per-azimuth overlap fraction ─────────────────────
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        cmap_rg = plt.get_cmap("RdYlGn")
        norm_rg = Normalize(vmin=0, vmax=1)
        azr = np.radians(az_neutron)
        daz = azr[1] - azr[0] if len(azr) > 1 else np.radians(2)
        ax3.set_theta_zero_location("N"); ax3.set_theta_direction(-1)
        for j in range(len(azr)):
            col = cmap_rg(norm_rg(overlap_az[j]))
            ax3.bar(azr[j], overlap_az[j], width=daz, color=col,
                    alpha=0.85, align="edge")
        azp = np.append(azr, azr[0])
        ax3.plot(azp, np.ones(len(azp)), "k--", lw=1, alpha=0.5)
        ax3.set_rmax(1.05); ax3.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax3.set_rlabel_position(45)
        ax3.set_title("Per-azimuth mean overlap fraction\n"
                      "Green=full / Red=deficit\n"
                      "W(r)-weighted average along each ray",
                      pad=20, fontsize=11)
        for azl, lbl in [(0,"N"),(90,"E"),(180,"S"),(270,"W")]:
            ax3.text(np.radians(azl), 1.18, lbl,
                     ha="center", va="center", fontsize=11, fontweight="bold")
        sm = ScalarMappable(cmap=cmap_rg, norm=norm_rg)
        sm.set_array([])
        plt.colorbar(sm, ax=ax3, fraction=0.03, pad=0.1,
                     label="Overlap fraction")

        fig.suptitle("CRNS Neutron Footprint — Soil Volume Loss by Direction",
                     fontsize=14, fontweight="bold")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {path}")


def plot_horizon(azimuths, horizon, kappa_muon, per_az_muon, path):
    """
    Left: polar horizon elevation angle.
    Right: polar per-azimuth muon sensitivity fraction.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                                  subplot_kw=dict(projection="polar"))
        azr  = np.radians(azimuths)
        azp  = np.append(azr, azr[0])
        hp   = np.append(horizon, horizon[0])
        mup  = np.append(per_az_muon, per_az_muon[0])

        # Left: horizon angle
        ax = axes[0]
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.fill(azp, hp, alpha=0.40, color="#c0392b")
        ax.plot(azp, hp, color="#c0392b", lw=2)
        ax.set_rlabel_position(45)
        ax.set_title("Horizon elevation angle psi (deg)\n"
                     "Radial axis = degrees above horizontal\n"
                     "A flat site would be a circle at 0 deg",
                     pad=20, fontsize=11)
        for azl, lbl in [(0,"N"),(90,"E"),(180,"S"),(270,"W")]:
            ax.text(np.radians(azl), ax.get_rmax()*1.13, lbl,
                    ha="center", va="center", fontsize=11, fontweight="bold")

        # Right: per-azimuth muon sensitivity
        ax2 = axes[1]
        ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
        ax2.fill(azp, mup, alpha=0.40, color="#2980b9")
        ax2.plot(azp, mup, color="#2980b9", lw=2)
        ax2.plot(azp, np.ones_like(azp), "k--", lw=1, label="Unobstructed ref.")
        ax2.set_rmax(1.05); ax2.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_rlabel_position(45)
        ax2.set_title(f"Per-azimuth muon sensitivity\n"
                      f"(fraction of unobstructed flux)\n"
                      f"Mean over all azimuths = kappa_muon = {kappa_muon:.4f}",
                      pad=20, fontsize=11)
        ax2.legend(loc="lower right", fontsize=9)
        for azl, lbl in [(0,"N"),(90,"E"),(180,"S"),(270,"W")]:
            ax2.text(np.radians(azl), 1.13, lbl,
                     ha="center", va="center", fontsize=11, fontweight="bold")

        fig.suptitle("Horizon Angles & Muon FOV Correction",
                     fontsize=14, fontweight="bold", y=1.01)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {path}")


def plot_fov_detail(azimuths, horizon, per_az_muon, kappa_muon,
                    az_neutron, overlap_az, r86, z86_cm, kappa_topo, path,
                    lat, lon, sensor_alt):
    """
    Detailed azimuth vs elevation angle FOV maps for both muons and neutrons.
    X-axis: azimuth 0-360 deg (N=0, E=90, S=180, W=270).
    Y-axis: elevation angle 0-90 deg (0=horizontal, 90=vertical).
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(22, 8))
        az_fine = np.linspace(0, 360, 720)
        daz     = az_fine[1] - az_fine[0]

        hor_i = np.interp(az_fine, azimuths,
                          np.append(horizon, horizon[0])[:len(azimuths)],
                          period=360)
        ovl_i = np.interp(az_fine, az_neutron,
                          np.append(overlap_az, overlap_az[0])[:len(az_neutron)],
                          period=360)
        mu_i  = np.interp(az_fine, azimuths,
                          np.append(per_az_muon, per_az_muon[0])[:len(azimuths)],
                          period=360)

        # ── LEFT: neutron overlap fraction bar chart by azimuth ───────────
        ax = axes[0]
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        cmap_rg = plt.get_cmap("RdYlGn")
        norm_rg = Normalize(vmin=0, vmax=1)
        for j, az in enumerate(az_fine):
            col = cmap_rg(norm_rg(ovl_i[j]))
            ax.bar(az, ovl_i[j], width=daz*1.05, color=col,
                   align="edge", linewidth=0)
        ax.axhline(1.0, color="gray", ls="--", lw=1.5, label="Full ref. (=1)")
        ax.axhline(kappa_topo, color="#e67e22", ls="-.", lw=2.5,
                   label=f"kappa_topo = {kappa_topo:.4f}")
        ax.set_xlabel("Azimuth (deg,  N=0  E=90  S=180  W=270)", fontsize=11)
        ax.set_ylabel("W(r)-weighted mean overlap fraction", fontsize=11)
        ax.set_title(f"Neutron: soil contribution per compass direction\n"
                     f"kappa_topo={kappa_topo:.4f}   r86={r86:.0f}m   z86={z86_cm:.1f}cm\n"
                     f"Green=full slab  Yellow=partial  Red=deficit (no contribution)",
                     fontsize=11)
        ax.set_xlim(0, 360); ax.set_ylim(0, 1.15)
        ax.set_xticks([0,45,90,135,180,225,270,315,360])
        ax.set_xticklabels(["N","NE","E","SE","S","SW","W","NW","N"])
        ax.legend(fontsize=10)
        for xv in [90,180,270]:
            ax.axvline(xv, color="gray", ls=":", lw=0.8)
        sm1 = ScalarMappable(cmap=cmap_rg, norm=norm_rg)
        sm1.set_array([])
        plt.colorbar(sm1, ax=ax, fraction=0.03, label="Overlap fraction")

        # ── RIGHT: muon FOV — azimuth vs elevation angle ──────────────────
        ax2 = axes[1]
        el_fine   = np.linspace(0, 90, 400)
        AZ2, EL2  = np.meshgrid(az_fine, el_fine)
        theta_z   = np.radians(90 - el_fine)
        muon_w    = np.cos(theta_z)**2 * np.sin(theta_z)
        muon_w   /= muon_w.max()
        muon_map  = np.outer(muon_w, np.ones(len(az_fine)))
        blocked   = EL2 < hor_i[np.newaxis, :]
        muon_map[blocked] = np.nan

        c2 = ax2.pcolormesh(az_fine, el_fine, muon_map,
                             cmap="Blues", shading="auto", vmin=0, vmax=1)
        plt.colorbar(c2, ax=ax2, fraction=0.03).set_label(
            "cos²(θ_z)·sin(θ_z)  (muon angular weight)\n"
            "peak ~45°, zero at zenith and horizon", fontsize=9)
        ax2.fill_between(az_fine, 0, hor_i, color="#95a5a6",
                          alpha=0.88, label="Terrain-blocked region")
        ax2.plot(az_fine, hor_i, "k-", lw=2.5, label="Horizon psi(az)")
        ax2.plot(az_fine, mu_i * 90, color="orange", lw=2.5,
                 label=f"Per-az muon fraction × 90°  (mean={kappa_muon:.3f})")
        ax2.set_xlabel("Azimuth (deg,  N=0  E=90  S=180  W=270)", fontsize=11)
        ax2.set_ylabel("Elevation angle above horizontal (deg)", fontsize=11)
        ax2.set_title(f"Muon FOV — visible sky weighted by cos²(θ_z)\n"
                      f"kappa_muon={kappa_muon:.4f}  (flat site=1.0000)\n"
                      f"Blue=muon-weighted sky  Grey=blocked  Orange=per-az fraction",
                      fontsize=11)
        ax2.set_xlim(0, 360); ax2.set_ylim(0, 90)
        ax2.set_xticks([0,45,90,135,180,225,270,315,360])
        ax2.set_xticklabels(["N","NE","E","SE","S","SW","W","NW","N"])
        ax2.legend(fontsize=10, loc="upper right")
        for xv in [90,180,270]:
            ax2.axvline(xv, color="gray", ls=":", lw=0.8)

        fig.suptitle(
            f"Detailed FOV  |  {lat:.4f}N {lon:.4f}E  |  "
            f"Alt={sensor_alt:.0f}m  r86={r86:.0f}m",
            fontsize=14, fontweight="bold")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {path}")



# ── helpers shared by new plot functions ─────────────────────────────────────
_MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
           'Jul','Aug','Sep','Oct','Nov','Dec']
_MX = np.arange(1, 13)


def plot_climate(site_climate, thermal, path, lat, lon, sensor_alt):
    """Six-panel climate summary: radiation, temperature, precipitation,
    wind, relative humidity, pressure — all monthly."""
    sc = site_climate
    th = thermal or {}

    def _a(d, key, fill=np.nan):
        v = d.get(key)
        return np.asarray(v, dtype=float) if v is not None else np.full(12, fill)

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(20, 18))
        gs  = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

        # (0,0) Solar radiation
        ax  = fig.add_subplot(gs[0, 0])
        ghi = _a(sc, 'GHI_monthly_kWh_m2')
        dni = _a(sc, 'DNI_monthly_kWh_m2')
        dhi = _a(sc, 'DHI_monthly_kWh_m2')
        poa = _a(sc, 'POA_monthly_kWh_m2')
        w   = 0.25
        ax.bar(_MX - w, ghi, width=w, color='#f39c12', label='GHI', align='center')
        ax.bar(_MX,     dni, width=w, color='#e74c3c', label='DNI', align='center')
        ax.bar(_MX + w, dhi, width=w, color='#3498db', label='DHI', align='center')
        ax.plot(_MX, poa, 'k^-', lw=2, ms=7, label='POA (tilted)')
        ax.set_xticks(_MX); ax.set_xticklabels(_MONTHS, fontsize=9)
        ax.set_ylabel('Irradiation (kWh/m\u00b2/month)')
        ax.set_title('Monthly Solar Radiation\n(GHI / DNI / DHI + POA tilted plane)')
        ax.legend(fontsize=9, ncol=2)

        # (0,1) Temperature
        ax     = fig.add_subplot(gs[0, 1])
        t_mean = _a(th, 'T_mean_corrected_C') if th.get('T_mean_corrected_C') is not None \
                 else _a(sc, 'T_mean_monthly_C')
        t_min  = _a(th, 'T_min_corrected_C')  if th.get('T_min_corrected_C')  is not None \
                 else _a(sc, 'T_min_monthly_C')
        t_max  = _a(th, 'T_max_corrected_C')  if th.get('T_max_corrected_C')  is not None \
                 else _a(sc, 'T_max_monthly_C')
        frost  = _a(th, 'frost_days_monthly', 0.0) if th.get('frost_days_monthly') is not None \
                 else _a(sc, 'frost_days_monthly', 0.0)
        ax.fill_between(_MX, t_min, t_max, alpha=0.18, color='#e67e22', label='T_min \u2013 T_max')
        ax.plot(_MX, t_mean, 'o-',  color='#e67e22', lw=2.5, ms=7, label='T_mean')
        ax.plot(_MX, t_min,  's--', color='#3498db', lw=1.5, ms=5, label='T_min')
        ax.plot(_MX, t_max,  '^--', color='#e74c3c', lw=1.5, ms=5, label='T_max')
        ax.axhline(0, color='gray', ls=':', lw=1)
        ax2 = ax.twinx()
        ax2.bar(_MX, frost, color='#2980b9', alpha=0.28, width=0.7, label='Frost days')
        ax2.set_ylabel('Frost days/month', color='#2980b9', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='#2980b9')
        ax2.set_ylim(0, max(float(np.nanmax(frost)) * 2.5, 5))
        ax.set_xticks(_MX); ax.set_xticklabels(_MONTHS, fontsize=9)
        ax.set_ylabel('Temperature (\u00b0C)')
        ax.set_title('Monthly Temperature (site-corrected)\n+ frost days')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2)

        # (1,0) Precipitation
        ax    = fig.add_subplot(gs[1, 0])
        prcp  = _a(sc, 'precip_monthly_mm')
        rdays = _a(sc, 'rainy_days_monthly', 0.0)
        ax.bar(_MX, prcp, color='#2980b9', alpha=0.75, width=0.7, label='Precipitation (mm)')
        ax.axhline(30,  color='gray',    ls='--', lw=1, label='Dry threshold (30 mm)')
        ax.axhline(100, color='#2c3e50', ls='--', lw=1, label='Wet threshold (100 mm)')
        ax2 = ax.twinx()
        ax2.plot(_MX, rdays, 'ko-', lw=2, ms=7, label='Rainy days')
        ax2.set_ylabel('Rainy days/month', fontsize=10)
        ax2.set_ylim(0, max(float(np.nanmax(rdays)) * 2.0, 5))
        ax.set_xticks(_MX); ax.set_xticklabels(_MONTHS, fontsize=9)
        ax.set_ylabel('Precipitation (mm/month)')
        ax.set_title('Monthly Precipitation (ERA5 ~31 km)\n+ rainy days (> 1 mm)')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2)

        # (1,1) Wind speed
        ax      = fig.add_subplot(gs[1, 1])
        ws_mean = _a(sc, 'WS_mean_monthly_ms')
        ws_p95  = _a(sc, 'WS_p95_monthly_ms')
        ws_max  = _a(sc, 'WS_max_monthly_ms')
        ax.fill_between(_MX, ws_mean, ws_p95, alpha=0.18, color='#27ae60', label='Mean \u2013 P95')
        ax.plot(_MX, ws_mean, 'o-',  color='#27ae60', lw=2.5, ms=7, label='WS mean')
        ax.plot(_MX, ws_p95,  's--', color='#2c3e50', lw=1.8, ms=5, label='WS P95')
        ax.plot(_MX, ws_max,  '^:',  color='#e74c3c', lw=1.5, ms=5, label='WS max')
        ax.set_xticks(_MX); ax.set_xticklabels(_MONTHS, fontsize=9)
        ax.set_ylabel('Wind speed (m/s)')
        ax.set_title('Monthly Wind Speed\n(mean / P95 / max)')
        ax.legend(fontsize=9, ncol=2)

        # (2,0) Relative humidity
        ax      = fig.add_subplot(gs[2, 0])
        rh_mean = _a(sc, 'RH_mean_monthly_pct')
        rh_min  = _a(sc, 'RH_min_monthly_pct')
        rh_max  = _a(sc, 'RH_max_monthly_pct')
        ax.fill_between(_MX, rh_min, rh_max, alpha=0.18, color='#9b59b6', label='RH min \u2013 max')
        ax.plot(_MX, rh_mean, 'o-',  color='#9b59b6', lw=2.5, ms=7, label='RH mean')
        ax.plot(_MX, rh_min,  's--', color='#8e44ad', lw=1.5, ms=5, label='RH min')
        ax.plot(_MX, rh_max,  '^--', color='#6c3483', lw=1.5, ms=5, label='RH max')
        ax.set_ylim(0, 110); ax.axhline(100, color='gray', ls=':', lw=1)
        ax.set_xticks(_MX); ax.set_xticklabels(_MONTHS, fontsize=9)
        ax.set_ylabel('Relative humidity (%)')
        ax.set_title('Monthly Relative Humidity\n(mean / min / max)')
        ax.legend(fontsize=9, ncol=2)

        # (2,1) Pressure
        ax = fig.add_subplot(gs[2, 1])
        sp = _a(sc, 'SP_mean_monthly_hPa')
        ax.plot(_MX, sp, 'o-', color='#1abc9c', lw=2.5, ms=8)
        ax.fill_between(_MX, np.nanmin(sp) * 0.999, sp, alpha=0.18, color='#1abc9c')
        ax.axhline(float(np.nanmean(sp)), color='gray', ls='--', lw=1.2,
                   label=f'Annual mean {np.nanmean(sp):.1f} hPa')
        ax.set_xticks(_MX); ax.set_xticklabels(_MONTHS, fontsize=9)
        ax.set_ylabel('Surface pressure (hPa)')
        ax.set_title('Monthly Mean Surface Pressure')
        ax.legend(fontsize=9)

        fig.suptitle(
            f"Site Climate Summary  |  {lat:.4f}N {lon:.4f}E  |  "
            f"Alt={sensor_alt:.0f} m  (PVGIS TMY + ERA5 Open-Meteo)",
            fontsize=14, fontweight='bold', y=1.01)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    print(f"  Saved: {path}")


def plot_soil(soil, path, lat, lon):
    """Four-panel soil characterization:
    heatmap of all 9 properties x 6 depths, physical profiles,
    geochemical profiles, CRNS-weighted summary bar chart."""
    PROPS  = ['bdod','clay','sand','silt','soc','phh2o','cec','cfvo','nitrogen']
    PLBLS  = ['Bulk density\n(g/cm\u00b3)','Clay\n(%)','Sand\n(%)','Silt\n(%)',
              'SOC\n(g/kg)','pH\n(H\u2082O)','CEC\n(cmol/kg)',
              'Coarse frags\n(%)','Nitrogen\n(g/kg)']
    DEPTHS = ['0-5','5-15','15-30','30-60','60-100','100-200']
    DMID   = np.array([2.5, 10.0, 22.5, 45.0, 80.0, 150.0])

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(22, 16))
        gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

        # (0,0) Normalised heatmap
        ax  = fig.add_subplot(gs[0, 0])
        mat = np.full((len(PROPS), 6), np.nan)
        for i, p in enumerate(PROPS):
            vals = np.asarray(soil.get(f'{p}_profile', {}).get('mean',
                              np.full(6, np.nan)), dtype=float)
            mat[i, :] = vals
        mat_n = np.full_like(mat, np.nan)
        for i in range(len(PROPS)):
            row = mat[i]
            lo, hi = np.nanmin(row), np.nanmax(row)
            if not np.isnan(lo) and hi > lo:
                mat_n[i] = (row - lo) / (hi - lo)
            elif not np.isnan(lo):
                mat_n[i] = 0.5
        im = ax.imshow(mat_n, aspect='auto', cmap='YlOrBr', vmin=0, vmax=1,
                       interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(
            'Normalised value (0=min, 1=max per property)', fontsize=9)
        ax.set_xticks(range(6)); ax.set_xticklabels(DEPTHS, fontsize=9)
        ax.set_yticks(range(len(PROPS))); ax.set_yticklabels(PLBLS, fontsize=9)
        ax.set_xlabel('Depth layer (cm)')
        ax.set_title('Soil Property Heatmap (normalised per property)\nSoilGrids v2.0 ISRIC 250 m')
        for i in range(len(PROPS)):
            for j in range(6):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2g}', ha='center', va='center', fontsize=7.5,
                            color='white' if (not np.isnan(mat_n[i,j]) and mat_n[i,j] > 0.6) else 'black')

        # (0,1) Physical profiles: bdod, clay, sand, silt
        ax = fig.add_subplot(gs[0, 1])
        for prop, col, lbl in [('bdod','#e67e22','Bulk density (g/cm\u00b3)'),
                                ('clay','#c0392b','Clay (%)'),
                                ('sand','#f1c40f','Sand (%)'),
                                ('silt','#95a5a6','Silt (%)')]:
            prof = soil.get(f'{prop}_profile', {})
            vals = np.asarray(prof.get('mean', np.full(6, np.nan)), dtype=float)
            unc  = np.asarray(prof.get('uncertainty', np.zeros(6)), dtype=float)
            mask = ~np.isnan(vals)
            if np.any(mask):
                ax.plot(vals[mask], DMID[mask], 'o-', color=col, lw=2, ms=7, label=lbl)
                ax.fill_betweenx(DMID[mask], np.maximum(0, vals[mask]-unc[mask]),
                                 vals[mask]+unc[mask], alpha=0.15, color=col)
        ax.invert_yaxis(); ax.set_ylim(200, 0)
        ax.set_xlabel('Value'); ax.set_ylabel('Depth midpoint (cm)')
        ax.set_title('Physical Soil Profiles\n(mean \u00b1 uncertainty shading)')
        ax.legend(fontsize=9)

        # (1,0) Geochemical profiles: soc, pH, CEC, cfvo, nitrogen
        ax = fig.add_subplot(gs[1, 0])
        for prop, col, lbl in [('soc',     '#27ae60','SOC (g/kg)'),
                                ('phh2o',  '#8e44ad','pH (H\u2082O)'),
                                ('cec',    '#2980b9','CEC (cmol/kg)'),
                                ('cfvo',   '#7f8c8d','Coarse frags (%)'),
                                ('nitrogen','#16a085','Nitrogen (g/kg)')]:
            prof = soil.get(f'{prop}_profile', {})
            vals = np.asarray(prof.get('mean', np.full(6, np.nan)), dtype=float)
            unc  = np.asarray(prof.get('uncertainty', np.zeros(6)), dtype=float)
            mask = ~np.isnan(vals)
            if np.any(mask):
                ax.plot(vals[mask], DMID[mask], 's-', color=col, lw=2, ms=7, label=lbl)
                ax.fill_betweenx(DMID[mask], np.maximum(0, vals[mask]-unc[mask]),
                                 vals[mask]+unc[mask], alpha=0.15, color=col)
        ax.invert_yaxis(); ax.set_ylim(200, 0)
        ax.set_xlabel('Value'); ax.set_ylabel('Depth midpoint (cm)')
        ax.set_title('Geochemical Soil Profiles\n(SOC / pH / CEC / coarse frags / N)')
        ax.legend(fontsize=9)

        # (1,1) CRNS-weighted means horizontal bar chart
        ax    = fig.add_subplot(gs[1, 1])
        means, uncs, lbls = [], [], []
        COLS  = ['#e67e22','#c0392b','#f1c40f','#95a5a6',
                 '#27ae60','#8e44ad','#2980b9','#7f8c8d','#16a085']
        for p, lbl in zip(PROPS, PLBLS):
            v = soil.get(f'{p}_crns')
            u = soil.get(f'{p}_crns_unc')
            means.append(float(v) if v is not None and not np.isnan(float(v)) else np.nan)
            uncs.append( float(u) if u is not None and not np.isnan(float(u)) else 0.0)
            lbls.append(lbl.replace('\n',' '))
        y = np.arange(len(PROPS))
        ax.barh(y, means, xerr=uncs, color=COLS, alpha=0.82, edgecolor='white',
                height=0.7, capsize=4, error_kw=dict(lw=1.5, capthick=1.5))
        ax.set_yticks(y); ax.set_yticklabels(lbls, fontsize=9)
        ax.set_xlabel('CRNS-weighted mean value')
        ax.set_title('CRNS-Weighted Soil Properties\n(z86 exponential depth weighting)')
        lw_gg   = soil.get('lattice_water_gg', np.nan)
        lw_str  = f'{float(lw_gg):.4f} g/g' if lw_gg is not None and not np.isnan(float(lw_gg)) else 'N/A'
        ax.text(0.98, 0.03,
                f"Texture: {soil.get('texture_class','N/A')}\n"
                f"WRB: {soil.get('wrb_class','N/A')}\n"
                f"Lattice water: {lw_str}",
                transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', alpha=0.85))

        fig.suptitle(
            f"Soil Characterization  |  {lat:.4f}N {lon:.4f}E  |  SoilGrids v2.0 ISRIC 250 m",
            fontsize=14, fontweight='bold', y=1.01)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    print(f"  Saved: {path}")


def plot_thermal(site_climate, thermal, path, lat, lon, sensor_alt):
    """Four-panel thermal correction: ERA5 vs corrected T_mean,
    min/max corrected, correction components, frost days comparison."""
    sc = site_climate
    th = thermal or {}

    def _a12(d, key):
        v = d.get(key)
        return np.asarray(v, dtype=float) if v is not None else np.full(12, np.nan)

    def _f(d, key, default=np.nan):
        v = d.get(key)
        return float(v) if v is not None and not np.isnan(float(v)) else default

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(20, 14))
        gs  = GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.35)

        # (0,0) T_mean: ERA5 raw vs site-corrected
        ax      = fig.add_subplot(gs[0, 0])
        t_era5  = _a12(sc, 'T_mean_monthly_C')
        t_corr  = _a12(th, 'T_mean_corrected_C')
        dT      = t_corr - t_era5
        ax.plot(_MX, t_era5, 'o--', color='#7f8c8d', lw=2,   ms=7, label='T_mean ERA5 (raw)')
        ax.plot(_MX, t_corr, 'o-',  color='#e67e22', lw=2.5, ms=8, label='T_mean site-corrected')
        ax.axhline(0, color='gray', ls=':', lw=1)
        ax2 = ax.twinx()
        ax2.bar(_MX, dT, color=['#e74c3c' if d > 0 else '#3498db' for d in dT],
                alpha=0.40, width=0.6, label='\u0394T (corr \u2212 ERA5)')
        ax2.axhline(0, color='gray', ls=':', lw=1)
        ax2.set_ylabel('\u0394T (\u00b0C)', fontsize=10)
        ax.set_xticks(_MX); ax.set_xticklabels(_MONTHS, fontsize=9)
        ax.set_ylabel('Temperature (\u00b0C)')
        ax.set_title('T_mean: ERA5 vs Site-Corrected\n(bar = site \u2212 ERA5 difference)')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='upper left')

        # (0,1) Corrected T_min / T_mean / T_max
        ax     = fig.add_subplot(gs[0, 1])
        t_min  = _a12(th, 'T_min_corrected_C')
        t_max  = _a12(th, 'T_max_corrected_C')
        ax.fill_between(_MX, t_min, t_max, alpha=0.18, color='#c0392b', label='T_min \u2013 T_max')
        ax.plot(_MX, t_min,  's-', color='#2980b9', lw=2, ms=7,
                label=f"T_min  (ann {_f(th,'T_min_annual_corrected_C'):.1f}\u00b0C)")
        ax.plot(_MX, t_corr, 'o-', color='#e67e22', lw=2, ms=7,
                label=f"T_mean (ann {_f(th,'T_mean_annual_corrected_C'):.1f}\u00b0C)")
        ax.plot(_MX, t_max,  '^-', color='#e74c3c', lw=2, ms=7,
                label=f"T_max  (ann {_f(th,'T_max_annual_corrected_C'):.1f}\u00b0C)")
        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.set_xticks(_MX); ax.set_xticklabels(_MONTHS, fontsize=9)
        ax.set_ylabel('Temperature (\u00b0C)')
        ax.set_title('Corrected Monthly Temperatures\n(T_mean / T_min / T_max)')
        ax.legend(fontsize=9)

        # (1,0) Correction components
        ax       = fig.add_subplot(gs[1, 0])
        dT_lapse = _f(th, 'dT_lapse_C', 0.0)
        dT_pool  = _f(th, 'dT_cold_pool_C', 0.0)
        dT_pisr  = _f(th, 'dT_pisr_C', 0.0)
        dT_tot   = dT_lapse + dT_pool + dT_pisr
        vals  = [dT_lapse, dT_pool, dT_pisr, dT_tot]
        lbls  = ['Lapse rate\n(altitude)', 'Cold-pool\n(valley)',
                 'PISR\n(insolation)', 'Total \u0394T']
        cols  = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
        cols[-1] = '#e67e22'
        bars = ax.bar(range(4), vals, color=cols, edgecolor='white', lw=1.5, width=0.55)
        ax.axhline(0, color='gray', ls='--', lw=1.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + (0.04 if v >= 0 else -0.10),
                        f'{v:+.2f}\u00b0C', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')
        ax.set_xticks(range(4)); ax.set_xticklabels(lbls, fontsize=10)
        ax.set_ylabel('Temperature correction \u0394T (\u00b0C)')
        ax.set_title('Thermal Correction Components\n(red=warming / blue=cooling)')
        ax.text(0.98, 0.97,
                f"\u03b3 lapse = {_f(th,'gamma_used'):.4f} \u00b0C/m\n"
                f"SVF = {_f(th,'svf'):.3f}\n"
                f"Concavity = {_f(th,'concavity_m'):.1f} m\n"
                f"Total unc. \u00b1{_f(th,'uncertainty_C'):.2f} \u00b0C",
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', alpha=0.85))

        # (1,1) Frost days: ERA5 vs corrected
        ax        = fig.add_subplot(gs[1, 1])
        fr_era5   = _a12(sc, 'frost_days_monthly')
        fr_corr   = _a12(th, 'frost_days_monthly')
        w = 0.38
        ax.bar(_MX - w/2, fr_era5, width=w, color='#7f8c8d', alpha=0.82,
               label=f"ERA5  (ann {_f(sc,'frost_days_annual'):.0f} d)")
        ax.bar(_MX + w/2, fr_corr, width=w, color='#2980b9', alpha=0.82,
               label=f"Site-corrected  (ann {_f(th,'frost_days_annual'):.0f} d)")
        ax.set_xticks(_MX); ax.set_xticklabels(_MONTHS, fontsize=9)
        ax.set_ylabel('Frost days / month')
        ax.set_title('Frost Days: ERA5 vs Site-Corrected\n'
                     '(accounts for cold-pool trapping in valleys)')
        ax.legend(fontsize=9)
        ax.text(0.98, 0.97,
                f"Cold-pool index = {_f(th,'cold_pool_index'):.3f}\n"
                f"\u0394z (site \u2212 ERA5) = {_f(th,'delta_elevation_m'):+.0f} m",
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', alpha=0.85))

        fig.suptitle(
            f"Thermal Correction  |  {lat:.4f}N {lon:.4f}E  |  Alt={sensor_alt:.0f} m",
            fontsize=14, fontweight='bold', y=1.01)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    print(f"  Saved: {path}")


def plot_twi(twi, elev, dx_grid, dy_grid, dist_grid, r86, path, lat, lon):
    """Four-panel TWI analysis: 2D TWI map, 2D slope map,
    TWI class histogram, radial TWI + slope profile."""
    with plt.rc_context(STYLE):
        fig  = plt.figure(figsize=(20, 14))
        gs   = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.38)
        th   = np.linspace(0, 2*np.pi, 360)
        clip = min(1.5 * r86, 800.0)

        twi_map   = twi.get('twi_map')
        slope_map = twi.get('slope_map_deg')
        fracs     = np.asarray(twi.get('twi_class_fractions', np.full(5, np.nan)), dtype=float)
        twi_w     = twi.get('twi_weighted',    np.nan)
        twi_mean  = twi.get('twi_mean_fp',     np.nan)
        twi_std   = twi.get('twi_std_fp',      np.nan)
        twi_lo    = twi.get('twi_min_fp',      np.nan)
        twi_hi    = twi.get('twi_max_fp',      np.nan)
        sl_mean   = twi.get('slope_mean_fp_deg', np.nan)

        def _circle(ax):
            ax.plot(r86*np.sin(th), r86*np.cos(th), 'k--', lw=2, label=f'r86={r86:.0f} m')
            ax.plot(0, 0, 'k^', ms=10, zorder=5, label='Sensor')

        # (0,0) TWI map
        ax = fig.add_subplot(gs[0, 0])
        if twi_map is not None:
            vlo, vhi = np.nanpercentile(twi_map, 2), np.nanpercentile(twi_map, 98)
            cm = ax.pcolormesh(dx_grid, dy_grid, twi_map,
                               cmap='RdYlGn', shading='auto', vmin=vlo, vmax=vhi)
            plt.colorbar(cm, ax=ax, fraction=0.03, pad=0.02).set_label(
                'TWI  ln[a / tan\u03b2]', fontsize=9)
        _circle(ax)
        ax.set_aspect('equal')
        ax.set_xlim(-clip, clip); ax.set_ylim(-clip, clip)
        ax.set_xlabel('Easting offset (m)'); ax.set_ylabel('Northing offset (m)')
        ax.set_title(f'Topographic Wetness Index\nCRNS-weighted = {float(twi_w):.2f}'
                     f'  |  FP mean = {float(twi_mean):.2f}')
        ax.legend(fontsize=9)

        # (0,1) Slope map
        ax = fig.add_subplot(gs[0, 1])
        if slope_map is not None:
            vhi_sl = np.nanpercentile(slope_map, 98)
            cm2 = ax.pcolormesh(dx_grid, dy_grid, slope_map,
                                cmap='YlOrRd', shading='auto', vmin=0, vmax=vhi_sl)
            plt.colorbar(cm2, ax=ax, fraction=0.03, pad=0.02).set_label('Slope (\u00b0)', fontsize=9)
        _circle(ax)
        ax.set_aspect('equal')
        ax.set_xlim(-clip, clip); ax.set_ylim(-clip, clip)
        ax.set_xlabel('Easting offset (m)'); ax.set_ylabel('Northing offset (m)')
        ax.set_title(f'Slope Map\nMean within footprint = {float(sl_mean):.1f}\u00b0')
        ax.legend(fontsize=9)

        # (1,0) TWI class histogram
        ax     = fig.add_subplot(gs[1, 0])
        cls_n  = ['Very dry\n(ridges)', 'Dry', 'Moderate', 'Moist', 'Very wet\n(hollows)']
        cls_c  = ['#e74c3c','#e67e22','#f1c40f','#2980b9','#1a5276']
        bars   = ax.bar(range(5), fracs * 100, color=cls_c, edgecolor='white', lw=1.5, width=0.65)
        for bar, v in zip(bars, fracs):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, v*100 + 0.5,
                        f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(5)); ax.set_xticklabels(cls_n, fontsize=9)
        ax.set_ylabel('Fraction of footprint area (%)')
        ax.set_title('TWI Class Distribution within Footprint\n'
                     '(5-percentile classes of full DEM extent)')
        ax.text(0.98, 0.97,
                f"CRNS-weighted TWI: {float(twi_w):.2f}\n"
                f"FP mean: {float(twi_mean):.2f} \u00b1 {float(twi_std):.2f}\n"
                f"Range: {float(twi_lo):.1f} \u2013 {float(twi_hi):.1f}",
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', alpha=0.85))

        # (1,1) Radial TWI + slope profile
        ax    = fig.add_subplot(gs[1, 1])
        r_bins = np.arange(0, r86 + 20, 20.0)
        r_mid  = 0.5 * (r_bins[:-1] + r_bins[1:])
        twi_r  = np.full(len(r_mid), np.nan)
        sl_r   = np.full(len(r_mid), np.nan)
        for k, (r0, r1) in enumerate(zip(r_bins[:-1], r_bins[1:])):
            ring = (dist_grid >= r0) & (dist_grid < r1)
            if not np.any(ring): continue
            if twi_map is not None:
                tv = twi_map[ring]; tv = tv[~np.isnan(tv)]
                if len(tv): twi_r[k] = float(np.mean(tv))
            if slope_map is not None:
                sv = slope_map[ring]; sv = sv[~np.isnan(sv)]
                if len(sv): sl_r[k] = float(np.mean(sv))
        ax.plot(r_mid, twi_r, 'o-', color='#27ae60', lw=2.5, ms=7, label='Mean TWI')
        ax.axhline(float(twi_mean), color='#27ae60', ls='--', lw=1.2, alpha=0.7,
                   label=f'FP mean TWI={float(twi_mean):.2f}')
        ax.set_xlabel('Distance from sensor (m)')
        ax.set_ylabel('Mean TWI', color='#27ae60')
        ax.tick_params(axis='y', labelcolor='#27ae60')
        ax2 = ax.twinx()
        ax2.plot(r_mid, sl_r, 's-', color='#e74c3c', lw=2.5, ms=7, label='Mean slope (\u00b0)')
        ax2.set_ylabel('Mean slope (\u00b0)', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        ax.axvline(r86, color='gray', ls=':', lw=1.5, label=f'r86={r86:.0f} m')
        ax.set_title('Radial Profile: TWI and Slope\n(mean per 20 m ring from sensor)')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8)

        fig.suptitle(f"Topographic Wetness Index  |  {lat:.4f}N {lon:.4f}E",
                     fontsize=14, fontweight='bold', y=1.01)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    print(f"  Saved: {path}")


def plot_kappa_budget(results, path, lat, lon):
    """Two-panel kappa budget + expected flux chain:
    Left: kappa decomposition (pieno/sopra/vuoto + muon + total).
    Right: expected CRNS count rates (sea-level vs site, N0)."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(20, 8))
        gs  = GridSpec(1, 2, figure=fig, wspace=0.40)

        def _f(key, default=np.nan):
            v = results.get(key, default)
            return float(v) if v is not None else default

        kappa_pieno = _f('kappa_pieno')
        kappa_sopra = _f('kappa_sopra')
        kappa_vuoto = _f('kappa_vuoto')
        kappa_topo  = _f('kappa_topo')
        kappa_muon  = _f('kappa_muon')
        kappa_tot   = _f('kappa_total')
        V0          = _f('V0')
        Veff        = _f('Veff')
        ki          = results.get('kappa_info', {})

        # LEFT: kappa decomposition bar chart
        ax   = fig.add_subplot(gs[0, 0])
        vals = [kappa_pieno, kappa_sopra, kappa_vuoto, kappa_topo, kappa_muon, kappa_tot]
        lbls = ['\u03ba_pieno\n(full vol.)', '\u03ba_sopra\n(above ref.)',
                '\u03ba_vuoto\n(below slab)', '\u03ba_topo\n(total)',
                '\u03ba_muon\n(sky FOV)', '\u03ba_total\n(combined)']
        cols = ['#e74c3c' if v > 1 else '#3498db' for v in vals]
        cols[3] = '#e67e22'   # kappa_topo
        cols[5] = '#8e44ad'   # kappa_total
        bars = ax.bar(range(6), vals, color=cols, edgecolor='white', lw=1.5, width=0.60)
        ax.axhline(1.0, color='gray', ls='--', lw=1.5, label='Reference = 1.0')
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + 0.01 if v >= 0 else v - 0.02,
                        f'{v:.4f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        ax.set_xticks(range(6)); ax.set_xticklabels(lbls, fontsize=10)
        ax.set_ylabel('Correction factor \u03ba')
        ax.set_title('\u03ba Decomposition: Neutron + Muon Budgets\n'
                     '(red > 1 overestimate / blue < 1 underestimate)')
        ax.set_ylim(0, max([v for v in vals if not np.isnan(v)] + [1.0]) * 1.30)
        ax.legend(fontsize=9)
        veff_pct = Veff / V0 * 100 if not (np.isnan(V0) or np.isnan(Veff) or V0 == 0) else np.nan
        ann = (f"V0 = {V0:.2f} m\u00b3  |  Veff = {Veff:.2f} m\u00b3  ({veff_pct:.1f}%)\n"
               f"Rays: {ki.get('n_rays_total','?')} total / {ki.get('n_rays_hit','?')} hit DEM\n"
               f"Pieno: {ki.get('n_pieno','?')} px  |  Sopra: {ki.get('n_sopra','?')} px  "
               f"|  Vuoto: {ki.get('n_vuoto','?')} px")
        ax.text(0.01, 0.98, ann, transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', alpha=0.85))

        # RIGHT: expected flux chain
        ax2  = fig.add_subplot(gs[0, 1])
        sf   = results.get('site_fluxes', {})
        def _sf(k, d=np.nan): return float(sf.get(k, d)) if sf.get(k) is not None else d
        N_muon_sl = _sf('N_muon_sl', 4000.0)
        N_neut_sl = _sf('N_neut_sl',  900.0)
        N_muon    = _sf('N_muon_site')
        N_neut    = _sf('N_neut_site')
        N0        = _sf('N0_theoretical')
        x_lbls = ['N_muon\nsea-level', 'N_muon\nsite', 'N_neut\nsea-level',
                  'N_neut\nsite', 'N0\n(dry soil)']
        x_vals = [N_muon_sl, N_muon, N_neut_sl, N_neut, N0]
        x_cols = ['#7f8c8d','#2980b9','#95a5a6','#27ae60','#8e44ad']
        brs = ax2.bar(range(5), x_vals, color=x_cols, edgecolor='white', lw=1.5, width=0.6)
        for bar, v in zip(brs, x_vals):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width()/2, v * 1.02,
                         f'{v:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.set_xticks(range(5)); ax2.set_xticklabels(x_lbls, fontsize=10)
        ax2.set_ylabel('Count rate (cph)')
        ax2.set_title('Expected CRNS Count Rates\n'
                      '(sea-level \u2192 altitude + \u03ba corrections applied)')
        ax2.text(0.98, 0.97,
                 f"P = {_f('pressure'):.1f} hPa\n"
                 f"Rc = {_sf('Rc_gv'):.2f} GV  |  f(Rc) = {_sf('f_Rc'):.4f}\n"
                 f"Alt factor muon = {_sf('alt_factor_muon'):.4f}\n"
                 f"Alt factor neut = {_sf('alt_factor_neut'):.4f}\n"
                 f"r86 = {_f('r86'):.0f} m  |  z86 = {_f('z86'):.1f} cm",
                 transform=ax2.transAxes, fontsize=9, ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', alpha=0.85))

        fig.suptitle(f"\u03ba Budget & Expected Fluxes  |  {lat:.4f}N {lon:.4f}E",
                     fontsize=14, fontweight='bold')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    print(f"  Saved: {path}")


def plot_water(water, dx_grid, dy_grid, dist_grid, r86, path, lat, lon):
    """Three-panel JRC surface water analysis:
    Left:   2D occurrence map on DEM grid.
    Centre: radial occurrence profile with W(r) weighting.
    Right:  eta + N_correction summary."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    eta         = float(water.get('eta', 0.0))
    ncf         = float(water.get('N_correction_factor', 1.0))
    occ_map     = water.get('occ_map_dem')          # 2D array [0-100] float
    wm_fp       = water.get('water_mask_fp')        # bool 2D
    occ_w       = float(water.get('occ_weighted_mean', 0.0))
    pix_w       = int(water.get('pixels_water_fp', 0))
    pix_t       = int(water.get('pixels_total_fp', 1))
    area_m2     = float(water.get('water_area_m2', 0.0))
    tile        = water.get('tile_name', 'N/A')
    f_w         = float(water.get('f_water_used', 0.03))
    from_cache  = water.get('from_cache', False)

    th   = np.linspace(0, 2*np.pi, 360)
    clip = min(1.5 * r86, 600.0)

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(22, 8))
        gs  = plt.GridSpec(1, 3, figure=fig, wspace=0.38)

        # ── LEFT: occurrence map ──────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        if occ_map is not None:
            occ_frac = occ_map / 100.0
            cm = ax.pcolormesh(dx_grid, dy_grid, occ_frac,
                               cmap='Blues', shading='auto', vmin=0, vmax=1)
            plt.colorbar(cm, ax=ax, fraction=0.03, pad=0.02).set_label(
                'JRC Occurrence fraction (0=never, 1=always)', fontsize=9)
            # water pixels highlighted
            if wm_fp is not None:
                occ_hl = np.where(wm_fp, occ_frac, np.nan)
                ax.contour(dx_grid, dy_grid, (~np.isnan(occ_hl)).astype(float),
                           levels=[0.5], colors=['#e74c3c'], linewidths=1.2,
                           linestyles='--')
        ax.plot(r86 * np.sin(th), r86 * np.cos(th), 'k--', lw=2,
                label=f'r86={r86:.0f} m')
        ax.plot(0, 0, 'k^', ms=10, zorder=5, label='Sensor')
        ax.set_aspect('equal')
        ax.set_xlim(-clip, clip); ax.set_ylim(-clip, clip)
        ax.set_xlabel('Easting offset (m)'); ax.set_ylabel('Northing offset (m)')
        ax.set_title(f'JRC Surface Water Occurrence\n'
                     f'(tile: {tile}  |  {"from cache" if from_cache else "freshly downloaded"})\n'
                     f'Red dashes = water pixels inside footprint')
        ax.legend(fontsize=9)

        # ── CENTRE: radial occurrence profile ────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        r_bins = np.arange(0, r86 + 15, 15.0)
        r_mid  = 0.5 * (r_bins[:-1] + r_bins[1:])
        occ_r  = np.zeros(len(r_mid))
        w_r    = np.zeros(len(r_mid))

        if occ_map is not None:
            occ_frac = occ_map / 100.0
            lam = r86 / 3.0
            for k, (r0, r1) in enumerate(zip(r_bins[:-1], r_bins[1:])):
                ring = (dist_grid >= r0) & (dist_grid < r1)
                if not np.any(ring): continue
                r_px = dist_grid[ring]
                W    = np.where(r_px < 1e-3, 0.0, np.exp(-r_px / lam))
                of   = occ_frac[ring]
                Ws   = W.sum()
                occ_r[k] = float(np.sum(W * of) / Ws) if Ws > 0 else 0.0
                w_r[k]   = float(Ws)

        bar_cols = [plt.get_cmap('Blues')(v * 0.9 + 0.1) for v in occ_r]
        ax.bar(r_mid, occ_r * 100, width=13, color=bar_cols,
               edgecolor='white', align='center', label='W(r)-weighted occurrence (%)')
        wn = w_r / max(w_r.max(), 1e-9)
        ax.plot(r_mid, wn * occ_r.max() * 100 if occ_r.max() > 0 else wn,
                'k--', lw=1.8, label='W(r) shape (normalised)')
        ax.axhline(occ_w * 100, color='#e74c3c', ls='-.', lw=2,
                   label=f'Weighted mean = {occ_w*100:.2f}%')
        ax.set_xlabel('Distance from sensor (m)')
        ax.set_ylabel('Mean occurrence (%)')
        ax.set_title('Radial Water Occurrence Profile\n'
                     '(W(r)-weighted, 15 m rings)\n'
                     'Blue bars = mean water frequency per ring')
        ax.set_xlim(0, r86); ax.legend(fontsize=9)

        # ── RIGHT: eta + correction summary ──────────────────────────────
        ax = fig.add_subplot(gs[0, 2])
        bar_lbls = ['\u03b7\n(count reduction)', 'N correction\nfactor \u22121 (%)']
        bar_vals = [eta, (ncf - 1.0) * 100]
        bar_cols2 = ['#e74c3c' if eta > 0.01 else '#f39c12' if eta > 0.001 else '#27ae60',
                     '#2980b9']
        bars = ax.bar(range(2), bar_vals, color=bar_cols2, edgecolor='white',
                      lw=1.5, width=0.55)
        ax.axhline(0, color='gray', ls='--', lw=1)
        for bar, v in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + max(bar_vals) * 0.03,
                    f'{v:.4f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(range(2)); ax.set_xticklabels(bar_lbls, fontsize=10)
        ax.set_ylabel('\u03b7  or  (N_corr / N_obs \u2212 1)  [%]')

        if eta < 0.001:
            interp = 'Negligible  (< 0.1%)'
            icol   = '#27ae60'
        elif eta < 0.01:
            interp = 'Minor  (0.1 \u2013 1%)'
            icol   = '#f39c12'
        elif eta < 0.05:
            interp = 'Significant  (1 \u2013 5%)'
            icol   = '#e67e22'
        else:
            interp = 'LARGE  (> 5%) \u2014 correction required'
            icol   = '#e74c3c'

        ax.set_title(f'Water Correction Budget\n'
                     f'N_corrected = N_obs \u00d7 {ncf:.4f}\n'
                     f'Impact: {interp}', color=icol if eta > 0.001 else 'black')

        info_txt = (f"f_water = {f_w:.3f}  (Zreda 2012)\n"
                    f"Occ weighted mean: {occ_w*100:.2f}%\n"
                    f"Water pixels: {pix_w} / {pix_t}  "
                    f"({pix_w/max(pix_t,1)*100:.1f}%)\n"
                    f"Water area \u2248 {area_m2/1e4:.2f} ha\n"
                    f"N_corrected = N_obs / (1\u2212\u03b7)")
        ax.text(0.98, 0.97, info_txt, transform=ax.transAxes,
                fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', alpha=0.85))

        fig.suptitle(
            f"JRC Surface Water Correction (\u03b7)  |  {lat:.4f}N {lon:.4f}E  |  "
            f"JRC Global Surface Water v1.4 (Pekel 2016)",
            fontsize=14, fontweight='bold')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    print(f"  Saved: {path}")
