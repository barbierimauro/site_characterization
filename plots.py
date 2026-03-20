
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
              results, path):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(20, 14))
        gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

        # DEM + footprint circles
        ax1  = fig.add_subplot(gs[0, :2])
        dist = np.sqrt(dx_grid**2 + dy_grid**2)
        em   = np.where(dist <= DEM_RADIUS_M,
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
        ax1.set_title(f"DEM & Neutron Footprint  ({LAT:.4f}N, {LON:.4f}E)  "
                      f"alt={results['sensor_alt']:.0f} m")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.set_xlim(-DEM_RADIUS_M*0.6, DEM_RADIUS_M*0.6)
        ax1.set_ylim(-DEM_RADIUS_M*0.6, DEM_RADIUS_M*0.6)

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
        ax3.set_xlim(-DEM_RADIUS_M*0.5, DEM_RADIUS_M*0.5)

        # N-S profile
        ax4 = fig.add_subplot(gs[1, 1])
        #mc = np.argmin(np.abs(dx_grid[:, dx_grid.shape[1]//2]))
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
        ax4.set_xlim(-DEM_RADIUS_M*0.5, DEM_RADIUS_M*0.5)

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
            f"{LAT:.4f}N {LON:.4f}E  |  "
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
            W   = weight_radial(r_r, r86)
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
                    az_neutron, overlap_az, r86, z86_cm, kappa_topo, path):
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
            f"Detailed FOV  |  {LAT:.4f}N {LON:.4f}E  |  "
            f"Alt={_results_alt:.0f}m  r86={r86:.0f}m",
            fontsize=14, fontweight="bold")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {path}")

_results_alt = 0.0  # filled in main before calling plot_fov_detail
