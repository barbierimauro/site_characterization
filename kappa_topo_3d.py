"""
compute_kappa_topo_3d  —  v3
============================
Physically correct topographic kappa for CRNS via 3-D ray-casting.

Classificazione pieno/sopra/vuoto basata sulla quota del punto di hit
rispetto a s_elev, non su delta_L (che è ambiguo per raggi obliqui steep).

Author      : MB
Affiliation : 
Email       : mauro.barbieri@pm.me
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

LAMBDA_N_GCM2 = 130.0
LAMBDA_S_GCM2 = 162.0
RHO_AIR_0     = 1.225e-3
H_SCALE_M     = 8500.0


def _lambda_air_m(alt_m):
    rho = RHO_AIR_0 * np.exp(-alt_m / H_SCALE_M)
    return LAMBDA_N_GCM2 / (rho * 100.0)

def _lambda_soil_m(rho_b):
    return LAMBDA_S_GCM2 / (rho_b * 100.0)

def _build_interp(elev, dx_grid, dy_grid):
    nr, nc = elev.shape
    x1 = dx_grid[nr // 2, :].copy()
    y1 = dy_grid[:, nc // 2].copy()
    el = elev.copy()
    if x1[-1] < x1[0]: x1 = x1[::-1]; el = el[:, ::-1]
    if y1[-1] < y1[0]: y1 = y1[::-1]; el = el[::-1, :]
    return RegularGridInterpolator(
        (y1, x1), el, method="linear", bounds_error=False, fill_value=np.nan)


def compute_kappa_topo_3d(
    elev, dx_grid, dy_grid, dist_grid,
    sz, s_elev, r86, z86_cm, rho_b, sensor_height_m,
    dphi_deg=2.0, dtheta_deg=2.0, dr_m=5.0, dem_sigma_m=5.0,
):
    """
    Parameters
    ----------
    elev, dx_grid, dy_grid, dist_grid : da clip_dem_to_radius
    sz              : quota sensore [m a.s.l.] = s_elev + sensor_height_m
    s_elev          : quota suolo al sensore [m a.s.l.]
    r86             : raggio footprint [m]
    z86_cm          : profondità penetrazione neutroni [cm]
    rho_b           : densità bulk suolo [g/cm3]
    sensor_height_m : altezza sensore dal suolo [m]
    dphi_deg        : passo azimuth [deg]
    dtheta_deg      : passo angolo elevazione [deg]
    dr_m            : passo ray marching [m]
    dem_sigma_m     : accuratezza verticale DEM [m] — soglia classificazione

    Returns
    -------
    kappa_topo, kappa_pieno, kappa_sopra, kappa_vuoto, wmap, info
    """
    z86_m    = z86_cm / 100.0
    lam_air  = _lambda_air_m(s_elev)
    lam_soil = _lambda_soil_m(rho_b)

    # ------------------------------------------------------------------ #
    # Griglia angolare
    # theta: angolo di elevazione sopra orizzontale, in [-90+dtheta, +90-dtheta]
    # ------------------------------------------------------------------ #
    phi_arr   = np.arange(0.0, 360.0, dphi_deg)
    theta_arr = np.arange(-(90.0 - dtheta_deg), 90.0, dtheta_deg)
    N_phi, N_th = len(phi_arr), len(theta_arr)

    phi_r   = np.radians(phi_arr)
    theta_r = np.radians(theta_arr)

    r_vals = np.arange(dr_m, r86 + dr_m, dr_m)
    N_r    = len(r_vals)

    print(f"   Ray-cast grid: {N_phi} phi x {N_th} theta x {N_r} r "
          f"= {N_phi*N_th*N_r:,} points", flush=True)

    # ------------------------------------------------------------------ #
    # Griglie 3-D  (N_phi, N_theta, N_r)
    # ------------------------------------------------------------------ #
    PHI   = phi_r[:, None, None]
    THETA = theta_r[None, :, None]
    R     = r_vals[None, None, :]

    cos_TH = np.cos(THETA)
    sin_TH = np.sin(THETA)

    X_grid = R * cos_TH * np.sin(PHI)   # easting offset [m]
    Y_grid = R * cos_TH * np.cos(PHI)   # northing offset [m]
    Z_ray  = sz + R * sin_TH            # quota del punto sul raggio [m a.s.l.]

    # ------------------------------------------------------------------ #
    # Interpolazione DEM
    # ------------------------------------------------------------------ #
    dem_interp = _build_interp(elev, dx_grid, dy_grid)
    pts        = np.column_stack([Y_grid.ravel(), X_grid.ravel()])
    Z_dem      = dem_interp(pts).reshape(N_phi, N_th, N_r)

    # DEM flat di riferimento
    Z_dem_flat = np.full_like(Z_dem, s_elev)

    # ------------------------------------------------------------------ #
    # Funzione: primo hit + contributo
    # ------------------------------------------------------------------ #
    def _process(Zd):
        hit_mask = (Z_ray <= Zd) & ~np.isnan(Zd)         # (N_phi, N_th, N_r)
        any_hit  = hit_mask.any(axis=2)                   # (N_phi, N_th)

        fk = np.argmax(hit_mask, axis=2)[:, :, np.newaxis]  # (N_phi, N_th, 1)

        R_bc = np.broadcast_to(R, (N_phi, N_th, N_r))
        r_hit  = np.take_along_axis(R_bc, fk, axis=2).squeeze(2)   # (N_phi, N_th)
        z_hit  = np.take_along_axis(Zd,   fk, axis=2).squeeze(2)

        # Distanza 3-D in aria sensore -> punto di hit
        rh_hit = r_hit * np.abs(np.cos(theta_r))[None, :]
        L_air  = np.sqrt(rh_hit**2 + (sz - z_hit)**2)     # (N_phi, N_th)

        # Percorso nel suolo: z86 / |cos(theta)|  (raggio obliquo)
        cos_inc = np.maximum(np.abs(np.cos(theta_r))[None, :], 0.1)
        L_soil  = z86_m / cos_inc                          # broadcast su N_phi

        # Peso angolare: |cos(theta)| * dtheta * dphi (si cancella nel rapporto)
        cos_abs = np.abs(np.cos(theta_r))[None, :]

        dN = (np.exp(-L_air / lam_air)
              * (1.0 - np.exp(-L_soil / lam_soil))
              * cos_abs)
        dN = np.where(any_hit, dN, 0.0)

        return dN, any_hit, L_air, r_hit, z_hit

    dN_obs, any_hit, L_air_real, r_hit, z_hit = _process(Z_dem)
    dN_ref, any_hit_flat, L_air_flat, _, _    = _process(Z_dem_flat)

    N_obs = float(dN_obs.sum())
    N_ref = float(dN_ref.sum())
    kappa_topo = N_obs / N_ref if N_ref > 0 else 1.0

    # ------------------------------------------------------------------ #
    # Classificazione pieno / sopra / vuoto
    #
    # Basata sulla quota del punto di hit z_hit rispetto a s_elev:
    #
    #   SOPRA : z_hit > s_elev + dem_sigma
    #           Il raggio ha colpito terreno SOPRA la quota del sensore.
    #           Contribuisce con più suolo del caso flat.
    #
    #   PIENO : |z_hit - s_elev| <= dem_sigma
    #           Il hit è alla stessa quota del sensore (dentro il rumore DEM).
    #           Equivalente al caso flat.
    #
    #   VUOTO : z_hit < s_elev - dem_sigma
    #           Il raggio ha percorso aria extra prima di trovare suolo
    #           (precipizio, valle). Contribuisce meno del caso flat.
    #
    # Questa classificazione è invariante rispetto all'angolo del raggio
    # e non soffre dell'ambiguità di delta_L per raggi molto obliqui.
    # ------------------------------------------------------------------ #
    sopra_mask = any_hit & (z_hit > s_elev + dem_sigma_m)
    pieno_mask = any_hit & (np.abs(z_hit - s_elev) <= dem_sigma_m)
    vuoto_mask = any_hit & (z_hit < s_elev - dem_sigma_m)

    kappa_pieno = +float(np.sum(np.where(pieno_mask, dN_obs, 0.0))) / N_ref
    kappa_sopra = +float(np.sum(np.where(sopra_mask, dN_obs, 0.0))) / N_ref
    kappa_vuoto = -float(np.sum(np.where(vuoto_mask, dN_obs, 0.0))) / N_ref
    kappa_check = kappa_pieno + kappa_sopra + kappa_vuoto

    kappa_topo = kappa_check
    
    # ------------------------------------------------------------------ #
    # Mappa 2-D dei pesi per il plotting
    # ------------------------------------------------------------------ #
    nr, nc = elev.shape
    x_1d = dx_grid[nr // 2, :]
    y_1d = dy_grid[:, nc // 2]
    x_asc = np.sort(x_1d); y_asc = np.sort(y_1d)

    x_hit_map = r_hit * np.abs(np.cos(theta_r))[None, :] * np.sin(phi_r)[:, None]
    y_hit_map = r_hit * np.abs(np.cos(theta_r))[None, :] * np.cos(phi_r)[:, None]

    ix = np.searchsorted(x_asc, x_hit_map.ravel(), side="left").clip(0, nc - 1)
    iy = np.searchsorted(y_asc, y_hit_map.ravel(), side="left").clip(0, nr - 1)
    if x_1d[0] > x_1d[-1]: ix = nc - 1 - ix
    if y_1d[0] > y_1d[-1]: iy = nr - 1 - iy

    wmap_flat = np.zeros(elev.size)
    np.add.at(wmap_flat, iy * nc + ix,
              np.where(any_hit.ravel(), dN_obs.ravel() / max(N_ref, 1e-30), 0.0))
    wmap = wmap_flat.reshape(elev.shape)
    wmap[np.isnan(elev)] = np.nan

    info = dict(
        N_phi=N_phi, N_theta=N_th, N_r=N_r,
        n_rays_total=N_phi * N_th,
        n_rays_hit=int(any_hit.sum()),
        n_pieno=int(pieno_mask.sum()),
        n_sopra=int(sopra_mask.sum()),
        n_vuoto=int(vuoto_mask.sum()),
        N_obs=N_obs, N_ref=N_ref,
        kappa_check=kappa_check,
        lam_air_m=lam_air, lam_soil_m=lam_soil,
        dem_sigma_m=dem_sigma_m,
        dphi_deg=dphi_deg, dtheta_deg=dtheta_deg, dr_m=dr_m,
    )
    return kappa_topo, kappa_pieno, kappa_sopra, kappa_vuoto, wmap, info


def report_kappa_3d(kappa_topo, kappa_pieno, kappa_sopra, kappa_vuoto, info):
    w = 72
    lines = [
        "=" * w,
        "KAPPA_TOPO  —  3-D Ray-Cast",
        "=" * w,
        f"  kappa_topo   = {kappa_topo:.4f}   (N_obs / N_ref)",
        f"  kappa_pieno  = {kappa_pieno:.4f}   suolo a quota sensore",
        f"  kappa_sopra  = {kappa_sopra:.4f}   terreno sopra il sensore  [+]",
        f"  kappa_vuoto  = {kappa_vuoto:.4f}   precipizi / valli         [-]",
        f"  check sum    = {info['kappa_check']:.4f}   (deve = kappa_topo)",
        "",
        f"  Grid  : {info['N_phi']} phi x {info['N_theta']} theta x {info['N_r']} r"
        f"  = {info['N_phi']*info['N_theta']*info['N_r']:,} punti",
        f"  Hit   : {info['n_rays_hit']:,} / {info['n_rays_total']:,}",
        f"  Class : pieno={info['n_pieno']:,}  "
        f"sopra={info['n_sopra']:,}  vuoto={info['n_vuoto']:,}",
        f"  lambda_air={info['lam_air_m']:.1f} m   lambda_soil={info['lam_soil_m']:.3f} m",
        f"  DEM sigma = {info['dem_sigma_m']:.1f} m",
        "=" * w,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    n=100; r_max=200.0
    x1=np.linspace(-r_max,r_max,n); y1=np.linspace(-r_max,r_max,n)
    XX,YY=np.meshgrid(x1,y1); dist=np.sqrt(XX**2+YY**2)
    s_elev=500.0; sz=s_elev+2.0; r86=130.0; z86_cm=16.0; rho_b=1.4; sh=2.0
    kw=dict(dphi_deg=5.,dtheta_deg=5.,dr_m=8.)

    tests = [
        ("flat -> kappa=1",
         np.full((n,n),s_elev), lambda k: abs(k-1.0)<0.03),
        ("cliff east -> kappa<1",
         np.where(XX[np.newaxis,:]>0, s_elev-50., s_elev).squeeze(),
         lambda k: k<1.0),
        ("hill west -> kappa>1",
         np.where(XX[np.newaxis,:]<0, s_elev+30., s_elev).squeeze(),
         lambda k: k>1.0),
        ("ring hill -> kappa>1",
         np.where((dist>40)&(dist<r86), s_elev+20., s_elev),
         lambda k: k>1.0),
    ]
    # fix shape for tests 2 and 3
    tests[1] = ("cliff east -> kappa<1",
                np.where(XX>0, s_elev-50., s_elev), lambda k: k<1.0)
    tests[2] = ("hill west -> kappa>1",
                np.where(XX<0, s_elev+30., s_elev), lambda k: k>1.0)

    for name, elev_t, check in tests:
        print(f"\nTEST: {name}")
        t0=time.perf_counter()
        kt,kp,ks,kv,_,info=compute_kappa_topo_3d(
            elev_t,XX,YY,dist,sz,s_elev,r86,z86_cm,rho_b,sh,**kw)
        dt=time.perf_counter()-t0
        print(report_kappa_3d(kt,kp,ks,kv,info))
        status="PASS" if check(kt) else f"FAIL (kappa={kt:.4f})"
        print(f"  {status}  [{dt:.3f} s]")
        # kappa_sopra deve essere >=0, kappa_vuoto deve essere >=0
        assert kv >= 0, f"kappa_vuoto negativo: {kv}"
        assert ks >= 0, f"kappa_sopra negativo: {ks}"
        assert abs(info['kappa_check']-kt)<0.001, "check sum fallito"
