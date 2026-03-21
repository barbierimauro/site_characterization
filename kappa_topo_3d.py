"""
compute_kappa_topo_3d  —  v4
============================
Physically correct topographic kappa for CRNS via 3-D ray-casting.

Fix rispetto a v3
-----------------
1. Peso angolare cos²θ invece di cosθ
   Il flusso di neutroni epitermali emessi dal suolo scala come cosθ
   (distribuzione lambertiana). L'elemento di angolo solido dΩ = cosθ dθ dφ
   aggiunge un secondo cosθ. Il peso totale è quindi cos²θ.
   In v3 si usava cosθ, che sottostima i raggi quasi-verticali.

2. L_soil calcolato sulla normale locale al DEM nel punto di hit
   In v3: L_soil = z86 / |cosθ|  (assume superficie sempre orizzontale).
   In v4: si calcola il gradiente del DEM nel punto di hit, si costruisce
   la normale locale n̂ = normalize(-dz/dx, -dz/dy, 1), e si usa:
       cos_inc = |ray_dir · n̂|
       L_soil  = z86 / cos_inc
   Corretto per pendii, dirupi, superfici quasi-verticali.
   Clamp: cos_inc >= COS_INC_MIN = 0.05 (raggi radenti).

3. λ_air calcolata alla quota media del percorso (sz + z_hit)/2
   In v3 era fissa a s_elev. Correzione piccola (<1%) ma fisicamente coerente.

4. L_air = r_hit (distanza 3-D al primo hit, già corretta in v3).

5. kappa_topo = N_obs / N_ref è il valore fisico autoritativo.
   La decomposizione pieno/sopra/vuoto è DIAGNOSTICA (non somma esattamente
   a kappa_topo — vedi nota algebra nel docstring). La deviazione attesa
   è < 5% in casi realistici.

6. kappa_vuoto definito ≥ 0 come deficit rispetto al flat.
   kappa_topo ≈ kappa_pieno + kappa_sopra - kappa_vuoto.

Nota algebra: perché check_sum ≠ kappa_topo in generale
--------------------------------------------------------
I raggi con θ > 0 (verso l'alto) colpiscono un eventuale rilievo nel DEM
reale (sopra_mask) ma non colpiscono mai il DEM flat → dN_ref = 0 per
quei raggi. Essi contribuiscono a N_obs ma non a N_ref.

I raggi con θ < 0 colpiscono il DEM flat → contribuiscono a N_ref.
Nel DEM reale possono trovare suolo più lontano o assente (vuoto_mask).

Algebra:
  check_sum = pieno + sopra - vuoto
            = [N_obs - Σ_vuoto(dN_ref)] / N_ref
            = kappa_topo - Σ_vuoto(dN_ref)/N_ref

La differenza kappa_topo - check_sum = Σ_vuoto(dN_ref)/N_ref è il peso
del flat corrispondente ai raggi classificati "vuoto nel reale". È piccola
quando i vuoti sono profondi (suolo lontano → dN_ref piccola per
attenuazione) e maggiore per vuoti poco profondi vicini al sensore.
Per DEM flat: Σ_vuoto = 0 → check_sum = kappa_topo = 1.0. ✓

Author      : MB
Affiliation : 
Email       : mauro.barbieri@pm.me
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ── Costanti fisiche ─────────────────────────────────────────────────────────
LAMBDA_N_GCM2 = 130.0       # MFP neutroni epitermali in aria   [g/cm²]
LAMBDA_S_GCM2 = 162.0       # MFP neutroni epitermali nel suolo [g/cm²]
RHO_AIR_0     = 1.225e-3    # densità aria al livello del mare  [g/cm³]
H_SCALE_M     = 8500.0      # altezza di scala atmosferica      [m]
COS_INC_MIN   = 0.05        # clamp su cos_incidenza (raggio radente)


# ── Funzioni di supporto ──────────────────────────────────────────────────────

def _lambda_air_m(alt_m):
    """MFP neutroni in aria [m] alla quota alt_m [m a.s.l.]."""
    rho = RHO_AIR_0 * np.exp(-alt_m / H_SCALE_M)
    return LAMBDA_N_GCM2 / (rho * 100.0)


def _lambda_soil_m(rho_b):
    """MFP neutroni nel suolo [m] per densità bulk rho_b [g/cm³]."""
    return LAMBDA_S_GCM2 / (rho_b * 100.0)


def _build_interp(elev, dx_grid, dy_grid):
    """RegularGridInterpolator sull'elevazione con assi crescenti garantiti."""
    nr, nc = elev.shape
    x1 = dx_grid[nr // 2, :].copy()
    y1 = dy_grid[:, nc // 2].copy()
    el = elev.copy()
    if x1[-1] < x1[0]: x1 = x1[::-1]; el = el[:, ::-1]
    if y1[-1] < y1[0]: y1 = y1[::-1]; el = el[::-1, :]
    return RegularGridInterpolator(
        (y1, x1), el, method="linear", bounds_error=False, fill_value=np.nan)


def _build_gradient_interps(elev, dx_grid, dy_grid):
    """
    Restituisce due RegularGridInterpolator per (dz/dx, dz/dy)
    calcolati con differenze finite centrali sul DEM.
    """
    nr, nc = elev.shape
    x1 = dx_grid[nr // 2, :].copy()
    y1 = dy_grid[:, nc // 2].copy()
    el = elev.copy()
    if x1[-1] < x1[0]: x1 = x1[::-1]; el = el[:, ::-1]
    if y1[-1] < y1[0]: y1 = y1[::-1]; el = el[::-1, :]
    dzdx = np.gradient(el, x1, axis=1)
    dzdy = np.gradient(el, y1, axis=0)
    kw = dict(method="linear", bounds_error=False, fill_value=0.0)
    return (RegularGridInterpolator((y1, x1), dzdx, **kw),
            RegularGridInterpolator((y1, x1), dzdy, **kw))


# ── Funzione principale ───────────────────────────────────────────────────────

def compute_kappa_topo_3d(
    elev, dx_grid, dy_grid, dist_grid,
    sz, s_elev, r86, z86_cm, rho_b, sensor_height_m,
    dphi_deg=2.0, dtheta_deg=2.0, dr_m=5.0, dem_sigma_m=5.0,
):
    """
    Calcola kappa_topo via ray-casting 3-D fisicamente corretto.

    Parameters
    ----------
    elev            ndarray (nr, nc)  elevazione DEM [m a.s.l.]
    dx_grid         ndarray (nr, nc)  coordinate x (easting offset) [m]
    dy_grid         ndarray (nr, nc)  coordinate y (northing offset) [m]
    dist_grid       ndarray (nr, nc)  distanza dal sensore [m]
    sz              float  quota sensore [m a.s.l.] = s_elev + sensor_height_m
    s_elev          float  quota suolo sotto il sensore [m a.s.l.]
    r86             float  raggio footprint 86% [m]
    z86_cm          float  profondità penetrazione neutroni [cm]
    rho_b           float  densità bulk suolo [g/cm³]
    sensor_height_m float  altezza sensore dal suolo [m]
    dphi_deg        float  passo azimuth [deg]           (default 2)
    dtheta_deg      float  passo angolo elevazione [deg] (default 2)
    dr_m            float  passo ray-marching [m]        (default 5)
    dem_sigma_m     float  accuratezza verticale DEM [m] — dead-band (default 5)

    Returns
    -------
    kappa_topo  float   N_obs / N_ref  — valore fisico da usare per correzione.
                        > 1 se c'è terreno extra sopra il sensore (rilievi).
                        < 1 se manca terreno (valli, dirupi, orizzonte aperto).
    kappa_pieno float   Contributo diagnostico da terreno a quota ≈ s_elev [≥0]
    kappa_sopra float   Contributo diagnostico da terreno sopra s_elev      [≥0]
    kappa_vuoto float   Deficit diagnostico da terreno sotto/assente         [≥0]
                        Contribuisce sottraendo: kappa ≈ pieno + sopra - vuoto
    wmap        ndarray (nr, nc)  mappa 2-D pesi normalizzati su N_ref
    info        dict    diagnostica numerica
    """
    z86_m    = z86_cm / 100.0
    lam_soil = _lambda_soil_m(rho_b)

    # ── Griglia angolare ──────────────────────────────────────────────────────
    phi_arr   = np.arange(0.0, 360.0, dphi_deg)
    theta_arr = np.arange(-(90.0 - dtheta_deg), 90.0, dtheta_deg)
    N_phi, N_th = len(phi_arr), len(theta_arr)

    phi_r   = np.radians(phi_arr)
    theta_r = np.radians(theta_arr)

    r_vals = np.arange(dr_m, r86 + dr_m, dr_m)
    N_r    = len(r_vals)

    print(f"   Ray-cast grid: {N_phi} phi × {N_th} theta × {N_r} r "
          f"= {N_phi * N_th * N_r:,} punti", flush=True)

    # ── Griglie 3-D  (N_phi, N_theta, N_r) ───────────────────────────────────
    PHI   = phi_r[:, None, None]
    THETA = theta_r[None, :, None]
    R3    = r_vals[None, None, :]

    cos_TH = np.cos(THETA)
    sin_TH = np.sin(THETA)

    # Versore direzione raggio (easting x, northing y, up z)
    ray_ex = cos_TH * np.sin(PHI)   # (N_phi, N_th, 1)
    ray_ey = cos_TH * np.cos(PHI)
    ray_ez = sin_TH

    # Posizioni 3-D lungo i raggi
    X_grid = R3 * ray_ex             # (N_phi, N_th, N_r)
    Y_grid = R3 * ray_ey
    Z_ray  = sz + R3 * ray_ez        # quota assoluta [m a.s.l.]

    # ── Interpolatori DEM ─────────────────────────────────────────────────────
    dem_interp             = _build_interp(elev, dx_grid, dy_grid)
    grad_x_interp, \
    grad_y_interp          = _build_gradient_interps(elev, dx_grid, dy_grid)

    pts_xy = np.column_stack([Y_grid.ravel(), X_grid.ravel()])
    Z_dem  = dem_interp(pts_xy).reshape(N_phi, N_th, N_r)
    DZDX   = grad_x_interp(pts_xy).reshape(N_phi, N_th, N_r)
    DZDY   = grad_y_interp(pts_xy).reshape(N_phi, N_th, N_r)

    Z_dem_flat = np.full_like(Z_dem, s_elev)   # DEM di riferimento (piano)

    # Versori direzione in 2-D per cos_incidenza  (N_phi, N_th)
    ex2d = ray_ex[:, :, 0]
    ey2d = ray_ey[:, :, 0]
    ez2d = ray_ez[:, :, 0]

    # ── Funzione elaborazione per un DEM dato ─────────────────────────────────
    def _process(Zd, dzdx_3d, dzdy_3d):
        """
        Trova il primo hit lungo ogni raggio e calcola il contributo dN.

        Zd, dzdx_3d, dzdy_3d : (N_phi, N_th, N_r)

        Ritorna: dN, any_hit, r_hit, z_hit  tutti (N_phi, N_th)
        """
        hit_mask = (Z_ray <= Zd) & ~np.isnan(Zd)
        any_hit  = hit_mask.any(axis=2)

        fk = np.argmax(hit_mask, axis=2)[:, :, np.newaxis]

        R_bc   = np.broadcast_to(R3, (N_phi, N_th, N_r))
        r_hit  = np.take_along_axis(R_bc,    fk, axis=2).squeeze(2)
        z_hit  = np.take_along_axis(Zd,      fk, axis=2).squeeze(2)
        dzdx_h = np.take_along_axis(dzdx_3d, fk, axis=2).squeeze(2)
        dzdy_h = np.take_along_axis(dzdy_3d, fk, axis=2).squeeze(2)

        # L_air = distanza 3-D sensore → hit (r_hit è già distanza 3-D)
        L_air = r_hit.copy()

        # λ_air alla quota media del percorso
        lam_air_m = _lambda_air_m((sz + z_hit) / 2.0)

        # Normale locale alla superficie: n̂ = normalize(-dz/dx, -dz/dy, 1)
        nx = -dzdx_h
        ny = -dzdy_h
        nz = np.ones_like(r_hit)
        n_norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= n_norm;  ny /= n_norm;  nz /= n_norm

        # cos angolo di incidenza raggio–normale locale
        cos_inc = np.abs(ex2d * nx + ey2d * ny + ez2d * nz)
        cos_inc = np.maximum(cos_inc, COS_INC_MIN)

        # Percorso nel suolo lungo la direzione del raggio
        L_soil = z86_m / cos_inc

        # Peso angolare cos²θ (distribuzione lambertiana × elemento solido)
        cos2_th = (np.cos(theta_r)**2)[None, :]

        dN = (np.exp(-L_air / lam_air_m)
              * (1.0 - np.exp(-L_soil / lam_soil))
              * cos2_th)
        dN = np.where(any_hit, dN, 0.0)

        return dN, any_hit, r_hit, z_hit

    # ── Esecuzione sui due DEM ────────────────────────────────────────────────
    dN_obs, any_hit,      r_hit, z_hit = _process(Z_dem, DZDX, DZDY)
    dN_ref, any_hit_flat, _,    _      = _process(
        Z_dem_flat, np.zeros_like(DZDX), np.zeros_like(DZDY))

    N_obs = float(dN_obs.sum())
    N_ref = float(dN_ref.sum())

    kappa_topo = N_obs / N_ref if N_ref > 0 else 1.0

    # ── Classificazione diagnostica ───────────────────────────────────────────
    sopra_mask = any_hit      & (z_hit >  s_elev + dem_sigma_m)
    pieno_mask = any_hit      & (np.abs(z_hit - s_elev) <= dem_sigma_m)
    vuoto_mask = any_hit_flat & (~any_hit | (z_hit < s_elev - dem_sigma_m))

    kappa_sopra = float(np.sum(np.where(sopra_mask, dN_obs,           0.0))) / N_ref
    kappa_pieno = float(np.sum(np.where(pieno_mask, dN_obs,           0.0))) / N_ref
    kappa_vuoto = float(np.sum(np.where(vuoto_mask, dN_ref - dN_obs,  0.0))) / N_ref

    kappa_check = kappa_pieno + kappa_sopra - kappa_vuoto

    # ── Mappa 2-D dei pesi ────────────────────────────────────────────────────
    nr, nc = elev.shape
    x_1d  = dx_grid[nr // 2, :]
    y_1d  = dy_grid[:, nc // 2]
    x_asc = np.sort(x_1d)
    y_asc = np.sort(y_1d)

    x_hit_map = r_hit * np.cos(theta_r)[None, :] * np.sin(phi_r)[:, None]
    y_hit_map = r_hit * np.cos(theta_r)[None, :] * np.cos(phi_r)[:, None]

    ix = np.searchsorted(x_asc, x_hit_map.ravel(), side="left").clip(0, nc - 1)
    iy = np.searchsorted(y_asc, y_hit_map.ravel(), side="left").clip(0, nr - 1)
    if x_1d[0] > x_1d[-1]: ix = nc - 1 - ix
    if y_1d[0] > y_1d[-1]: iy = nr - 1 - iy

    wmap_flat = np.zeros(elev.size)
    np.add.at(wmap_flat, iy * nc + ix,
              np.where(any_hit.ravel(), dN_obs.ravel() / max(N_ref, 1e-30), 0.0))
    wmap = wmap_flat.reshape(elev.shape)
    wmap[np.isnan(elev)] = np.nan

    # ── Info diagnostica ──────────────────────────────────────────────────────
    info = dict(
        N_phi=N_phi, N_theta=N_th, N_r=N_r,
        n_rays_total=N_phi * N_th,
        n_rays_hit=int(any_hit.sum()),
        n_rays_hit_flat=int(any_hit_flat.sum()),
        n_pieno=int(pieno_mask.sum()),
        n_sopra=int(sopra_mask.sum()),
        n_vuoto=int(vuoto_mask.sum()),
        N_obs=N_obs, N_ref=N_ref,
        kappa_check=kappa_check,
        lam_soil_m=lam_soil,
        dem_sigma_m=dem_sigma_m,
        dphi_deg=dphi_deg, dtheta_deg=dtheta_deg, dr_m=dr_m,
    )
    return kappa_topo, kappa_pieno, kappa_sopra, kappa_vuoto, wmap, info


# ── Report testuale ───────────────────────────────────────────────────────────

def report_kappa_3d(kappa_topo, kappa_pieno, kappa_sopra, kappa_vuoto, info):
    w   = 72
    dev = kappa_topo - info["kappa_check"]
    lines = [
        "=" * w,
        "KAPPA_TOPO  —  3-D Ray-Cast  v4",
        "=" * w,
        f"  kappa_topo   = {kappa_topo:.4f}   (N_obs / N_ref)  ← valore da usare",
        f"  Diagnostica (pieno + sopra - vuoto ≈ kappa_topo):",
        f"    kappa_pieno  = {kappa_pieno:.4f}   suolo a quota sensore    [+]",
        f"    kappa_sopra  = {kappa_sopra:.4f}   terreno sopra sensore    [+]",
        f"    kappa_vuoto  = {kappa_vuoto:.4f}   deficit valle/dirupo     [-]",
        f"    check sum    = {info['kappa_check']:.4f}   "
        f"dev = {dev:+.4f}",
        "",
        f"  Grid  : {info['N_phi']} phi × {info['N_theta']} theta × "
        f"{info['N_r']} r = {info['N_phi']*info['N_theta']*info['N_r']:,} punti",
        f"  Hit reale : {info['n_rays_hit']:,} / {info['n_rays_total']:,}   "
        f"hit flat : {info['n_rays_hit_flat']:,}",
        f"  Class : pieno={info['n_pieno']:,}  "
        f"sopra={info['n_sopra']:,}  vuoto={info['n_vuoto']:,}",
        f"  lambda_soil={info['lam_soil_m']:.3f} m   "
        f"DEM sigma={info['dem_sigma_m']:.1f} m",
        "=" * w,
    ]
    return "\n".join(lines)


# ── Test suite ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    n = 100; r_max = 200.0
    x1 = np.linspace(-r_max, r_max, n)
    y1 = np.linspace(-r_max, r_max, n)
    XX, YY = np.meshgrid(x1, y1)
    dist   = np.sqrt(XX**2 + YY**2)

    s_elev = 500.0; sz = s_elev + 2.0
    r86 = 130.0; z86_cm = 16.0; rho_b = 1.4; sh = 2.0
    kw  = dict(dphi_deg=5., dtheta_deg=5., dr_m=8.)

    tests = [
        ("flat → kappa≈1",
         np.full((n, n), s_elev),
         lambda kt, kp, ks, kv: abs(kt - 1.0) < 0.03),

        ("cliff est → kappa<1",
         np.where(XX > 0, s_elev - 50., s_elev),
         lambda kt, kp, ks, kv: kt < 1.0),

        ("hill ovest → kappa>1",
         np.where(XX < 0, s_elev + 30., s_elev),
         lambda kt, kp, ks, kv: kt > 1.0),

        ("ring hill → kappa>1",
         np.where((dist > 40) & (dist < r86), s_elev + 20., s_elev),
         lambda kt, kp, ks, kv: kt > 1.0),

        ("pendio 20° → sopra>0 e vuoto>0",
         s_elev + XX * np.tan(np.radians(20.)),
         lambda kt, kp, ks, kv: ks > 0 and kv > 0),

        ("dirupo ovest → kappa<1",
         np.where(XX < -20, s_elev - 100., s_elev),
         lambda kt, kp, ks, kv: kt < 1.0),
    ]

    all_pass = True
    for name, elev_t, check in tests:
        print(f"\nTEST: {name}")
        t0 = time.perf_counter()
        kt, kp, ks, kv, _, info = compute_kappa_topo_3d(
            elev_t, XX, YY, dist, sz, s_elev, r86, z86_cm, rho_b, sh, **kw)
        dt = time.perf_counter() - t0
        print(report_kappa_3d(kt, kp, ks, kv, info))

        assert kv >= 0, f"kappa_vuoto negativo: {kv:.4f}"
        assert ks >= 0, f"kappa_sopra negativo: {ks:.4f}"
        assert kp >= 0, f"kappa_pieno negativo: {kp:.4f}"

        ok = check(kt, kp, ks, kv)
        status = "PASS" if ok else f"FAIL (kappa={kt:.4f})"
        if not ok: all_pass = False
        print(f"  {status}  [{dt:.3f} s]")

    print("\n" + ("ALL PASS" if all_pass else "SOME FAILURES"))
