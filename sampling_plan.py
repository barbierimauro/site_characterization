"""
sampling_plan.py
================
Piano di campionamento ottimale del suolo per la calibrazione CRNS,
basato sulla distribuzione radiale della sensibilità W(r).

Protocollo
----------
Lo standard CRNS field protocol (Zreda et al. 2012, Franz et al. 2012)
prevede campioni di suolo su anelli concentrici ponderati per W(r):
    - Anello interno  : r ≈ 0.25 * r86
    - Anello medio    : r ≈ 0.50 * r86
    - Anello esterno  : r ≈ 0.75 * r86
    - Anello distale  : r ≈ 1.00 * r86

Per ogni anello si campionano 4–8 punti equidistanti in azimut.
Il numero di campioni per anello è proporzionale al peso radiale dell'anello.

Ogni campione viene mediato pesato da W(r_i) per ottenere theta_v_ref.

Range dinamico r86
------------------
r86 varia da ~180 m (suolo secco, theta_v=0) a ~75 m (suolo saturo,
theta_v=0.45) per condizioni tipiche. Il grafico r86(theta_v) con la
land cover sovrapposta mostra come il footprint "si restringe" in
condizioni umide, potenzialmente escludendo coperture distali diverse.

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

KOHLI_A1 = 29.13    # m · m³/m³
KOHLI_A2 = 0.0578   # m³/m³
P0_HPA   = 1013.25  # hPa

# Profondità di campionamento raccomandate [cm]: corrispondono agli strati CRNS
SAMPLE_DEPTHS_CM = [(0, 5), (5, 15), (15, 30)]

# Numero di punti per anello (ridotto per anelli piccoli)
POINTS_PER_RING = [4, 6, 8, 8]

# Frazioni di r86 per i 4 anelli
RING_FRACS = [0.25, 0.50, 0.75, 1.00]


# ---------------------------------------------------------------------------
# Peso radiale
# ---------------------------------------------------------------------------

def _W(r, r86):
    lam = r86 / 3.0
    return np.where(r < 1e-3, 0.0, np.exp(-r / lam))


# ---------------------------------------------------------------------------
# r86 in funzione di theta_v
# ---------------------------------------------------------------------------

def r86_at_theta_v(theta_v, pressure_hpa=P0_HPA):
    """r86 [m] da Köhli 2015 / Desilets in funzione di theta_v."""
    return KOHLI_A1 / (np.asarray(theta_v) + KOHLI_A2) * (P0_HPA / pressure_hpa)


# ---------------------------------------------------------------------------
# Funzione principale: piano di campionamento
# ---------------------------------------------------------------------------

def compute_sampling_plan(r86, theta_v_init=0.20, theta_wp=0.05,
                           theta_fc=0.35, pressure_hpa=P0_HPA,
                           lulc_info=None):
    """
    Calcola il piano ottimale di campionamento del suolo per la calibrazione CRNS.

    Il piano è basato sulla ponderazione W(r) = exp(-r / (r86/3)).
    I campioni sono disposti su anelli concentrici; la frazione di peso
    di ogni anello determina quanti campioni sono necessari per rappresentare
    correttamente quella zona del footprint.

    Parameters
    ----------
    r86          : float — raggio footprint 86% [m] alla theta_v_init
    theta_v_init : float — SM di riferimento [m³/m³]
    theta_wp     : float — punto di appassimento [m³/m³]
    theta_fc     : float — capacità di campo [m³/m³]
    pressure_hpa : float — pressione al sito [hPa]
    lulc_info    : dict opzionale — da get_lulc(), per sovrapporre land cover

    Returns
    -------
    dict con:
        rings               : lista di dict per ogni anello
        sample_locations    : lista di (r_m, az_deg, weight) per tutti i punti
        total_samples       : int — numero totale di campioni raccomandato
        theta_v_range       : array — range SM per curva r86
        r86_range           : array — r86 corrispondente
        r86_dry             : float — r86 a theta_v=0 (massimo footprint)
        r86_sat             : float — r86 a theta_v=0.45 (footprint minimo)
        sample_weights      : array — peso normalizzato di ogni campione
    """
    # --- Anelli e pesi ---
    rings = []
    sample_locations = []
    total_weight_check = 0.0

    # Calcolo peso integrale per ogni anello
    # W_ring(i) = integral_{r_inner}^{r_outer} W(r) * 2π*r dr  (approssimato)
    ring_inner = [0.0] + [f * r86 for f in RING_FRACS[:-1]]
    ring_outer = [f * r86 for f in RING_FRACS]

    n_rings = len(RING_FRACS)
    ring_weights = np.zeros(n_rings)
    for i in range(n_rings):
        r_test = np.linspace(ring_inner[i] + 1, ring_outer[i], 30)
        ring_weights[i] = np.trapezoid(_W(r_test, r86) * r_test, r_test)
    ring_weights /= ring_weights.sum()  # normalizza

    for i in range(n_rings):
        r_mean = 0.5 * (ring_inner[i] + ring_outer[i])
        n_pts  = POINTS_PER_RING[i]
        W_r    = float(_W(r_mean, r86))
        azimuths = np.linspace(0, 360, n_pts, endpoint=False)

        # Contributo per campione di questo anello
        w_per_sample = ring_weights[i] / n_pts

        ring_info = dict(
            ring_id         = i + 1,
            r_inner_m       = float(ring_inner[i]),
            r_outer_m       = float(ring_outer[i]),
            r_mean_m        = float(r_mean),
            n_samples       = n_pts,
            ring_weight_frac = float(ring_weights[i]),
            W_r             = W_r,
            sample_depths   = SAMPLE_DEPTHS_CM,
        )
        rings.append(ring_info)

        for az in azimuths:
            sample_locations.append(dict(
                r_m       = float(r_mean),
                az_deg    = float(az),
                weight    = float(w_per_sample),
                ring_id   = i + 1,
            ))
        total_weight_check += ring_weights[i]

    # Pesi normalizzati dei singoli campioni
    all_weights = np.array([s['weight'] for s in sample_locations])
    all_weights /= all_weights.sum()
    for i, s in enumerate(sample_locations):
        s['weight_norm'] = float(all_weights[i])

    # --- Range dinamico r86 ---
    theta_arr = np.linspace(max(0.01, theta_wp - 0.02),
                             min(0.50, theta_fc + 0.10), 50)
    r86_arr   = r86_at_theta_v(theta_arr, pressure_hpa)
    r86_dry   = float(r86_at_theta_v(0.0, pressure_hpa))
    r86_sat   = float(r86_at_theta_v(0.45, pressure_hpa))

    return dict(
        rings            = rings,
        sample_locations = sample_locations,
        total_samples    = sum(r['n_samples'] for r in rings),
        theta_v_range    = theta_arr,
        r86_range        = r86_arr,
        r86_dry          = r86_dry,
        r86_sat          = r86_sat,
        r86_ref          = float(r86),
        theta_v_init     = float(theta_v_init),
        theta_wp         = float(theta_wp),
        theta_fc         = float(theta_fc),
        pressure_hpa     = float(pressure_hpa),
        sample_weights   = all_weights,
    )


# ---------------------------------------------------------------------------
# Report testuale
# ---------------------------------------------------------------------------

def report_sampling_plan(sp):
    w = 72
    L = [
        "=" * w,
        "OPTIMAL SOIL SAMPLING PLAN FOR CRNS CALIBRATION",
        "=" * w,
        f"  r86 di riferimento  : {sp['r86_ref']:.0f} m  (a θ_v={sp['theta_v_init']:.2f} m³/m³)",
        f"  r86 a suolo secco   : {sp['r86_dry']:.0f} m  (θ_v → 0)",
        f"  r86 a saturazione   : {sp['r86_sat']:.0f} m  (θ_v = 0.45 m³/m³)",
        f"  Campioni totali     : {sp['total_samples']}  (su {len(sp['rings'])} anelli)",
        f"  Profondità campione : " +
        "  ".join(f"{t}-{b}cm" for t, b in SAMPLE_DEPTHS_CM),
        "",
        f"  {'Anello':>7} {'r interno':>10} {'r esterno':>10} {'r medio':>8} "
        f"{'N camp':>7} {'Peso %':>7} {'W(r)':>7}",
        "  " + "-" * 60,
    ]
    for r in sp['rings']:
        L.append(
            f"  {r['ring_id']:>7} {r['r_inner_m']:10.0f} {r['r_outer_m']:10.0f}"
            f" {r['r_mean_m']:8.0f} {r['n_samples']:>7} "
            f"{r['ring_weight_frac']*100:7.1f} {r['W_r']:7.3f}")
    L += [
        "",
        "  Istruzioni campo:",
        "  1. Campionare sui 4 anelli concentrici centrati sul sensore.",
        "  2. Per ogni anello, prelevare N campioni di suolo ad azimutti equidistanti.",
        f"  3. Profondità: 0-5 cm, 5-15 cm, 15-30 cm (max z86≈{sp['r86_ref']:.0f}/6 cm).",
        "  4. Pesare i campioni come indicato (θ_v_ref = Σ w_i * θ_v_i / Σ w_i).",
        "  5. Calibrare in condizioni di SM stabile (non subito dopo pioggia).",
        "  6. Ideale: calibrazione a θ_v intermedio tra θ_WP e θ_FC.",
        "=" * w,
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

STYLE = {
    "figure.dpi": 100, "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f6", "axes.grid": True,
    "grid.color": "white", "grid.linewidth": 1.2,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
}


def plot_sampling_plan(sp, elev, dx_grid, dy_grid, dist_grid, path,
                        site_name="", lulc_wc_dem=None, wc_classes=None):
    """
    Figura con:
      - Pannello sinistro: mappa DEM con campioni e anelli sovrapposti
      - Pannello destro:   curva r86(theta_v) con range dinamico
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="white")

        # ---- Pannello sx: mappa con campioni ----
        ax = axes[0]
        r86 = sp['r86_ref']

        # DEM sfondo
        dem_show = np.where(dist_grid <= r86 * 1.3, elev, np.nan)
        ext = [dx_grid.min(), dx_grid.max(), dy_grid.min(), dy_grid.max()]
        if not np.all(np.isnan(dem_show)):
            ax.imshow(dem_show, extent=ext, origin="upper",
                      cmap="terrain", aspect="equal", interpolation="bilinear",
                      alpha=0.5)

        # Anelli
        th = np.linspace(0, 2 * np.pi, 360)
        ring_colors = ["#1a9641", "#fdae61", "#d7191c", "#2c7bb6"]
        for i, ring in enumerate(sp['rings']):
            r = ring['r_outer_m']
            ax.plot(r * np.sin(th), r * np.cos(th),
                    color=ring_colors[i], lw=1.5, ls='--',
                    label=f"Anello {ring['ring_id']} ({r:.0f} m)")

        # Campioni
        for s in sp['sample_locations']:
            az_r = np.radians(s['az_deg'])
            x = s['r_m'] * np.sin(az_r)
            y = s['r_m'] * np.cos(az_r)
            ci = ring_colors[s['ring_id'] - 1]
            sz = 80 + 200 * s['weight_norm'] * sp['total_samples']
            ax.scatter(x, y, s=sz, color=ci, edgecolors='k',
                       linewidths=0.8, zorder=5)
            ax.text(x * 1.05, y * 1.05,
                    f"{s['weight_norm']*100:.1f}%",
                    fontsize=7, ha='center', va='center', color='k')

        ax.plot(0, 0, 'r^', ms=14, zorder=10, label='Sensore')
        ax.set_aspect('equal')
        ax.set_xlabel('Easting offset (m)')
        ax.set_ylabel('Northing offset (m)')
        ax.set_title(f'Piano di campionamento CRNS\n'
                     f'{sp["total_samples"]} campioni su 4 anelli  '
                     f'(r86={r86:.0f} m, θ_v={sp["theta_v_init"]:.2f})',
                     fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        lim = r86 * 1.25
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

        # ---- Pannello dx: r86(theta_v) ----
        ax2 = axes[1]
        tv  = sp['theta_v_range']
        r86v = sp['r86_range']
        ax2.plot(tv, r86v, 'b-', lw=2.5, label='r86(θ_v)')
        ax2.axvline(sp['theta_wp'], color='orange', ls='--', lw=1.5,
                    label=f"θ_WP = {sp['theta_wp']:.2f}")
        ax2.axvline(sp['theta_fc'], color='red', ls='--', lw=1.5,
                    label=f"θ_FC = {sp['theta_fc']:.2f}")
        ax2.axvline(sp['theta_v_init'], color='green', ls='-', lw=1.5,
                    label=f"θ_init = {sp['theta_v_init']:.2f}")
        ax2.axhline(sp['r86_ref'], color='green', ls=':', lw=1.2)
        ax2.fill_betweenx([sp['r86_sat'], sp['r86_dry']],
                           sp['theta_wp'], sp['theta_fc'],
                           alpha=0.12, color='blue',
                           label='Range operativo SM')

        # Annotazioni
        ax2.annotate(f"r86_max = {sp['r86_dry']:.0f} m\n(suolo secco)",
                     xy=(0.02, sp['r86_dry']), fontsize=9, color='gray')
        ax2.annotate(f"r86_sat = {sp['r86_sat']:.0f} m\n(saturo 0.45)",
                     xy=(0.40, sp['r86_sat']+2), fontsize=9, color='gray')

        ax2.set_xlabel('θ_v — Soil moisture (m³/m³)')
        ax2.set_ylabel('r86 — Footprint radius (m)')
        ax2.set_title('Range dinamico del footprint r86\n'
                      'Il footprint si restringe all\'aumentare di θ_v',
                      fontsize=12)
        ax2.legend(fontsize=9)
        ax2.set_xlim(0, 0.52)

        if site_name:
            fig.suptitle(f'CRNS Sampling Plan — {site_name}', fontsize=13,
                         fontweight='bold')
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"   Sampling plan plot -> {path}", flush=True)
