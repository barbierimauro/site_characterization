# =============================================================================
# REPORT
# =============================================================================

import numpy as np
from kappa_topo_3d      import report_kappa_3d
from lulc               import report_lulc
from site_fluxes        import report_site_fluxes, report_desilets_curve
from site_climate       import report_site_climate, report_power_budget
from terrain_indices    import report_twi, report_thermal_index
from get_soil_properties import report_soil_properties
from vegetation_indices  import report_snow_cover


def write_report(path, params, results):
    W = 72
    L = []
    def s(x=""): L.append(x)
    def h(t):    s(t); s("-"*40)

    s("="*W); s("CRNS TOPOGRAPHIC & FOV CORRECTION REPORT"); s("="*W); s()

    h("INPUT PARAMETERS")
    s(f"  Sensor location      : {params['lat']:.6f} N, {params['lon']:.6f} E")
    s(f"  Sensor height        : {params['h']} m a.g.l.")
    s(f"  Soil bulk density    : {params['rho_b']} g/cm3")
    s(f"  Initial SM (theta_v) : {params['theta_v_init']:.3f} m3/m3")
    s(f"  DEM download radius  : {params['dem_radius']} m")
    s(f"  DEM source           : {params['dem_source']}")
    s(f"  Azimuth step         : {params['az_step']} deg")
    s(f"  N cores              : {params['n_cores']}")
    s()

    h("DEM / SITE PROPERTIES")
    s(f"  Sensor altitude      : {results['sensor_alt']:.1f} m a.s.l.")
    s(f"  Pressure at site     : {results['pressure']:.2f} hPa")
    s(f"  Air density at site  : {results['rho_air']:.6f} g/cm3")
    s(f"  Mean slope (footprint): {np.degrees(results['mean_slope_rad']):.2f} deg")
    s(f"  Max horizon angle    : {results['max_horizon']:.2f} deg")
    s(f"  Mean horizon angle   : {results['mean_horizon']:.2f} deg")
    s()

    h("NEUTRON FOOTPRINT PARAMETERS")
    s(f"  r86 (sea level ref)  : {results['r86_sealevel']:.1f} m")
    s(f"  r86 (at site, P-cor) : {results['r86']:.1f} m")
    s(f"  z86 (at site)        : {results['z86']:.2f} cm")
    s(f"  V0 (flat reference)  : {results['V0']:.2f} m3")
    s(f"    (cylinder: pi*r86^2 * z86, used as denominator of kappa)")
    s(f"  V_eff (actual soil)  : {results['Veff']:.2f} m3")
    s(f"    (DEM-cell integral of visible soil inside V0)")
    s()

    h("CORRECTION FACTORS")
    s(f"  kappa_topo           : {results['kappa_topo']:.4f}")
    s(f"    Method: DEM-cell summation — no iteration.")
    s(f"    For each pixel within r86, the vertical overlap between the")
    s(f"    reference slab [sensor - z86, sensor] and the actual soil")
    s(f"    column is computed and weighted by W(r) = exp(-r/r86*3).")
    s(f"    kappa_topo = V_eff / V0.")
    s(f"    kappa_topo < 1: terrain drops away (cliffs, valleys) —")
    s(f"      air replaces soil inside V0 — sensor underestimates SM.")
    s(f"    kappa_topo > 1: terrain rises above sensor level —")
    s(f"      extra soil inside V0 — sensor overestimates SM.")
    s()
    s(f"  kappa_muon           : {results['kappa_muon']:.4f}")
    s(f"    Method: analytical integration of cos^2(theta_z) spectrum")
    s(f"    over the visible sky fraction at each azimuth.")
    s(f"    kappa_muon is ALWAYS <= 1 (obstruction can only reduce flux).")
    s(f"    kappa_muon = 1.0 means open sky in all directions.")
    s(f"    kappa_muon < 1: surrounding topography blocks low-angle muons.")
    s(f"    Effect: measured muon rate is lower than expected for flat site.")
    s(f"    Consequence for normalisation: if muons are used as cosmic-ray")
    s(f"    reference monitor, the apparent muon rate at this site is")
    s(f"    reduced by kappa_muon = {results['kappa_muon']:.4f} relative to a flat site.")
    s(f"    The muon-normalised neutron count must be DIVIDED by kappa_muon")
    s(f"    to recover the flux equivalent to an open-sky reference station.")
    s()
    s(f"  kappa_total          : {results['kappa_total']:.4f}")
    s(f"    = kappa_topo x kappa_muon (both effects combined)")
    s()

    h("SOIL MOISTURE CORRECTION")
    s(f"  theta_v apparent     : {params['theta_v_init']:.4f} m3/m3")
    s(f"  theta_v corrected    : {results['theta_v_corrected']:.4f} m3/m3")
    s(f"    = theta_v_apparent / kappa_topo")
    s(f"  SM relative bias     : {(results['kappa_topo']-1)*100:+.1f} %")
    s(f"    (positive = overestimate, negative = underestimate)")
    s()

    h("KAPPA_TOPO 3-D RAY-CASTING DETAIL")
    s(report_kappa_3d(
        results['kappa_topo'], results['kappa_pieno'],
        results['kappa_sopra'], results['kappa_vuoto'],
        results['kappa_info']))
    s()

    h("SITE FLUXES")
    s(report_site_fluxes(results['site_fluxes']))
    s()

    h("DESILETS CURVE — N(θ_v) EXPECTED RANGE")
    s(report_desilets_curve(results['desilets_curve']))
    s()

    h("SITE CLIMATE")
    s(report_site_climate(results['site_climate']))
    s()

    h("POWER BUDGET — SOLAR PANEL & BATTERY")
    s(report_power_budget(results['power_budget']))
    s()

    h("SOIL PROPERTIES")
    s(report_soil_properties(results['soil']))
    s()

    h("TOPOGRAPHIC WETNESS INDEX")
    s(report_twi(results['twi']))
    s()

    h("THERMAL INDEX")
    sc = results['site_climate']
    s(report_thermal_index(
        results['thermal'],
        sc['T_mean_monthly_C'],
        sc['T_min_monthly_C'],
        sc['T_max_monthly_C']))
    s()

    h("SNOW COVER — MODIS MOD10A1")
    s(report_snow_cover(results['snow']))
    s()

    h("LULC — LAND USE / LAND COVER")
    s(report_lulc(results['lulc']))
    s()

    h("OUTPUT FILES — DESCRIPTION")
    s(f"  crns_topo_main.png")
    s(f"    Panel 1 (top left): DEM map centred on sensor with three")
    s(f"    concentric circles at 0.5*r86, r86, 1.5*r86. The red triangle")
    s(f"    marks the sensor. Colour = elevation (m a.s.l.).")
    s(f"    Panel 2 (top right): theoretical kappa vs slope angle for a")
    s(f"    uniform hilltop (red) and dolina (blue), with horizontal line")
    s(f"    showing this site's kappa_topo.")
    s(f"    Panel 3 (bottom left): E-W terrain cross-section through")
    s(f"    sensor, with r86 marked.")
    s(f"    Panel 4 (bottom centre): N-S terrain cross-section.")
    s(f"    Panel 5 (bottom right): bar chart of kappa_topo, kappa_muon,")
    s(f"    kappa_total. Red = above 1, blue = below 1.")
    s()
    s(f"  crns_footprint.png")
    s(f"    Left panel: pixel classification within r86. Each DEM cell")
    s(f"    coloured by contribution type: RED=terrain deficit (z_DEM")
    s(f"    below slab bottom, zero overlap), ORANGE=partial overlap,")
    s(f"    GREEN=full contribution (terrain at or above reference).")
    s(f"    Grey shading intensity = radial weight W(r). Dashed = r86.")
    s(f"    Centre panel: radial overlap profile. W(r)-weighted mean")
    s(f"    soil overlap fraction vs distance from sensor, in 15m shells.")
    s(f"    Value 1.0 = full slab in that shell. Dashed = W(r) shape.")
    s(f"    Right panel: per-azimuth mean overlap fraction, polar diagram.")
    s(f"    North up, clockwise. Colour red (deficit) to green (full).")
    s(f"    Shows which compass directions lose most soil volume.")
    s(f"  crns_horizon.png")
    s(f"    Left panel: polar diagram of horizon elevation angle psi(phi)")
    s(f"    for each azimuth (North up, clockwise). The radial axis is")
    s(f"    the terrain elevation angle above horizontal as seen from the")
    s(f"    sensor. A flat site would show a circle at 0 deg.")
    s(f"    Right panel: per-azimuth muon sensitivity f(phi)/f_ref, where")
    s(f"    f(phi) = integral of cos^2(theta_z) over the visible zenith arc")
    s(f"    [0, 90-psi(phi)]. A flat site would show a circle at 1.0.")
    s(f"    Directions with high horizon angle appear depressed (less muon")
    s(f"    flux from those directions). The mean over all azimuths gives")
    s(f"    kappa_muon = {results['kappa_muon']:.4f}.")
    s()
    s(f"  crns_fov_detail.png")
    s(f"    Left panel: neutron soil contribution by azimuth. Each bar")
    s(f"    shows the W(r)-weighted mean overlap fraction per compass")
    s(f"    direction (0=no soil, 1=full slab). Colour red->green.")
    s(f"    Orange dashed = kappa_topo. Immediately shows which sectors")
    s(f"    dominate the SM bias and in which direction.")
    s(f"    Right panel: muon FOV map. X=azimuth, Y=elevation angle above")
    s(f"    horizontal (0=horizon, 90=zenith). Blue = muon angular weight")
    s(f"    cos^2(theta_z)*sin(theta_z), peak near 45 deg. Grey = blocked")
    s(f"    by terrain. Black line = horizon profile psi(az). Orange line =")
    s(f"    per-azimuth muon fraction scaled to elevation axis (0->0, 1->90)")
    s(f"    shows which directions contribute less to normalisation rate.")
    s()

    s("="*W)
    text = "\n".join(L)
    with open(path, "w") as f:
        f.write(text)
    return text
