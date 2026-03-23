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
from crns_corrections   import report_crns_corrections
from geology            import report_geology
from sampling_plan      import report_sampling_plan
from radiofreq          import report_rf
from evapotranspiration import report_et
from electrical_conductivity import report_ec


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
    s(f"  z86 (at site, +lw)   : {results['z86']:.2f} cm")
    s(f"    z86 = 8.3 / (rho_b * (0.0564 + theta_v + lw))")
    s(f"    lw (lattice water) = {results.get('lw', 0.0):.4f} g/g")
    s(f"    theta_v_SOC (sup.) = {results['site_fluxes'].get('theta_v_soc', 0.0):.4f} m³/m³")
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
    s(f"  kappa_lulc           : {results.get('kappa_lulc', 1.0):.4f}")
    s(f"    Fonte: ESA WorldCover 10m — rapporto H footprint / H suolo baseline.")
    s(f"    kappa_lulc > 1: alta presenza di acqua/vegetazione nel footprint.")
    s(f"    kappa_lulc < 1: bassa presenza di H (asfalto, roccia nuda).")
    s(f"    Condizione di riferimento: θ_v_init = {params['theta_v_init']:.3f} m³/m³.")
    s()
    s(f"  kappa_total          : {results['kappa_total']:.4f}")
    s(f"    = kappa_topo × kappa_muon × kappa_lulc (tutti gli effetti combinati)")
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

    h("GEOLOGY (MACROSTRAT API)")
    if 'geology' in results:
        s(report_geology(results['geology']))
    else:
        s("  [non disponibile]")
    s()

    h("CRNS SUPPLEMENTARY CORRECTIONS (WV, AGBH, SWE)")
    if 'crns_corrections' in results:
        z86 = results.get('z86', 15.0)
        s(report_crns_corrections(results['crns_corrections'], z86_cm=z86))
    else:
        s("  [non disponibile]")
    s()

    h("EVAPOTRANSPIRATION (FAO-56 Penman-Monteith)")
    s(report_et(results.get("et")))
    s()

    h("CONDUCIBILITA' ELETTRICA APPARENTE (ECa)")
    s(report_ec(results.get("ec")))
    s()

    h("RF ANALYSIS (OpenCelliD + OSM RFI)")
    s(report_rf(results.get("rf")))
    s()

    h("OPTIMAL SOIL SAMPLING PLAN")
    if 'sampling' in results:
        s(report_sampling_plan(results['sampling']))
    else:
        s("  [non disponibile]")
    s()

    s("="*W)
    text = "\n".join(L)
    with open(path, "w") as f:
        f.write(text)
    return text
