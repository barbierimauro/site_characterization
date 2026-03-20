# CRNS Site Characterization Tool

A Python toolkit for the complete scientific characterization of a
**Cosmic-Ray Neutron Sensor (CRNS)** installation site. The tool
computes topographic and field-of-view correction factors (κ_topo,
κ_muon), expected count rates, local climate, soil properties, and
terrain indices — providing everything needed to correctly interpret
raw CRNS neutron counts as soil moisture.

---

## Scientific Background

### Cosmic-Ray Neutron Sensing

CRNS measures soil moisture by detecting **epithermal neutrons** produced
when high-energy cosmic-ray particles collide with atmospheric nuclei and
reach the ground. Hydrogen atoms in liquid water are extremely efficient
neutron moderators: the higher the soil water content, the more neutrons
are absorbed, and the lower the count rate measured by the detector.

The relationship between corrected count rate *N* and volumetric soil
moisture *θ_v* (Desilets et al. 2010) is:

```
θ_v = a1 / (N/N0 − a0) − a2

  a0 = 0.0808,  a1 = 0.372,  a2 = 0.115
```

where **N0** is the reference count rate for dry soil at the site. This
inversion requires that *N* has been correctly normalised for all
confounding factors — which is precisely what this tool computes.

### Footprint Geometry

The sensor integrates signal over a roughly circular footprint whose
characteristic radius **r86** (the distance enclosing 86 % of the
sensitivity) depends on soil moisture and atmospheric pressure
(Kohli et al. 2015):

```
r86 = (A1 / (θ_v + A2)) × (P0 / P)

  A1 = 29.13,  A2 = 0.0578,  P0 = 1013.25 hPa
```

At sea level and θ_v = 0.20 m³/m³, r86 ≈ 113 m. The sensor also
integrates vertically to a characteristic depth **z86** (Köhli/Bogena):

```
z86 = NUM / (ρ_b × (α + θ_v))    [cm]

  NUM = 8.3,  α = 0.0564
```

At typical conditions z86 ≈ 16 cm. Both r86 and z86 are pressure- and
moisture-dependent — they define the reference soil volume that a
perfectly flat site would sample.

The **radial sensitivity weight** W(r) decays exponentially from the
sensor:

```
W(r) = exp(−r / λ),   λ = r86 / 3
```

so pixels close to the sensor contribute ~3× more than those at r86.

### Why Topographic Corrections Matter

Real terrain is never flat. Hills, valleys, cliffs, and depressions mean
that the actual volume of soil inside the theoretical footprint cylinder
differs from the flat-site reference. Two independent corrections are
needed:

| Factor | Symbol | Physical meaning |
|--------|--------|-----------------|
| Topographic volume correction | κ_topo | Ratio of actual soil volume inside footprint to the flat reference |
| Muon field-of-view correction | κ_muon | Fraction of cosmic-muon flux reaching the sensor after terrain obstruction |

The corrected soil moisture is recovered as:

```
θ_v_corrected = θ_v_apparent / κ_topo
```

and the combined correction on the muon-normalised count rate is:

```
κ_total = κ_topo × κ_muon
```

---

## Modules

### `kappa_topo_3d.py` — 3-D Ray-Casting Topographic Correction

This is the physically rigorous core of the tool. Instead of geometric
approximations, it traces neutron rays from the sensor through a full
3-D angular grid (azimuth φ, elevation angle θ, radial distance r)
and uses **exponential attenuation physics** in both air and soil.

For each ray the signal contribution is:

```
dN = exp(−L_air / λ_air) × (1 − exp(−L_soil / λ_soil)) × |cos(θ)|

  λ_air  = 130 g/cm²    neutron attenuation length in air
  λ_soil = 162 g/cm²    neutron attenuation length in soil
  L_air  = oblique path in air from sensor to DEM surface [g/cm²]
  L_soil = z86 / |cos(θ)|   oblique penetration depth in soil
  |cos(θ)|               geometric projection factor
```

The result is compared to the flat-site reference (identical integral
over a perfectly horizontal plane at sensor altitude):

```
κ_topo = N_observed / N_reference
```

The algorithm also decomposes pixel contributions into three classes:

- **PIENO** (full): terrain at approximately the reference level → ideal
  signal, κ contribution ≈ 1
- **SOPRA** (above): terrain higher than sensor level → extra soil enters
  the footprint, κ > 1 risk (SM overestimate)
- **VUOTO** (void): terrain below the slab bottom → air instead of soil,
  κ < 1 (SM underestimate)

A depression (dolina) or cliff edge produces κ_topo < 1; a hilltop or
raised terrace produces κ_topo > 1.

---

### `site_fluxes.py` — Expected Count Rates and N0

Translates site coordinates and κ corrections into physically expected
CRNS count rates. Two key scaling effects are applied:

**Barometric scaling** — cosmic-ray flux increases with altitude as
atmospheric shielding decreases:

```
N_muon = N_muon,sl × exp(β_muon × ΔP/P0) × κ_muon       β_muon = 1.15
N_neut = N_neut,sl × f_Rc × exp(β_neut × ΔP/P0) × κ_topo  β_neut = 2.3
```

where ΔP = P0 − P_site (positive at altitude: less pressure, more flux).

**Geomagnetic rigidity cutoff** — charged cosmic-ray primaries are
deflected by Earth's magnetic field. Sites at low geomagnetic latitude
(closer to the equator) have a higher rigidity cutoff and experience a
lower primary flux (Hawdon 2014):

```
f_Rc = exp(α × (Rc_site − Rc_ref))     α = −0.075 GV⁻¹
```

Rigidity cutoff Rc is retrieved from the Smart & Shea (2019) database
via the `crnpy` package. The theoretical **N0** (count rate for
completely dry soil) is then:

```
N0 = N_neut / (a0 + a1/a2)     ≈ N_neut / 3.43
```

This N0 estimate is site-specific and accounts for both topographic
and geomagnetic effects, making it directly usable for the Desilets
soil moisture retrieval.

---

### `site_climate.py` — Local Climate Characterisation

Provides a complete 12-month climate profile from two external APIs.

#### PVGIS TMY (JRC / European Commission)

Hourly typical meteorological year (ERA5/SARAH3, 2005–2020) providing
global horizontal irradiance (GHI), direct normal irradiance (DNI),
diffuse horizontal irradiance (DHI), air temperature, relative humidity,
wind speed and direction, and atmospheric pressure.

When available, the 30 m DEM horizon profile is passed to PVGIS to
correct irradiance for local terrain shading — substantially more
accurate than the internal ~90 m horizon used by default.

Irradiance on a tilted surface is computed with the **Perez transposition
model** (direct + diffuse sky + ground albedo). Panel cell temperature
uses the **Faiman (1994) thermal model**:

```
T_cell = T_air + POA / (U0 + U1 × WS)

  U0 = 25 W m⁻² K⁻¹       constant thermal loss coefficient
  U1 = 6.84 W m⁻² K⁻¹ (m/s)⁻¹   wind-induced convective cooling
```

#### Open-Meteo ERA5 Precipitation (~31 km)

Daily historical precipitation (2005–2020) aggregated into monthly
totals (mm/month) and mean number of rainy days (threshold: > 1 mm/day).
The ERA5 grid-cell elevation is extracted from the API response and
passed to the thermal correction module.

#### Frost Day Estimation

A logistic model gives the probability that the daily minimum temperature
crosses 0 °C:

```
P(frost) = 1 / (1 + exp(T_min / 3))
```

Monthly outputs (12-element arrays, January = index 0) include:
GHI, DNI, DHI, POA, PV energy yield, T_mean/min/max, relative humidity,
pressure, wind speed statistics, precipitation total, rainy days, and
estimated frost days.

---

### `terrain_indices.py` — TWI and Thermal Index

#### Topographic Wetness Index (TWI)

TWI quantifies the tendency of each DEM cell to accumulate water
(Beven & Kirkby 1979):

```
TWI = ln(a / tan(β))

  a  = specific upstream drainage area [m]   (D8 flow accumulation)
  β  = local slope [rad]
```

The D8 algorithm uses **Wang & Liu (2006)** priority-queue depression
filling (O(n log n)) before routing flow in the direction of steepest
descent among the 8 neighbours. Within the CRNS footprint (r ≤ r86)
the **radial-sensitivity-weighted mean TWI** is computed using W(r),
giving a single representative value of wetness potential for what the
neutron sensor actually integrates.

| TWI | Drainage class |
|-----|---------------|
| < 5 | Well-drained ridges and upper slopes |
| 5–8 | Moderate drainage, transitional |
| > 8 | Convergence zones, riparian areas, wet soils |

#### Thermal Index — Local Temperature Correction

ERA5 climate data represents a ~31 km grid cell that may differ
substantially in elevation and topographic exposure from the actual
sensor site. Three physical corrections are applied to produce
site-specific monthly temperature arrays:

**1. Lapse rate** — elevation difference between site and ERA5 grid cell:

```
ΔT_lapse = −γ × (z_site − z_ERA5)     γ = 6.5 × 10⁻³ °C/m  (ICAO standard)
```

Applied to T_mean, T_min, and T_max.

**2. Cold-air pooling** — nocturnal valley temperature inversion:

```
ΔT_pool = −K × (1 − SVF) × tanh(concavity / 50)

  K          = 3.0 °C      maximum intensity of cold-pool effect
  SVF        = 1 − mean(sin²(ψ))   Sky View Factor from DEM horizon angles
  concavity  = z̄(50–200 m ring) − z̄(< 50 m)   positive in depressions
```

Applied only to T_min (purely nocturnal radiative cooling). The
hyperbolic tangent saturates the correction for very deep basins
(concavity > 150 m → full K amplitude).

**3. Local insolation (PISR)** — aspect and horizon-controlled heating:

```
ΔT_pisr = α × (PISR_site − PISR_ERA5) / PISR_ERA5     α = 4.0 °C
```

PISR_site is the annual POA irradiance from PVGIS with the 30 m DEM
horizon; PISR_ERA5 is normalised to open-sky conditions via SVF.
Applied to T_mean and T_max (daytime heating effect).

Total uncertainty on corrected temperatures: ~2.5 °C (quadrature sum
of ±1 °C lapse, ±2 °C cold-pooling, ±1.5 °C PISR components).

---

### `get_soil_properties.py` — Soil Properties from SoilGrids

Retrieves nine soil properties from **SoilGrids v2.0** (ISRIC, 250 m
resolution) at six standard depth intervals (0–5, 5–15, 15–30, 30–60,
60–100, 100–200 cm):

| Property | Symbol | Units |
|----------|--------|-------|
| Bulk density | ρ_b | g/cm³ |
| Clay content | — | % |
| Sand content | — | % |
| Silt content | — | % |
| Soil organic carbon | SOC | g/kg |
| pH in water | pH | — |
| Cation exchange capacity | CEC | cmol/kg |
| Coarse fragments volume | CF | % |
| Total nitrogen | N_tot | g/kg |

Raw SoilGrids values are rescaled to SI units and averaged using
**CRNS neutron-attenuation weights**:

```
W(z) ∝ exp(−z × ρ_b / λ_s)     λ_s = 162 g/cm²
```

This weighting mirrors the actual depth sensitivity of the neutron
signal: shallow layers contribute exponentially more than deeper ones,
matching the physical measurement.

Additional derived quantities:

- **Lattice water** (Köhli et al. 2021):
  `lw = 0.097 × (clay/100) + 0.033  [g/g]`
  Water bound to clay mineral lattice that does not contribute to free
  pore water and biases the neutron calibration if unaccounted for.
- **USDA texture class**: classified from the clay/sand/silt triangle
  (12 classes from Sand to Silty Clay).
- **WRB soil class**: retrieved directly from the SoilGrids API response
  (e.g., Cambisol, Podzol, Leptosol).

---

## Data Sources

| Data | Provider | Resolution | Period |
|------|----------|-----------|--------|
| Digital Elevation Model | Copernicus GLO-30 (AWS S3) | 30 m | — |
| Climate: radiation, temperature, wind, RH | PVGIS TMY (JRC/EC) | Point | 2005–2020 |
| Precipitation | Open-Meteo ERA5 | ~31 km | 2005–2020 |
| Soil properties | SoilGrids v2.0 (ISRIC) | 250 m | — |
| Geomagnetic rigidity cutoff | Smart & Shea (2019) via crnpy | Global grid | — |

DEM tiles are cached locally (`.npz` + `.json`) keyed by a SHA-256 hash
of coordinates, radius, and source — subsequent runs use the cache and
skip the download.

---

## Workflow

```
Input: LAT, LON, sensor height, bulk density, initial θ_v
          │
          ▼
  [1] Download & cache Copernicus GLO-30 DEM (2 km radius)
          │
          ▼
  [2] Compute r86, z86 from θ_v and local pressure
          │
          ▼
  [3] κ_topo — 3-D ray-casting over full DEM
          │
          ▼
  [4] Horizon angles — parallel KD-tree scan (2° azimuth steps)
          │
          ▼
  [5] κ_muon — analytical cos²(θ_z) integral over visible sky
          │
          ▼
  [6] Site fluxes — expected N_muon, N_neut, N0
          │
          ▼
  [7] Site climate — PVGIS TMY + Open-Meteo ERA5 precipitation
          │
          ▼
  [8] Soil properties — SoilGrids REST API + neutron weights
          │
          ▼
  [9] TWI — D8 depression-filled flow accumulation
          │
          ▼
 [10] Thermal index — lapse + cold pooling + PISR corrections
          │
          ▼
 [11] Neutron per-azimuth FOV — W(r)-weighted radial profiles
          │
          ▼
 [12] Report (crns_report.txt) + 4 figures (PNG)
```

---

## Outputs

### Text Report — `crns_report.txt`

A self-contained 72-column report covering all twelve workflow steps.
Key quantities reported:

- Site altitude, pressure, air density
- r86, z86, reference volume V0, effective volume V_eff
- κ_topo, κ_muon, κ_total with physical interpretation
- Soil moisture bias in percent and corrected θ_v
- Monthly climate tables (radiation, temperature, precipitation)
- Soil property profiles and CRNS-weighted averages with uncertainty
- TWI statistics and wetness-class distribution
- Corrected monthly temperature arrays with component-wise uncertainty

### Figures

**`crns_topo_main.png`** — Topographic overview.
DEM map with concentric r86 circles (0.5×, 1×, 1.5×); theoretical κ
curves for hilltop (1/cos³α, overestimate) and dolina (cos³α,
underestimate) geometries; E–W and N–S terrain cross-sections through
the sensor; summary bar chart of κ_topo / κ_muon / κ_total.

**`crns_footprint.png`** — Neutron footprint analysis.
2-D pixel classification map (deficit / partial / full soil
contribution) overlaid with W(r) contour shading; radial overlap
profile in 15 m shells; polar diagram of per-azimuth mean overlap
fraction showing which compass directions lose or gain soil volume.

**`crns_horizon.png`** — Horizon and muon FOV.
Polar plot of terrain horizon elevation angle ψ(φ) and per-azimuth
muon sensitivity fraction. The mean of the sensitivity plot equals κ_muon.

**`crns_fov_detail.png`** — Detailed FOV maps.
Azimuth bar chart of neutron soil contribution per compass direction
(immediately shows where the SM bias originates); 2-D heatmap of the
muon angular weight cos²(θ_z)·sin(θ_z) as function of azimuth and
elevation angle, with terrain-blocked regions greyed out and the
horizon profile overlaid.

---

## Configuration

All parameters are set at the top of `main.py`:

```python
LAT              = 46.2799      # decimal degrees WGS84
LON              = 11.8857      # decimal degrees WGS84
SENSOR_HEIGHT_M  = 2.0          # m above ground level
RHO_BULK         = 1.4          # g/cm³   soil bulk density
THETA_V_INIT     = 0.20         # m³/m³   initial soil moisture estimate
DEM_RADIUS_M     = 2000.0       # m       DEM download and analysis radius
AZIMUTH_STEP_DEG = 2.0          # deg     horizon scan resolution
OUTPUT_DIR       = "output"     # output subdirectory
DEM_SOURCE       = "copernicus_aws"
N_CORES          = 4            # parallel cores for horizon computation
```

---

## Key References

- Desilets, D., Zreda, M., & Ferré, T. P. A. (2010). Nature's neutron probe:
  Land surface hydrology at an elusive scale with cosmic rays.
  *Water Resources Research*, 46, W11cocite.
- Kohli, M., Schrön, M., Zreda, M., Schmidt, U., Dietrich, P., &
  Zacharias, S. (2015). Footprint characteristics revised for field-scale
  soil moisture monitoring with cosmic-ray neutrons.
  *Water Resources Research*, 51, 5772–5790.
- Köhli, M., et al. (2021). Soil moisture and air humidity dependence of
  the above-ground cosmic-ray neutron intensity.
  *Frontiers in Water*, 2, 544847.
- Hawdon, A., McJannet, D., & Wallace, J. (2014). Calibration and
  correction procedures for cosmic-ray neutron soil moisture probes
  located across Australia. *Water Resources Research*, 50, 5029–5043.
- Beven, K. J., & Kirkby, M. J. (1979). A physically based variable
  contributing area model of basin hydrology.
  *Hydrological Sciences Bulletin*, 24(1), 43–69.
- Wang, L., & Liu, H. (2006). An efficient method for identifying and
  filling surface depressions in digital elevation models.
  *International Journal of Geographical Information Science*, 20(2), 193–213.
- Faiman, D. (1994). Assessing the outdoor operating temperature of
  photovoltaic modules. *Progress in Photovoltaics*, 16(4).
- Smart, D. F., & Shea, M. A. (2019). Geomagnetic cutoff rigidity
  calculations for cosmic-ray studies. Various geomagnetic field models.
