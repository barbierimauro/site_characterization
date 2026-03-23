# CRNS Site Characterization Tool — Analisi del Codice

**Branch:** `claude/analyze-new-branch-qIwOJ`
**Data analisi:** 2026-03-23
**Analizzato da:** Claude Code (claude-sonnet-4-6)

---

## 1. Scopo e Dominio

Il progetto è un toolkit scientifico Python per la **caratterizzazione completa di siti di installazione di sensori CRNS** (Cosmic-Ray Neutron Sensor). Trasforma i conteggi grezzi di neutroni in stime accurate di umidità del suolo calcolando correzioni topografiche e parametri ambientali site-specific.

**Dominio scientifico:** Monitoraggio dell'umidità del suolo tramite fisica dei raggi cosmici. I CRNS misurano neutroni epitermali moderati dall'idrogeno nell'acqua del suolo, basandosi sulla teoria di Desilets et al. (2010) e correzioni 3-D moderne.

---

## 2. Architettura e Struttura dei Moduli

### Moduli di Fisica

| Modulo | Funzione | Algoritmo chiave |
|--------|----------|-----------------|
| `kappa_topo_3d.py` | Correzione volume topografico | Ray-casting 3-D con attenuazione esponenziale, pesi cos²θ, normali locali alla superficie |
| `site_fluxes.py` | Tassi di conteggio neutroni/muoni, calcolo N0 | Scalatura baroscopica (β_neut=2.3, β_muon=1.15), rigidità geomagnetica (Hawdon 2014) |
| `terrain_indices.py` | TWI e correzioni termiche | Accumulo D8 (Wang & Liu 2006), PISR, pool di aria fredda, SVF |
| `crns_corrections.py` | Correzioni supplementari | Vapore acqueo (WV), biomassa aerea (AGBH), neve (SWE) |
| `smphysics.py` | Fisica dell'umidità del suolo | PTF Saxton & Rawls, rescaling pedologico/topografico/LULC |

### Moduli di Acquisizione Dati

| Modulo | Fonte dati | Risoluzione |
|--------|-----------|-------------|
| `site_climate.py` | PVGIS TMY + Open-Meteo ERA5 | Oraria / 31 km griglia |
| `vegetation_indices.py` | Landsat 8/9 L2 + MODIS MCD15A3H (Planetary Computer) | 30 m / 500 m |
| `get_soil_properties.py` | SoilGrids v2.0 (ISRIC REST API) | 250 m, 6 strati di profondità |
| `lulc.py` | ESA WorldCover 10m + OSM Overpass | 10 m, fattori f_H per idrogeno equivalente |
| `geology.py` | Macrostrat v2 API | Globale, litologia, età, ambiente deposizionale |
| `water.py` | JRC Global Surface Water v1.4 | 30 m, occorrenza 1984–2021 |
| `era5sm.py` | Open-Meteo ERA5-Land | Oraria, 3 strati di suolo, ~31 km |
| `radiofreq.py` | OpenCelliD + OSM + DEM-90 | Interferenza RF, potenza del segnale cellulare |

### Moduli di Output

| Modulo | Funzione |
|--------|----------|
| `config_parser.py` | Configurazione da file (tipi: numerici, stringhe, tuple) |
| `sampling_plan.py` | Piano ottimale di campionamento del suolo basato su W(r) |
| `plots.py` | 6+ figure di pubblicazione (DEM, footprint, orizzonte, FOV, clima, suolo) |
| `vegetation_plots.py` | Cicli stagionali, serie temporali, mappe 2D degli indici di vegetazione |
| `reports.py` | Report testuale strutturato a 72 colonne (12 sezioni) |
| `main.py` | Pipeline di orchestrazione, configurazione, strategia di caching |

---

## 3. Algoritmi Chiave

### A. Correzione Topografica — `kappa_topo_3d.py`

**Ray-casting fisico 3-D:**
1. Griglia angolare: azimuth φ (0–360°), elevazione θ (−90° a +90°), radiale r (0 a r₈₆)
2. Direzione del raggio: (cos θ sin φ, cos θ cos φ, sin θ)
3. Fisica dell'attenuazione:
   - L_air = distanza del raggio al primo impatto DEM [m]
   - L_soil = z₈₆ / cos(angolo di incidenza rispetto alla normale locale)
   - dN ∝ exp(−L_air / λ_air) × [1 − exp(−L_soil / λ_soil)] × cos²θ
4. Risultato: κ_topo = N_osservato / N_riferimento

**Miglioramenti vs. metodi semplificati:**
- Normali locali alla superficie → gestione corretta di cliff/terreni ripidi
- Peso cos²θ (non cos θ) → emissione Lambertiana + elemento di angolo solido
- λ_air(altitude) → correzione baroscopica alla lunghezza di attenuazione

### B. Footprint e Peso Radiale (Kohli et al. 2015)
```
r₈₆ = (A₁ / (θ_v + A₂)) × (P₀ / P)
W(r) = exp(−r / λ),  λ = r₈₆ / 3
```

### C. TWI — `terrain_indices.py`
1. Riempimento depressioni: algoritmo priority-queue O(n log n) (Wang & Liu 2006)
2. Direzione di flusso D8: discesa più ripida tra 8 vicini
3. Accumulazione di flusso
4. TWI: ln(specific_drainage_area / tan(slope))

### D. Correzioni Termiche (3 componenti additive)
1. **Lapse rate:** ΔT = −γ × (z_site − z_ERA5), γ = 6.5×10⁻³ °C/m
2. **Cold-air pooling:** ΔT = −K × (1 − SVF) × tanh(concavity / 50)
3. **Insolazione locale (PISR):** ΔT = α × (PISR_site − PISR_ERA5) / PISR_ERA5

---

## 4. Dipendenze Esterne

### Servizi Cloud (COG / STAC API)
- **Copernicus GLO-30 DEM:** AWS S3, 30 m, globale
- **Planetary Computer STAC:** Landsat Collection 2 L2, MODIS MCD15A3H
- **JRC Global Surface Water v1.4:** Google Cloud Storage
- **ESA WorldCover 2021:** Planetary Computer, 10 m

### REST API (pubbliche, senza auth salvo indicato)
- **PVGIS TMY:** JRC/Commissione Europea
- **Open-Meteo Archive:** ERA5-Land, gratuito
- **SoilGrids v2.0:** ISRIC, 250 m
- **Macrostrat v2:** Geologia, gratuito
- **OpenCelliD:** Torri cellulari (richiede registrazione)
- **OSM Overpass:** Infrastrutture

---

## 5. Qualità del Codice — Problemi Identificati

### Problemi Critici

#### 1. Bug import `os` in `config_parser.py`
```python
# Linea 398: os.path.exists() chiamato PRIMA che os sia importato (linea 465)
# Causa: NameError: name 'os' is not defined
# Fix: spostare import os in cima al file
```

#### 2. Validazione `rasterio` incompleta in `main.py`
```python
HAS_RASTERIO = False  # linee 111-115
# ... ma rasterio_merge() viene chiamata a linea 249 senza controllo HAS_RASTERIO
# Fix: aggiungere controllo esplicito prima delle chiamate rasterio
```

#### 3. DEM sintetico crea dati falsi senza avviso
```python
# Se il download Copernicus fallisce, si usa un DEM piatto → κ_topo ≈ 1.0 FALSO
# Non c'è flag nell'output per avvisare l'utente
# Fix: aggiungere warning prominente e flag nei risultati
```

#### 4. Mancanza di `requirements.txt`
- Dipendenze obbligatorie (rasterio, pvlib, crnpy, planetary_computer, pystac_client) non dichiarate
- Fix: creare `requirements.txt` con versioni pinned

### Problemi Moderati

#### 5. Coordinate hardcoded in `main.py` (linee 26–59)
```python
# Più siti commentati sovrascrivono LON/LAT sequenzialmente
# Facile eseguire il sito sbagliato per errore
# Fix: usare esclusivamente file config o argomenti CLI
```

#### 6. Precisione float nelle chiavi di cache (`terrain_indices.py`)
```python
# Cache key usa f"{np.nanmean(elev):.4f}" → solo 4 decimali
# Due siti vicini con stessa media elevazione ±0.00005 m → collisione cache
# Fix: usare 6+ decimali o includere bounding box geografico
```

#### 7. Nessuna validazione coordinate di ingresso
```python
# Accetta qualsiasi float; non controlla bounds WGS84 (±90 lat, ±180 lon)
# Fix: assert -90 <= lat <= 90 and -180 <= lon <= 180
```

#### 8. Rischio OOM nel download parallelo Landsat
```python
# vegetation_indices.py: PARALLEL_WINDOW=9 scene contemporanee
# 9 scene × 200 MB = 1.8 GB in memoria simultaneamente
# Fix: ridurre PARALLEL_WINDOW o usare memory-mapped arrays
```

### Code Smell / Stile

#### 9. Commenti in italiano misti a docstring in inglese
- `terrain_indices.py`: "Depressioni rimosse", "Peso radiale"
- Fix: standardizzare a inglese per codice scientifico internazionale

#### 10. Magic numbers dispersi
- `KOHLI_A1 = 29.13`, `DESILETS_ALPHA = 0.0564` presenti in più file
- Fix: creare `constants.py` centralizzato per tutti i parametri fisici

#### 11. Gestione errori non coerente
- Mix di `print(..., flush=True)` e `raise Exception`
- Fix: usare il modulo `logging` con livelli (DEBUG, INFO, WARNING, ERROR)

#### 12. Nessun test unitario
- Solo `config_parser.py` ha uno smoke test nel blocco `__main__`
- Fix: aggiungere test per moduli fisici (`test_kappa_topo_3d.py`, `test_site_fluxes.py`)

### Problemi di Dati / Edge Case

#### 13. Propagazione NaN nei valori pesati
- Se TUTTI i valori sono NaN, la media pesata ritorna NaN senza warning
- Il codice downstream può fallire silenziosamente (es. divisione per zero nella formula Desilets)

#### 14. `water.py` non integrato in κ_topo
- Il fattore η di riduzione acqua è applicato post-hoc
- Se il footprint è parzialmente inondato, il vero "volume suolo" è diverso ma κ_topo lo ignora

#### 15. Chiavi API inline
```python
# main.py linea 66: OPENTOPO_API_KEY = ""
# Rischio: sviluppatore potrebbe scrivere la chiave inline e committarla
# Fix: usare variabili d'ambiente o file .env (con .gitignore)
```

---

## 6. Riepilogo per File

| File | Stato | Note principali |
|------|-------|-----------------|
| `main.py` | ⚠️ RICHIEDE FIX | Coordinate hardcoded, check rasterio incompleto |
| `config_parser.py` | ✓ BUONO | Ben strutturato; bug `import os` da correggere |
| `kappa_topo_3d.py` | ✓ ECCELLENTE | Ray-casting fisicamente rigoroso; mancano test |
| `site_fluxes.py` | ✓ BUONO | Correzioni barometriche e rigidità corrette |
| `site_climate.py` | ✓ BUONO | Integrazione PVGIS + OpenMeteo solida |
| `terrain_indices.py` | ⚠️ MODERATO | TWI/termico corretto; precisione cache debole; commenti IT |
| `vegetation_indices.py` | ⚠️ MODERATO | Fast-reject GDAL intelligente; rischio OOM; cache non rivalidata |
| `get_soil_properties.py` | ✓ BUONO | Integrazione SoilGrids; peso CRNS corretto |
| `lulc.py` | ✓ BUONO | WorldCover + OSM; fattori f_H ben referenziati |
| `water.py` | ⚠️ MODERATO | Lettura JRC elegante; η non integrato in κ_topo |
| `geology.py` | ✓ BUONO | Macrostrat; fallback di scala sensato |
| `era5sm.py` | ✓ BUONO | Download anno per anno + cache; tracking metadata |
| `smphysics.py` | ✓ BUONO | PTF e correzioni chiare |
| `sampling_plan.py` | ✓ BUONO | Disegno ottimale degli anelli; visualizzazione r86 |
| `plots.py` | ✓ BUONO | 6+ figure di pubblicazione; stile coerente |
| `vegetation_plots.py` | ✓ BUONO | Cicli stagionali, serie temporali, mappe |
| `reports.py` | ✓ BUONO | Formato 72 colonne; sezioni complete |
| `crns_corrections.py` | ✓ BUONO | Correzioni WV, AGBH, SWE ben documentate |
| `radiofreq.py` | ✓ BUONO | Analisi interferenza RF; ray-marching viewshed |

---

## 7. Raccomandazioni Prioritizzate

### Priorità 1 — Fix Immediati
1. Spostare `import os` in cima a `config_parser.py`
2. Aggiungere controllo `HAS_RASTERIO` prima delle chiamate rasterio in `main.py`
3. Aggiungere warning esplicito se DEM sintetico viene usato
4. Creare `requirements.txt` con tutte le dipendenze e versioni pinned

### Priorità 2 — Robustezza
5. Rifattorizzare coordinate hardcoded → solo da config file / CLI
6. Aggiungere validazione coordinate (±90 lat, ±180 lon)
7. Implementare modulo `logging` al posto del mix print/exception
8. Aggiungere test unitari per moduli fisici

### Priorità 3 — Miglioramenti
9. Integrare `water.py` (η factor) nel calcolo di κ_topo (maschera pixel inondati)
10. Aggiungere opzione force-recompute nella cache vegetazione
11. Creare `constants.py` centralizzato per parametri fisici
12. Standardizzare la lingua dei commenti all'inglese

---

## 8. Conclusioni

Questo è un **tool scientifico ben ingegnerizzato**:
- ✅ Fisica corretta (ray-casting, attenuazione, pesatura)
- ✅ Acquisizione dati moderna (Cloud COG, STAC API, caching)
- ✅ Struttura modulare chiara (fisica / dati / visualizzazione)
- ✅ Risultati riproducibili (hash deterministico, config persistente)
- ✅ Figure di qualità pubblicazione

**Caveat:** Per uso in produzione, è necessario correggere il bug di import, aggiungere gestione delle dipendenze e spostare la configurazione delle coordinate fuori dal codice sorgente.

---

*Analisi generata da Claude Code — [segnala problemi](https://github.com/anthropics/claude-code/issues)*
