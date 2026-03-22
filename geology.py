"""
geology.py
==========
Caratterizzazione geologica di un sito CRNS tramite Macrostrat API.

Macrostrat (https://macrostrat.org) è un database globale di unità
geologiche che copre la maggior parte del mondo a diverse scale.
È liberamente accessibile via REST API senza registrazione.

Endpoint usato:
    https://macrostrat.org/api/v2/geologic_units/map
    ?lat={lat}&lng={lon}&scale=large

Se l'endpoint non restituisce dati (copertura assente), si prova con
scale=medium e poi scale=small.

Informazioni estratte:
    - Litologia (lith): tipo di roccia dominante e secondaria
    - Età geologica: era e periodo
    - Nome dell'unità geologica
    - Ambiente di deposizione

Rilevanza per CRNS
------------------
La litologia influenza:
  1. La composizione chimica del suolo (quantità di minerali idrati come
     argille smectitiche, zeoliti, feldspatidi alterati) → lattice water
  2. Il tenore di uranio e torio per la radiazione di fondo gamma → rumore
  3. La produzione locale di neutroni da fissione spontanea (U, Th in graniti)
  4. La densità apparente del substrato roccioso per z86

Il modulo non corregge automaticamente i valori — fornisce informazione
qualitativa da integrare nella valutazione manuale del sito.

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import requests
import hashlib
import json
import os

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

MACROSTRAT_URL = "https://macrostrat.org/api/v2/geologic_units/map"
MACROSTRAT_SCALES = ["large", "medium", "small"]

# Impatto qualitativo sulla lattice water (lw) per litologia
# Valori indicativi — scala 1 (basso) → 3 (alto)
# Basati su mineralogia media di suoli sviluppati su quella roccia
LITH_LW_IMPACT = {
    "granite"         : 1,   # feldspati poco alterati
    "granodiorite"    : 1,
    "gneiss"          : 1,
    "schist"          : 2,
    "phyllite"        : 2,
    "shale"           : 3,   # alto tenore di argille
    "mudstone"        : 3,
    "claystone"       : 3,
    "siltstone"       : 2,
    "sandstone"       : 1,
    "limestone"       : 1,
    "dolostone"       : 1,
    "dolomite"        : 1,
    "basalt"          : 2,   # plagioclasi alterabili
    "andesite"        : 2,
    "rhyolite"        : 1,
    "tuff"            : 2,
    "till"            : 2,   # depositi glaciali misti
    "glacial"         : 2,
    "alluvium"        : 2,
    "colluvium"       : 2,
    "peat"            : 3,   # alto contenuto organico
    "loess"           : 2,
    "chalk"           : 1,
    "coal"            : 3,
    "evaporite"       : 1,
    "serpentinite"    : 2,
    "peridotite"      : 1,
    "unknown"         : 0,
}

# Radiogenicità relativa (produzione neutroni cosmogenici + fondo gamma)
# 1=bassa, 2=media, 3=alta
LITH_RADIOACTIVITY = {
    "granite"    : 3,   # alto U, Th
    "rhyolite"   : 3,
    "granodiorite": 2,
    "gneiss"     : 2,
    "schist"     : 2,
    "basalt"     : 1,
    "limestone"  : 1,
    "sandstone"  : 1,
    "shale"      : 2,
    "coal"       : 1,
    "unknown"    : 0,
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _geo_cache_path(cache_dir, lat, lon):
    tag = f"{lat:.5f}_{lon:.5f}"
    h   = hashlib.sha256(tag.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"geology_{h}.json")


def _save_geo_cache(result, cache_dir, lat, lon):
    os.makedirs(cache_dir, exist_ok=True)
    with open(_geo_cache_path(cache_dir, lat, lon), "w") as f:
        json.dump(result, f, indent=2)
    print(f"   Geology cached -> {_geo_cache_path(cache_dir, lat, lon)}", flush=True)


def _load_geo_cache(cache_dir, lat, lon):
    path = _geo_cache_path(cache_dir, lat, lon)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        print(f"   Geology from cache: {path}", flush=True)
        return d
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fetch Macrostrat
# ---------------------------------------------------------------------------

def _fetch_macrostrat(lat, lon, timeout_s=30):
    """
    Interroga Macrostrat API per le unità geologiche al punto (lat, lon).
    Prova 3 scale (large, medium, small) fino a ottenere dati.

    Returns
    -------
    list of dict con i campi dell'unità, oppure [] se nessun dato.
    """
    for scale in MACROSTRAT_SCALES:
        params = {
            "lat"   : round(lat, 5),
            "lng"   : round(lon, 5),
            "scale" : scale,
        }
        try:
            resp = requests.get(MACROSTRAT_URL, params=params, timeout=timeout_s)
            if resp.status_code != 200:
                continue
            data = resp.json()
            items = (data.get("success", {}).get("data", []) or
                     data.get("data", []))
            if items:
                return items, scale
        except Exception as e:
            print(f"   Macrostrat {scale}: {e}", flush=True)
    return [], "none"


def _parse_macrostrat_unit(unit):
    """
    Estrae i campi rilevanti da una unità Macrostrat.

    Returns
    -------
    dict semplificato
    """
    # Litologia: lista di dict con tipo, classe, colore
    liths = unit.get("liths", []) or []
    lith_names = [l.get("lith", "unknown") for l in liths if l.get("lith")]
    lith_types  = [l.get("lith_type", "") for l in liths if l.get("lith_type")]
    lith_classes = [l.get("lith_class", "") for l in liths if l.get("lith_class")]

    # Età stratigrafica
    t_age  = unit.get("t_age")   # top (giovane) [Ma]
    b_age  = unit.get("b_age")   # base (antico) [Ma]

    # Nome e descrizione
    name   = (unit.get("name") or unit.get("strat_name") or
               unit.get("unit_name") or "N/A")
    desc   = unit.get("descrip") or unit.get("comments") or ""

    # Ambiente deposizionale
    envs   = unit.get("environ") or []
    env_names = [e.get("environ", "") for e in envs if isinstance(e, dict)]

    return dict(
        unit_name    = str(name)[:80],
        description  = str(desc)[:200],
        lith_names   = lith_names,
        lith_types   = lith_types,
        lith_classes = lith_classes,
        environments = env_names,
        t_age_ma     = float(t_age) if t_age is not None else None,
        b_age_ma     = float(b_age) if b_age is not None else None,
    )


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def get_geology(lat, lon, cache_dir=None, timeout_s=30):
    """
    Caratterizzazione geologica del sito tramite Macrostrat.

    Parameters
    ----------
    lat, lon   : coordinate WGS84
    cache_dir  : directory cache locale
    timeout_s  : timeout HTTP [s]

    Returns
    -------
    dict con:
        units          : lista di unità geologiche (spesso 1-3)
        scale_used     : scala Macrostrat usata (large/medium/small/none)
        dominant_lith  : litologia dominante (stringa)
        lith_lw_impact : impatto sulla lattice water (0=ignoto, 1-3)
        lith_radioact  : radiogenicità relativa (0=ignoto, 1-3)
        age_era        : era geologica approssimativa
        source         : 'macrostrat_api' o 'not_available'
        coverage_note  : avvertimento sulla copertura geografica
    """
    if cache_dir is not None:
        cached = _load_geo_cache(cache_dir, lat, lon)
        if cached is not None:
            return cached

    print("   Geology: querying Macrostrat API ...", flush=True)
    items, scale = _fetch_macrostrat(lat, lon, timeout_s)

    if not items:
        result = dict(
            units          = [],
            scale_used     = "none",
            dominant_lith  = "unknown",
            lith_lw_impact = 0,
            lith_radioact  = 0,
            age_era        = "unknown",
            source         = "not_available",
            coverage_note  = ("Macrostrat non ha dati per questa area. "
                              "Consultare la carta geologica regionale."),
        )
        if cache_dir is not None:
            _save_geo_cache(result, cache_dir, lat, lon)
        return result

    units = [_parse_macrostrat_unit(u) for u in items[:5]]

    # Litologia dominante: quella più comune tra tutte le unità
    all_liths = [l.lower() for u in units for l in u["lith_names"]]
    dom_lith  = all_liths[0] if all_liths else "unknown"

    # Trova impatti nella lookup table (ricerca parziale)
    def _lookup(lith_str, table):
        for key, val in table.items():
            if key in lith_str:
                return val
        return 0

    lw_impact = _lookup(dom_lith, LITH_LW_IMPACT)
    radioact  = _lookup(dom_lith, LITH_RADIOACTIVITY)

    # Era geologica approssimativa
    t_age = next((u["t_age_ma"] for u in units if u["t_age_ma"] is not None), None)
    b_age = next((u["b_age_ma"] for u in units if u["b_age_ma"] is not None), None)
    age_era = _age_to_era(t_age, b_age)

    result = dict(
        units          = units,
        scale_used     = scale,
        dominant_lith  = dom_lith,
        lith_lw_impact = lw_impact,
        lith_radioact  = radioact,
        age_era        = age_era,
        source         = "macrostrat_api",
        coverage_note  = ("Macrostrat globale — qualità variabile. "
                          "Per siti europei verificare con carta geologica "
                          "regionale 1:50.000 o 1:100.000."),
    )
    if cache_dir is not None:
        _save_geo_cache(result, cache_dir, lat, lon)

    print(f"   Geology: {dom_lith}, scale={scale}, lw_impact={lw_impact}",
          flush=True)
    return result


def _age_to_era(t_age_ma, b_age_ma):
    """Converte età [Ma] in era geologica semplificata."""
    if t_age_ma is None:
        return "unknown"
    if t_age_ma <= 2.6:
        return "Quaternary"
    if t_age_ma <= 23:
        return "Neogene"
    if t_age_ma <= 66:
        return "Paleogene"
    if t_age_ma <= 145:
        return "Cretaceous"
    if t_age_ma <= 201:
        return "Jurassic"
    if t_age_ma <= 252:
        return "Triassic"
    if t_age_ma <= 299:
        return "Permian"
    if t_age_ma <= 359:
        return "Carboniferous"
    return f"Pre-Carboniferous (>{t_age_ma:.0f} Ma)"


# ---------------------------------------------------------------------------
# Report testuale
# ---------------------------------------------------------------------------

_LW_LABELS = {0: "ignoto", 1: "basso", 2: "medio", 3: "alto"}
_RA_LABELS = {0: "ignoto", 1: "bassa", 2: "media", 3: "alta"}


def report_geology(res):
    w = 72
    L = [
        "=" * w,
        "GEOLOGY (Macrostrat API)",
        "=" * w,
        f"  Scala usata      : {res['scale_used']}",
        f"  Sorgente         : {res['source']}",
        f"  Litologia dom.   : {res['dominant_lith']}",
        f"  Era geologica    : {res['age_era']}",
        f"  Impatto su lw    : {_LW_LABELS.get(res['lith_lw_impact'],'?')} "
        f"(0=ignoto, 1=basso, 3=alto)",
        f"  Radiogenicità    : {_RA_LABELS.get(res['lith_radioact'],'?')} "
        f"(U/Th nei minerali)",
        f"  Nota             : {res['coverage_note']}",
        "",
    ]
    if res['units']:
        L.append("  Unità geologiche trovate:")
        for i, u in enumerate(res['units'][:3], 1):
            L.append(f"    [{i}] {u['unit_name']}")
            if u['lith_names']:
                L.append(f"        Litologia: {', '.join(u['lith_names'][:4])}")
            if u['environments']:
                L.append(f"        Ambiente : {', '.join(u['environments'][:3])}")
            if u['t_age_ma'] is not None:
                L.append(f"        Età      : {u['b_age_ma']:.0f}–{u['t_age_ma']:.0f} Ma")
            if u['description']:
                L.append(f"        Descrip. : {u['description'][:100]}")
    else:
        L.append("  Nessun dato geologico disponibile per questo sito.")
        L.append("  Consultare la carta geologica regionale.")
    L.append("=" * w)
    return "\n".join(L)
