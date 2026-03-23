"""
config_parser.py
================
Parser per file di configurazione sito CRNS.

Formato:
    # commento
    VAR_NUM   = tipo, valore, err+, err-, unita, descrizione
    VAR_STRG  = STRG, "stringa", descrizione
    VAR_TUPLE = TUPLE, dim, [[...]], descrizione

Tipi numerici (case insensitive): INT, DBLE, REAL, CPLX
Tipo stringa : STRG
Tipo tupla   : TUPLE

Dimensione tupla: n1;n2;n3  (es. "3;4" per array 3x4)

Tupla contenuto: lista annidata JSON-like, ma:
  - complessi scritti con 'i' invece di 'j'  (es. 3i+4, 2i)
  - stringhe interne con doppi apici

Esempio file:
    OPENCELLID_TOKEN = STRG, "abc123", "Token OpenCelliD"
    LAT   = DBLE, 46.9255, 0.0001, 0.0001, deg, "Latitudine WGS84"
    N0    = INT,  1200, 50, 50, cph, "Conteggio N0"
    BBOX  = TUPLE, 4, [11.80, 46.90, 11.92, 46.96], "Bounding box"
    MAT2D = TUPLE, 2;3, [[1,2,3],[4,5,6]], "Matrice 2x3"

Author      : MB
Affiliation :
Email       : mauro.barbieri@pm.me
"""

import re
import numpy as np

# Tipi numerici riconosciuti
_NUM_TYPES = {"int", "dble", "real", "cplx"}
_PYTYPE    = {"int": int, "dble": float, "real": float, "cplx": complex}


# ---------------------------------------------------------------------------
# Conversione singolo valore scalare
# ---------------------------------------------------------------------------

def _parse_scalar(s, typ):
    """
    Converte stringa s nel tipo Python corrispondente a typ.
    typ: 'int' | 'dble' | 'real' | 'cplx'
    """
    s = s.strip()
    if typ == "int":
        return int(s)
    if typ in ("dble", "real"):
        return float(s)
    if typ == "cplx":
        return _parse_complex(s)
    raise ValueError(f"Tipo sconosciuto: '{typ}'")


def _parse_complex(s):
    """
    Parsa notazione Fortran con 'i' per parte immaginaria.
    Supporta: 3i+4, 4+3i, 3i, -2i, 4.0, 3.5i+4.2, 1e2i+3
    """
    s = s.strip().replace(" ", "")
    _num = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
    _pos = r'(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'

    # Forma im+re: 3i+4, 3.5i-2.1
    m = re.fullmatch(
        rf'({_num})i([+-]{_pos})', s)
    if m:
        return complex(float(m.group(2)), float(m.group(1)))

    # Forma re+im*i: 4+3i, 4.2+3.5i
    m = re.fullmatch(
        rf'({_num})([+-]{_pos})i', s)
    if m:
        return complex(float(m.group(1)), float(m.group(2)))

    # Puro immaginario: 3i, -2i
    m = re.fullmatch(rf'({_num})i', s)
    if m:
        return complex(0, float(m.group(1)))

    # Puro reale: 4, 3.14
    try:
        return complex(float(s), 0)
    except ValueError:
        raise ValueError(
            f"Formato complesso non riconosciuto: '{s}'")


# ---------------------------------------------------------------------------
# Parser della tupla
# ---------------------------------------------------------------------------

def _parse_tuple_content(s, typ_hint=None):
    """
    Parsa una stringa che rappresenta una lista annidata.
    Supporta: interi, float, complessi (con 'i'), stringhe con "".
    Ritorna oggetto Python (list annidata).

    Strategia: tokenizzazione manuale per gestire complessi e stringhe.
    """
    s = s.strip()

    # ---- Stringa ----
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]

    # ---- Lista annidata ----
    if s.startswith('[') and s.endswith(']'):
        inner   = s[1:-1].strip()
        if not inner:
            return []
        items   = _split_list(inner)
        return [_parse_tuple_content(it.strip(), typ_hint)
                for it in items]

    # ---- Scalare ----
    s = s.strip()

    # Stringa interna
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]

    # Complesso (contiene 'i' non preceduto o seguito da lettera)
    if re.search(r'(?<![a-zA-Z])i(?![a-zA-Z])', s):
        try:
            return _parse_complex(s)
        except ValueError:
            pass

    # Intero
    try:
        if '.' not in s and 'e' not in s.lower():
            return int(s)
    except ValueError:
        pass

    # Float
    try:
        return float(s)
    except ValueError:
        pass

    # Fallback: stringa grezza
    return s


def _split_list(s):
    """
    Divide una stringa CSV rispettando le parentesi [] annidate
    e le stringhe tra "".
    Es: '[1,2],[3,4],"a,b",5'  ->  ['[1,2]', '[3,4]', '"a,b"', '5']
    """
    items   = []
    depth   = 0
    in_str  = False
    current = []

    for ch in s:
        if ch == '"' and depth == 0:
            in_str = not in_str
            current.append(ch)
        elif in_str:
            current.append(ch)
        elif ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ']':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            items.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        items.append(''.join(current).strip())

    return [it for it in items if it]


def _to_numpy(nested, shape):
    """
    Converte lista annidata in numpy array con la shape indicata.
    Sceglie dtype automaticamente.
    """
    flat = _flatten(nested)
    if not flat:
        return np.array([])

    # Dtype — str ha priorità assoluta (tupla mista)
    if any(isinstance(v, str) for v in flat):
        dtype = object
    elif any(isinstance(v, complex) for v in flat):
        dtype = complex
    elif all(isinstance(v, int) for v in flat):
        dtype = int
    else:
        dtype = float

    arr = np.array(flat, dtype=dtype)
    try:
        return arr.reshape(shape)
    except ValueError:
        return arr   # restituisce flat se shape incompatibile


def _flatten(lst):
    """Appiattisce lista annidata arbitrariamente profonda."""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Parser di una singola riga
# ---------------------------------------------------------------------------

def _parse_line(line):
    """
    Parsa una riga 'NOME = ...'.
    Ritorna (nome, record) dove record è un dict con i campi parsed.
    Lancia ValueError se la riga non è valida.
    """
    # Separa nome e valore
    if '=' not in line:
        raise ValueError(f"Nessun '=' nella riga: '{line}'")

    name_part, value_part = line.split('=', 1)
    name = name_part.strip()
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
        raise ValueError(f"Nome variabile non valido: '{name}'")

    # Estrai il tipo come primo token prima della virgola
    # ma attenzione: la tupla ha '[' con virgole interne
    # -> usa _split_top_level per separare i campi al top level
    fields = _split_top_level(value_part.strip())

    if not fields:
        raise ValueError(f"Nessun campo dopo '=' per '{name}'")

    typ = fields[0].strip().lower()

    # ---- STRG ----
    if typ == "strg":
        # STRG, "valore", descrizione
        if len(fields) < 2:
            raise ValueError(f"STRG '{name}': manca il valore")
        raw_val = fields[1].strip()
        if raw_val.startswith('"') and raw_val.endswith('"'):
            val = raw_val[1:-1]
        else:
            val = raw_val
        desc = fields[2].strip().strip('"') if len(fields) > 2 else ""
        return name, {
            "type": "STRG", "value": val,
            "description": desc,
        }

    # ---- TUPLE ----
    if typ == "tuple":
        # TUPLE, dim, [[...]], descrizione
        if len(fields) < 3:
            raise ValueError(f"TUPLE '{name}': formato incompleto")
        dim_str  = fields[1].strip()
        arr_str  = fields[2].strip()
        desc     = fields[3].strip().strip('"') if len(fields) > 3 else ""

        # Dimensioni: "3;4" -> (3, 4)
        try:
            shape = tuple(int(d) for d in dim_str.split(';') if d.strip())
        except ValueError:
            raise ValueError(
                f"TUPLE '{name}': dimensione '{dim_str}' non valida")

        nested = _parse_tuple_content(arr_str)
        arr    = _to_numpy(nested, shape)

        return name, {
            "type": "TUPLE", "value": arr,
            "shape": shape, "description": desc,
        }

    # ---- Tipi numerici ----
    if typ in _NUM_TYPES:
        # tipo, valore, err+, err-, unita, descrizione
        if len(fields) < 2:
            raise ValueError(
                f"Numerico '{name}': manca il valore")

        conv = _PYTYPE[typ]
        try:
            val = _parse_scalar(fields[1].strip(), typ)
        except Exception as e:
            raise ValueError(
                f"'{name}': valore '{fields[1]}' non convertibile: {e}")

        err_pos = None
        err_neg = None
        unit    = ""
        desc    = ""

        if len(fields) >= 3:
            try:
                err_pos = _parse_scalar(fields[2].strip(), typ)
            except Exception:
                err_pos = None

        if len(fields) >= 4:
            try:
                err_neg = _parse_scalar(fields[3].strip(), typ)
            except Exception:
                err_neg = None

        if len(fields) >= 5:
            unit = fields[4].strip().strip('"')

        if len(fields) >= 6:
            desc = fields[5].strip().strip('"')

        return name, {
            "type"       : typ.upper(),
            "value"      : val,
            "err_pos"    : err_pos,
            "err_neg"    : err_neg,
            "unit"       : unit,
            "description": desc,
        }

    raise ValueError(
        f"Tipo '{fields[0]}' non riconosciuto per '{name}'")


def _split_top_level(s):
    """
    Divide s per virgole al livello zero (non dentro [] o "").
    Restituisce lista di stringhe.
    """
    items   = []
    depth   = 0
    in_str  = False
    current = []

    for ch in s:
        if ch == '"' and depth == 0:
            in_str = not in_str
            current.append(ch)
        elif in_str:
            current.append(ch)
        elif ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ']':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0 and not in_str:
            items.append(''.join(current))
            current = []
        else:
            current.append(ch)

    if current:
        items.append(''.join(current))

    return items


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def load_config(filepath):
    """
    Legge un file di configurazione CRNS e ritorna un dict.

    Chiavi: nome variabile (stringa)
    Valori: dict con almeno 'type' e 'value', più campi opzionali
            (err_pos, err_neg, unit, description, shape).

    Accesso rapido al valore: config['VAR']['value']

    Lancia FileNotFoundError se il file non esiste.
    Lancia ValueError con messaggio descrittivo per errori di parsing.
    """
    if not isinstance(filepath, str):
        filepath = str(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: '{filepath}'")

    config = {}
    errors = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # Salta righe vuote e commenti
            if not line or line.startswith('#'):
                continue

            try:
                name, record = _parse_line(line)
                config[name] = record
            except ValueError as e:
                errors.append(f"  Line {lineno}: {e}")

    if errors:
        raise ValueError(
            f"Errori nel file '{filepath}':\n" + "\n".join(errors))

    return config


def get(config, name, default=None):
    """Accesso rapido: ritorna il valore di config[name]['value']."""
    if name not in config:
        return default
    return config[name]["value"]


def summary(config):
    """Stampa un sommario leggibile della configurazione."""
    w = 60
    lines = ["="*w, "Configuration summary", "="*w]
    for name, rec in config.items():
        typ = rec["type"]
        val = rec["value"]
        desc= rec.get("description","")
        unit= rec.get("unit","")

        if typ == "STRG":
            lines.append(f"  {name:<24} = \"{val}\"")
        elif typ == "TUPLE":
            shp = rec.get("shape","?")
            lines.append(f"  {name:<24} = TUPLE{shp}  dtype={val.dtype}")
        else:
            ep  = rec.get("err_pos")
            en  = rec.get("err_neg")
            u   = f" [{unit}]" if unit else ""
            err = (f" +{ep}/-{en}" if ep is not None else "")
            lines.append(f"  {name:<24} = {val}{err}{u}")

        if desc:
            lines.append(f"  {'':24}   # {desc}")

    lines.append("="*w)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

import os

if __name__ == "__main__":
    import tempfile

    TEST_CONFIG = """
# Test file configurazione CRNS
OPENCELLID_TOKEN = STRG, "abc123xyz", "Token OpenCelliD"
SITE_NAME        = STRG, "Malga Fadner", "Nome del sito"
LAT              = DBLE, 46.9255, 0.0001, 0.0001, deg, "Latitudine WGS84"
LON              = DBLE, 11.8614, 0.0001, 0.0001, deg, "Longitudine WGS84"
ALT              = REAL, 1102.0, 2.0, 2.0, m, "Quota DEM mediana"
N0               = INT,  1200, 50, 50, cph, "Conteggio N0"
RHO_BULK         = DBLE, 1.40, 0.05, 0.05, g/cm3, "Densita bulk suolo"
FREQ_COMPLEX     = CPLX, 3i+4, 0.1, 0.1, Hz, "Frequenza complessa test"
BBOX             = TUPLE, 4, [11.80, 46.90, 11.92, 46.96], "Bounding box"
MATRIX2D         = TUPLE, 2;3, [[1,2,3],[4,5,6]], "Matrice intera 2x3"
MIXED_TUPLE      = TUPLE, 3, [1.5, 2i+1, "hello"], "Tupla mista"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg',
                                      delete=False) as f:
        f.write(TEST_CONFIG)
        fname = f.name

    try:
        cfg = load_config(fname)
        print(summary(cfg))
        print()

        # Verifica valori
        assert get(cfg,"OPENCELLID_TOKEN") == "abc123xyz"
        assert get(cfg,"SITE_NAME")        == "Malga Fadner"
        assert abs(get(cfg,"LAT") - 46.9255) < 1e-9
        assert get(cfg,"N0") == 1200
        assert isinstance(get(cfg,"N0"), int)
        assert abs(get(cfg,"FREQ_COMPLEX") - complex(4,3)) < 1e-9
        assert cfg["LAT"]["err_pos"]  == 0.0001
        assert cfg["LAT"]["unit"]     == "deg"
        assert cfg["BBOX"]["value"].shape == (4,)
        assert cfg["MATRIX2D"]["value"].shape == (2,3)
        assert cfg["MATRIX2D"]["value"][1,2]  == 6

        print("All tests PASSED")

    finally:
        os.unlink(fname)
