"""
net_utils.py
============
HTTP helpers con retry automatico su errori transitori.

Transient codes: 429, 500, 502, 503, 504 + ConnectionError + Timeout.
Strategia: retry fino a `retries` volte con backoff 2s, 4s, 8s, …

Uso:
    from net_utils import http_get, http_post

    resp = http_get(url, params={...})   # ritorna Response o None
    if resp is None:
        # fallback
    data = resp.json()

Author : MB
"""

import time
import requests

TRANSIENT_CODES = {429, 500, 502, 503, 504}
_DEFAULT_RETRIES = 3


def http_get(url, params=None, timeout=120, retries=_DEFAULT_RETRIES,
             verbose=True, **kwargs):
    """
    GET con retry su errori transitori.
    Solleva HTTPError su codici non transitori (4xx, 501, …).
    Ritorna None solo se tutti i retry falliscono su errori transitori
    o su ConnectionError / Timeout.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, **kwargs)
            if resp.status_code in TRANSIENT_CODES and attempt < retries:
                delay = 2 ** attempt
                if verbose:
                    print(f"   [WARN] HTTP {resp.status_code} — retry "
                          f"{attempt+1}/{retries} in {delay}s ...", flush=True)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < retries:
                delay = 2 ** attempt
                if verbose:
                    print(f"   [WARN] {type(exc).__name__} — retry "
                          f"{attempt+1}/{retries} in {delay}s ...", flush=True)
                time.sleep(delay)
            else:
                raise
        except requests.exceptions.HTTPError:
            raise
    raise last_exc


def http_post(url, data=None, timeout=120, retries=_DEFAULT_RETRIES,
              verbose=True, **kwargs):
    """
    POST con retry su errori transitori.
    Stessa semantica di http_get.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, data=data, timeout=timeout, **kwargs)
            if resp.status_code in TRANSIENT_CODES and attempt < retries:
                delay = 2 ** attempt
                if verbose:
                    print(f"   [WARN] HTTP {resp.status_code} — retry "
                          f"{attempt+1}/{retries} in {delay}s ...", flush=True)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < retries:
                delay = 2 ** attempt
                if verbose:
                    print(f"   [WARN] {type(exc).__name__} — retry "
                          f"{attempt+1}/{retries} in {delay}s ...", flush=True)
                time.sleep(delay)
            else:
                raise
        except requests.exceptions.HTTPError:
            raise
    raise last_exc
