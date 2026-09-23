#!/usr/bin/env python
"""
sre_psgc_crosswalk.py
=====================
Attach a PSGC code to every row of the SRE panel.

Inputs
------
    data/processed/sre_panel.parquet
    data/processed/psgc_lgu_master.parquet

Outputs
-------
    data/processed/sre_psgc_crosswalk.parquet / .csv
    data/processed/sre_panel_with_psgc.parquet / .csv
    data/processed/sre_unmatched.csv
    data/processed/sre_unmatched_ranked.csv
    data/processed/sre_unmatched_diagnostic.txt   (--diagnose)
    data/processed/sre_psgc_crosswalk.log
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SRE  = Path("data/processed/sre_panel.parquet")
DEFAULT_PSGC = Path("data/processed/psgc_lgu_master.parquet")
DEFAULT_OUT  = Path("data/processed")

LOG = logging.getLogger("crosswalk")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_WS_RE      = re.compile(r"\s+")
_PUNCT_RE   = re.compile(r"[^a-z0-9 ]+")
_PAREN_RE   = re.compile(r"\([^)]*\)")
_PAREN_INNER_RE = re.compile(r"\(([^)]+)\)")
_FOOTNOTE_RE = re.compile(r"[\*\u2020\u2021]+\s*$")

CITY_PREFIXES = ("city of ", "lungsod ng ", "dakbayan sa ")
MUN_PREFIXES  = ("municipality of ", "mun of ", "bayan ng ", "munisipalidad ng ")
CITY_SUFFIXES = (" city",)
MUN_SUFFIXES  = (" municipality",)

ABBREV = {
    "sta": "santa", "sto": "santo", "gen": "general",
    "gov": "governor", "pres": "president", "mt": "mountain", "st": "santo",
}


def _clean(name) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).replace("\u00a0", " ").replace("牋", " ")
    s = unicodedata.normalize("NFKC", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _FOOTNOTE_RE.sub("", s).strip()
    return s


def _strip_affixes(name: str) -> str:
    for p in CITY_PREFIXES + MUN_PREFIXES:
        if name.startswith(p):
            name = name[len(p):].strip()
            break
    for sfx in CITY_SUFFIXES + MUN_SUFFIXES:
        if name.endswith(sfx):
            name = name[: -len(sfx)].strip()
            break
    return name


def _ascii_lower(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii").lower()


def make_key(name) -> str:
    s = _clean(name)
    s = _PAREN_RE.sub(" ", s).strip()
    s = _ascii_lower(s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _strip_affixes(s)
    s = _WS_RE.sub(" ", s).strip()
    return " ".join(ABBREV.get(t, t) for t in s.split())


def lgu_type_norm(t: str) -> str:
    t = str(t).strip().lower()
    if t.startswith("prov"): return "province"
    if t.startswith("cit"):  return "city"
    if t.startswith("mun"):  return "municipality"
    return t


# ---------------------------------------------------------------------------
# Province aliases
# ---------------------------------------------------------------------------

PROVINCE_ALIASES: dict[str, list[str]] = {
    # ---- SRE/BLGF abbreviations AND typos ----
    "zamb del norte":       ["zamboanga del norte"],
    "zamb del sur":         ["zamboanga del sur"],
    "zamboanga norte":      ["zamboanga del norte"],
    "zamboanga sur":        ["zamboanga del sur"],
    "western samar":        ["samar"],
    "mt province":          ["mountain province"],
    "mtn province":         ["mountain province"],
    "mountain":             ["mountain province"],
    "metro manila":         [""],
    "ncr":                  [""],
    "north cotobato":       ["north cotabato", "cotabato"],
    "south cotobato":       ["south cotabato"],
    "negros occ":           ["negros occidental"],
    "neg occ":              ["negros occidental"],
    "misamis occ":          ["misamis occidental"],
    "agusan sur":           ["agusan del sur"],
    "agusan norte":         ["agusan del norte"],
    "lanao sur":            ["lanao del sur"],
    "lanao norte":          ["lanao del norte"],
    "com valley":           ["compostela valley", "davao de oro"],
    "compostella valley":   ["compostela valley", "davao de oro"],
    "suriagao":             ["surigao"],
    "suriagao del norte":   ["surigao del norte"],
    "surigao norte":        ["surigao del norte"],
    "surigao sur":          ["surigao del sur"],

    # ---- reverse-direction safety nets ----
    "misamis occidental":   ["misamis occ"],
    "negros occidental":    ["negros occ"],
    "mountain province":    ["mtn province", "mt province"],
    "agusan del norte":     ["agusan norte"],
    "agusan del sur":       ["agusan sur"],
    "lanao del norte":      ["lanao norte"],
    "lanao del sur":        ["lanao sur"],

    # ---- historical splits / successor provinces ----
    "zamboanga del norte":  ["zamboanga norte", "zamb del norte"],
    "zamboanga del sur":    ["zamboanga sur", "zamb del sur",
                             "zamboanga sibugay"],
    "surigao del norte":    ["surigao norte", "suriagao del norte",
                             "dinagat islands"],
    "surigao del sur":      ["surigao sur"],
    "maguindanao":          ["maguindanao del norte", "maguindanao del sur"],
    "shariff kabunsuan":    ["maguindanao", "maguindanao del norte",
                             "maguindanao del sur"],
    "kalinga apayao":       ["kalinga", "apayao"],
    "davao del sur":        ["davao occidental"],
    "davao del norte":      ["davao de oro", "compostela valley"],
    "south cotabato":       ["sarangani"],
    "iloilo":               ["guimaras"],
    "samar":                ["northern samar", "eastern samar"],

    # ---- renames ----
    "north cotabato":       ["cotabato"],
    "davao":                ["davao del norte"],
    "compostela valley":    ["davao de oro", "com valley"],
    "davao de oro":         ["com valley", "compostela valley"],
    "negros del norte":     ["negros occidental"],
}


def expand_province_key(key: str, max_depth: int = 3) -> list[str]:
    if key is None:
        return []
    if key == "":
        return [""]
    seen = {key}
    levels = [[key]]
    for _ in range(max_depth):
        new_level = []
        for k in levels[-1]:
            for alias in PROVINCE_ALIASES.get(k, []):
                if alias not in seen:
                    seen.add(alias)
                    new_level.append(alias)
        if not new_level:
            break
        levels.append(new_level)
    out: list[str] = []
    for lvl in levels:
        out.extend(lvl)
    return out


# ---------------------------------------------------------------------------
# SRE name aliases (post-make_key)
# ---------------------------------------------------------------------------
# NOTE: Babak/Kaputian/Island Garden City of Samal are handled in
# SRE_PROV_NAME_ALIASES instead of here — a global "babak→samal" alias
# would incorrectly route Davao del Norte rows to Samal, Bataan
# (PSGC 0300812000) via tier-2, because Samal-the-municipality only
# exists in Bataan; Davao's Samal is now a city with a longer name.

SRE_NAME_ALIASES: dict[str, str] = {
    # ---- spelling ----
    "pililia":      "pililla",
    "ozamis":       "ozamiz",
    "lavesares":    "lavezares",
    "belizon":      "belison",
    "legaspi":      "legazpi",
    "talugtog":     "talugtug",
    "sasmoan":      "sasmuan",
    "calanugas":    "calanogas",
    "polilio":      "polillo",
    "zaragosa":     "zaragoza",
    "tangkal":      "tangcal",
    "mendez nunez": "mendez",
    "cordoba":      "cordova",
    "linacapan":    "linapacan",
    "colombio":     "columbio",
    "baliuag":      "baliwag",
    "taboso":       "toboso",
    "jetafe":       "getafe",
    "kalookan":     "caloocan",
    "cullon":       "culion",
    "nogpog":       "mogpog",
    "banawe":       "banaue",
    "diniao":       "dimiao",
    "fany":         "famy",
    "pa paz":       "la paz",
    "davis":        "dauis",
    "abordo":       "aborlan",
    "antipolo ciy": "antipolo",
    "a castaneda":  "alfonso castaneda",
    "masbately":    "masbate",
    "supiden":      "sudipen",
    "tagun":        "tagum",
    "tabajon":      "tubajon",

    # ---- renames ----
    "bumbaran":                        "amai manabilang",
    "bacungan":                        "leon b postigo",
    "maganoy":                         "shariff aguak",
    "sultan gumander":                 "picong",
    "sultan gumendar":                 "picong",
    "karomatan":                       "sultan naga dimaporo",
    "sultan naga dimaporo karomatan":  "sultan naga dimaporo",
    "marungas":                        "hadji panglima tahil",
    "new valencia":                    "nueva valencia",
    "dna r trinidad":                  "dona remedios trinidad",
    "dona r trinidad":                 "dona remedios trinidad",
    "potia":                           "alfonso lista",
    "sergio osmena":                   "sergio osmena sr",
    "datu montawal":                   "datu abdullah sangki",
    "montawal":                        "datu abdullah sangki",
    "pagagawan":                       "datu abdullah sangki",
    "montawal pagagawan":              "datu abdullah sangki",
    "bacolod grande":                  "bacolod kalawi",
    "s k pendatun":                    "sultan kudarat",
    "s k pendatuan":                   "sultan kudarat",
    "sk pendatun":                     "sultan kudarat",
    "espiritu":                        "banna",
    "mariano marcos":                  "president quirino",
    "don mariano marcos":              "president quirino",
    "panamao":                         "old panamao",
    "panamao old":                     "old panamao",
    "munoz":                           "science city of munoz",
    "munoz city":                      "science city of munoz",
    "don victoriano chiongbian":       "don victoriano",
    "don victoriano f chiongbian":     "don victoriano",
    "s benedicto":                     "don salvador benedicto",
    "san benedicto":                   "don salvador benedicto",
    "san isidro davao del norte":      "santo tomas",
    "datu odin":                       "datu odin sinsuat",
    "dimas":                           "dimataling",
    "gian":                            "glan",
    "mohammad ajul":                   "hadji mohammad ajul",
    "r magsaysay":                     "ramon magsaysay",
    "v sagun":                         "vincenzo a sagun",

    # ---- first-name initial -> full name ----
    "general natividad":  "general mamerto natividad",
    "general alvarez":    "general mariano alvarez",
    "gov generoso":       "governor generoso",
    "e villanueva":       "enrique villanueva",
    "e viallanueva":      "enrique villanueva",
    "g h del pilar":      "gregorio del pilar",
    "r romualdez":        "remedios t romualdez",
    "r t romualdez":      "remedios t romualdez",
    "gen aguinaldo":      "general emilio aguinaldo",
    "general aguinaldo":  "general emilio aguinaldo",
    "e b magalona":       "enrique b magalona",

    # ---- president-name shorthand ----
    "pres roxas":                        "president manuel a roxas",
    "president roxas":                   "president manuel a roxas",
    "pres c garcia":                     "president carlos p garcia",
    "president c garcia":                "president carlos p garcia",
    "pitogo president carlos p garcia":  "president carlos p garcia",
}


# (province_key, lgu_key) -> psgc_key   — province-scoped overrides
SRE_PROV_NAME_ALIASES: dict[tuple, str] = {
    ("bohol",              "pitogo"):             "president carlos p garcia",
    ("bohol",              "pres c garcia"):      "president carlos p garcia",
    ("bohol",              "president c garcia"): "president carlos p garcia",
    ("lanao del sur",      "tagoloan"):           "tagoloan ii",
    ("tawi-tawi",          "balimbing"):          "panglima sugala",
    ("palawan",            "rizal"):              "dr jose p rizal",
    ("palawan",            "rizal marcos"):       "dr jose p rizal",
    ("surigao del norte",  "rizal"):              "san francisco",
    ("sorsogon",           "bacon"):              "sorsogon",
    ("davao del norte",    "san isidro"):         "santo tomas",
    ("davao del norte",    "san vicente"):        "laak",
    ("sultan kudarat",     "mariano marcos"):     "president quirino",
    ("sultan kudarat",     "don mariano marcos"): "president quirino",
    ("compostela valley",  "san vicente"):        "laak",
    ("compostella valley", "san vicente"):        "laak",
    ("davao de oro",       "san vicente"):        "laak",
    ("sulu",               "panamao"):            "old panamao",
    ("misamis occidental", "don victoriano chiongbian"): "don victoriano",
    ("misamis occidental", "don victoriano f chiongbian"): "don victoriano",
    ("misamis occ",        "don victoriano chiongbian"): "don victoriano",
    ("misamis occ",        "don victoriano f chiongbian"): "don victoriano",
    ("negros occidental",  "talisay city negros occ"):   "talisay",
    ("negros occ",         "talisay city negros occ"):   "talisay",

    # -----------------------------------------------------------------
    # Samal Island, Davao del Norte.
    #   "Samal" alone is ambiguous: the only *municipality* named Samal
    #   is Samal, Bataan (0300812000).  Davao's Samal was merged with
    #   Babak and Kaputian (RA 8471, 1998) and is now the City of
    #   Island Garden City of Samal (1102317000).
    #   Route all three names through the province-scoped path so tier-1
    #   hits the correct LGU and tier-2 never falls back to Bataan.
    # -----------------------------------------------------------------
    ("davao del norte",    "babak"):             "island garden city of samal",
    ("davao del norte",    "kaputian"):          "island garden city of samal",
    ("davao del norte",    "samal"):             "island garden city of samal",
    ("davao del norte",    "samal city"):        "island garden city of samal",
    ("davao del norte",    "island garden city of samal"): "island garden city of samal",
}


# ---------------------------------------------------------------------------
# Candidate keys
# ---------------------------------------------------------------------------

def _candidate_lgu_keys(lgu_name) -> list[str]:
    raw = _clean(lgu_name)
    inlined = raw.replace("(", " ").replace(")", " ")
    cands: list[str] = []
    for variant in (inlined, raw):
        k = make_key(variant)
        if k and k not in cands:
            cands.append(k)
    m = _PAREN_INNER_RE.search(raw)
    if m:
        k = make_key(m.group(1))
        if k and k not in cands:
            cands.append(k)
    return cands


def _paren_province_candidates(lgu_name) -> list[str]:
    raw = _clean(lgu_name)
    m = _PAREN_INNER_RE.search(raw)
    if not m:
        return []
    inner = make_key(m.group(1))
    if not inner:
        return []
    return expand_province_key(inner)


def _resolve_keys(lgu_name, province, lgu_type):
    pk = make_key(province) if province else ""
    lt = lgu_type_norm(lgu_type)

    raw_cands = _candidate_lgu_keys(lgu_name)

    prov_cands: list[str] = []
    for pc in _paren_province_candidates(lgu_name):
        if pc not in prov_cands:
            prov_cands.append(pc)
    for pc in expand_province_key(pk):
        if pc not in prov_cands:
            prov_cands.append(pc)

    resolved: list[str] = []
    for c in raw_cands:
        c_glob = SRE_NAME_ALIASES.get(c, c)
        prov_scoped = None
        for pc in prov_cands:
            hit = SRE_PROV_NAME_ALIASES.get((pc, c))
            if hit:
                prov_scoped = hit
                break
        c_prov = prov_scoped if prov_scoped else c
        c_name = SRE_NAME_ALIASES.get(c_prov, c_prov)
        for cand in (c, c_glob, c_prov, c_name):
            if cand and cand not in resolved:
                resolved.append(cand)

    if lt == "province":
        expanded: list[str] = []
        for c in resolved:
            for ec in expand_province_key(c):
                if ec not in expanded:
                    expanded.append(ec)
        resolved = expanded

    return resolved, prov_cands, lt


# ---------------------------------------------------------------------------
# Fuzzy
# ---------------------------------------------------------------------------

def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _best_fuzzy_2nd(key: str, keys: list[str]):
    best_i, best_s, second_s = -1, 0.0, 0.0
    for i, k in enumerate(keys):
        s = ratio(key, k)
        if s > best_s:
            second_s = best_s
            best_s, best_i = s, i
        elif s > second_s:
            second_s = s
    return best_i, best_s, second_s


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def build_psgc_lookups(psgc: pd.DataFrame) -> dict:
    psgc = psgc.copy()
    psgc["x_key"] = psgc["name"].map(make_key)
    psgc["x_lvl"] = psgc["level"]

    lgu = psgc[psgc["level"].isin(["province", "city", "municipality"])].copy()
    lgu["x_prov_key"] = lgu["province_name"].map(make_key)
    is_prov = lgu["level"] == "province"
    lgu.loc[is_prov, "x_prov_key"] = lgu.loc[is_prov, "x_key"]

    lgu_by_prov: dict[tuple, str] = {}
    for row in lgu.itertuples(index=False):
        lgu_by_prov.setdefault((row.x_prov_key, row.x_key, row.x_lvl), row.psgc10)

    global_map: dict[tuple, list[str]] = defaultdict(list)
    for row in lgu.itertuples(index=False):
        global_map[(row.x_key, row.x_lvl)].append(row.psgc10)
    global_map = dict(global_map)

    global_any_level: dict[str, list[str]] = defaultdict(list)
    for row in lgu.itertuples(index=False):
        global_any_level[row.x_key].append(row.psgc10)
    global_any_level = dict(global_any_level)

    level_of = dict(zip(psgc["psgc10"], psgc["level"]))

    cat = lgu[["x_prov_key", "x_key", "x_lvl", "psgc10", "name"]].copy()
    cat.columns = ["prov_key", "lgu_key", "level", "psgc10", "psgc_name"]
    cat = cat.reset_index(drop=True)

    cat_by_level = {lvl: sub.reset_index(drop=True) for lvl, sub in cat.groupby("level")}

    cat_by_prov_level: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
    for row in cat.itertuples(index=False):
        cat_by_prov_level[(row.prov_key, row.level)].append(
            (row.lgu_key, row.psgc10, row.psgc_name))

    cat_by_prov: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for row in cat.itertuples(index=False):
        cat_by_prov[row.prov_key].append(
            (row.lgu_key, row.psgc10, row.psgc_name, row.level))

    return {
        "lgu_by_prov":       lgu_by_prov,
        "global_map":        global_map,
        "global_any_level":  global_any_level,
        "level_of":          level_of,
        "cat":               cat,
        "cat_by_level":      cat_by_level,
        "cat_by_prov_level": dict(cat_by_prov_level),
        "cat_by_prov":       dict(cat_by_prov),
    }


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

def match_one(lgu_name, province, lgu_type, L: dict) -> dict:
    lk_cands, prov_cands, lt = _resolve_keys(lgu_name, province, lgu_type)

    # Tier 1
    for pc in prov_cands:
        for lc in lk_cands:
            hit = L["lgu_by_prov"].get((pc, lc, lt))
            if hit:
                return {"psgc10": hit, "tier": 1, "score": 1.0, "matched_name": None}

        # Tier 1b — city<->municipality level swap ONLY.
    # Deliberately excludes the province level: a city/municipality never
    # shares its PSGC with its parent province.  Allowing the upgrade
    # caused "Isabela City | Isabela | City" to be matched to the PSGC
    # of Isabela *Province* (0203100000), producing duplicate
    # (psgc10, fiscal_year) rows in the panel.
    CROSS_LEVELS = {
        "city":         ("municipality",),
        "municipality": ("city",),
    }
    for pc in prov_cands:
        for lc in lk_cands:
            for lvl in CROSS_LEVELS.get(lt, ()):
                hit = L["lgu_by_prov"].get((pc, lc, lvl))
                if hit:
                    return {"psgc10": hit, "tier": 1, "score": 1.0,
                            "matched_name": None}

    # Tier 2
    for lc in lk_cands:
        hits = L["global_map"].get((lc, lt), [])
        if len(hits) == 1:
            return {"psgc10": hits[0], "tier": 2, "score": 1.0, "matched_name": None}

    # Tier 2b
    for lc in lk_cands:
        hits = L["global_any_level"].get(lc, [])
        if len(hits) == 1:
            return {"psgc10": hits[0], "tier": 2, "score": 1.0, "matched_name": None}
        if len(hits) > 1 and lt != "province":
            non_prov = [h for h in hits if L["level_of"].get(h) != "province"]
            if len(non_prov) == 1:
                return {"psgc10": non_prov[0], "tier": 2, "score": 1.0,
                        "matched_name": None}

    # Tier 3
    for pc in prov_cands:
        pool = L["cat_by_prov_level"].get((pc, lt), [])
        if not pool:
            continue
        keys = [k for k, _, _ in pool]
        for lc in lk_cands:
            idx, score, _ = _best_fuzzy_2nd(lc, keys)
            if idx >= 0 and score >= 0.85:
                _, pg, nm = pool[idx]
                return {"psgc10": pg, "tier": 3, "score": round(score, 4),
                        "matched_name": nm}

    # Tier 3b
    for pc in prov_cands:
        pool = L["cat_by_prov"].get(pc, [])
        if not pool:
            continue
        keys = [k for k, _, _, _ in pool]
        for lc in lk_cands:
            idx, score, second = _best_fuzzy_2nd(lc, keys)
            if idx >= 0 and score >= 0.85 and (score - second) > 0.05:
                _, pg, nm, _lvl = pool[idx]
                return {"psgc10": pg, "tier": 3, "score": round(score, 4),
                        "matched_name": nm}

    # Tier 4
    pool = L["cat_by_level"].get(lt, pd.DataFrame())
    if len(pool):
        keys = pool["lgu_key"].tolist()
        for lc in lk_cands:
            idx, score, second = _best_fuzzy_2nd(lc, keys)
            if idx >= 0 and score >= 0.94 and (score - second) > 0.03:
                row = pool.iloc[idx]
                return {"psgc10": row["psgc10"], "tier": 4,
                        "score": round(score, 4), "matched_name": row["psgc_name"]}

    # Tier 5
    keys = L["cat"]["lgu_key"].tolist()
    for lc in lk_cands:
        idx, score, second = _best_fuzzy_2nd(lc, keys)
        if idx >= 0 and score >= 0.92 and (score - second) > 0.05:
            row = L["cat"].iloc[idx]
            return {"psgc10": row["psgc10"], "tier": 5,
                    "score": round(score, 4), "matched_name": row["psgc_name"]}

    return {"psgc10": None, "tier": 0, "score": 0.0, "matched_name": None}


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def diagnose_unmatched(unmatched: pd.DataFrame, L: dict, n_per: int = 8) -> str:
    lines = []
    for r in unmatched.itertuples(index=False):
        lk_cands, prov_cands, lt = _resolve_keys(r.lgu_name, r.province, r.lgu_type)
        lines.append("=" * 78)
        lines.append(f"SRE: {r.lgu_name!r} | province={r.province!r} | "
                     f"region={r.region!r} | type={r.lgu_type!r}")
        lines.append(f"     lgu_keys={lk_cands}  prov_keys={prov_cands}  level={lt}")

        lines.append("     -- PSGC contents of candidate (province, level) pools --")
        for pc in prov_cands:
            pool = L["cat_by_prov_level"].get((pc, lt), [])
            label = pc or "<blank>"
            lines.append(f"       [{label}] level={lt}: {len(pool)} entries")
            sample = sorted(pool, key=lambda t: t[0])[:12]
            for k, pg, nm in sample:
                lines.append(f"          key={k!r:35s}  name={nm!r} ({pg})")

        cands = []
        for pc in prov_cands:
            pool = L["cat_by_prov"].get(pc, [])
            for lc in lk_cands:
                for k, pg, nm, lvl in pool:
                    cands.append((ratio(lc, k), pc or "<blank>", nm, lvl, pg))
        for lc in lk_cands:
            for row in L["cat"].itertuples(index=False):
                cands.append((ratio(lc, row.lgu_key), "GLOBAL",
                              row.psgc_name, row.level, row.psgc10))
        cands.sort(key=lambda t: -t[0])

        lines.append("     -- top fuzzy candidates --")
        seen, shown = set(), 0
        for s, pc, nm, lvl, pg in cands:
            if pg in seen:
                continue
            seen.add(pg)
            lines.append(f"     [{pc:25s}] {s:.3f}  {nm} ({lvl}, {pg})")
            shown += 1
            if shown >= n_per:
                break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------

def build_crosswalk(sre: pd.DataFrame, L: dict) -> pd.DataFrame:
    uniq = (sre[["lgu_name", "province", "region", "lgu_type"]]
            .drop_duplicates()
            .reset_index(drop=True))
    LOG.info("Distinct LGU tuples to match: %d", len(uniq))

    rows = []
    for i, r in uniq.iterrows():
        m = match_one(r["lgu_name"], r["province"], r["lgu_type"], L)
        rows.append({
            "lgu_name": r["lgu_name"], "province": r["province"],
            "region": r["region"], "lgu_type": r["lgu_type"],
            "psgc10": m["psgc10"], "tier": m["tier"],
            "score": m["score"], "matched_name": m["matched_name"],
        })
        if (i + 1) % 500 == 0:
            LOG.info("  matched %d / %d", i + 1, len(uniq))

    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sre",    type=Path, default=DEFAULT_SRE)
    p.add_argument("--psgc",   type=Path, default=DEFAULT_PSGC)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.outdir / "sre_psgc_crosswalk.log",
                                mode="w", encoding="utf-8"),
        ],
        force=True,
    )

    LOG.info("Loading SRE panel: %s", args.sre)
    sre = pd.read_parquet(args.sre)
    LOG.info("  %d rows", len(sre))

    LOG.info("Loading PSGC master: %s", args.psgc)
    psgc = pd.read_parquet(args.psgc)
    LOG.info("  %d LGU rows", len(psgc))

    L = build_psgc_lookups(psgc)
    xw = build_crosswalk(sre, L)

    xw["matched"] = xw["psgc10"].notna()
    LOG.info("Distinct tuple match rate: %.2f%%", xw["matched"].mean() * 100)
    LOG.info("Tier distribution:\n%s",
             xw["tier"].value_counts().sort_index().to_string())

    xw.to_parquet(args.outdir / "sre_psgc_crosswalk.parquet", index=False)
    xw.to_csv(args.outdir / "sre_psgc_crosswalk.csv", index=False)

    unmatched = xw[~xw["matched"]].copy()
    unmatched.to_csv(args.outdir / "sre_unmatched.csv", index=False)
    LOG.info("Unmatched tuples: %d", len(unmatched))

    if len(unmatched):
        sre_keys = sre[["lgu_name", "province", "region", "lgu_type"]].copy()
        rank = (sre_keys
                .merge(unmatched,
                       on=["lgu_name", "province", "region", "lgu_type"],
                       how="inner")
                .groupby(["lgu_name", "province", "lgu_type"])
                .size()
                .reset_index(name="n_rows")
                .sort_values("n_rows", ascending=False))
        rank.to_csv(args.outdir / "sre_unmatched_ranked.csv", index=False)
        LOG.info("Top 30 unmatched:\n%s", rank.head(30).to_string(index=False))

        if args.diagnose:
            diag = diagnose_unmatched(unmatched, L, n_per=8)
            (args.outdir / "sre_unmatched_diagnostic.txt").write_text(
                diag, encoding="utf-8")
            LOG.info("Wrote sre_unmatched_diagnostic.txt")

    LOG.info("Merging back to panel...")
    sre_out = sre.merge(
        xw[["lgu_name", "province", "region", "lgu_type",
            "psgc10", "tier", "score", "matched_name"]],
        on=["lgu_name", "province", "region", "lgu_type"],
        how="left",
    )
    row_match = sre_out["psgc10"].notna().mean() * 100
    LOG.info("Panel row match rate: %.2f%%", row_match)

    sre_out.to_parquet(args.outdir / "sre_panel_with_psgc.parquet", index=False)
    sre_out.to_csv(args.outdir / "sre_panel_with_psgc.csv", index=False)
    LOG.info("Wrote sre_panel_with_psgc.parquet / .csv (%d rows)", len(sre_out))

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())