"""
装置の値と系統（Issue #280 で数える前に固定した対応表、#287 で共通のモジュールに移した。中身は変えていない）。

- inst_value: instrument_type の生の値。欠け（None・NaN・空・nan・N/A・none・null）は None
- family: 装置の系統。大小文字を無視した部分文字列の規則を上から順に当てる。欠けは None
"""
from __future__ import annotations

import numpy as np

MISSING = {"", "nan", "n/a", "none", "null"}


def inst_value(v) -> str | None:
    """instrument_type の生の値。欠け（None・NaN・空・nan・N/A など）は None。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v)
    return None if s.strip().lower() in MISSING else s


def family(s: str | None) -> str | None:
    """装置の系統（#280 で数える前に固定）。大小文字を無視した部分文字列の規則を上から順に当てる。欠けは None。"""
    s = inst_value(s)
    if s is None:
        return None
    t = s.lower()
    if "tims" in t:
        return "timsTOF"
    if ("orbitrap" in t or "exactive" in t or "qehf" in t or "qft" in t or "itft" in t or "hybrid ft" in t
            or t.endswith("-ft") or t.endswith(" ft") or t.endswith("qit-ft") or "hrms" in t):
        return "Orbitrap/FT"
    if "tof" in t or "maxis" in t or "impact" in t or "synapt" in t:
        return "TOF"
    if "qq" in t:
        return "QQQ"
    if "trap" in t or "qit" in t or t in ("lit", "lc-esi-it", "ltq") or "ltq" in t:
        return "IonTrap"
    return "other"
