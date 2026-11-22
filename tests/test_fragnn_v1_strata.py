import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))

import fragnn_v1_strata as fs  # noqa: E402


def test_is_multimer():
    assert fs.is_multimer("[2M+Na]+")
    assert fs.is_multimer("[3M-H]-")
    assert fs.is_multimer("[ 2M+H]+")
    assert not fs.is_multimer("[M+H]+")
    assert not fs.is_multimer("[M+2H]2+")
    assert not fs.is_multimer(None)


def test_bins():
    assert [fs.n_spec_bin(n) for n in (1, 2, 3, 5, 6, 16)] == ["1-2", "1-2", "3-5", "3-5", "6+", "6+"]
    assert [fs.peaks_bin(x) for x in (0, 49.5, 50, 199, 200)] == ["<50", "<50", "50-199", "50-199", "200+"]


def test_molecule_strata_and_summary():
    rows = [
        dict(molecule_id="a", adduct="[M+H]+", n_peaks=10),
        dict(molecule_id="a", adduct="[2M+Na]+", n_peaks=30),
        dict(molecule_id="b", adduct="[M-H]-", n_peaks=300),
        dict(molecule_id="b", adduct="[M-H]-", n_peaks=250),
        dict(molecule_id="b", adduct="[M+Cl]-", n_peaks=100),
    ]
    st = fs.molecule_strata(rows)
    assert st["a"] == dict(multimer="with_multimer", n_spec="1-2", peaks="<50")
    assert st["b"] == dict(multimer="no_multimer", n_spec="3-5", peaks="200+")
    rr_ctl = {"a": 1.0, "b": 0.5, "c": 0.0}
    rr_new = {"a": 0.5, "b": 1.0}  # c は新に無い → 0
    s = fs.summarize(rr_new, rr_ctl, st)
    assert s["all"]["n"] == 3 and s["all"]["mean_diff"] == 0.0
    assert s["multimer"]["with_multimer"] == dict(n=1, mean_diff=-0.5, mean_ctrl=1.0, mean_new=0.5,
                                                  n_better=0, n_worse=1)
    assert s["multimer"]["no_multimer"]["mean_diff"] == 0.5
    assert s["multimer"]["missing"]["n"] == 1
