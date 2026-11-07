import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ref_last", ROOT / "scripts/submit/ref_last.py")
rl = importlib.util.module_from_spec(_spec)
sys.modules["ref_last"] = rl
_spec.loader.exec_module(rl)

KEYS = {"a": {"KA"}, "b": {"KB"}, "c": {"KC"}, "d": set()}
keys_of = lambda s: KEYS[s]


def test_refs_move_last_stably_when_library_match_is_weak():
    new, changed = rl.ref_last(["a", "b", "c", "d"], keys_of, {"KA", "KC"}, lib_max=0.3)
    assert new == ["b", "d", "a", "c"] and changed


def test_untouched_when_library_match_is_strong():
    new, changed = rl.ref_last(["a", "b"], keys_of, {"KA"}, lib_max=0.7)
    assert new == ["a", "b"] and not changed


def test_missing_lib_is_weak_and_no_refs_is_noop():
    assert rl.ref_last(["a", "b"], keys_of, {"KA"}, lib_max=None)[0] == ["b", "a"]
    assert rl.ref_last(["a", "b"], keys_of, set(), lib_max=0.0) == (["a", "b"], False)


def test_rdkit_keys_include_raw_inchikey14():
    k = rl.make_keys_of()
    assert "LFQSCWFLJHTTHZ" in k("CCO")
    assert k("not a smiles") == frozenset()
