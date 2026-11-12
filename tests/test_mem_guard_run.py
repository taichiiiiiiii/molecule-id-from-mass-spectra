import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'train'))
import mem_guard_run as mg  # noqa: E402


def test_ok_run_records_exit_zero(tmp_path):
    r = mg.run_guarded([sys.executable, '-c', 'print("hi")'], str(tmp_path / 'a.log'), 8 * 1024 ** 3, 60, 10,
                       poll_s=0.2, free_pct_fn=lambda: 50, use_time=False)
    assert r['returncode'] == 0 and r['killed_reason'] is None
    assert 'hi' in (tmp_path / 'a.log').read_text()


def test_nonzero_exit_is_kept(tmp_path):
    r = mg.run_guarded([sys.executable, '-c', 'import sys; sys.exit(3)'], str(tmp_path / 'b.log'), 8 * 1024 ** 3, 60,
                       10, poll_s=0.2, free_pct_fn=lambda: 50, use_time=False)
    assert r['returncode'] == 3 and r['killed_reason'] is None


def test_low_free_pct_kills(tmp_path):
    r = mg.run_guarded([sys.executable, '-c', 'import time; time.sleep(30)'], str(tmp_path / 'c.log'), 8 * 1024 ** 3,
                       60, 10, poll_s=0.2, free_pct_fn=lambda: 5, use_time=False)
    assert r['killed_reason'].startswith('free_pct 5') and r['returncode'] != 0 and r['wall_s'] < 25


def test_unreadable_free_pct_kills(tmp_path):
    r = mg.run_guarded([sys.executable, '-c', 'import time; time.sleep(30)'], str(tmp_path / 'd.log'), 8 * 1024 ** 3,
                       60, 10, poll_s=0.2, free_pct_fn=lambda: None, use_time=False)
    assert r['killed_reason'] == 'free_pct_unreadable'


def test_wall_limit_kills(tmp_path):
    r = mg.run_guarded([sys.executable, '-c', 'import time; time.sleep(30)'], str(tmp_path / 'e.log'), 8 * 1024 ** 3,
                       0.5, 10, poll_s=0.2, free_pct_fn=lambda: 50, use_time=False)
    assert r['killed_reason'].startswith('wall')


def test_rss_limit_kills(tmp_path):
    r = mg.run_guarded([sys.executable, '-c', 'import time; time.sleep(30)'], str(tmp_path / 'f.log'), 1, 60, 10,
                       poll_s=0.2, free_pct_fn=lambda: 50, use_time=False)
    assert r['killed_reason'].startswith('tree_rss')


def test_free_regex_and_time_regex():
    assert mg.FREE_RE.search('System-wide memory free percentage: 59%').group(1) == '59'
    txt = '        0.10 real 0.0 user\n   12345678  maximum resident set size\n'
    assert mg.TIME_RSS_RE.findall(txt) == ['12345678']


def test_main_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(mg, 'free_pct_macos', lambda: 50)
    rep = tmp_path / 'r.json'
    rc = mg.main(['--log', str(tmp_path / 'g.log'), '--report', str(rep), '--poll-s', '0.2', '--',
                  sys.executable, '-c', 'pass'])
    d = json.loads(rep.read_text())
    assert rc == 0 and d['returncode'] == 0
