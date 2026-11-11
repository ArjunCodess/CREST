"""Packaging smoke test for the crest-traffic wheel.

This is packaging smoke, not a clean dependency-resolution proof: the test
assumes the active pytest environment already satisfies runtime dependencies
(torch, numpy, etc.). A temporary venv uses ``--system-site-packages`` so pip
installs only the wheel with ``--no-deps`` and never hits the network.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MINI_DEV_PROTOCOL = {
    'protocol_id': 'wheel_smoke_dev',
    'protocol_role': 'development',
    'seeds': [11, 22, 33, 44, 55],
    'scenario_params': {
        'n_links': 5,
        'n_steps': 6,
        'demand_vph': 700.0,
        'noise_std': 10.0,
        'missing_rate': 0.1,
        'capacity_vehicles': 200.0,
    },
    'stress_variants': ['baseline'],
}

_WHEEL_BUILD_TIMEOUT_S = 180
_INSTALL_TIMEOUT_S = 120
_RUNNER_TIMEOUT_S = 600


def _venv_python(venv: Path) -> Path:
    if sys.platform == 'win32':
        return venv / 'Scripts' / 'python.exe'
    return venv / 'bin' / 'python'


def _isolated_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    return env


def test_installed_wheel_smoke_packaging(tmp_path: Path) -> None:
    wheel_dir = tmp_path / 'wheelhouse'
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            '-m',
            'pip',
            'wheel',
            str(ROOT),
            '--no-deps',
            '--no-build-isolation',
            '--no-index',
            '-w',
            str(wheel_dir),
        ],
        cwd=ROOT,
        check=True,
        timeout=_WHEEL_BUILD_TIMEOUT_S,
    )
    wheels = sorted(wheel_dir.glob('*.whl'))
    assert wheels, 'expected a wheel artifact in temporary wheelhouse'

    venv = tmp_path / 'testvenv'
    subprocess.run(
        [sys.executable, '-m', 'venv', '--system-site-packages', str(venv)],
        check=True,
        timeout=60,
    )
    py = _venv_python(venv)

    subprocess.run(
        [
            str(py),
            '-m',
            'pip',
            'install',
            '--no-deps',
            '--no-index',
            '--force-reinstall',
            '--disable-pip-version-check',
            str(wheels[-1]),
        ],
        check=True,
        timeout=_INSTALL_TIMEOUT_S,
    )

    work_cwd = tmp_path / 'empty_cwd'
    work_cwd.mkdir()
    env = _isolated_env()
    checkout_research = (ROOT / 'src' / 'research').resolve()

    probe = subprocess.run(
        [
            str(py),
            '-c',
            (
                'import importlib.metadata as im\n'
                'import site\n'
                'import sys\n'
                'from pathlib import Path\n'
                'import src.research\n'
                f'checkout = Path({str(checkout_research)!r}).resolve()\n'
                'mod = Path(src.research.__file__).resolve()\n'
                'if mod == checkout or checkout in mod.parents:\n'
                "    raise SystemExit(f'module loaded from checkout, not wheel: {mod}')\n"
                'venv_site = [Path(p).resolve() for p in site.getsitepackages()]\n'
                'if not any(mod.is_relative_to(sp) for sp in venv_site if sp.is_dir()):\n'
                "    raise SystemExit(f'module not under venv site-packages: {mod} list={venv_site}')\n"
                'dist = im.distribution("crest-traffic")\n'
                'dist_path = Path(dist.locate_file("src/research/__init__.py")).resolve()\n'
                'if mod != dist_path:\n'
                "    raise SystemExit(f'package metadata file mismatch: mod={mod} dist={dist_path}')\n"
                'ver = im.version("crest-traffic")\n'
                'print(f"OK wheel crest-traffic=={ver} module={mod}")\n'
            ),
        ],
        cwd=work_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert 'OK wheel' in probe.stdout

    bundled = subprocess.run(
        [
            str(py),
            '-c',
            (
                'from src.research.config_paths import bundled_config_path, load_raw_config\n'
                'path = bundled_config_path("dev_pilot.json")\n'
                'assert path.is_file(), path\n'
                'raw, resolved = load_raw_config(None, default_bundled="dev_pilot.json")\n'
                'assert raw["protocol_id"] == "dev_pilot"\n'
                'assert raw["protocol_role"] == "development"\n'
                'print(resolved)\n'
            ),
        ],
        cwd=work_cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert 'dev_pilot' in bundled.stdout

    mini_config = tmp_path / 'mini_dev_protocol.json'
    mini_config.write_text(json.dumps(MINI_DEV_PROTOCOL), encoding='utf-8')
    out = tmp_path / 'run_out'
    subprocess.run(
        [
            str(py),
            '-m',
            'src.research.runner',
            '--mode',
            'development',
            '--config',
            str(mini_config),
            '--output',
            str(out),
            '--force',
        ],
        cwd=work_cwd,
        env=env,
        check=True,
        timeout=_RUNNER_TIMEOUT_S,
    )
    assert (out / 'pilot_results.json').is_file()
