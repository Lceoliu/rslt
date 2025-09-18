import sys
import types
from pathlib import Path


def pytest_sessionstart(session):
    # Ensure project root is importable (dataset/, model/, training/)
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Stub optional heavy deps so imports succeed in lean envs
    if 'av' not in sys.modules:
        sys.modules['av'] = types.ModuleType('av')
    if 'matplotlib' not in sys.modules:
        m = types.ModuleType('matplotlib')
        m.use = lambda *a, **k: None
        sys.modules['matplotlib'] = m
    if 'matplotlib.pyplot' not in sys.modules:
        sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')
