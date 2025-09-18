import sys
import types


def pytest_sessionstart(session):
    # Stub optional heavy deps so imports succeed in lean envs
    if 'av' not in sys.modules:
        sys.modules['av'] = types.ModuleType('av')
    if 'matplotlib' not in sys.modules:
        m = types.ModuleType('matplotlib')
        m.use = lambda *a, **k: None
        sys.modules['matplotlib'] = m
    if 'matplotlib.pyplot' not in sys.modules:
        sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')

