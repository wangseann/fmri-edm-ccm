import numpy as np

from src.ccm import ccm_pair


def test_ccm_pair_outputs():
    X = np.sin(np.linspace(0, 10, 200))
    Y = np.roll(X, 1)
    result = ccm_pair(X, Y, E=3, tau=1, theiler=3, L_grid=[60, 120, 180])
    assert "skill_curve" in result
    assert len(result["skill_curve"]) == 3
    assert "convergence" in result
    assert "poslag_asym" in result
