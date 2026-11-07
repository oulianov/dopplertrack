"""fit_event's two objectives. On a uniform spectrogram "brightness" (mean brightness under the curve)
scores the level itself, while "contrast" (the curve minus its sidebands) is zero away from the crop's
edges and at most about half the level at them, so the two cannot be confused."""
import numpy as np
import pytest

from dopplertrack import cli, tracks
from dopplertrack.audio import Spectrogram
from dopplertrack.tracks import EventFit, fit_event, grid_search


def _uniform(level=1.0):
    t = np.arange(0, 400, 0.5); f = np.arange(40, 1000, 0.5)
    return Spectrogram(t, f, np.full((len(f), len(t)), level, np.float32))


def test_grid_search_objectives_differ():
    sp = _uniform()
    sb, _ = grid_search(sp, 100, 140, 100, 150, 340.0, objective="brightness")
    sc, _ = grid_search(sp, 100, 140, 100, 150, 340.0, objective="contrast")
    assert sb == pytest.approx(1.0) and sc == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        grid_search(sp, 100, 140, 100, 150, 340.0, objective="loudness")


def test_fit_event_objectives_differ():
    sp = _uniform()
    b = fit_event(sp, 100, 140, 100, 150, c=340.0, whiten=False, objective="brightness")
    c = fit_event(sp, 100, 140, 100, 150, c=340.0, whiten=False, objective="contrast")
    assert b.score == pytest.approx(1.0, abs=1e-3)
    assert c.score < 0.75


def test_fit_cli_defaults_to_brightness(tmp_path, monkeypatch):
    spec = tmp_path / "rec.npz"
    _uniform().save(spec)
    seen = []

    def fake_fit_event(sp, t0, t1, f_lo, f_hi, harmonics=1, objective="contrast", device=None):
        seen.append(objective)
        return EventFit(t_cpa=0, t_cpa_emit=0, f0=1, v=1, d=1, t_start=t0, t_end=t1, score=0, f_pre=1, f_post=1)

    monkeypatch.setattr(tracks, "fit_event", fake_fit_event)
    cli.main(["fit", str(spec), "--window", "100", "140", "--band", "100", "150"])
    cli.main(["fit", str(spec), "--window", "100", "140", "--band", "100", "150", "--objective", "contrast"])
    assert seen == ["brightness", "contrast"]
