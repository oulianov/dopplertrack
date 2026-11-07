"""End-to-end on synthetic data: two straight flights, a synthetic recording, a day index.
detect must find both events, fit must recover their speeds, the search must rank the true
observer first, and refine with an extra event must pick the right flight."""
import numpy as np
import pandas as pd

from dopplertrack import audio, doppler, pipeline, search, tracks
from dopplertrack.adsb import Trace
from dopplertrack.geo import enu_to_lla, haversine_m


def _synth_trace(icao, t, enu, lat0, lon0, gs):
    lat, lon, alt = enu_to_lla(enu[:, 0], enu[:, 1], enu[:, 2], lat0, lon0)
    return Trace(icao, None, "C172", "synthetic", t, lat, lon, alt, np.full_like(t, gs), np.zeros(len(t), bool))


def test_true_observer_ranks_first(tmp_path):
    c = 340.0
    lat0, lon0 = 40.8765, -73.525
    sr = 4000
    T0 = 1_700_000_000.0                     # recording starts at this unix time
    # flight 1: east-bound, passes 2.5 km north of origin at 1500 m, t_cpa_emit = T0+300
    t1 = T0 + np.arange(0, 900, 5.0)
    enu1 = np.stack([110.0 * (t1 - (T0 + 300)), np.full_like(t1, 2500.0), np.full_like(t1, 1500.0)], -1)
    # flight 2: north-bound with a turn, passes ~3 km east at 1200 m, t_cpa_emit ≈ T0+1220. The turn
    # matters: two perfectly straight passes have a mirror-image observer that scores the same.
    t2 = T0 + np.arange(900, 1800, 5.0)
    tau = t2 - (T0 + 1220)
    enu2 = np.stack([3000.0 + 0.1 * tau**2, 95.0 * tau, np.full_like(t2, 1200.0)], -1)
    traces = [_synth_trace("aaa001", t1, enu1, lat0, lon0, 110 / 0.5144), _synth_trace("aaa002", t2, enu2, lat0, lon0, 95 / 0.5144)]
    idx = search.DayIndex.build(traces, lat0, lon0)
    assert [s.icao for s in idx.segs] == ["aaa001", "aaa002"]

    # synthesize the audio the observer at the origin hears: f0=120 Hz and 130 Hz
    obs = np.zeros(3)
    dur = 1800
    tt = np.arange(0, dur, 1 / sr)
    x = 0.4 * np.random.default_rng(1).standard_normal(len(tt)).astype(np.float32)
    for seg, f0 in zip(idx.segs, (120.0, 130.0)):
        to, ratio, _ = doppler.curve_from_trajectory(seg.t, seg.enu, obs, f0=1.0, c=c)
        finst = np.interp(tt + T0, to, ratio, left=np.nan, right=np.nan) * f0
        m = np.isfinite(finst)
        ph = 2 * np.pi * np.cumsum(np.nan_to_num(finst, nan=0.0)) / sr
        x[m] += 0.05 * np.sin(ph[m])
    sp = audio.compute_spectrogram(x, sr, nperseg=8192, hop_s=0.5, fmin=60, fmax=300, whiten_window_s=None)

    found = tracks.detect_events(sp, 90, 180, c=c, top=4, refine=False)
    cpas = sorted(e.t_cpa for e in found[:2])
    assert abs(cpas[0] - (300 + 2500 / c)) < 15 and abs(cpas[1] - (1220 + 3000 / c)) < 15, [e.t_cpa for e in found]

    ev1 = tracks.fit_event(sp, 230, 400, 90, 170, c=c)
    ev2 = tracks.fit_event(sp, 1150, 1310, 100, 180, c=c)
    assert abs(ev1.v - 110) < 25 and abs(ev2.v - 95) < 25
    evb = tracks.fit_event(sp, 1150, 1310, 100, 180, c=c, objective="brightness")    # what `fit` uses
    assert abs(evb.v - 95) < 25 and abs(evb.t_cpa - ev2.t_cpa) < 6
    dt_meas = ev2.t_cpa - ev1.t_cpa
    assert abs(dt_meas - (1220 - 300 + 3000 / c - 2500 / c)) < 6   # heard-CPA offset

    res = search.run_search(sp, ev1, ev2, idx, c=c, workers=1, silence_d_m=None, log=lambda m: None)
    assert len(res) > 0
    best = res.iloc[0]
    assert best.icao1 == "aaa001" and best.icao2 == "aaa002"
    assert haversine_m(best.lat, best.lon, lat0, lon0) < 600.0, f"best candidate {haversine_m(best.lat, best.lon, lat0, lon0):.0f} m from truth"

    # refine around the winner with event 2 passed again as an "extra" event: pick_flight must
    # choose flight 2, and its locked score must equal s2 at every fine point
    fine = pipeline.refine(sp, ev1, ev2, idx, best, spacing_m=250.0, half_m=500.0, extra=[ev2], c=c, log=lambda m: None)
    assert fine.attrs["extra_icao"] == ["aaa002"]
    assert np.allclose(fine.s3, fine.s2) and np.allclose(fine.score, fine.s1 + fine.s2 + fine.s3)
    assert haversine_m(fine.lat[0], fine.lon[0], lat0, lon0) < 600.0

    # the CLI's refine reads the CSV the search wrote and the index it ran on
    idx.save(tmp_path / "day.npz"); res.to_csv(tmp_path / "res.csv", index=False)
    row = pipeline.best_per_pair(pd.read_csv(tmp_path / "res.csv")).iloc[0]
    back = search.DayIndex.load(tmp_path / "day.npz")
    assert back.icao[int(row.seg1)] == row.icao1 and back.icao[int(row.seg2)] == row.icao2
