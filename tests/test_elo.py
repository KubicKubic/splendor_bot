import jax
import numpy as np
import pytest

from splendor import env, network
from splendor.elo import analyze, fit_elo, make_match, match_summary


def test_elo_known_probabilities_and_anchor():
    truth = np.array([800., 1000., 1200.])
    pairs = [(0, 1), (0, 2), (1, 2)]
    counts = np.full(3, 1e6)
    wins = np.array([1 / (1 + 10 ** ((truth[b] - truth[a]) / 400)) for a, b in pairs]) * counts
    fitted = fit_elo(3, pairs, wins, counts, anchor=1)
    np.testing.assert_allclose(fitted, truth, atol=.01)
    reversed_fit = fit_elo(3, [(b, a) for a, b in pairs], counts - wins, counts, anchor=1)
    np.testing.assert_allclose(fitted, reversed_fit, atol=1e-6)
    assert fitted[1] == 1000.


def test_elo_sweeps_finite_and_disconnected_rejected():
    elo = fit_elo(2, [(0, 1)], [100.], [100.], anchor=1)
    assert np.all(np.isfinite(elo)) and elo[0] > elo[1]
    with pytest.raises(ValueError, match='connect'):
        fit_elo(3, [(0, 1)], [50.], [100.], anchor=0)


def test_match_accounting_excludes_timeouts_and_counts_ties():
    result = dict(score=np.array([1., .5, 0., 0.]), done=np.array([True, True, True, False]),
                  seat_a=np.array([0, 0, 1, 1]), turns=np.array([50, 60, 40, 100]), decisions_executed=4000)
    s = match_summary(result)
    assert (s['a_wins'], s['draws'], s['b_wins'], s['unfinished']) == (1, 1, 1, 1)
    assert s['a_score_completed'] == .5
    assert s['a_score_all_bounds'] == [.375, .625]


def test_bootstrap_and_elo_direction():
    score = np.array([0., 0., 0., 1.] * 16)
    result = dict(score=score, done=np.ones(64, bool))
    models = [dict(name='old', checkpoint='old'), dict(name='new', checkpoint='new')]
    report = analyze(models, [(0, 1)], [result], anchor=0, bootstrap=100)
    assert report['ratings'][0]['elo'] == 1000.
    assert report['ratings'][1]['elo'] > 1100.
    assert report['adjacent_changes'][0]['ci95'][0] > 0.


def test_actual_checkpoint_match_and_reproducibility():
    params = network.init(jax.random.PRNGKey(0), env.observe(env.reset(jax.random.PRNGKey(1))).shape[0], 32)
    run = make_match(games=64, max_decisions=256, bf16=False)
    result = jax.device_get(run(params, params, 20260920))
    repeat = jax.device_get(run(params, params, 20260920))
    for name in result:
        np.testing.assert_array_equal(result[name], repeat[name])
    assert np.all(result['seat_a'][:32] == 0) and np.all(result['seat_a'][32:] == 1)
    assert set(result['score'].tolist()) <= {0., .5, 1.}
    s = match_summary(result)
    assert s['a_wins'] + s['draws'] + s['b_wins'] + s['unfinished'] == 64
