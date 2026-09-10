import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from splendor import env
from splendor.train import advantages

ROOT = Path(__file__).resolve().parents[1]
STEP = jax.jit(env.step)
MASK = jax.jit(env.legal_mask)


def oracle(requests):
    result = subprocess.run(['node', str(ROOT / 'tests/hullqin_oracle.cjs')],
        input=json.dumps(requests), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def view(state):
    s = jax.device_get(state)
    n = int(s.nplayers)
    return dict(waitFor=int(s.player), waitThrowing=int(s.phase) == env.DISCARD,
        waitNoble=int(s.phase) == env.CHOOSE_NOBLE, bankGem=s.bank.tolist(),
        bankCard=(s.market + 1).tolist(), bankNoble=(s.nobles[:n + 1] + 1).tolist(),
        playerGem=s.gems[:n].tolist(), playerCard=[np.where(s.bought == p)[0].tolist() for p in range(n)],
        playerBooked=[sorted(r[r >= 0].tolist()) for r in s.reserved[:n]],
        playerNoble=[np.where(s.noble_owner == p)[0].tolist() for p in range(n)],
        playerCardCount=s.bonuses[:n].tolist(), playerScore=s.scores[:n].tolist(),
        bankLeftCard=[sorted(row[int(cursor):int(size)].tolist()) for row, cursor, size in zip(s.decks, s.cursor, [40, 30, 20])],
        nobleCandidates=np.where(np.asarray(env.noble_candidates(state))[:n + 1])[0].tolist(),
        lastOp={}, enableCt=0, costs=[0] * n, pt=0, ptOwner=None)


def compare(expected, actual):
    got = view(actual)
    for k in ('waitFor', 'waitThrowing', 'waitNoble', 'bankGem', 'bankCard', 'bankNoble', 'playerGem',
              'playerCard', 'playerBooked', 'playerNoble', 'playerCardCount', 'playerScore', 'bankLeftCard'):
        assert got[k] == expected[k], k
    assert (np.where(np.asarray(env.winners(actual)))[0] + 1).tolist() == expected['winner']


def invariant(s):
    s = jax.device_get(s)
    n = int(s.nplayers)
    supply = {2: 4, 3: 5, 4: 7}[n]
    np.testing.assert_array_equal(s.bank + s.gems.sum(0), [supply] * 5 + [5])
    assert np.all(s.bank >= 0) and np.all(s.gems >= 0)
    cards = list(s.market[s.market >= 0]) + list(s.reserved[s.reserved >= 0]) + np.where(s.bought >= 0)[0].tolist()
    for row, cursor, size in zip(s.decks, s.cursor, [40, 30, 20]):
        cards += row[int(cursor):size].tolist()
    assert sorted(cards) == list(range(90))
    assert np.any(np.asarray(MASK(s)))


def test_exact_card_table():
    t = oracle([dict(kind='table')])[0]
    for key, ours in [('cost', env.COST), ('points', env.POINT), ('bonus', env.BONUS), ('tier', env.TIER), ('nobles', env.NOBLE)]:
        np.testing.assert_array_equal(t[key], np.asarray(ours))


@pytest.mark.parametrize('players', [2, 3, 4])
def test_differential_games(players):
    rng = np.random.default_rng(4100 + players)
    s = env.reset(jax.random.PRNGKey(players), players)
    requests, states = [], []
    kinds = set()
    for i in range(320):
        if bool(s.done):
            s = env.reset(jax.random.PRNGKey(i + 42), players)
        before = s
        v = view(s)
        legal = np.flatnonzero(np.asarray(MASK(s)))
        # Bias toward purchases to exercise finishing, nobles and higher tiers.
        buys = legal[(legal >= 31) & (legal < 46)]
        a = int(rng.choice(buys if len(buys) and rng.random() < .9 else legal))
        s = STEP(s, jnp.int32(a))
        request = dict(view=v)
        if int(before.phase) == env.NORMAL:
            if a == 0:
                request.update(kind='pass', action=0)
            elif a < 31:
                request.update(kind='take', action=np.repeat(np.arange(6), np.asarray(env.TAKES[a - 1])).tolist())
            elif a < 46:
                slot = a - 31
                cid = int(env.targets(before)[slot])
                request.update(kind='buy', action=[cid, slot if slot >= 12 else slot % 4, slot >= 12])
                while int(s.phase) == env.PAYMENT:
                    pay = int(rng.choice(np.flatnonzero(np.asarray(MASK(s)))))
                    s = STEP(s, jnp.int32(pay))
                payment = np.asarray(before.gems[int(before.player)] - s.gems[int(before.player)])
                request['payment'] = dict(spend=payment[:5].tolist(), gold=int(payment[5]))
                if slot < 12:
                    tier = slot // 4
                    drawn = int(s.market.ravel()[slot])
                    request['drawIndex'] = v['bankLeftCard'][tier].index(drawn) if drawn >= 0 else 0
            else:
                slot = a - 46
                if slot < 12:
                    request.update(kind='reserve', action=[int(before.market.ravel()[slot]), slot % 4])
                    tier = slot // 4
                    drawn = int(s.market.ravel()[slot])
                else:
                    tier = slot - 12
                    request.update(kind='blind', action=tier)
                    drawn = list(set(np.asarray(s.reserved[int(before.player)]).tolist()) - set(v['playerBooked'][int(before.player)]) - {-1})[0]
                request['drawIndex'] = v['bankLeftCard'][tier].index(drawn) if drawn >= 0 else 0
        elif int(before.phase) == env.CHOOSE_NOBLE:
            request.update(kind='noble', action=a - 67)
        else:
            while int(s.phase) == env.DISCARD:
                a = int(rng.choice(np.flatnonzero(np.asarray(MASK(s)))))
                s = STEP(s, jnp.int32(a))
            request.update(kind='discard', action=np.asarray(before.gems[int(before.player)] - s.gems[int(before.player)]).tolist())
        kinds.add(request['kind'])
        invariant(s)
        requests.append(request)
        states.append(s)
    expected = oracle(requests)
    for i, (e, actual) in enumerate(zip(expected, states)):
        try:
            compare(e, actual)
        except AssertionError as exc:
            raise AssertionError(f'players={players}, operation={i}, {requests[i]}') from exc
    assert {'take', 'buy', 'reserve', 'blind', 'discard', 'pass'} <= kinds


def test_masks_partial_take_double_gold_and_pass():
    s = env.reset(jax.random.PRNGKey(0))
    mask = np.asarray(MASK(s))
    assert mask[0] and mask[1:31].all()
    s = s._replace(bank=s.bank.at[0].set(3))
    assert not bool(MASK(s)[26])  # first double
    assert len(env.TAKES) == 30 and not np.asarray(env.TAKES)[:, 5].any()


def test_noble_choice_before_discard_and_round_end_tie():
    s = env.reset(jax.random.PRNGKey(0))
    s = s._replace(bonuses=s.bonuses.at[0].set(4), gems=s.gems.at[0, 0].set(10))
    s = STEP(s, jnp.int32(1))
    assert int(s.phase) == env.CHOOSE_NOBLE and int(s.player) == 0
    s = STEP(s, jnp.int32(67 + np.flatnonzero(np.asarray(env.noble_candidates(s)))[0]))
    assert int(s.phase) == env.DISCARD and int(s.scores[0]) == 3
    s = STEP(s, jnp.int32(61))
    assert int(s.player) == 1
    s = s._replace(scores=jnp.array([15, 15, 0, 0]), bonuses=s.bonuses.at[1].set(3))
    s = STEP(s, jnp.int32(0))
    assert bool(s.done) and np.asarray(env.winners(s)).tolist() == [False, True, False, False]
    s = s._replace(bonuses=s.bonuses.at[1].set(4))
    np.testing.assert_array_equal(env.outcome(s), np.zeros(4))


def test_terminal_rewards_are_zero_sum_fractional_tie_payoffs():
    base = env.reset(jax.random.PRNGKey(820), 4)._replace(done=jnp.array(True))
    # Two winners: each +1/2; each of the two remaining players -1/2.
    tied = base._replace(scores=jnp.array([16, 16, 15, 14]), bonuses=jnp.zeros((4, 5), jnp.int32))
    np.testing.assert_allclose(env.outcome(tied), [.5, .5, -.5, -.5])
    # Three winners in a four-player game: +1/3 each and -1 for the loser.
    three_way = base._replace(scores=jnp.array([16, 16, 16, 15]), bonuses=jnp.zeros((4, 5), jnp.int32))
    np.testing.assert_allclose(env.outcome(three_way), [1 / 3, 1 / 3, 1 / 3, -1.])
    # All-way terminal tie is defined as neutral because n - m is zero.
    all_way = base._replace(scores=jnp.array([16, 16, 16, 16]), bonuses=jnp.zeros((4, 5), jnp.int32))
    np.testing.assert_allclose(env.outcome(all_way), np.zeros(4))


def test_shaped_rewards_remain_zero_sum():
    s = env.reset(jax.random.PRNGKey(821), 4)
    for action in (1, 2, 3, 4):
        ns = env.step(s, jnp.int32(action))
        reward = env.outcome(ns) + .25 * (env.potential(ns) - env.potential(s))
        np.testing.assert_allclose(reward.sum(), 0., atol=1e-7)
        s = ns


def test_observation_exposes_all_reserved_cards_but_not_hidden_deck_order():
    s = env.reset(jax.random.PRNGKey(9))
    t = s._replace(reserved=s.reserved.at[1, 0].set(0))
    # Opponent reserved-card identities are public and must affect observation.
    u = t._replace(reserved=t.reserved.at[1, 0].set(39))
    assert not np.array_equal(env.observe(t), env.observe(u))
    # Shuffled face-down identities remain hidden when the public count agrees.
    hidden = t._replace(decks=jnp.flip(t.decks, -1), bought=t.bought.at[4].set(1))
    np.testing.assert_array_equal(env.observe(t), env.observe(hidden))


def test_partial_reserved_rows_keep_empty_slots_distinct_from_real_cards():
    s = env.reset(jax.random.PRNGKey(905), 4)._replace(
        reserved=jnp.array([[0, -1, -1], [1, 39, -1], [-1, -1, -1], [40, 41, 42]], jnp.int32))
    encoded = env.card_features(s.reserved)
    np.testing.assert_array_equal(encoded[..., -1], s.reserved >= 0)
    np.testing.assert_array_equal(encoded[s.reserved < 0], np.zeros((6, encoded.shape[-1])))
    # A partially filled opponent row is part of the public observation too.
    added = s._replace(reserved=s.reserved.at[1, 2].set(2))
    assert not np.array_equal(env.observe(s), env.observe(added))


def test_observation_schema_keeps_legacy_checkpoints_usable():
    s = env.reset(jax.random.PRNGKey(906), 4)
    assert env.observe(s, 1).shape == (363,)
    assert env.observe(s, 2).shape == (498,)
    assert env.observe(s, 3).shape == (498,)


@pytest.mark.parametrize('players', [2, 3])
def test_current_observation_zeros_reserved_cards_for_padding_seats(players):
    s = env.reset(jax.random.PRNGKey(907 + players), players)._replace(
        reserved=jnp.full((4, 3), -1, jnp.int32)
            .at[0, 0].set(0).at[1, 0].set(1).at[2, 0].set(2))
    feature_width = env.card_features(jnp.array(0)).shape[-1]
    player_width = 6 + 5 + 1 + 3 + 1
    offset = 4 * player_width + 6 + 12 * feature_width
    size = 4 * 3 * feature_width
    current = np.asarray(env.observe(s, 3))[offset:offset + size].reshape(4, 3, feature_width)
    legacy = np.asarray(env.observe(s, 2))[offset:offset + size].reshape(4, 3, feature_width)
    assert np.any(current[:players])
    np.testing.assert_array_equal(current[players:], 0.)
    # v2 remains bit-for-bit compatible with the historical duplicated slots.
    assert np.any(legacy[players:])


@pytest.mark.parametrize('tier,blind_action', [(0, 58), (1, 59), (2, 60)])
def test_face_down_count_is_observed_and_empty_tier_cannot_be_reserved(tier, blind_action):
    s = env.reset(jax.random.PRNGKey(910 + tier), 4)
    exhausted = s._replace(cursor=s.cursor.at[tier].set(env.DECK_SIZE[tier]))
    assert not np.array_equal(env.observe(s), env.observe(exhausted))
    assert bool(env.legal_mask(s)[blind_action])
    assert not bool(env.legal_mask(exhausted)[blind_action])


def test_optional_payments_match_every_site_combination():
    s = env.reset(jax.random.PRNGKey(77))
    s = s._replace(gems=s.gems.at[0].set(jnp.array([2, 2, 2, 2, 2, 3])))
    legal = np.flatnonzero(np.asarray(MASK(s))[31:46])
    assert len(legal)
    slot = int(legal[0])
    cid = int(env.targets(s)[slot])
    expected = oracle([dict(kind='payments', view=view(s), action=cid)])[0]
    root = STEP(s, jnp.int32(31 + slot))
    assert int(root.phase) == env.PAYMENT
    payments = []
    def walk(state):
        if int(state.phase) != env.PAYMENT:
            paid = np.asarray(s.gems[0] - state.gems[0])
            payments.append(tuple(paid.tolist()))
            return
        for action in np.flatnonzero(np.asarray(MASK(state))):
            walk(STEP(state, jnp.int32(action)))
    walk(root)
    assert set(payments) == {tuple(p['spend'] + [p['gold']]) for p in expected}


def test_gpu_batched_invariants():
    @jax.jit
    def rollout(key):
        states = env.batch_reset(jax.random.split(key, 128), 4)
        def body(carry, i):
            s, rng = carry
            rng, ka = jax.random.split(rng)
            mask = env.batch_mask(s)
            a = jax.random.categorical(ka, jnp.where(mask, 0., -1e9)).astype(jnp.int32)
            ns = env.batch_step(s, a)
            conserved = jnp.all(ns.bank + ns.gems.sum(1) == jnp.array([7, 7, 7, 7, 7, 5]))
            nonnegative = jnp.all(ns.bank >= 0) & jnp.all(ns.gems >= 0)
            return (ns, rng), conserved & nonnegative & jnp.all(mask.any(-1))
        (_, _), checks = jax.lax.scan(body, (states, key), jnp.arange(256))
        return checks
    assert np.asarray(rollout(jax.random.PRNGKey(78))).all()


def test_vector_gae_same_player_and_terminal():
    r = jnp.array([[[0., 0., 0., 0.]], [[1., -1., 0., 0.]]])
    adv, ret = advantages(r, jnp.zeros_like(r), jnp.ones((1, 4)), jnp.array([[1.], [0.]]),
                          jnp.array([[1.], [0.]]), jnp.ones((2, 1)))
    np.testing.assert_array_equal(adv[:, 0, :2], [[1, -1], [1, -1]])
    np.testing.assert_array_equal(adv, ret)
