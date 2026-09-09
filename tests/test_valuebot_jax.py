import jax
import jax.numpy as jnp
import numpy as np

from splendor import env
from splendor.external_bot import ExternalBot, to_native
from splendor.valuebot_jax import card_values, choose_path, plan, act
from splendor.evaluate_external import make_match
from splendor import network


@jax.jit
def sample_states():
    key = jax.random.PRNGKey(20261011)
    states = env.batch_reset(jax.random.split(key, 16), 2)
    def body(carry, _):
        s, rng = carry
        rng, ka, kr = jax.random.split(rng, 3)
        mask = env.batch_mask(s)
        weight = jnp.where((jnp.arange(env.N_ACTIONS) >= 31) & (jnp.arange(env.N_ACTIONS) < 46), 3., 0.)
        a = jax.random.categorical(ka, jnp.where(mask, weight, -1e9)).astype(jnp.int32)
        ns = env.batch_step(s, a)
        fresh = env.batch_reset(jax.random.split(kr, 16), 2)
        ns = jax.tree.map(lambda x, y: jnp.where(ns.done.reshape((16,) + (1,) * (x.ndim - 1)), y, x), ns, fresh)
        return (ns, rng), s
    _, history = jax.lax.scan(body, (states, key), None, length=128)
    return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), history)


def test_valuebot_matches_original_targets_actions_and_returns():
    history = sample_states()
    host = jax.device_get(history)
    plans = jax.device_get(jax.jit(jax.vmap(plan))(history))
    values = jax.device_get(jax.jit(jax.vmap(lambda s: card_values(s, s.player)[0]))(history))
    masks = jax.device_get(env.batch_mask(history))
    checked = 0
    for index in np.flatnonzero((host.phase == env.NORMAL) & ~host.done)[::2]:
        s = jax.tree.map(lambda x: x[index], host)
        native = to_native(s)
        bot = ExternalBot()
        ids = np.concatenate((s.market.ravel(), s.reserved[int(s.player)]))
        for slot, cid in enumerate(ids):
            if cid >= 0:
                card = native.GetReservedOrRevealedCardById(f'c{cid}')
                np.testing.assert_allclose(values[index, slot], bot.native._GetSelfCardValue(card, native), rtol=2e-6)
        target = bot.native._GetNextPurchaseTarget(native)
        expected = [int(c.asset_id[1:]) for c in target]
        actual = [int(ids[plans.primary[index]])]
        if plans.secondary[index] >= 0:
            actual.append(int(ids[plans.secondary[index]]))
        assert actual == expected, (index, actual, expected)
        assert bool(plans.valid[index]), index
        assert int(plans.action[index]) == bot.act(s, masks[index]), index
        np.testing.assert_array_equal(plans.returned[index], bot.returned, err_msg=f'index={index}')
        checked += 1
    assert checked >= 500


def test_primitive_matches_original_through_games():
    step = jax.jit(env.step)
    choose = jax.jit(act)
    mask_fn = jax.jit(env.legal_mask)
    s = env.reset(jax.random.PRNGKey(312))
    carry = jnp.zeros(6, jnp.int32)
    reference = ExternalBot()
    for i in range(180):
        if bool(s.done):
            s = env.reset(jax.random.PRNGKey(i))
            carry = jnp.zeros(6, jnp.int32)
            reference = ExternalBot()
        host = jax.device_get(s)
        expected = reference.act(host, np.asarray(mask_fn(s)))
        actual, carry, valid = choose(s, carry)
        assert bool(valid) and int(actual) == expected, (i, int(actual), expected)
        s = step(s, actual)


def test_compiled_batched_match_is_legal_and_seat_paired():
    key = jax.random.PRNGKey(91)
    params = network.init(key, env.observe(env.reset(key)).shape[0], 32)
    result = make_match(games=8, max_decisions=800, bf16=False)(params, 20261012)
    assert int(result['bot_invalid_actions']) == 0
    np.testing.assert_array_equal(result['seat_a'], [0, 0, 0, 0, 1, 1, 1, 1])
    assert np.all(np.asarray(result['turns']) >= 0)
