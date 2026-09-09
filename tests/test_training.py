import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

from splendor import env, network
from splendor.agent import Agent, from_hullqin_view
from splendor.train import Config, load, make_update, save
from test_rules import view


def test_checkpoint_resume_reproduces_next_update(tmp_path):
    cfg = Config(envs=8, horizon=8, epochs=1, minibatches=2, width=32)
    key, kp, ke = jax.random.split(jax.random.PRNGKey(3), 3)
    states = env.batch_reset(jax.random.split(ke, cfg.envs), 2)
    params = network.init(kp, env.observe(env.reset(ke)).shape[0], cfg.width)
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(cfg.lr, eps=1e-5))
    opt_state = opt.init(params)
    run = make_update(cfg, opt)
    params, opt_state, states, key, _ = run(params, opt_state, states, key)
    path = tmp_path / 'state.npz'
    save(path, params, cfg, 1, opt_state, states, key)
    result = run(params, opt_state, states, key)
    restored, restored_cfg = load(path)
    assert cfg == restored_cfg
    with np.load(path, allow_pickle=False) as data:
        tree = jax.tree.structure(opt.init(restored))
        restored_opt = jax.tree.unflatten(tree, [jnp.asarray(data[f'opt_{i}']) for i in range(tree.num_leaves)])
        restored_states = env.State(**{f: jnp.asarray(data[f'state_{f}']) for f in env.State._fields})
        restored_key = jnp.asarray(data['key'])
    repeated = run(restored, restored_opt, restored_states, restored_key)
    for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(repeated)):
        np.testing.assert_array_equal(a, b)


def test_mixed_player_update_and_widening():
    cfg = Config(envs=9, horizon=8, epochs=1, minibatches=1, width=48,
                 players=4, mixed_players=True)
    key, kp, ke = jax.random.split(jax.random.PRNGKey(31), 3)
    small = network.init(kp, env.observe(env.reset(ke)).shape[0], 32)
    exact = network.widen(small, 48, jax.random.PRNGKey(32), noise=0.)
    state = env.reset(jax.random.PRNGKey(33), 3)
    before = network.apply(small, env.observe(state), env.legal_mask(state))[0]
    after = network.apply(exact, env.observe(state), env.legal_mask(state))[0]
    np.testing.assert_allclose(before, after, rtol=2e-5, atol=2e-5)
    params = network.widen(small, 48, jax.random.PRNGKey(34))
    counts = 2 + jnp.arange(cfg.envs) % 3
    expected_counts = np.asarray(counts)
    states = jax.vmap(env.reset)(jax.random.split(ke, cfg.envs), counts)
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(cfg.lr, eps=1e-5))
    result = make_update(cfg, opt)(params, opt.init(params), states, key)
    stats = result[-1]
    assert int(stats['games_by_players'].sum()) == int(stats['games'])
    np.testing.assert_array_equal(result[2].nplayers, expected_counts)


def test_resnet_checkpoint_roundtrip(tmp_path):
    cfg = Config(width=48, residual_blocks=6, residual_taper=True, value_head_width=32,
                 value_head_layers=2, value_loss_coef=1.)
    state = env.reset(jax.random.PRNGKey(35))
    params = network.init(jax.random.PRNGKey(36), env.observe(state).shape[0], cfg.width,
                          cfg.residual_blocks, cfg.residual_taper, cfg.value_head_width,
                          cfg.value_head_layers)
    logits, values = network.apply(params, env.observe(state), env.legal_mask(state))
    path = tmp_path / 'resnet_policy.npz'
    save(path, params, cfg, 0)
    restored, restored_cfg = load(path)
    restored_logits, restored_values = network.apply(restored, env.observe(state), env.legal_mask(state))
    assert restored_cfg == cfg
    assert logits.shape == (env.N_ACTIONS,) and values.shape == (4,)
    assert network.parameter_count(params) > 0
    np.testing.assert_array_equal(logits, restored_logits)
    np.testing.assert_array_equal(values, restored_values)


def test_site_adapter_and_complete_actions(tmp_path):
    cfg = Config(width=32)
    s = env.reset(jax.random.PRNGKey(5))
    params = network.init(jax.random.PRNGKey(7), env.observe(s).shape[0], cfg.width)
    path = tmp_path / 'policy.npz'
    save(path, params, cfg, 0)
    agent = Agent(path, deterministic=True)
    np.testing.assert_array_equal(env.observe(s), env.observe(from_hullqin_view(view(s))))
    action = agent.plan_view(view(s), jax.random.PRNGKey(8))
    assert action['kind'] in ('take', 'reserve', 'reserve_blind', 'pass')
    s = s._replace(phase=jnp.int32(env.DISCARD), gems=s.gems.at[0].set(jnp.array([3, 3, 3, 2, 1, 1])))
    action = agent.plan_view(view(s), jax.random.PRNGKey(9))
    assert action['kind'] == 'discard' and sum(action['gems']) == 3
    assert np.all(np.asarray(action['gems']) <= np.asarray(s.gems[0]))
    # Force the actor's normal-phase decision to a buy; subsequent payment
    # microdecisions still come from the checkpoint and must form a site payment.
    s = env.reset(jax.random.PRNGKey(5))._replace(gems=s.gems.at[0].set(jnp.array([2, 2, 2, 2, 2, 3])), phase=jnp.int32(0))
    slot = int(np.flatnonzero(np.asarray(env.legal_mask(s))[31:46])[0])
    original_act = agent.act
    agent.act = lambda state, key: 31 + slot if int(state.phase) == env.NORMAL else original_act(state, key)
    action = agent.plan_view(view(s), jax.random.PRNGKey(10))
    assert action['kind'] == 'buy'
    payment = action['payment']
    paid = np.asarray(payment['spend'])
    cost = np.asarray(jnp.maximum(env.COST[action['card_id']] - s.bonuses[0], 0))
    assert np.all(paid <= np.asarray(s.gems[0, :5])) and np.all(paid <= cost)
    assert int((cost - paid).sum()) == payment['gold'] <= int(s.gems[0, 5])
