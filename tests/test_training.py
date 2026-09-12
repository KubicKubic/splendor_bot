import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

from splendor import env, network
from splendor.agent import Agent, from_hullqin_view
from splendor.train import (Config, advantages, load, make_update,
                            reconcile_metrics, resume_mismatches, save,
                            value_statistics)
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


def test_epoch_count_can_change_only_at_resume_boundary():
    original = Config(epochs=3, updates=500, save_every=500, log_every=10)
    continued = Config(epochs=1, updates=100000, save_every=250, log_every=50)
    assert resume_mismatches(original, continued) == {}
    lower_lr = Config(lr=2e-5)
    assert resume_mismatches(original, lower_lr) == {}
    ablated_turns = Config(zero_turn_feature=True)
    assert resume_mismatches(original, ablated_turns) == {}
    incompatible = Config(epochs=1, gamma=.99)
    assert resume_mismatches(original, incompatible) == {'gamma': (original.gamma, .99)}


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
    assert int(stats['reset_overflows']) == 0
    np.testing.assert_array_equal(result[2].nplayers, expected_counts)


def test_four_hidden_layer_mlp_roundtrip(tmp_path):
    cfg = Config(width=32, mlp_hidden_layers=4, mlp_activation='gelu')
    state = env.reset(jax.random.PRNGKey(341), 4)
    params = network.init_from_config(jax.random.PRNGKey(342), env.observe(state).shape[0], cfg)
    assert len(params['flat_gelu_layers']) == 5
    logits, values = network.apply(params, env.observe(state), env.legal_mask(state))
    path = tmp_path / 'deep_mlp.npz'
    save(path, params, cfg, 0)
    restored, restored_cfg = load(path)
    restored_logits, restored_values = network.apply(restored, env.observe(state), env.legal_mask(state))
    assert restored_cfg == cfg
    np.testing.assert_array_equal(logits, restored_logits)
    np.testing.assert_array_equal(values, restored_values)


def test_four_hidden_layer_gelu_mlp_runs_through_ppo_update():
    cfg = Config(envs=6, horizon=4, epochs=1, minibatches=2, width=32,
                 mlp_hidden_layers=4, mlp_activation='gelu', players=4,
                 mixed_players=True)
    key, kp, ke = jax.random.split(jax.random.PRNGKey(343), 3)
    counts = 2 + jnp.arange(cfg.envs) % 3
    states = jax.vmap(env.reset)(jax.random.split(ke, cfg.envs), counts)
    first_state = jax.tree.map(lambda value: value[0], states)
    params = network.init_from_config(kp, env.observe(first_state, cfg.observation_version).shape[-1], cfg)
    optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.adam(cfg.lr, eps=1e-5))
    result = make_update(cfg, optimizer)(params, optimizer.init(params), states, key)
    assert np.isfinite(np.asarray(result[-1]['loss'])).all()
    assert np.isfinite(np.asarray(result[-1]['value_prediction_mse']))


def test_mixed_player_absolute_value_rotation_uses_each_environment_count():
    relative = jnp.array([[10., 11., 90., 91.], [20., 21., 22., 92.], [30., 31., 32., 33.]])
    player = jnp.array([1, 2, 3])
    nplayers = jnp.array([2, 3, 4])
    absolute = network.absolute_values(relative, player, nplayers)
    np.testing.assert_array_equal(absolute, [[.5, -.5, 0, 0], [0, 1, -1, 0], [-.5, .5, 1.5, -1.5]])
    np.testing.assert_allclose(absolute.sum(-1), 0.)


def test_standard_explained_variance_separates_bias_from_variance():
    target = jnp.array([[[-1., 1., 0., 0.], [-.5, .5, 0., 0.]]])
    prediction = target + jnp.array([[[2., 2., 0., 0.], [2., 2., 0., 0.]]])
    active = jnp.array([[[True, True, False, False], [True, True, False, False]]])
    mse, ev, _, _, bias = value_statistics(target, prediction, active)
    np.testing.assert_allclose(mse, 4.)
    np.testing.assert_allclose(ev, 1.)
    np.testing.assert_allclose(bias, -2.)


def test_lambda_one_targets_telescope_to_rollout_bootstrap():
    value = jnp.array([[[.2, -.2, 0., 0.]], [[.4, -.4, 0., 0.]], [[.1, -.1, 0., 0.]]])
    last = jnp.array([[.7, -.7, 0., 0.]])
    zeros = jnp.zeros_like(value)
    live = jnp.ones((3, 1))
    _, target = advantages(zeros, value, last, live, live, live)
    np.testing.assert_allclose(target, jnp.broadcast_to(last, value.shape), atol=1e-6)
    np.testing.assert_allclose(target.sum(-1), 0., atol=1e-7)


def test_max_turns_is_diagnostic_not_an_artificial_terminal():
    cfg = Config(envs=6, horizon=8, epochs=1, minibatches=1, width=32,
                 players=2, max_turns=1)
    key, kp, ke = jax.random.split(jax.random.PRNGKey(131), 3)
    states = env.batch_reset(jax.random.split(ke, cfg.envs), 2)
    params = network.init(kp, env.observe(env.reset(ke)).shape[0], cfg.width)
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(cfg.lr, eps=1e-5))
    result = make_update(cfg, opt)(params, opt.init(params), states, key)
    stats = result[-1]
    assert int(stats['long_games']) > 0
    assert int(stats['resets']) == int(stats['games'])
    assert np.any(np.asarray(result[2].turns) > cfg.max_turns)


def test_update_uses_configured_observation_version(monkeypatch):
    cfg = Config(envs=6, horizon=4, epochs=1, minibatches=1, width=32,
                 players=2, observation_version=2)
    key, kp, ke = jax.random.split(jax.random.PRNGKey(132), 3)
    states = env.batch_reset(jax.random.split(ke, cfg.envs), 2)
    params = network.init(kp, env.observe(env.reset(ke), 2).shape[0], cfg.width)
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(cfg.lr, eps=1e-5))
    observed_versions = []
    original = env.batch_observe_for_version

    def record_version(batch, version, zero_turn_feature=False):
        observed_versions.append((version, zero_turn_feature))
        return original(batch, version, zero_turn_feature)

    monkeypatch.setattr(env, 'batch_observe_for_version', record_version)
    make_update(cfg, opt)(params, opt.init(params), states, key)
    assert observed_versions and set(observed_versions) == {(2, False)}


def test_reconcile_metrics_matches_resumed_checkpoint(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    rows = [dict(update=1, value='old'), dict(update=2, value='old'),
            dict(update=2, value='latest'), dict(update=3, value='unrecoverable')]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    reconcile_metrics(path, 2)
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        dict(update=1, value='old'), dict(update=2, value='latest')]


def test_resnet_checkpoint_roundtrip(tmp_path):
    cfg = Config(width=48, residual_blocks=6, residual_taper=True, residual_stage_widths='48,30,18',
                 policy_head_width=36, policy_head_layers=2, value_head_width=32,
                 value_head_layers=2, value_loss_coef=1.)
    state = env.reset(jax.random.PRNGKey(35))
    params = network.init(jax.random.PRNGKey(36), env.observe(state).shape[0], cfg.width,
                          cfg.residual_blocks, cfg.residual_taper, cfg.value_head_width,
                          cfg.value_head_layers, cfg.policy_head_width, cfg.policy_head_layers,
                          cfg.residual_stage_widths)
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


def test_transformer_tokens_cover_every_public_feature_and_fit_budget(tmp_path):
    # Activate each scalar observation feature in isolation. Every row must
    # reach at least one typed input; this catches silent holes and bad slices.
    basis = jnp.eye(network.OBSERVATION_V3_DIM)
    typed, validity = network.structured_observation(basis)
    coverage = sum(jnp.abs(value).reshape((len(basis), -1)).sum(-1)
                   for value in typed.values())
    assert np.asarray(coverage > 0).all()
    assert validity.shape == (network.OBSERVATION_V3_DIM, network.TOKEN_COUNT)

    cfg = Config(width=64, architecture='transformer', transformer_layers=4,
        transformer_heads=4, transformer_ff_dim=256, token_embed_width=64,
        policy_head_width=640, policy_head_layers=2,
        value_head_width=448, value_head_layers=2, observation_version=3)
    state = env.reset(jax.random.PRNGKey(801), 4)
    params = network.init_from_config(
        jax.random.PRNGKey(802), env.observe(state, 3).shape[-1], cfg)
    assert network.parameter_count(params) == 1_037_203
    logits, values = network.apply(params, env.observe(state, 3), env.legal_mask(state), True)
    assert logits.shape == (env.N_ACTIONS,) and values.shape == (4,)
    assert np.isfinite(np.asarray(values)).all()

    path = tmp_path / 'transformer_policy.npz'
    save(path, params, cfg, 0)
    restored, restored_cfg = load(path)
    restored_logits, restored_values = network.apply(
        restored, env.observe(state, 3), env.legal_mask(state), True)
    assert restored_cfg == cfg
    np.testing.assert_array_equal(logits, restored_logits)
    np.testing.assert_array_equal(values, restored_values)


def test_transformer_runs_through_complete_ppo_update():
    cfg = Config(envs=6, horizon=4, epochs=1, minibatches=2, width=32,
        architecture='transformer', transformer_layers=2, transformer_heads=4,
        transformer_ff_dim=64, token_embed_width=16,
        policy_head_width=32, policy_head_layers=1,
        value_head_width=24, value_head_layers=1,
        players=4, mixed_players=True, observation_version=3)
    key, kp, ke = jax.random.split(jax.random.PRNGKey(803), 3)
    counts = 2 + jnp.arange(cfg.envs) % 3
    states = jax.vmap(env.reset)(jax.random.split(ke, cfg.envs), counts)
    first_state = jax.tree.map(lambda value: value[0], states)
    params = network.init_from_config(kp, env.observe(first_state, 3).shape[-1], cfg)
    optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.adam(cfg.lr, eps=1e-5))
    result = make_update(cfg, optimizer)(params, optimizer.init(params), states, key)
    stats = result[-1]
    assert np.isfinite(np.asarray(stats['loss'])).all()
    assert np.isfinite(np.asarray(stats['value_prediction_mse']))
    assert np.isfinite(np.asarray(stats['value_explained_variance']))


def test_site_adapter_and_complete_actions(tmp_path):
    cfg = Config(width=32)
    s = env.reset(jax.random.PRNGKey(5))
    params = network.init(jax.random.PRNGKey(7), env.observe(s).shape[0], cfg.width)
    path = tmp_path / 'policy.npz'
    save(path, params, cfg, 0)
    agent = Agent(path, deterministic=True)
    np.testing.assert_array_equal(env.observe(s), env.observe(from_hullqin_view(view(s))))
    deploy_view = view(s)
    deploy_view.pop('playerCard')
    np.testing.assert_array_equal(env.observe(s), env.observe(from_hullqin_view(deploy_view)))
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
