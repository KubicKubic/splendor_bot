"""Benchmark ~1M-parameter mixed-player PPO configurations on one GPU."""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from splendor import env, network
from splendor.train import Config, make_update


def main():
    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')
    results = []
    for minibatches in (32, 64, 128):
        cfg = Config(envs=8192, horizon=128, epochs=3, minibatches=minibatches,
                     width=800, players=4, mixed_players=True, gamma=1., bf16=True)
        key, kp, ke = jax.random.split(jax.random.PRNGKey(41), 3)
        counts = 2 + jnp.arange(cfg.envs) % 3
        states = jax.vmap(env.reset)(jax.random.split(ke, cfg.envs), counts)
        params = network.init(kp, env.observe(env.reset(ke)).shape[0], cfg.width)
        optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.adam(cfg.lr, eps=1e-5))
        opt_state = optimizer.init(params)
        update = make_update(cfg, optimizer)
        for _ in range(3):
            params, opt_state, states, key, stats = update(params, opt_state, states, key)
        jax.block_until_ready(stats)
        elapsed = []
        for _ in range(8):
            started = time.perf_counter()
            params, opt_state, states, key, stats = update(params, opt_state, states, key)
            jax.block_until_ready(stats)
            elapsed.append(time.perf_counter() - started)
        seconds = float(np.median(elapsed))
        row = dict(width=cfg.width, parameters=sum(x.size for x in jax.tree.leaves(params)),
                   envs=cfg.envs, horizon=cfg.horizon, epochs=cfg.epochs,
                   minibatches=minibatches,
                   minibatch_size=cfg.envs * cfg.horizon // minibatches,
                   median_seconds=seconds,
                   decisions_per_second=cfg.envs * cfg.horizon / seconds,
                   device=jax.devices()[0].device_kind)
        results.append(row)
        print(json.dumps(row), flush=True)
        del params, opt_state, states, update
        jax.clear_caches()
    with open('runs/benchmark_mixed_scale.json', 'w') as f:
        json.dump(results, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
