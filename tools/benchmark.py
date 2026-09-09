"""Benchmark complete PPO updates, synchronize, exclude compilation and warmup."""
import json
from pathlib import Path
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
    for n, bf16 in [(512, False), (2048, False), (2048, True), (4096, True), (8192, True)]:
        cfg = Config(envs=n, bf16=bf16, minibatches=max(8, n // 64))
        key, kp, ke = jax.random.split(jax.random.PRNGKey(41), 3)
        states = env.batch_reset(jax.random.split(ke, n), cfg.players)
        params = network.init(kp, env.observe(env.reset(ke)).shape[0], cfg.width)
        optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.adam(cfg.lr, eps=1e-5))
        opt_state = optimizer.init(params)
        update = make_update(cfg, optimizer)
        # Five warm-up updates also populate mid-game states.
        for _ in range(5):
            params, opt_state, states, key, stats = update(params, opt_state, states, key)
        jax.block_until_ready(stats)
        elapsed = []
        for _ in range(20):
            start = time.perf_counter()
            params, opt_state, states, key, stats = update(params, opt_state, states, key)
            jax.block_until_ready(stats)
            elapsed.append(time.perf_counter() - start)
        row = dict(envs=n, bf16=bf16, horizon=cfg.horizon, epochs=cfg.epochs,
                   minibatches=cfg.minibatches, median_seconds=float(np.median(elapsed)),
                   median_decisions_per_second=n * cfg.horizon / float(np.median(elapsed)),
                   device=jax.devices()[0].device_kind)
        results.append(row)
        print(json.dumps(row), flush=True)
        del params, opt_state, states, update
        jax.clear_caches()
    out = Path('runs/benchmark.json')
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
