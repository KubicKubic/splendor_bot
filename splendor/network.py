"""Small fused MLP actor / multi-seat critic with optional BF16 matrix products."""
import jax
import jax.numpy as jnp

from .env import N_ACTIONS


def init(key, obs_dim, width=256):
    sizes = [obs_dim, width, width, N_ACTIONS + 4]
    keys = jax.random.split(key, 3)
    result = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(keys[i], (a, b)) * jnp.sqrt(2. / a)
        if i == 2:
            w = w.at[:, :N_ACTIONS].multiply(.01)
            w = w.at[:, N_ACTIONS:].multiply(.5)
        result.append(dict(w=w, b=jnp.zeros(b)))
    return result


def apply(params, obs, mask, bf16=False):
    dtype = jnp.bfloat16 if bf16 else jnp.float32
    x = obs.astype(dtype)
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer['w'].astype(dtype) + layer['b'].astype(dtype))
    layer = params[-1]
    out = (x @ layer['w'].astype(dtype) + layer['b'].astype(dtype)).astype(jnp.float32)
    return jnp.where(mask, out[..., :N_ACTIONS], -1e9), out[..., N_ACTIONS:]


def widen(params, width, key, noise=1e-4):
    """Net2Wider expansion of both hidden layers with tiny symmetry breaking.

    Repeated units have their outgoing weights divided by repeat count, which
    preserves the original function before the small perturbation is applied.
    """
    old = params[0]['b'].shape[0]
    if width < old:
        raise ValueError('widen cannot shrink a network')
    if width == old:
        return params
    k1, k2, kn1, kn2 = jax.random.split(key, 4)
    map1 = jnp.concatenate((jnp.arange(old), jax.random.randint(k1, (width - old,), 0, old)))
    map2 = jnp.concatenate((jnp.arange(old), jax.random.randint(k2, (width - old,), 0, old)))
    count1 = jnp.bincount(map1, length=old)
    count2 = jnp.bincount(map2, length=old)
    w0 = params[0]['w'][:, map1]
    b0 = params[0]['b'][map1]
    w1 = params[1]['w'][map1][:, map2] / count1[map1, None]
    b1 = params[1]['b'][map2]
    w2 = params[2]['w'][map2] / count2[map2, None]
    # Perturb only duplicate incoming columns. Original units remain exact and
    # duplicate activations separate immediately under gradient updates.
    duplicate1 = (jnp.arange(width) >= old)[None, :]
    duplicate2 = (jnp.arange(width) >= old)[None, :]
    w0 = w0 + noise * jax.random.normal(kn1, w0.shape) * duplicate1
    w1 = w1 + noise * jax.random.normal(kn2, w1.shape) * duplicate2
    return [dict(w=w0, b=b0), dict(w=w1, b=b1), dict(w=w2, b=params[2]['b'])]


def absolute_values(relative, player, nplayers):
    nplayers = jnp.asarray(nplayers)
    index = (jnp.arange(4) - player[..., None]) % nplayers[..., None]
    return (jnp.take_along_axis(relative, index, -1) *
            (jnp.arange(4) < nplayers[..., None]))
