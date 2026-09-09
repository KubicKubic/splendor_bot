"""Fused MLP or residual MLP actor / multi-seat critic with optional BF16 products."""
import jax
import jax.numpy as jnp

from .env import N_ACTIONS


def _linear(key, fan_in, fan_out, scale=1.):
    return dict(w=jax.random.normal(key, (fan_in, fan_out)) * (scale * jnp.sqrt(2. / fan_in)),
                b=jnp.zeros(fan_out))


def init(key, obs_dim, width=256, residual_blocks=0, residual_taper=False):
    """Initialize the legacy two-hidden-layer MLP or a pre-activation residual MLP.

    Each residual block contains two affine transforms.  Tapered networks split
    blocks across three stages, reducing width to 11/16 then 7/16 of the stem;
    transition blocks learn a projection residual connection.
    """
    if residual_blocks < 0:
        raise ValueError('residual_blocks must be non-negative')
    if residual_blocks:
        if residual_taper:
            if residual_blocks % 3:
                raise ValueError('tapered residual networks need a multiple of three blocks')
            stage_widths = (width, width * 11 // 16, width * 7 // 16)
            block_widths = sum(([stage] * (residual_blocks // 3) for stage in stage_widths), [])
        else:
            block_widths = [width] * residual_blocks
        keys = jax.random.split(key, 2 + 3 * residual_blocks)
        stem = _linear(keys[0], obs_dim, width)
        blocks = []
        in_width = width
        for i, out_width in enumerate(block_widths):
            block = dict(w1=_linear(keys[1 + 3 * i], in_width, out_width)['w'],
                         b1=jnp.zeros(out_width),
                         w2=_linear(keys[2 + 3 * i], out_width, out_width, scale=.1)['w'],
                         b2=jnp.zeros(out_width))
            if in_width != out_width:
                block['skip'] = _linear(keys[3 + 3 * i], in_width, out_width, scale=1.)['w']
            blocks.append(block)
            in_width = out_width
        head = _linear(keys[-1], in_width, N_ACTIONS + 4)
        head['w'] = head['w'].at[:, :N_ACTIONS].multiply(.01)
        head['w'] = head['w'].at[:, N_ACTIONS:].multiply(.5)
        return dict(stem=stem, blocks=blocks, head=head)
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
    if isinstance(params, dict):
        stem = params['stem']
        x = jax.nn.relu(x @ stem['w'].astype(dtype) + stem['b'].astype(dtype))
        for block in params['blocks']:
            residual = x
            x = jax.nn.relu(x @ block['w1'].astype(dtype) + block['b1'].astype(dtype))
            x = x @ block['w2'].astype(dtype) + block['b2'].astype(dtype)
            if 'skip' in block:
                residual = residual @ block['skip'].astype(dtype)
            x = jax.nn.relu(x + residual)
        layer = params['head']
        out = (x @ layer['w'].astype(dtype) + layer['b'].astype(dtype)).astype(jnp.float32)
        return jnp.where(mask, out[..., :N_ACTIONS], -1e9), out[..., N_ACTIONS:]
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
    if not isinstance(params, list):
        raise ValueError('widen only supports legacy MLP parameters')
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


def parameter_count(params):
    return sum(value.size for value in jax.tree.leaves(params))


def absolute_values(relative, player, nplayers):
    nplayers = jnp.asarray(nplayers)
    index = (jnp.arange(4) - player[..., None]) % nplayers[..., None]
    return (jnp.take_along_axis(relative, index, -1) *
            (jnp.arange(4) < nplayers[..., None]))
