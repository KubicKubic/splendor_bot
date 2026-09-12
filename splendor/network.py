"""Legacy MLP/ResNet and typed-token Transformer actor-critic networks."""
import jax
import jax.numpy as jnp

from .env import N_ACTIONS


def _linear(key, fan_in, fan_out, scale=1.):
    return dict(w=jax.random.normal(key, (fan_in, fan_out)) * (scale * jnp.sqrt(2. / fan_in)),
                b=jnp.zeros(fan_out))


def _xavier(key, fan_in, fan_out, scale=1.):
    return dict(w=jax.random.normal(key, (fan_in, fan_out)) * (scale / jnp.sqrt(fan_in)),
                b=jnp.zeros(fan_out))


# Observation v3 is deliberately parsed here rather than reconstructed from
# State. PPO stores compact BF16 observations, and deployment also goes through
# env.observe. These widths are asserted by structured_observation so a future
# schema change cannot silently shift a token boundary.
OBSERVATION_V3_DIM = 498
TOKEN_COUNT = 47
FUSED_ATTENTION_TOKEN_COUNT = 64
POOLED_LATENT_COUNT = 16
POOLED_LATENT_LAYERS = 4


def structured_observation(obs):
    """Losslessly split observation v3 into semantically typed token inputs.

    Every one of the 498 public features is present in at least one returned
    tensor. Padding/empty validity is returned separately for attention keys.
    Purchased-card identities and hidden deck order are not in observation v3
    and therefore cannot leak into the model.
    """
    if obs.shape[-1] != OBSERVATION_V3_DIM:
        raise ValueError(f'Typed-token Transformer requires observation v3 ({OBSERVATION_V3_DIM} features)')
    cursor = 0

    def take(width):
        nonlocal cursor
        value = obs[..., cursor:cursor + width]
        cursor += width
        return value

    players = take(4 * 16).reshape(obs.shape[:-1] + (4, 16))
    bank = take(6)[..., None, :]
    market = take(12 * 15).reshape(obs.shape[:-1] + (12, 15))
    reserved = take(12 * 15).reshape(obs.shape[:-1] + (12, 15))
    nobles = take(5 * 6).reshape(obs.shape[:-1] + (5, 6))
    decks = take(3)[..., :, None]
    context = take(4 + 5 + 5 + 15 + 4 + 2)[..., None, :]
    if cursor != OBSERVATION_V3_DIM:
        raise AssertionError(f'Observation parser consumed {cursor} features')

    active = players[..., 15:16]
    inputs = dict(
        market=market,
        reserved=reserved,
        nobles=nobles,
        gems=jnp.concatenate((players[..., :6], active), -1),
        discounts=jnp.concatenate((players[..., 6:11], active), -1),
        players=jnp.concatenate((players[..., 11:15], active), -1),
        bank=bank,
        decks=decks,
        context=context,
    )
    validity = jnp.concatenate((
        jnp.ones(obs.shape[:-1] + (1,), bool),  # learned global token
        market[..., -1] > 0,
        reserved[..., -1] > 0,
        nobles[..., -1] > 0,
        jnp.concatenate((active[..., 0] > 0,) * 3, -1),
        jnp.ones(obs.shape[:-1] + (1 + 3 + 1,), bool),  # bank, decks, context
    ), -1)
    return inputs, validity


def _init_mlp(key, fan_in, hidden, fan_out):
    k1, k2 = jax.random.split(key)
    return [_xavier(k1, fan_in, hidden), _xavier(k2, hidden, fan_out)]


def _init_transformer(key, obs_dim, model_dim, layers, heads, ff_dim, embed_width,
                      value_head_width, value_head_layers, policy_head_width,
                      policy_head_layers, attention='dense'):
    if obs_dim != OBSERVATION_V3_DIM:
        raise ValueError('Typed-token Transformer only supports observation v3')
    if min(model_dim, layers, heads, ff_dim, embed_width) <= 0 or model_dim % heads:
        raise ValueError('Invalid Transformer dimensions or attention head count')
    if not value_head_width or not value_head_layers or not policy_head_width or not policy_head_layers:
        raise ValueError('Typed-token Transformer requires explicit policy/value heads')
    if attention not in ('dense', 'star'):
        raise ValueError(f'Unknown Transformer attention pattern {attention!r}')

    input_widths = dict(market=15, reserved=15, nobles=6, gems=7, discounts=6,
                        players=5, bank=6, decks=1, context=35)
    key_count = (len(input_widths) + 1 + layers * (4 + (attention == 'star'))
                 + policy_head_layers + 1 + value_head_layers + 1)
    keys = iter(jax.random.split(key, key_count))
    embedders = {name: _init_mlp(next(keys), width, embed_width, model_dim)
                 for name, width in input_widths.items()}
    position_key = next(keys)
    transformer = []
    for _ in range(layers):
        layer = dict(
            norm1=dict(scale=jnp.ones(model_dim), bias=jnp.zeros(model_dim)),
            qkv=_xavier(next(keys), model_dim, 3 * model_dim),
            projection=_xavier(next(keys), model_dim, model_dim),
            norm2=dict(scale=jnp.ones(model_dim), bias=jnp.zeros(model_dim)),
            ff1=_xavier(next(keys), model_dim, ff_dim),
            ff2=_xavier(next(keys), ff_dim, model_dim),
        )
        # Star attention has one learned information-routing hub.  It attends
        # over all semantic tokens then broadcasts its updated state back to
        # them, retaining multi-hop token communication without T^2 attention.
        if attention == 'star':
            layer['broadcast'] = _xavier(next(keys), model_dim, model_dim)
        transformer.append(layer)
    policy_layers = []
    width = model_dim
    for _ in range(policy_head_layers):
        policy_layers.append(_linear(next(keys), width, policy_head_width))
        width = policy_head_width
    policy_out = _linear(next(keys), width, N_ACTIONS)
    policy_out['w'] = policy_out['w'] * .01
    value_layers = []
    width = 3 * model_dim  # attended player summary + its gems + discounts
    for _ in range(value_head_layers):
        value_layers.append(_linear(next(keys), width, value_head_width))
        width = value_head_width
    value_out = _linear(next(keys), width, 1, scale=.5)
    return dict(
        token_embedders=embedders,
        global_token=jax.random.normal(position_key, (1, model_dim)) * .02,
        positions=jax.random.normal(jax.random.fold_in(position_key, 1), (TOKEN_COUNT, model_dim)) * .02,
        **({'star_transformer_layers': transformer} if attention == 'star'
           else {'transformer_layers': transformer}),
        final_norm=dict(scale=jnp.ones(model_dim), bias=jnp.zeros(model_dim)),
        policy_layers=policy_layers,
        policy_out=policy_out,
        value_layers=value_layers,
        value_out=value_out,
        attention_head_scale=jnp.ones((heads,), jnp.float32),
    )


def _init_pooled_transformer(key, obs_dim, model_dim, layers, ff_dim, embed_width,
                             value_head_width, value_head_layers, policy_head_width,
                             policy_head_layers):
    """Set Transformer pooling followed by a wide, GPU-efficient trunk."""
    if obs_dim != OBSERVATION_V3_DIM:
        raise ValueError('Pooled Set Transformer only supports observation v3')
    if min(model_dim, layers, ff_dim, embed_width) <= 0:
        raise ValueError('Invalid Pooled Set Transformer dimensions')
    input_widths = dict(market=15, reserved=15, nobles=6, gems=7, discounts=6,
                        players=5, bank=6, decks=1, context=35)
    latent_layers = POOLED_LATENT_LAYERS
    key_count = (len(input_widths) + 1 + latent_layers * 4 + 1 + layers * 2
                 + policy_head_layers + 1 + value_head_layers + 1)
    keys = iter(jax.random.split(key, key_count))
    embedders = {name: _init_mlp(next(keys), input_widths[name], embed_width, embed_width)
                 for name in input_widths}
    latent_queries = jax.random.normal(next(keys), (POOLED_LATENT_COUNT, embed_width)) * .02
    latent_blocks = []
    for _ in range(latent_layers):
        latent_blocks.append(dict(
            norm1=dict(scale=jnp.ones(embed_width), bias=jnp.zeros(embed_width)),
            qkv=_xavier(next(keys), embed_width, 3 * embed_width),
            projection=_xavier(next(keys), embed_width, embed_width),
            norm2=dict(scale=jnp.ones(embed_width), bias=jnp.zeros(embed_width)),
            ff1=_xavier(next(keys), embed_width, 2 * embed_width),
            ff2=_xavier(next(keys), 2 * embed_width, embed_width),
        ))
    stem = _linear(next(keys), POOLED_LATENT_COUNT * embed_width, model_dim)
    blocks = [dict(w1=_xavier(next(keys), model_dim, ff_dim)['w'],
                   b1=jnp.zeros(ff_dim),
                   w2=_xavier(next(keys), ff_dim, model_dim)['w'],
                   b2=jnp.zeros(model_dim)) for _ in range(layers)]
    policy_layers, width = [], model_dim
    for _ in range(policy_head_layers):
        policy_layers.append(_linear(next(keys), width, policy_head_width)); width = policy_head_width
    policy_out = _linear(next(keys), width, N_ACTIONS); policy_out['w'] *= .01
    value_layers, width = [], model_dim + 3 * embed_width
    for _ in range(value_head_layers):
        value_layers.append(_linear(next(keys), width, value_head_width)); width = value_head_width
    return dict(token_embedders=embedders, latent_queries=latent_queries,
                latent_blocks=latent_blocks, stem=stem, pooled_blocks=blocks, policy_layers=policy_layers,
                policy_out=policy_out, value_layers=value_layers,
                value_out=_linear(next(keys), width, 1, scale=.5))


def init(key, obs_dim, width=256, residual_blocks=0, residual_taper=False,
         value_head_width=0, value_head_layers=0, policy_head_width=0,
         policy_head_layers=0, residual_stage_widths='', architecture='mlp',
         transformer_layers=4, transformer_heads=4, transformer_ff_dim=384,
         token_embed_width=64, mlp_hidden_layers=2, mlp_activation='relu',
         transformer_attention='dense'):
    """Initialize a flat MLP, a residual MLP, or the typed-token Transformer.

    Each residual block contains two affine transforms.  Tapered networks split
    blocks across three stages, reducing width to 11/16 then 7/16 of the stem;
    transition blocks learn a projection residual connection.
    """
    if architecture == 'transformer':
        return _init_transformer(key, obs_dim, width, transformer_layers, transformer_heads,
                                 transformer_ff_dim, token_embed_width, value_head_width,
                                 value_head_layers, policy_head_width, policy_head_layers,
                                 transformer_attention)
    if architecture == 'pooled_transformer':
        return _init_pooled_transformer(key, obs_dim, width, transformer_layers,
                                        transformer_ff_dim, token_embed_width,
                                        value_head_width, value_head_layers,
                                        policy_head_width, policy_head_layers)
    if architecture != 'mlp':
        raise ValueError(f'Unknown architecture {architecture!r}')
    if residual_blocks < 0:
        raise ValueError('residual_blocks must be non-negative')
    if residual_blocks:
        if (bool(value_head_width) != bool(value_head_layers)
                or bool(policy_head_width) != bool(policy_head_layers)):
            raise ValueError('head widths and depths must both be positive or both be zero')
        if residual_taper:
            if residual_blocks % 3:
                raise ValueError('tapered residual networks need a multiple of three blocks')
            stage_widths = tuple(int(x) for x in residual_stage_widths.split(',')) if residual_stage_widths else (
                width, width * 11 // 16, width * 7 // 16)
            if len(stage_widths) != 3 or min(stage_widths) <= 0 or stage_widths[0] != width:
                raise ValueError('residual_stage_widths must be three positive widths beginning with width')
            block_widths = sum(([stage] * (residual_blocks // 3) for stage in stage_widths), [])
        else:
            block_widths = [width] * residual_blocks
        policy_keys = policy_head_layers + 1 if policy_head_width else 1
        value_keys = value_head_layers + 1 if value_head_width else 0
        keys = jax.random.split(key, 1 + 3 * residual_blocks + policy_keys + value_keys)
        cursor = 0
        stem = _linear(keys[cursor], obs_dim, width); cursor += 1
        blocks = []
        in_width = width
        for i, out_width in enumerate(block_widths):
            block = dict(w1=_linear(keys[cursor], in_width, out_width)['w'],
                         b1=jnp.zeros(out_width),
                         w2=_linear(keys[cursor + 1], out_width, out_width, scale=.1)['w'],
                         b2=jnp.zeros(out_width))
            if in_width != out_width:
                block['skip'] = _linear(keys[cursor + 2], in_width, out_width, scale=1.)['w']
            blocks.append(block)
            in_width = out_width
            cursor += 3
        if not value_head_width and not policy_head_width:
            head = _linear(keys[cursor], in_width, N_ACTIONS + 4)
            head['w'] = head['w'].at[:, :N_ACTIONS].multiply(.01)
            head['w'] = head['w'].at[:, N_ACTIONS:].multiply(.5)
            return dict(stem=stem, blocks=blocks, head=head)
        if policy_head_width:
            policy_layers = []
            policy_width = in_width
            for _ in range(policy_head_layers):
                policy_layers.append(_linear(keys[cursor], policy_width, policy_head_width))
                cursor += 1
                policy_width = policy_head_width
            policy_out = _linear(keys[cursor], policy_width, N_ACTIONS); cursor += 1
            policy_out['w'] = policy_out['w'] * .01
        else:
            policy_head = _linear(keys[cursor], in_width, N_ACTIONS); cursor += 1
            policy_head['w'] = policy_head['w'] * .01
        value_layers = []
        for _ in range(value_head_layers):
            value_layers.append(_linear(keys[cursor], in_width, value_head_width))
            cursor += 1
            in_width = value_head_width
        result = dict(stem=stem, blocks=blocks, value_layers=value_layers,
                      value_out=_linear(keys[cursor], in_width, 4, scale=.5))
        if policy_head_width:
            result.update(policy_layers=policy_layers, policy_out=policy_out)
        else:
            result['policy_head'] = policy_head
        return result
    if mlp_hidden_layers < 1:
        raise ValueError('Flat MLP needs at least one hidden layer')
    if mlp_activation not in ('relu', 'gelu'):
        raise ValueError(f'Unsupported flat MLP activation {mlp_activation!r}')
    sizes = [obs_dim] + [width] * mlp_hidden_layers + [N_ACTIONS + 4]
    keys = jax.random.split(key, len(sizes) - 1)
    result = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(keys[i], (a, b)) * jnp.sqrt(2. / a)
        if i == len(sizes) - 2:
            w = w.at[:, :N_ACTIONS].multiply(.01)
            w = w.at[:, N_ACTIONS:].multiply(.5)
        result.append(dict(w=w, b=jnp.zeros(b)))
    # Retain the historical list representation for two-layer ReLU MLPs so
    # all existing checkpoints retain bit-identical inference.  New variants
    # use a typed dictionary so apply can select their activation without
    # putting non-array metadata into the optimizer pytree.
    if mlp_hidden_layers == 2 and mlp_activation == 'relu':
        return result
    return {f'flat_{mlp_activation}_layers': result}


def init_from_config(key, obs_dim, cfg):
    """Initialize from a training config while keeping legacy call sites valid."""
    return init(key, obs_dim, cfg.width, cfg.residual_blocks, cfg.residual_taper,
                cfg.value_head_width, cfg.value_head_layers, cfg.policy_head_width,
                cfg.policy_head_layers, cfg.residual_stage_widths, cfg.architecture,
                cfg.transformer_layers, cfg.transformer_heads, cfg.transformer_ff_dim,
                cfg.token_embed_width, cfg.mlp_hidden_layers, cfg.mlp_activation,
                cfg.transformer_attention)


def _apply_affine(layer, x, dtype):
    return x @ layer['w'].astype(dtype) + layer['b'].astype(dtype)


def _apply_embed(layers, x, dtype):
    for layer in layers:
        x = jax.nn.gelu(_apply_affine(layer, x, dtype))
    return x


def _layer_norm(layer, x):
    # Accumulate moments in FP32 even when the matrix products use BF16.
    source = x.astype(jnp.float32)
    mean = source.mean(-1, keepdims=True)
    variance = ((source - mean) ** 2).mean(-1, keepdims=True)
    normalized = (source - mean) * jax.lax.rsqrt(variance + 1e-5)
    return (normalized * layer['scale'] + layer['bias']).astype(x.dtype)


def _transformer_block(x, layer, valid, head_scale, dtype, attention_implementation):
    residual = x
    normalized = _layer_norm(layer['norm1'], x)
    qkv = _apply_affine(layer['qkv'], normalized, dtype)
    heads = head_scale.shape[0]
    qkv = qkv.reshape(qkv.shape[:-1] + (3, heads, qkv.shape[-1] // (3 * heads)))
    query, key, value = [qkv[..., i, :, :] for i in range(3)]
    query = query * head_scale.astype(dtype)[None, None, :, None]
    # cuDNN fused attention requires explicit Q and KV sequence dimensions;
    # broadcast is represented lazily and does not materialize per-head masks.
    key_mask = jnp.broadcast_to(valid[..., None, None, :],
                                valid.shape[:-1] + (1, valid.shape[-1], valid.shape[-1]))
    attended = jax.nn.dot_product_attention(
        query, key, value, mask=key_mask, implementation=attention_implementation)
    attended = attended.reshape(attended.shape[:-2] + (attended.shape[-2] * attended.shape[-1],))
    x = residual + _apply_affine(layer['projection'], attended, dtype)
    normalized = _layer_norm(layer['norm2'], x)
    hidden = jax.nn.gelu(_apply_affine(layer['ff1'], normalized, dtype))
    x = x + _apply_affine(layer['ff2'], hidden, dtype)
    return x * valid[..., None]


def _star_transformer_block(x, layer, valid, head_scale, dtype):
    """Linear-token-cost Transformer block with a global routing token.

    The global token is a query over every valid public-information token.
    Its updated representation is broadcast before the per-token FFN, letting
    any two cards/players communicate through the hub once per layer.  This
    avoids padding 47 tokens to a 64x64 dense attention matrix in PPO.
    """
    normalized = _layer_norm(layer['norm1'], x)
    qkv = _apply_affine(layer['qkv'], normalized, dtype)
    heads = head_scale.shape[0]
    qkv = qkv.reshape(qkv.shape[:-1] + (3, heads, qkv.shape[-1] // (3 * heads)))
    query = qkv[..., :1, 0, :, :] * head_scale.astype(dtype)[None, None, :, None]
    key, value = qkv[..., 1, :, :], qkv[..., 2, :, :]
    scores = jnp.einsum('bqhd,bkhd->bhqk', query, key).astype(jnp.float32)
    scores = scores / jnp.sqrt(jnp.asarray(query.shape[-1], jnp.float32))
    scores = jnp.where(valid[:, None, None, :], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1).astype(dtype)
    attended = jnp.einsum('bhqk,bkhd->bqhd', weights, value)
    attended = attended.reshape(attended.shape[:-2] + (attended.shape[-2] * attended.shape[-1],))
    global_x = x[..., :1, :] + _apply_affine(layer['projection'], attended, dtype)
    x = x.at[..., :1, :].set(global_x)

    residual = x
    normalized = _layer_norm(layer['norm2'], x)
    hidden = jax.nn.gelu(_apply_affine(layer['ff1'], normalized, dtype))
    x = residual + _apply_affine(layer['ff2'], hidden, dtype)
    broadcast = _apply_affine(layer['broadcast'], global_x, dtype)
    x = x.at[..., 1:, :].add(broadcast)
    return x * valid[..., None]


def _apply_transformer(params, obs, mask, bf16):
    dtype = jnp.bfloat16 if bf16 else jnp.float32
    unbatched = obs.ndim == 1
    if unbatched:
        obs = obs[None]
        mask = mask[None]
    inputs, valid = structured_observation(obs)
    embeddings = params['token_embedders']
    pieces = [jnp.broadcast_to(params['global_token'].astype(dtype), obs.shape[:-1] + params['global_token'].shape)]
    # This order is part of the checkpoint schema and matches validity above.
    for name in ('market', 'reserved', 'nobles', 'gems', 'discounts', 'players', 'bank', 'decks', 'context'):
        pieces.append(_apply_embed(embeddings[name], inputs[name].astype(dtype), dtype))
    x = jnp.concatenate(pieces, -2)
    if x.shape[-2] != TOKEN_COUNT:
        raise AssertionError(f'Expected {TOKEN_COUNT} tokens, received {x.shape[-2]}')
    x = (x + params['positions'].astype(dtype)) * valid[..., None]
    # Rematerializing each block keeps the large PPO minibatch practical: only
    # block inputs are retained and attention/FF intermediates are recomputed.
    attention_implementation = 'cudnn' if bf16 and jax.default_backend() == 'gpu' else 'xla'
    if attention_implementation == 'cudnn' and 'star_transformer_layers' not in params:
        # cuDNN flash attention on A100 accepts this head size at a 64-token
        # sequence boundary. The 17 padding tokens are never valid keys and do
        # not alter the model semantics or parameter count.
        padding = FUSED_ATTENTION_TOKEN_COUNT - TOKEN_COUNT
        x = jnp.pad(x, ((0, 0), (0, padding), (0, 0)))
        valid = jnp.pad(valid, ((0, 0), (0, padding)))
    if 'star_transformer_layers' in params:
        block = jax.checkpoint(
            lambda tokens, layer, token_valid, head_scale:
                _star_transformer_block(tokens, layer, token_valid, head_scale, dtype))
        for layer in params['star_transformer_layers']:
            x = block(x, layer, valid, params['attention_head_scale'])
    else:
        block = jax.checkpoint(
            lambda tokens, layer, token_valid, head_scale:
                _transformer_block(tokens, layer, token_valid, head_scale, dtype,
                                   attention_implementation))
        for layer in params['transformer_layers']:
            x = block(x, layer, valid, params['attention_head_scale'])
    x = _layer_norm(params['final_norm'], x)

    policy_x = x[..., 0, :]
    for layer in params['policy_layers']:
        # GELU keeps the Transformer heads smooth around zero.  In particular,
        # unlike ReLU it does not permanently discard a token interaction just
        # because its initially small policy/value feature is negative.
        policy_x = jax.nn.gelu(_apply_affine(layer, policy_x, dtype))
    logits = _apply_affine(params['policy_out'], policy_x, dtype).astype(jnp.float32)

    # Token offsets: CLS 0, market 1:13, reserve 13:25, noble 25:30,
    # gems 30:34, discounts 34:38, player summaries 38:42.
    value_x = jnp.concatenate((x[..., 38:42, :], x[..., 30:34, :], x[..., 34:38, :]), -1)
    for layer in params['value_layers']:
        value_x = jax.nn.gelu(_apply_affine(layer, value_x, dtype))
    values = _apply_affine(params['value_out'], value_x, dtype)[..., 0].astype(jnp.float32)
    logits = jnp.where(mask, logits, -1e9)
    return (logits[0], values[0]) if unbatched else (logits, values)


def _apply_pooled_transformer(params, obs, mask, bf16):
    """Attention-pool each semantic set, then reason in large dense matrices."""
    dtype = jnp.bfloat16 if bf16 else jnp.float32
    unbatched = obs.ndim == 1
    if unbatched:
        obs, mask = obs[None], mask[None]
    inputs, valid = structured_observation(obs)
    embedded, token_values = {}, []
    for name in ('market', 'reserved', 'nobles', 'gems', 'discounts', 'players', 'bank', 'decks', 'context'):
        value = _apply_embed(params['token_embedders'][name], inputs[name].astype(dtype), dtype)
        embedded[name] = value
        token_values.append(value)
    token_values = jnp.concatenate(token_values, -2)
    token_valid = valid[:, 1:]
    # Sixteen learned queries extract distinct card/player relations from all
    # 46 semantic tokens.  The following attention blocks communicate only
    # among these short latents, keeping the costly attention path tiny.
    scores = jnp.einsum('ld,btd->blt', params['latent_queries'].astype(dtype), token_values).astype(jnp.float32)
    scores = jnp.where(token_valid[:, None, :], scores, -1e9)
    x = jnp.einsum('blt,btd->bld', jax.nn.softmax(scores, -1).astype(dtype), token_values)
    heads, head_dim = 4, x.shape[-1] // 4
    for layer in params['latent_blocks']:
        residual = x
        qkv = _apply_affine(layer['qkv'], _layer_norm(layer['norm1'], x), dtype)
        qkv = qkv.reshape(qkv.shape[:-1] + (3, heads, head_dim))
        query, key, value = [qkv[..., i, :, :] for i in range(3)]
        logits = jnp.einsum('bqhd,bkhd->bhqk', query, key).astype(jnp.float32) / jnp.sqrt(head_dim)
        attended = jnp.einsum('bhqk,bkhd->bqhd', jax.nn.softmax(logits, -1).astype(dtype), value)
        attended = attended.reshape(attended.shape[:-2] + (attended.shape[-2] * attended.shape[-1],))
        x = residual + _apply_affine(layer['projection'], attended, dtype)
        residual = x
        x = jax.nn.gelu(_apply_affine(layer['ff1'], _layer_norm(layer['norm2'], x), dtype))
        x = residual + _apply_affine(layer['ff2'], x, dtype)
    x = jax.nn.gelu(_apply_affine(params['stem'], x.reshape(x.shape[0], -1), dtype))
    for block in params['pooled_blocks']:
        residual = x
        x = jax.nn.gelu(_apply_affine(dict(w=block['w1'], b=block['b1']), x, dtype))
        x = residual + _apply_affine(dict(w=block['w2'], b=block['b2']), x, dtype)
    policy_x = x
    for layer in params['policy_layers']:
        policy_x = jax.nn.gelu(_apply_affine(layer, policy_x, dtype))
    logits = _apply_affine(params['policy_out'], policy_x, dtype).astype(jnp.float32)
    global_x = jnp.broadcast_to(x[:, None, :], x.shape[:-1] + (4, x.shape[-1]))
    value_x = jnp.concatenate((global_x, embedded['gems'], embedded['discounts'], embedded['players']), -1)
    for layer in params['value_layers']:
        value_x = jax.nn.gelu(_apply_affine(layer, value_x, dtype))
    values = _apply_affine(params['value_out'], value_x, dtype)[..., 0].astype(jnp.float32)
    logits = jnp.where(mask, logits, -1e9)
    return (logits[0], values[0]) if unbatched else (logits, values)


def apply(params, obs, mask, bf16=False):
    if isinstance(params, dict) and 'pooled_blocks' in params:
        return _apply_pooled_transformer(params, obs, mask, bf16)
    if isinstance(params, dict) and 'token_embedders' in params:
        return _apply_transformer(params, obs, mask, bf16)
    dtype = jnp.bfloat16 if bf16 else jnp.float32
    flat_key = next((name for name in ('flat_relu_layers', 'flat_gelu_layers') if isinstance(params, dict) and name in params), None)
    if flat_key is not None:
        x = obs.astype(dtype)
        layers = params[flat_key]
        activation = jax.nn.gelu if flat_key == 'flat_gelu_layers' else jax.nn.relu
        for layer in layers[:-1]:
            x = activation(_apply_affine(layer, x, dtype))
        out = _apply_affine(layers[-1], x, dtype).astype(jnp.float32)
        return jnp.where(mask, out[..., :N_ACTIONS], -1e9), out[..., N_ACTIONS:]
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
        if 'head' in params:  # Pre-value-head residual checkpoint compatibility.
            layer = params['head']
            out = (x @ layer['w'].astype(dtype) + layer['b'].astype(dtype)).astype(jnp.float32)
            return jnp.where(mask, out[..., :N_ACTIONS], -1e9), out[..., N_ACTIONS:]
        if 'policy_layers' in params:
            policy_x = x
            for layer in params['policy_layers']:
                policy_x = jax.nn.relu(policy_x @ layer['w'].astype(dtype) + layer['b'].astype(dtype))
            policy = params['policy_out']
            logits = (policy_x @ policy['w'].astype(dtype) + policy['b'].astype(dtype)).astype(jnp.float32)
        else:
            policy = params['policy_head']
            logits = (x @ policy['w'].astype(dtype) + policy['b'].astype(dtype)).astype(jnp.float32)
        value_x = x
        for layer in params['value_layers']:
            value_x = jax.nn.relu(value_x @ layer['w'].astype(dtype) + layer['b'].astype(dtype))
        value = params['value_out']
        values = (value_x @ value['w'].astype(dtype) + value['b'].astype(dtype)).astype(jnp.float32)
        return jnp.where(mask, logits, -1e9), values
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
    if not isinstance(params, list) or len(params) != 3:
        raise ValueError('widen only supports two-hidden-layer legacy MLP parameters')
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
    active = jnp.arange(4) < nplayers[..., None]
    absolute = jnp.take_along_axis(relative, index, -1) * active
    # Every reward component is zero-sum, so the true multi-seat value lives
    # in this subspace.  Removing the unidentifiable common mode also keeps
    # lambda-return bootstraps exactly zero-sum.
    active_mean = absolute.sum(-1, keepdims=True) / nplayers[..., None]
    return jnp.where(active, absolute - active_mean, 0.)
