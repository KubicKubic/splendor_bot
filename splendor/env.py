"""Fixed-shape, pure JAX environment; all IDs are zero based (-1 is empty).

The public site's optional payments, partial takes, pass, noble-before-discard
order and tie rules are preserved. Microdecisions do not advance the turn.
All reserved-card identities and the three face-down deck counts are observable;
the shuffled face-down deck order is never exposed.
"""
from itertools import combinations
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

DATA = json.loads((Path(__file__).resolve().parents[1] / 'data/cards.json').read_text())
COST = jnp.array([c['cost'] for c in DATA['cards']], jnp.int32)
POINT = jnp.array([c['points'] for c in DATA['cards']], jnp.int32)
BONUS = jnp.array([c['bonus'] for c in DATA['cards']], jnp.int32)
TIER = jnp.array([c['tier'] for c in DATA['cards']], jnp.int32)
NOBLE = jnp.array(DATA['nobles'], jnp.int32)
DECK_SIZE = jnp.array([40, 30, 20], jnp.int32)
DECK_IDS = jnp.array([[c['id'] for c in DATA['cards'] if c['tier'] == t] + [-1] * (40 - n)
                      for t, n in enumerate([40, 30, 20])], jnp.int32)
_takes = []
for n in range(1, 4):
    for colors in combinations(range(5), n):
        _takes.append([int(i in colors) for i in range(6)])
_takes.extend([[2 * int(i == c) for i in range(6)] for c in range(5)])
TAKES = jnp.array(_takes, jnp.int32)
# 0 pass; 1:31 take; 31:46 buy (12 market + 3 reserve);
# 46:61 reserve (12 market + 3 blind); 61:67 discard; 67:72 noble;
# 72:78 choose how much gold to spend on the current color.
N_ACTIONS = 78
OBSERVATION_VERSION = 3
NORMAL, PAYMENT, CHOOSE_NOBLE, DISCARD = range(4)


class State(NamedTuple):
    bank: jax.Array             # [6]
    gems: jax.Array             # [4,6]
    bonuses: jax.Array          # [4,5]
    scores: jax.Array           # [4]
    reserved: jax.Array         # [4,3], sorted by card ID as on site
    bought: jax.Array           # [90], owner or -1 (audit, never observed)
    nobles: jax.Array           # [5], noble ID or -1
    noble_owner: jax.Array      # [10], owner or -1
    market: jax.Array           # [3,4]
    decks: jax.Array            # [3,40], includes initial face-up cards
    cursor: jax.Array           # [3]
    player: jax.Array           # scalar
    nplayers: jax.Array
    phase: jax.Array
    pending: jax.Array          # buy target 0..14
    pay_color: jax.Array
    pay_cost: jax.Array         # discounted cost, fixed throughout payment
    turns: jax.Array
    done: jax.Array


def reset(key, nplayers=2):
    kd, kn = jax.random.split(key)
    order = jnp.argsort(jnp.where(DECK_IDS >= 0, jax.random.uniform(kd, (3, 40)), 2.), axis=-1)
    decks = jnp.take_along_axis(DECK_IDS, order, axis=-1)
    nobles = jax.random.permutation(kn, 10)[:5]
    nobles = jnp.where(jnp.arange(5) < nplayers + 1, nobles, -1)
    z = lambda shape: jnp.zeros(shape, jnp.int32)
    return State(jnp.array([4, 4, 4, 4, 4, 5], jnp.int32).at[:5].set(
        jnp.where(nplayers == 2, 4, jnp.where(nplayers == 3, 5, 7))),
        z((4, 6)), z((4, 5)), z(4), jnp.full((4, 3), -1, jnp.int32),
        jnp.full(90, -1, jnp.int32), nobles, jnp.full(10, -1, jnp.int32),
        decks[:, :4], decks, jnp.full(3, 4, jnp.int32), z(()), jnp.asarray(nplayers, jnp.int32),
        z(()), z(()), z(()), z(5), z(()), jnp.array(False))


def targets(s):
    return jnp.concatenate((s.market.ravel(), s.reserved[s.player]))


def noble_candidates(s):
    return (s.nobles >= 0) & jnp.all(s.bonuses[s.player] >= NOBLE[jnp.maximum(s.nobles, 0)], -1)


def legal_mask(s):
    p = s.player
    ids = targets(s)
    need = jnp.maximum(COST[jnp.maximum(ids, 0)] - s.bonuses[p], 0)
    buy = (ids >= 0) & (jnp.maximum(need - s.gems[p, :5], 0).sum(-1) <= s.gems[p, 5])
    take = jnp.all(TAKES <= s.bank, -1) & jnp.all((TAKES != 2) | (s.bank >= 4), -1)
    reserve = jnp.concatenate((s.market.ravel() >= 0, s.cursor < DECK_SIZE)) & jnp.any(s.reserved[p] < 0)
    normal = jnp.concatenate((jnp.ones(1, bool), take, buy, reserve, jnp.zeros(17, bool)))
    discard = jnp.zeros(N_ACTIONS, bool).at[61:67].set(s.gems[p] > 0)
    noble = jnp.zeros(N_ACTIONS, bool).at[67:72].set(noble_candidates(s))
    color = jnp.minimum(s.pay_color, 4)
    need_gold = jnp.maximum(s.pay_cost - s.gems[p, :5], 0)
    future = jnp.where(jnp.arange(5) > color, need_gold, 0).sum()
    gold = jnp.arange(6)
    valid_gold = ((gold >= need_gold[color]) & (gold <= s.pay_cost[color]) &
                  (gold + future <= s.gems[p, 5]))
    payment = jnp.zeros(N_ACTIONS, bool).at[72:78].set(valid_gold)
    mask = jnp.stack((normal, payment, noble, discard))[s.phase]
    return jnp.where(s.done, jnp.arange(N_ACTIONS) == 0, mask)


def _advance(s):
    player = (s.player + 1) % s.nplayers
    return s._replace(player=player, phase=jnp.int32(NORMAL), turns=s.turns + 1,
                      done=(player == 0) & jnp.any(s.scores >= 15))


def _after_noble(s):
    return jax.lax.cond(s.gems[s.player].sum() > 10,
                        lambda x: x._replace(phase=jnp.int32(DISCARD)), _advance, s)


def _claim(s, slot):
    nid = s.nobles[slot]
    return s._replace(nobles=s.nobles.at[slot].set(-1),
                      noble_owner=s.noble_owner.at[nid].set(s.player),
                      scores=s.scores.at[s.player].add(3))


def _finish(s):
    candidates = noble_candidates(s)
    count = candidates.sum()
    s = jax.lax.cond(count == 1, lambda x: _claim(x, jnp.argmax(candidates)), lambda x: x, s)
    return jax.lax.cond(count > 1, lambda x: x._replace(phase=jnp.int32(CHOOSE_NOBLE)), _after_noble, s)


def _draw(s, tier):
    pos = jnp.minimum(s.cursor[tier], 39)
    cid = jnp.where(s.cursor[tier] < DECK_SIZE[tier], s.decks[tier, pos], -1)
    return s._replace(cursor=s.cursor.at[tier].set(jnp.minimum(s.cursor[tier] + 1, DECK_SIZE[tier]))), cid


def _remove_market(s, slot):
    tier, pos = slot // 4, slot % 4
    s, cid = _draw(s, tier)
    return s._replace(market=s.market.at[tier, pos].set(cid))


def _sort_reserved(cards):
    return jnp.where(jnp.sort(jnp.where(cards < 0, 99, cards)) == 99, -1,
                     jnp.sort(jnp.where(cards < 0, 99, cards)))


def _complete_buy(s):
    cid = targets(s)[s.pending]
    p = s.player
    s = s._replace(bonuses=s.bonuses.at[p, BONUS[cid]].add(1),
                    bought=s.bought.at[cid].set(p), scores=s.scores.at[p].add(POINT[cid]))
    def remove_reserved(x):
        row = _sort_reserved(x.reserved[p].at[x.pending - 12].set(-1))
        return x._replace(reserved=x.reserved.at[p].set(row))
    s = jax.lax.cond(s.pending < 12, lambda x: _remove_market(x, x.pending), remove_reserved, s)
    return _finish(s)


def _pay(s, action):
    color = s.pay_color
    gold = action - 72
    spend = jnp.zeros(6, jnp.int32).at[color].set(s.pay_cost[color] - gold).at[5].set(gold)
    s = s._replace(bank=s.bank + spend, gems=s.gems.at[s.player].add(-spend), pay_color=color + 1)
    return jax.lax.cond(color == 4, _complete_buy, lambda x: x, s)


def _normal(s, action):
    def take(x):
        amount = TAKES[jnp.clip(action - 1, 0, 29)]
        return _finish(x._replace(bank=x.bank - amount, gems=x.gems.at[x.player].add(amount)))
    def buy(x):
        slot = action - 31
        cost = jnp.maximum(COST[targets(x)[slot]] - x.bonuses[x.player], 0)
        # A unique payment can be done immediately. Only optional gold needs a subphase.
        spend = jnp.minimum(cost, x.gems[x.player, :5])
        gold = (cost - spend).sum()
        choice = (x.gems[x.player, 5] > gold) & (spend.sum() > 0)
        x = x._replace(pending=slot, pay_cost=cost, pay_color=jnp.int32(0), phase=jnp.int32(PAYMENT))
        def immediate(y):
            payment = jnp.concatenate((spend, gold[None]))
            return _complete_buy(y._replace(bank=y.bank + payment, gems=y.gems.at[y.player].add(-payment)))
        return jax.lax.cond(choice, lambda y: y, immediate, x)
    def reserve(x):
        slot = action - 46
        def face(y):
            cid = y.market.ravel()[slot]
            return _remove_market(y, slot), cid
        x, cid = jax.lax.cond(slot < 12, face, lambda y: _draw(y, slot - 12), x)
        row = _sort_reserved(x.reserved[x.player].at[jnp.argmax(x.reserved[x.player] < 0)].set(cid))
        gold = (x.bank[5] > 0).astype(jnp.int32)
        return _finish(x._replace(reserved=x.reserved.at[x.player].set(row), bank=x.bank.at[5].add(-gold),
                                  gems=x.gems.at[x.player, 5].add(gold)))
    kind = jnp.where(action == 0, 0, jnp.where(action < 31, 1, jnp.where(action < 46, 2, 3)))
    return jax.lax.switch(kind, (lambda x: _advance(x), take, buy, reserve), s)


def _discard(s, action):
    color = action - 61
    s = s._replace(bank=s.bank.at[color].add(1), gems=s.gems.at[s.player, color].add(-1))
    return jax.lax.cond(s.gems[s.player].sum() > 10, lambda x: x, _advance, s)


def step(s, action):
    """Caller must supply a masked legal action. Terminal states are absorbing."""
    return jax.lax.cond(s.done, lambda x: x,
        lambda x: jax.lax.switch(x.phase, (lambda y: _normal(y, action), lambda y: _pay(y, action),
                                          lambda y: _after_noble(_claim(y, action - 67)),
                                          lambda y: _discard(y, action)), x), s)


def winners(s):
    rank = jnp.where((jnp.arange(4) < s.nplayers) & (s.scores >= 15),
                     100 * s.scores - s.bonuses.sum(-1), -10000)
    return s.done & (rank == rank.max())


def outcome(s):
    win = winners(s)
    # Every winning seat gets +1/m and every losing seat gets -1/(n-m), where
    # n is the active player count and m is the number of winners.  This is
    # zero-sum for every partial tie; an all-way tie has no losers and is zero.
    active = jnp.arange(4) < s.nplayers
    winners_count = win.sum()
    losers_count = s.nplayers - winners_count
    payoff = jnp.where(win, 1. / jnp.maximum(winners_count, 1),
                       -1. / jnp.maximum(losers_count, 1))
    payoff = jnp.where(losers_count > 0, payoff, 0.)
    return jnp.where(s.done & active, payoff, 0.)


def potential(s):
    """Public score/engine potential, used only via potential-based reward shaping."""
    raw = s.scores / 15. + s.bonuses.sum(-1) / 60.
    active = jnp.arange(4) < s.nplayers
    centered = raw - jnp.sum(raw * active) / s.nplayers
    return jnp.where(active & ~s.done, centered, 0.)


def card_features(ids):
    valid = ids >= 0
    ids = jnp.maximum(ids, 0)
    features = jnp.concatenate((COST[ids] / 7., jax.nn.one_hot(BONUS[ids], 5),
                                POINT[ids, None] / 5., jax.nn.one_hot(TIER[ids], 3), valid[..., None]), -1)
    return features * valid[..., None]


def observe(s, version=OBSERVATION_VERSION):
    order = (jnp.arange(4) + s.player) % s.nplayers
    active = jnp.arange(4) < s.nplayers
    reserved = s.reserved[order]
    tiers = jax.nn.one_hot(TIER[jnp.maximum(reserved, 0)], 3) * (reserved >= 0)[..., None]
    players = jnp.concatenate((s.gems[order] / 10., s.bonuses[order] / 10., s.scores[order, None] / 20.,
                               tiers.sum(1) / 3., active[:, None]), -1) * active[:, None]
    nobles = jnp.concatenate((NOBLE[jnp.maximum(s.nobles, 0)] / 4., (s.nobles >= 0)[:, None]), -1)
    nobles *= (s.nobles >= 0)[:, None]
    # Reserved cards are public in the deployed game.  Keep the same
    # actor-relative seat order as the other per-player features, but encode
    # every card exactly rather than exposing only the acting player's hand.
    if version == 1:
        reserved_cards = card_features(s.reserved[s.player]).ravel()
    elif version == 2:
        # Preserve the historical v2 padding semantics exactly: modulo seat
        # ordering repeated active players into unused 2P/3P slots.  Existing
        # checkpoints must continue to receive the representation on which
        # they were trained.
        reserved_cards = card_features(reserved).ravel()
    elif version == OBSERVATION_VERSION:
        # Fixed-shape padding must carry no player information.  The other
        # per-seat features above already apply this active mask; v2 omitted
        # it for the exact reserved-card block.
        reserved_cards = (card_features(reserved) * active[:, None, None]).ravel()
    else:
        raise ValueError(f'Unsupported observation version {version}')
    # cursor points just past every card removed from the face-down portion,
    # so this is exactly the remaining face-down count in each tier.  Counts
    # are normalized only for conditioning; no deck identity/order is leaked.
    face_down_left = (DECK_SIZE - s.cursor) / DECK_SIZE
    return jnp.concatenate((players.ravel(), s.bank / 7., card_features(s.market.ravel()).ravel(),
        reserved_cards, nobles.ravel(), face_down_left,
        jax.nn.one_hot(s.phase, 4), jax.nn.one_hot(jnp.minimum(s.pay_color, 4), 5) * (s.phase == PAYMENT),
        s.pay_cost / 7. * (s.phase == PAYMENT), jax.nn.one_hot(s.pending, 15) * (s.phase == PAYMENT),
        jax.nn.one_hot(s.player, 4), jnp.array([s.nplayers / 4., jnp.any(s.scores >= 15)]))).astype(jnp.float32)


batch_reset = jax.vmap(reset, in_axes=(0, None))
batch_step = jax.vmap(step)
batch_observe = jax.vmap(observe)


def batch_observe_for_version(states, version):
    """Encode a batch with a checkpoint's immutable observation schema."""
    return jax.vmap(lambda state: observe(state, version))(states)


batch_mask = jax.vmap(legal_mask)
