"""JAX port of edwadli/splendor-ai ValueBuyBot (Apache-2.0).

Source: commit 580687e8f38377f00dff50cc97414308ec4e9c6e,
src/agents/value_man/value_buy_bot.py. Original strategy/constants retained.
Dense 15x16 path table replaces dictionaries and variable-length lists.
State passed here contains no use of hidden deck or opponent reserved IDs.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import env


class Plan(NamedTuple):
    action: jax.Array
    returned: jax.Array
    primary: jax.Array
    secondary: jax.Array
    valid: jax.Array


def card_values(s, player):
    ids = env.targets(s)
    costs = env.COST[jnp.maximum(ids, 0)]
    discounted = jnp.maximum(costs - s.bonuses[player], 0)
    missing = jnp.maximum(discounted - s.gems[player, :5], 0)
    # Upstream _GetTempoForCard uses self's missing colors even when valuing
    # an opponent's tempo. Preserve that implementation detail verbatim.
    own_missing = jnp.maximum(costs - s.bonuses[s.player] - s.gems[s.player, :5], 0)
    wanted = (own_missing > 0).sum(-1)
    unavailable = ((own_missing > 0) & (s.bank[:5] == 0)).sum(-1)
    penalty = jnp.where((wanted <= 3) & (unavailable > 0), unavailable,
                        jnp.where((wanted > 3) & (wanted - unavailable < 3), 3, 0))
    tempo_thirds = jnp.where(missing.max(-1) == 0, 3, 3 + 2 * missing.max(-1) + 3 * penalty)
    value = (env.POINT[jnp.maximum(ids, 0)] + .15) * 3. / tempo_thirds
    return jnp.where(ids >= 0, value, -1.), tempo_thirds


def choose_path(s):
    """Return slots for the primary / secondary target and a validity flag."""
    ids = env.targets(s)
    safe = jnp.maximum(ids, 0)
    cost = env.COST[safe]
    affordable = (ids >= 0) & (jnp.maximum(cost - s.bonuses[s.player], 0).sum(-1) <= 10)
    value, tempo = card_values(s, s.player)
    # Ancestors appear by required gem color, then by market/reserve iteration.
    ancestor_order = jnp.argsort(env.BONUS[safe] * 15 + jnp.arange(15), stable=True)
    target_order = jnp.argsort(-value, stable=True)
    ancestors = ((cost[:, env.BONUS[safe]] > 0) & affordable[None, :] &
                 (jnp.arange(15)[:, None] != jnp.arange(15)[None, :]))
    max_value = jnp.max(jnp.where(ancestors, value[None, :], -2.), -1)
    best = ancestors & (value[None, :] == max_value[:, None])
    single = ~ancestors.any(-1) & affordable
    valid = jnp.concatenate((single[:, None], best[:, ancestor_order]), -1)[target_order].ravel()
    primary = jnp.concatenate((jnp.arange(15)[:, None], jnp.broadcast_to(ancestor_order, (15, 15))), -1)[target_order].ravel()
    secondary = jnp.concatenate((jnp.full((15, 1), -1), jnp.broadcast_to(jnp.arange(15)[:, None], (15, 15))), -1)[target_order].ravel()
    valid &= ids[jnp.maximum(secondary, primary)] >= 0
    not_win = s.scores[s.player] + env.POINT[safe[primary]] < 15
    order = jnp.lexsort((jnp.arange(240), primary < 12, tempo[primary], not_win, ~valid))[:3]
    total_value = value[primary] + jnp.where(secondary >= 0, value[jnp.maximum(secondary, 0)], 0.)
    best_path = order[jnp.argmax(jnp.where(valid[order], total_value[order], -1.))]
    return primary[best_path], secondary[best_path], valid.any()


def _take_some(taken, candidates, bank):
    """Add scarce colors in stable white/blue/green/red/black order up to 3."""
    available = candidates & (bank[:5] > 0) & (taken[:5] == 0)
    order = jnp.argsort(jnp.where(available, bank[:5], 99), stable=True)
    ranks = jnp.argsort(order)
    added = available & (ranks < 3 - taken.sum())
    return taken.at[:5].add(added.astype(jnp.int32))


def _take(s, primary, secondary):
    ids = env.targets(s)
    cost = jnp.maximum(env.COST[jnp.maximum(ids[primary], 0)] - s.bonuses[s.player], 0)
    missing = jnp.maximum(cost - s.gems[s.player, :5], 0)
    second_cost = jnp.maximum(env.COST[jnp.maximum(ids[jnp.maximum(secondary, 0)], 0)] - s.bonuses[s.player], 0)
    first = jnp.argmax(missing)
    double_first = (s.bank[first] >= 4) & (missing[first] >= 2)
    taken = _take_some(jnp.zeros(6, jnp.int32), missing > 0, s.bank)
    second = jnp.argmax(second_cost)
    second_unaffordable = jnp.maximum(second_cost - s.gems[s.player, :5], 0).sum() > s.gems[s.player, 5]
    double_second = ((taken.sum() == 0) & (secondary >= 0) & second_unaffordable &
                     (s.bank[second] >= 4) & (second_cost[second] >= 2))
    taken = _take_some(taken, (second_cost > 0) & (secondary >= 0), s.bank)
    taken = _take_some(taken, jnp.ones(5, bool), s.bank)
    taken = jnp.where(double_second, jnp.zeros(6, jnp.int32).at[second].set(2), taken)
    return jnp.where(double_first, jnp.zeros(6, jnp.int32).at[first].set(2), taken)


def _returns(s, primary, taken):
    need = jnp.maximum(env.COST[jnp.maximum(env.targets(s)[primary], 0)] - s.bonuses[s.player], 0)
    owned = s.gems[s.player]
    total = owned + taken
    extra = jnp.maximum(total.sum() - 10, 0)
    unnecessary = jnp.concatenate((jnp.maximum(total[:5] - need, 0), jnp.zeros(1, jnp.int32)))
    gold_only = jnp.all(taken[:5] == 0)
    unnecessary = jnp.where(gold_only, total.at[5].set(0), unnecessary)
    def body(_, carry):
        total, unnecessary, returned, gold_buffer, valid = carry
        use_buffer = ~jnp.any(unnecessary > 0) & (gold_buffer > 0)
        possible = jnp.where(use_buffer, total > 0, unnecessary > 0)
        # Upstream reverse stable-sort: larger bank count, then later color.
        ranking = (s.bank + returned) * 8 + jnp.arange(6)
        color = jnp.argmax(jnp.where(possible, ranking, -999))
        do = returned.sum() < extra
        delta = jax.nn.one_hot(color, 6, dtype=jnp.int32) * do
        return total - delta, unnecessary - delta, returned + delta, gold_buffer - (use_buffer & do), valid & (~do | possible.any())
    _, _, returned, _, valid = jax.lax.fori_loop(0, 3, body,
        (total, unnecessary, jnp.zeros(6, jnp.int32), owned[5], jnp.array(True)))
    common = jnp.minimum(taken, returned)
    return taken - common, returned - common, valid


def plan(s):
    primary, secondary, valid = choose_path(s)
    ids = env.targets(s)
    safe = jnp.maximum(ids, 0)
    value, _ = card_values(s, s.player)
    opponent, _ = card_values(s, (s.player + 1) % s.nplayers)
    cost = jnp.maximum(env.COST[safe[primary]] - s.bonuses[s.player], 0)
    buy = jnp.maximum(cost - s.gems[s.player, :5], 0).sum() <= s.gems[s.player, 5]
    can_reserve = (s.reserved[s.player] < 0).any()
    reserve_first = can_reserve & (primary < 12) & (value[primary] >= .85) & (opponent[primary] >= value[primary])
    reserve_second = (can_reserve & (secondary >= 0) & (secondary < 12) & (s.bank[5] > 0) &
                      (value[jnp.maximum(secondary, 0)] >= 2.))
    reserve = reserve_first | reserve_second
    reserve_slot = jnp.where(reserve_first, primary, secondary)
    taken = jnp.where(reserve, jnp.zeros(6, jnp.int32).at[5].set((s.bank[5] > 0).astype(jnp.int32)), _take(s, primary, secondary))
    taken, returned, return_valid = _returns(s, primary, taken)
    same_take = jnp.all(env.TAKES == taken, -1)
    take_action = jnp.where(taken.sum() == 0, 0, jnp.argmax(same_take) + 1)
    action = jnp.where(buy, primary + 31, jnp.where(reserve, reserve_slot + 46, take_action)).astype(jnp.int32)
    valid &= buy | (return_valid & (reserve | same_take.any() | ((taken.sum() == 0) & (returned.sum() == 0))))
    valid &= env.legal_mask(s)[action]
    return Plan(action, jnp.where(buy, 0, returned), primary, secondary, valid)


def act(s, returned):
    """Returns primitive action, carried return plan, and a validity check."""
    result = plan(s)
    action = result.action
    keep = jnp.where(s.phase == env.NORMAL, result.returned, returned)
    action = jnp.where(s.phase == env.PAYMENT, jnp.argmax(env.legal_mask(s)[72:78]) + 72, action)
    action = jnp.where(s.phase == env.CHOOSE_NOBLE, jnp.argmax(env.noble_candidates(s)) + 67, action)
    color = jnp.argmax(keep > 0)
    action = jnp.where(s.phase == env.DISCARD, color + 61, action)
    valid = jnp.where(s.phase == env.NORMAL, result.valid, env.legal_mask(s)[action])
    valid &= (s.phase != env.DISCARD) | (keep > 0).any()
    keep = keep - jax.nn.one_hot(color, 6, dtype=jnp.int32) * (s.phase == env.DISCARD)
    return jnp.where(s.done, 0, action).astype(jnp.int32), keep, valid | s.done
