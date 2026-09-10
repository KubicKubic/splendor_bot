"""Checkpoint inference and a read-only adapter for the HullQin view object.

Produces an action description; never connects to or writes to a live game.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import env, network
from .train import load


def from_hullqin_view(v):
    """Input uses the site's view convention: market/noble IDs 1 based;
    playerBooked/playerCard IDs 0 based.  Reserved-card identities for every
    player and the remaining face-down count for every tier are required.
    """
    n = len(v['playerGem'])
    if n not in (2, 3, 4):
        raise ValueError('Expected 2..4 players')
    p = int(v['waitFor'])
    z = env.reset(jax.random.PRNGKey(0), n)
    booked = v.get('playerBooked')
    if booked is None or len(booked) != n:
        raise ValueError('playerBooked must expose reserved-card IDs for every player')
    reserved = np.full((4, 3), -1, dtype=np.int32)
    for i, row in enumerate(booked):
        if len(row) > 3:
            raise ValueError('A player cannot reserve more than three cards')
        reserved[i, :len(row)] = sorted(row)
    counts = v.get('bankLeftCardCount')
    if counts is None:
        if 'bankLeftCard' not in v:
            raise ValueError('bankLeftCardCount is required when bankLeftCard is absent')
        counts = [len(x) for x in v['bankLeftCard']]
    counts = np.asarray(counts, dtype=np.int32)
    if counts.shape != (3,) or np.any(counts < 0) or np.any(counts > np.asarray(env.DECK_SIZE)):
        raise ValueError('bankLeftCardCount must contain three valid face-down deck counts')
    phase = env.CHOOSE_NOBLE if v.get('waitNoble') else env.DISCARD if v.get('waitThrowing') else env.NORMAL
    return z._replace(bank=jnp.asarray(v['bankGem'], jnp.int32),
        gems=z.gems.at[:n].set(jnp.asarray(v['playerGem'], jnp.int32)),
        bonuses=z.bonuses.at[:n].set(jnp.asarray(v['playerCardCount'], jnp.int32)),
        scores=z.scores.at[:n].set(jnp.asarray(v['playerScore'], jnp.int32)),
        reserved=jnp.asarray(reserved), nobles=z.nobles.at[:n + 1].set(jnp.asarray(v['bankNoble']) - 1),
        market=jnp.asarray(v['bankCard'], jnp.int32) - 1, cursor=env.DECK_SIZE - jnp.asarray(counts, jnp.int32),
        player=jnp.int32(p), phase=jnp.int32(phase), done=jnp.bool_(bool(v.get('winner'))))


class Agent:
    def __init__(self, checkpoint, deterministic=False):
        self.params, self.config = load(checkpoint)
        self.deterministic = deterministic
        def choose(params, state, key):
            observation = env.observe(state, self.config.observation_version)
            logits, _ = network.apply(params, observation, env.legal_mask(state), self.config.bf16)
            return jnp.argmax(logits).astype(jnp.int32) if deterministic else jax.random.categorical(key, logits).astype(jnp.int32)
        self._choose = jax.jit(choose)

    def act(self, state, key):
        if (not self.config.mixed_players and int(state.nplayers) != self.config.players):
            raise ValueError('Use a checkpoint trained for this player count')
        if bool(state.done):
            raise ValueError('Game already ended')
        return int(self._choose(self.params, state, key))

    def plan_view(self, view, key):
        s = from_hullqin_view(view)
        def choose(state):
            nonlocal key
            key, subkey = jax.random.split(key)
            return self.act(state, subkey)
        action = choose(s)
        if int(s.phase) == env.DISCARD:
            returned = np.zeros(6, np.int32)
            while int(s.gems[s.player].sum()) > 10:
                color = action - 61
                returned[color] += 1
                s = s._replace(bank=s.bank.at[color].add(1), gems=s.gems.at[s.player, color].add(-1))
                if int(s.gems[s.player].sum()) > 10:
                    action = choose(s)
            return dict(kind='discard', gems=returned.tolist())
        if int(s.phase) == env.CHOOSE_NOBLE:
            return dict(kind='noble', position=action - 67)
        if action == 0:
            return dict(kind='pass')
        if action < 31:
            return dict(kind='take', gems=np.asarray(env.TAKES[action - 1]).tolist())
        if action >= 58:
            return dict(kind='reserve_blind', tier=action - 58)
        if action >= 46:
            slot = action - 46
            return dict(kind='reserve', card_id=int(s.market.ravel()[slot]), position=slot % 4)
        slot = action - 31
        cid = int(env.targets(s)[slot])
        cost = jnp.maximum(env.COST[cid] - s.bonuses[s.player], 0)
        spend = jnp.minimum(cost, s.gems[s.player, :5])
        gold = int((cost - spend).sum())
        if int(s.gems[s.player, 5]) > gold and int(spend.sum()) > 0:
            s = s._replace(phase=jnp.int32(env.PAYMENT), pending=jnp.int32(slot), pay_cost=cost)
            spend, gold = np.zeros(5, np.int32), 0
            for color in range(5):
                s = s._replace(pay_color=jnp.int32(color))
                g = choose(s) - 72
                spend[color] = int(cost[color]) - g
                gold += g
                payment = jnp.zeros(6, jnp.int32).at[color].set(spend[color]).at[5].set(g)
                s = s._replace(bank=s.bank + payment, gems=s.gems.at[s.player].add(-payment))
        return dict(kind='buy', card_id=cid, position=slot - 12 if slot >= 12 else slot % 4,
                    booked=slot >= 12, payment=dict(spend=np.asarray(spend).tolist(), gold=gold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--view', required=True, help='Local JSON file containing a HullQin view object')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    agent = Agent(args.checkpoint, args.deterministic)
    print(json.dumps(agent.plan_view(json.loads(Path(args.view).read_text()), jax.random.PRNGKey(args.seed)), indent=2))
