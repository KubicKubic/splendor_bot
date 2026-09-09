"""Adapter to the unmodified Apache-2.0 edwadli/splendor-ai ValueBuyBot.

Only the authoritative HullQin-aligned env performs game transitions.
No hidden deck or opponent reserved identities are passed to the bot.
"""
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

from . import env

VENDOR = Path(__file__).resolve().parents[1] / 'vendor/edwadli_splendor'
COMMIT = '580687e8f38377f00dff50cc97414308ec4e9c6e'
sys.path.insert(0, str(VENDOR))
from src.agents.value_man.value_buy_bot import ValueBuyBot
from src.data.game_rules import GAME_RULES
from src.game.player_game_state import PlayerGameState
from src.proto.development_card_proto import DevelopmentCard
from src.proto.game_state_proto import GameState
from src.proto.noble_tile_proto import NobleTile
from src.proto.player_state_proto import PlayerState

# Our W,B,G,R,K,gold -> upstream GemType WHITE,BLUE,GREEN,RED,BROWN,GOLD.
COLOR = [4, 1, 2, 3, 5, 6]
TAKES = np.asarray(env.TAKES)


def gems(values):
    # Keep a stable documented white/blue/green/red/black/gold ordering.
    return Counter({color: int(value) for color, value in zip(COLOR, values)})


CARDS = [DevelopmentCard(f'c{c["id"]}', c['tier'] + 1, c['points'], COLOR[c['bonus']], gems(c['cost']))
         for c in env.DATA['cards']]
NOBLES = [NobleTile(f'n{i}', 3, gems(cost)) for i, cost in enumerate(env.DATA['nobles'])]


def to_native(s):
    """s is a host (NumPy) State. No game transition uses the upstream engine."""
    p, n = int(s.player), int(s.nplayers)
    players = []
    for seat in range(n):
        reserved = [CARDS[int(cid)] for cid in s.reserved[seat] if cid >= 0] if seat == p else [None] * int((s.reserved[seat] >= 0).sum())
        players.append(PlayerState(gems(s.gems[seat]), [CARDS[int(cid)] for cid in np.flatnonzero(s.bought == seat)],
                                   [], reserved, [NOBLES[int(nid)] for nid in np.flatnonzero(s.noble_owner == seat)]))
    market = defaultdict(list, {tier + 1: [CARDS[int(cid)] for cid in s.market[tier] if cid >= 0] for tier in range(3)})
    state = GameState(gems(s.bank), market, [NOBLES[int(nid)] for nid in s.nobles if nid >= 0], players, p)
    return PlayerGameState(state, GAME_RULES)


class ExternalBot:
    def __init__(self):
        self.native = ValueBuyBot()
        self.returned = np.zeros(6, dtype=np.int32)
        self.preferred_noble = None

    def act(self, s, mask):
        """Choose a masked primitive; complete compound return/payment plans faithfully."""
        phase = int(s.phase)
        if phase == env.PAYMENT:
            # Upstream uses minimum necessary gold, with normal colors preferred.
            action = int(np.flatnonzero(mask[72:78])[0] + 72)
        elif phase == env.CHOOSE_NOBLE:
            eligible = np.flatnonzero(mask[67:72])
            preferred = [i for i in eligible if f'n{int(s.nobles[i])}' == self.preferred_noble]
            action = int((preferred or eligible.tolist())[0] + 67)
        elif phase == env.DISCARD:
            available = np.flatnonzero(self.returned > 0)
            if not len(available):
                raise ValueError('Upstream compound action did not specify required returns')
            color = int(available[0])
            self.returned[color] -= 1
            action = color + 61
        else:
            plan = self.native.PlayTurn(to_native(s))
            self.preferred_noble = plan.noble_tile_id
            self.returned = np.array([plan.gems_returned[c] for c in COLOR], np.int32)
            if plan.purchased_card_id is not None:
                cid = int(plan.purchased_card_id[1:])
                ids = np.concatenate((s.market.ravel(), s.reserved[int(s.player)]))
                action = int(np.flatnonzero(ids == cid)[0]) + 31
                # These returned gems are the purchase price, not a later discard.
                cost = np.maximum(np.asarray(env.COST[cid]) - s.bonuses[int(s.player)], 0)
                spend = np.minimum(cost, s.gems[int(s.player), :5])
                expected = np.concatenate((spend, [(cost - spend).sum()]))
                if not np.array_equal(self.returned, expected):
                    raise ValueError('Upstream payment does not match canonical legal payment')
                self.returned[:] = 0
            elif plan.reserved_card_id is not None:
                cid = int(plan.reserved_card_id[1:])
                action = int(np.flatnonzero(s.market.ravel() == cid)[0]) + 46
            elif plan.topdeck_level is not None:
                action = 58 + int(plan.topdeck_level) - 1
            else:
                take = np.array([plan.gems_taken[c] for c in COLOR], np.int32)
                if not take.any():
                    if self.returned.any():
                        raise ValueError('Cannot return gems without a corresponding take')
                    action = 0
                else:
                    candidates = np.flatnonzero(np.all(TAKES == take, axis=1))
                    if not len(candidates):
                        raise ValueError(f'Unsupported upstream take: {take}')
                    action = int(candidates[0]) + 1
        if not mask[action]:
            raise ValueError(f'Upstream proposed illegal primitive action {action} in phase {phase}')
        return action
