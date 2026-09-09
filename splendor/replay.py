"""Generate self-contained interactive HTML replays from a JAX checkpoint."""
import argparse
import html
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import env, network
from .train import load
from .valuebot_jax import act as bot_act


COLOR_ZH = ('白', '蓝', '绿', '红', '黑', '金')


def snapshot(s):
    h = jax.device_get(s)
    return dict(turn=int(h.turns), player=int(h.player), bank=h.bank.tolist(),
                gems=h.gems[:2].tolist(), bonuses=h.bonuses[:2].tolist(),
                scores=h.scores[:2].tolist(),
                reserved_count=(h.reserved[:2] >= 0).sum(-1).tolist(),
                market=h.market.tolist(), nobles=h.nobles.tolist(), done=bool(h.done))


def action_text(s, action):
    a = int(action)
    if int(s.phase) == env.PAYMENT:
        return f'支付：当前颜色用 {a - 72} 金代替'
    if int(s.phase) == env.CHOOSE_NOBLE:
        nid = int(s.nobles[a - 67])
        return f'选择贵族 N{nid}（+3 分）'
    if int(s.phase) == env.DISCARD:
        return f'弃 1 枚{COLOR_ZH[a - 61]}宝石'
    if a == 0:
        return '跳过（pass）'
    if a < 31:
        take = np.asarray(env.TAKES[a - 1])
        pieces = [f'{int(n)} {COLOR_ZH[i]}' for i, n in enumerate(take) if n]
        return '拿取 ' + '、'.join(pieces)
    if a < 46:
        slot = a - 31
        cid = int(env.targets(s)[slot])
        return f'购买 C{cid}' + ('（预定牌）' if slot >= 12 else '')
    slot = a - 46
    if slot < 12:
        return f'预定公开牌 C{int(s.market.ravel()[slot])}'
    return f'从第 {slot - 11} 层暗抽预定牌'


def play(checkpoint, seed, mode, max_decisions=4000):
    params, cfg = load(checkpoint)
    if cfg.players != 2:
        raise ValueError('HTML replay currently supports 2-player checkpoints')
    state = env.reset(jax.random.PRNGKey(seed), 2)
    key = jax.random.PRNGKey(seed + 1000003)
    bot_returns = jnp.zeros((2, 6), jnp.int32)

    @jax.jit
    def model_action(s, k):
        logits, _ = network.apply(params, env.observe(s), env.legal_mask(s), True)
        return jax.random.categorical(k, logits).astype(jnp.int32)

    choose_bot = jax.jit(bot_act)
    do_step = jax.jit(env.step)
    frames = [dict(state=snapshot(state), actor=None, action='初始局面')]
    actor = int(state.player)
    descriptions = []
    for _ in range(max_decisions):
        if bool(state.done):
            break
        player = int(state.player)
        is_bot = (mode == 'bot-seat0' and player == 1) or (mode == 'bot-seat1' and player == 0)
        key, ka = jax.random.split(key)
        if is_bot:
            action, carried, valid = choose_bot(state, bot_returns[player])
            if not bool(valid):
                raise RuntimeError(f'JAX bot selected an illegal action at turn {int(state.turns)}')
            bot_returns = bot_returns.at[player].set(carried)
        else:
            action = model_action(state, ka)
        descriptions.append(action_text(state, action))
        old_turn = int(state.turns)
        state = do_step(state, action)
        if int(state.turns) != old_turn or bool(state.done):
            frames.append(dict(state=snapshot(state), actor=actor,
                               action='；'.join(descriptions)))
            descriptions = []
            actor = int(state.player)
    if not bool(state.done):
        raise RuntimeError(f'game seed {seed} did not finish in {max_decisions} decisions')
    winners = np.flatnonzero(np.asarray(env.winners(state))).tolist()
    labels = ['最终模型', 'ValueBuyBot'] if mode == 'bot-seat0' else (
             ['ValueBuyBot', '最终模型'] if mode == 'bot-seat1' else ['最终模型 A', '最终模型 B'])
    return dict(seed=seed, mode=mode, labels=labels, winners=winners,
                total_turns=int(state.turns), frames=frames, cards=env.DATA['cards'],
                noble_costs=env.DATA['nobles'])


def render_html(game, checkpoint):
    payload = json.dumps(game, ensure_ascii=False, separators=(',', ':'))
    title = f"Splendor 回放 · seed {game['seed']}"
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--felt:#123d34;--panel:#102c28;--ink:#f7f3e8;--muted:#a8beb7;--gold:#e8bd54}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at top,#246958,var(--felt) 55%,#08241f);color:var(--ink);font:15px system-ui,sans-serif;min-height:100vh}}
header{{display:flex;justify-content:space-between;gap:18px;align-items:end;padding:18px 24px;background:#071d1acc;box-shadow:0 4px 20px #0007}} h1{{font-size:22px;margin:0}} .sub{{color:var(--muted);font-size:12px}}
.layout{{display:grid;grid-template-columns:240px 1fr 240px;gap:16px;padding:18px;max-width:1450px;margin:auto}} .panel{{background:var(--panel);border:1px solid #ffffff18;border-radius:14px;padding:14px;box-shadow:0 8px 24px #0005}}
.player.active{{outline:3px solid var(--gold)}} .score{{font-size:36px;font-weight:800;color:var(--gold)}} .tokens,.discounts{{display:flex;flex-wrap:wrap;gap:6px;margin-top:9px}}
.gem{{width:31px;height:31px;border-radius:50%;display:grid;place-items:center;font-weight:800;color:#fff;border:2px solid #ffffff55;text-shadow:0 1px 3px #000}} .c0{{background:#ddd;color:#222;text-shadow:none}}.c1{{background:#2775d7}}.c2{{background:#29975a}}.c3{{background:#cf4545}}.c4{{background:#20252b}}.c5{{background:#d0a72e}}
.market{{display:grid;gap:12px}} .tier{{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:10px}} .card{{min-height:112px;border-radius:11px;padding:9px;background:#eee;color:#17201e;border-top:12px solid var(--bonus);box-shadow:0 4px 10px #0005;position:relative}} .card.empty{{opacity:.18}} .pts{{font-size:28px;font-weight:900}} .id{{position:absolute;right:8px;top:7px;color:#555;font-size:12px}} .cost{{display:flex;gap:4px;position:absolute;bottom:8px;left:8px}} .cost .gem{{width:25px;height:25px;font-size:12px;border-width:1px}}
.nobles{{display:flex;gap:8px;margin-bottom:14px}} .noble{{background:#f0e4c9;color:#282018;border-radius:9px;padding:8px;min-width:92px}} .bank{{display:flex;gap:8px;justify-content:center;margin:12px}}
.controls{{position:sticky;bottom:0;background:#071d1af0;padding:12px 20px;display:grid;grid-template-columns:auto auto 1fr auto;gap:12px;align-items:center}} button{{border:0;border-radius:8px;padding:9px 14px;background:#e8bd54;color:#17201e;font-weight:800;cursor:pointer}} input{{width:100%}} #event{{font-size:17px;min-height:26px;text-align:center;color:#ffe39d}} .back{{display:inline-block;background:#253c51;border:2px solid #87a0b5;border-radius:6px;width:36px;height:50px;margin:4px}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr}}.player{{order:2}}.board{{order:1}}.tier{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<header><div><h1>{html.escape(title)}</h1><div class="sub">{html.escape(checkpoint)} · HullQin 规则 · 对手预定牌以背面显示</div></div><div id="status"></div></header>
<main class="layout"><section id="p0" class="panel player"></section><section class="panel board"><div id="nobles" class="nobles"></div><div id="bank" class="bank"></div><div id="market" class="market"></div></section><section id="p1" class="panel player"></section></main>
<div id="event"></div><div class="controls"><button id="play">▶ 播放</button><button id="prev">◀</button><input id="range" type="range" min="0" value="0"><span id="counter"></span></div>
<script>const G={payload}; let i=0,timer=null; const colors=['#ddd','#2775d7','#29975a','#cf4545','#20252b'];
const gem=(n,c)=>`<span class="gem c${{c}}">${{n}}</span>`;
function player(s,p){{let backs='';for(let k=0;k<s.reserved_count[p];k++)backs+='<span class="back"></span>';return `<h2>${{G.labels[p]}}</h2><div class="score">${{s.scores[p]}} 分</div><div class="sub">永久折扣</div><div class="discounts">${{s.bonuses[p].map((n,c)=>gem(n,c)).join('')}}</div><div class="sub">手中宝石</div><div class="tokens">${{s.gems[p].map((n,c)=>gem(n,c)).join('')}}</div><div class="sub">预定牌 ${{s.reserved_count[p]}} 张</div>${{backs}}`}}
function card(id){{if(id<0)return '<div class="card empty"></div>';let c=G.cards[id], cost=c.cost.map((n,k)=>n?gem(n,k):'').join('');return `<div class="card" style="--bonus:${{colors[c.bonus]}}"><span class="pts">${{c.points||''}}</span><span class="id">C${{id}} · T${{c.tier+1}}</span><div class="cost">${{cost}}</div></div>`}}
function draw(){{let f=G.frames[i],s=f.state;for(let p=0;p<2;p++){{let e=document.querySelector('#p'+p);e.innerHTML=player(s,p);e.classList.toggle('active',!s.done&&s.player===p)}}document.querySelector('#bank').innerHTML=s.bank.map((n,c)=>gem(n,c)).join('');document.querySelector('#market').innerHTML=[2,1,0].map(t=>`<div class="tier">${{s.market[t].map(card).join('')}}</div>`).join('');document.querySelector('#nobles').innerHTML=s.nobles.filter(x=>x>=0).map(id=>`<div class="noble"><b>N${{id}} · 3分</b><div>${{G.noble_costs[id].map((n,c)=>n?gem(n,c):'').join('')}}</div></div>`).join('');document.querySelector('#event').textContent=f.actor===null?f.action:`${{G.labels[f.actor]}}：${{f.action}}`;document.querySelector('#status').textContent=s.done?`结束 · 胜者：${{G.winners.map(x=>G.labels[x]).join('、')}}`:`第 ${{s.turn+1}} 回合 · ${{G.labels[s.player]}} 行动`;range.value=i;counter.textContent=`${{i}} / ${{G.frames.length-1}}`;}}
function stop(){{clearInterval(timer);timer=null;play.textContent='▶ 播放'}}function toggle(){{if(timer){{stop();return}}play.textContent='⏸ 暂停';timer=setInterval(()=>{{if(i>=G.frames.length-1){{stop();return}}i++;draw()}},850)}}
range.max=G.frames.length-1;range.oninput=()=>{{i=+range.value;draw()}};play.onclick=toggle;prev.onclick=()=>{{stop();i=Math.max(0,i-1);draw()}};document.addEventListener('keydown',e=>{{if(e.key===' ')toggle();if(e.key==='ArrowRight'){{i=Math.min(G.frames.length-1,i+1);draw()}}if(e.key==='ArrowLeft'){{i=Math.max(0,i-1);draw()}}}});draw();</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--out', default='replays')
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--modes', nargs='+', default=['bot-seat0', 'bot-seat1', 'self'],
                        choices=['bot-seat0', 'bot-seat1', 'self'])
    parser.add_argument('--max-decisions', type=int, default=4000)
    args = parser.parse_args()
    if len(args.seeds) != len(args.modes):
        parser.error('--seeds and --modes must have equal lengths')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    index = []
    for number, (seed, mode) in enumerate(zip(args.seeds, args.modes), 1):
        game = play(args.checkpoint, seed, mode, args.max_decisions)
        path = out / f'game_{number}_{mode}_seed{seed}.html'
        path.write_text(render_html(game, args.checkpoint), encoding='utf-8')
        index.append(dict(file=path.name, mode=mode, seed=seed,
                          turns=game['total_turns'], winners=game['winners']))
        print(json.dumps({**index[-1], 'file': str(path)}, ensure_ascii=False))
    (out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    links = ''.join(f'<li><a href="{html.escape(x["file"])}">对局 {i}</a> — '
                    f'{html.escape(x["mode"])}，seed {x["seed"]}，{x["turns"]} 回合</li>'
                    for i, x in enumerate(index, 1))
    (out / 'index.html').write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Splendor HTML 回放</title>'
        '<style>body{max-width:760px;margin:60px auto;padding:20px;background:#123d34;color:#f7f3e8;'
        'font:18px system-ui}a{color:#ffd66e}li{margin:18px 0}</style>'
        '<h1>第 8000 次更新模型 · Splendor 回放</h1><p>点击后可自动播放、暂停或拖动进度。</p>'
        f'<ol>{links}</ol></html>', encoding='utf-8')


if __name__ == '__main__':
    main()
