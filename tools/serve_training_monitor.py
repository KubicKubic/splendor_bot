"""Read-only live HTTP dashboard for training metrics and sparse Elo results."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import statistics


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Splendor JAX 训练监控</title>
<style>:root{--bg:#081b21;--panel:#102b34;--ink:#e8f5f3;--muted:#8eafb2;--cyan:#55d7d2;--gold:#f1c75b;--green:#67db8c}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui}header{padding:20px 24px;background:#0d252d;display:flex;justify-content:space-between;align-items:center}h1{margin:0;font-size:22px}.live{color:var(--green)}main{padding:18px;max-width:1500px;margin:auto}.cards{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}.card,.chart{background:var(--panel);border:1px solid #ffffff12;border-radius:12px;padding:14px}.label{color:var(--muted);font-size:12px}.big{font-size:25px;font-weight:750;margin-top:5px}.charts{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-top:14px}.chart h2{font-size:15px;margin:0 0 8px}canvas{width:100%;height:230px}.players{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px}.bar{height:8px;background:#1b4650;border-radius:4px;overflow:hidden;margin-top:9px}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}footer{color:var(--muted);padding:15px 3px}@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}.charts,.players{grid-template-columns:1fr}}</style></head>
<body><header><div><h1>Splendor · 混合人数长时训练</h1><div class="label">约 1M 参数 · JAX · 2P/3P/4P · gamma 1.0</div></div><div><span class="live">● LIVE</span> <span id="clock"></span></div></header><main>
<section class="cards"><div class="card"><div class="label">训练进度</div><div class="big" id="progress">—</div><div class="bar"><i id="progressBar"></i></div></div><div class="card"><div class="label">决策吞吐</div><div class="big" id="dps">—</div></div><div class="card"><div class="label">预计剩余</div><div class="big" id="eta">—</div></div><div class="card"><div class="label">最新 Elo</div><div class="big" id="elo">—</div></div><div class="card"><div class="label">KL / Clip</div><div class="big" id="stability">—</div></div><div class="card"><div class="label">策略熵</div><div class="big" id="entropy">—</div></div></section>
<section class="players" id="players"></section><section class="charts"><div class="chart"><h2>训练稳定性</h2><canvas id="loss"></canvas></div><div class="chart"><h2>分人数平均结束回合</h2><canvas id="turns"></canvas></div><div class="chart"><h2>稀疏 Ladder Elo</h2><canvas id="eloChart"></canvas></div><div class="chart"><h2>固定启发式胜率</h2><canvas id="win"></canvas></div></section><footer id="footer"></footer></main>
<script>
const palette=['#55d7d2','#f1c75b','#e879a9','#67db8c'];
function fmtSec(s){if(s==null)return '—';let h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}h ${m}m`:`${m}m`}
function plot(id,series,percent=false){let c=document.getElementById(id),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;let x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,w,h);let vals=series.flatMap(s=>s.y).filter(Number.isFinite);if(!vals.length)return;let lo=Math.min(...vals),hi=Math.max(...vals);if(percent){lo=0;hi=1}if(hi===lo){lo-=1;hi+=1}let pad=30;x.strokeStyle='#ffffff20';x.fillStyle='#8eafb2';x.font='11px system-ui';for(let k=0;k<5;k++){let yy=pad+(h-2*pad)*k/4;x.beginPath();x.moveTo(pad,yy);x.lineTo(w-pad,yy);x.stroke();let v=hi-(hi-lo)*k/4;x.fillText(percent?(v*100).toFixed(0)+'%':v.toFixed(3),2,yy+4)}series.forEach((s,j)=>{x.strokeStyle=palette[j%palette.length];x.lineWidth=2;x.beginPath();s.y.forEach((v,i)=>{let xx=pad+(w-2*pad)*(s.y.length===1?0:i/(s.y.length-1)),yy=pad+(h-2*pad)*(hi-v)/(hi-lo);i?x.lineTo(xx,yy):x.moveTo(xx,yy)});x.stroke();x.fillStyle=x.strokeStyle;x.fillText(s.name,pad+90*j,h-7)});}
async function refresh(){try{let d=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json()),m=d.latest,c=d.config;progress.textContent=`${m.update.toLocaleString()} / ${c.updates.toLocaleString()}`;progressBar.style.width=(100*m.update/c.updates)+'%';dps.textContent=(m.decisions_per_second/1e6).toFixed(2)+' M/s';eta.textContent=fmtSec(d.eta_seconds);elo.textContent=d.latest_elo?d.latest_elo.elo.toFixed(1)+' ± '+((d.latest_elo.ci95[1]-d.latest_elo.ci95[0])/2).toFixed(1):'等待评估';stability.textContent=m.approx_kl.toFixed(4)+' / '+(100*m.clip_fraction).toFixed(1)+'%';entropy.textContent=m.entropy.toFixed(3);players.innerHTML=['2','3','4'].map(p=>`<div class="card"><div class="label">${p} 人局 · 最新训练批次</div><div class="big">${m.mean_turns_by_players[p].toFixed(1)} 回合</div><div>${m.games_by_players[p].toLocaleString()} 局完成 · ${m.timeouts_by_players[p]} 截断</div></div>`).join('');plot('loss',[{name:'value loss',y:d.history.map(x=>x.value_loss)},{name:'KL',y:d.history.map(x=>x.approx_kl)}]);plot('turns',['2','3','4'].map(p=>({name:p+'P',y:d.history.map(x=>x.mean_turns_by_players[p])})));plot('eloChart',[{name:'Elo',y:d.ratings.map(x=>x.elo)}]);plot('win',['2','3','4'].map(p=>({name:p+'P',y:d.diagnostics.filter(x=>x.players==p).map(x=>x.score_all)})),true);footer.textContent=`自然完赛累计 ${d.completed_games.toLocaleString()} · 截断 ${d.timeouts.toLocaleString()} · 数据刷新 ${new Date().toLocaleTimeString()} · 每 5 秒自动刷新`;clock.textContent=new Date().toLocaleTimeString()}catch(e){clock.textContent='读取失败：'+e} }
refresh();setInterval(refresh,5000);
</script></body></html>'''


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def status(run, ladder):
    config = read_json(run / 'config.json', {})
    try:
        rows = [json.loads(x) for x in (run / 'metrics.jsonl').read_text().splitlines() if x]
    except (FileNotFoundError, json.JSONDecodeError):
        rows = []
    if not rows:
        return dict(config=config, latest={}, history=[], ratings=[], diagnostics=[],
                    latest_elo=None, eta_seconds=None, completed_games=0, timeouts=0)
    stride = max(1, len(rows) // 400)
    history = rows[::stride]
    if history[-1] is not rows[-1]:
        history.append(rows[-1])
    ratings_data = read_json(ladder / 'ratings.json', {})
    ratings = ratings_data.get('ratings', [])
    diagnostics = read_json(ladder / 'diagnostics.json', [])
    recent = [x['decisions_per_second'] for x in rows[-100:] if x['seconds'] < 2]
    dps = statistics.median(recent) if recent else rows[-1]['decisions_per_second']
    per_update = config.get('envs', 0) * config.get('horizon', 0)
    remaining = max(0, config.get('updates', rows[-1]['update']) - rows[-1]['update']) * per_update
    return dict(config=config, latest=rows[-1], history=history, ratings=ratings,
                diagnostics=diagnostics, latest_elo=ratings[-1] if ratings else None,
                eta_seconds=remaining / dps if dps else None,
                completed_games=sum(x['games'] for x in rows),
                timeouts=sum(x['timeouts'] for x in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/a100_mixed_1m')
    parser.add_argument('--ladder', default='runs/mixed_longrun_ladder')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(); run, ladder = Path(args.run), Path(args.ladder)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split('?')[0] == '/api/status':
                body = json.dumps(status(run, ladder), allow_nan=False).encode()
                mime = 'application/json'
            elif self.path.split('?')[0] == '/elo_live.png':
                try:
                    body = (ladder / 'elo_live.png').read_bytes()
                except FileNotFoundError:
                    self.send_error(404); return
                mime = 'image/png'
            elif self.path.split('?')[0] in ('/', '/index.html'):
                body = HTML.encode(); mime = 'text/html; charset=utf-8'
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(body)
        def log_message(self, format, *args):
            return

    print(f'http://{args.host}:{args.port}', flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
