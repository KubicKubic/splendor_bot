// Offline execution of the downloaded site's actual rule module. No network or DOM.
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(__dirname + '/../reference/ccbs.73a7734d.chunk.js', 'utf8');
const context = {self: {webpackChunkgame: []}};
vm.runInNewContext(source, context, {timeout: 1000});
const modules = context.self.webpackChunkgame[0][1];
let drawIndex = 0;
const cache = {};
function req(id) {
  if (id === 885) return {Z: x => x};
  if (id === 2982) return {Z: x => Array.from(x)};
  if (id === 3329 || id === 3515) return {};
  if (id === 4420) return {M: n => {if (drawIndex >= n) throw Error('draw index'); return drawIndex;}};
  if (!cache[id]) {
    const exports = {};
    modules[id]({}, exports, req);
    cache[id] = exports;
  }
  return cache[id];
}
req.d = (exports, getters) => {
  for (const name in getters) Object.defineProperty(exports, name, {get: getters[name]});
};
const rules = req(1364);
const table = req(3797);
const requests = JSON.parse(fs.readFileSync(0, 'utf8'));
const output = requests.map(request => {
  if (request.kind === 'table') return {cost: table.s2, bonus: table.dS, points: table.KI, tier: table.XO, nobles: table.Hf};
  drawIndex = request.drawIndex || 0;
  const v = request.view, a = request.action;
  switch (request.kind) {
    case 'pass': return rules.Uj(v);
    case 'take': return rules.L3(v, a);
    case 'reserve': return rules.Cv(v, a[0], a[1]);
    case 'blind': return rules.H1(v, a);
    case 'buy': return rules.y_(v, a[0], a[1], a[2], request.payment);
    case 'noble': return rules.kq(v, a);
    case 'discard': return rules.$u(v, a);
    case 'payments': return rules.h(v, a, v.waitFor);
    default: throw Error(request.kind);
  }
});
process.stdout.write(JSON.stringify(output));
