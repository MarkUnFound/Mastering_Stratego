// Replays games recorded from the Python environment in the browser engine of play.html and
// checks legal moves, action indices, observation planes and results ply by ply.
//     node web/test_engine.js <games.json>
// Run by test/test_web_engine.py.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, 'play.html'), 'utf8');
const src = /<script id="engine">([\s\S]*?)<\/script>/.exec(html)[1];
const ctx = { globalThis: {} };
vm.createContext(ctx);
vm.runInContext(src, ctx);
const E = ctx.globalThis.StrategoEngine;

const games = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const key = (m) => `${m[0][0]},${m[0][1]}>${m[1][0]},${m[1][1]}`;
const fail = (msg) => { console.error(msg); process.exit(1); };

function digest(obs) {
  const nz = [];
  for (let i = 0; i < 28 * 100; i++) if (obs[i] !== 0) nz.push(i);
  return { nz, noBattle: obs[28 * 100], progress: obs[29 * 100] };
}

let plies = 0;
games.forEach((g, gi) => {
  const game = new E.Game({ maxTurns: g.max_turns, noBattleLimit: g.no_battle }).reset(g.p1, g.p2, g.start);
  g.plies.forEach((p, pi) => {
    const where = `game ${gi} ply ${pi}`;
    if (game.current !== p.player) fail(`${where}: side to move ${game.current} != ${p.player}`);
    const legal = game.legalMoves(game.current).map(key).sort();
    const want = p.legal.map(key).sort();
    if (JSON.stringify(legal) !== JSON.stringify(want)) fail(`${where}: legal moves differ\n js ${legal}\n py ${want}`);
    const idx = p.legal.map((m) => E.encodeMove(m, p.player)).sort((a, b) => a - b);
    if (JSON.stringify(idx) !== JSON.stringify(p.indices)) fail(`${where}: action indices differ`);
    p.legal.forEach((m) => {
      if (key(E.decodeAction(E.encodeMove(m, p.player), p.player)) !== key(m)) fail(`${where}: decode(encode) != move`);
    });
    const d = digest(game.observation(p.player));
    if (JSON.stringify(d.nz) !== JSON.stringify(p.obs_nonzero)) fail(`${where}: observation planes differ`);
    if (Math.abs(d.noBattle - p.no_battle_plane) > 1e-6 || Math.abs(d.progress - p.progress_plane) > 1e-6) fail(`${where}: counter planes differ`);
    game.step(p.move);
    plies++;
  });
  if (!game.over) fail(`game ${gi}: not over after the recorded plies`);
  if (game.winner !== g.winner || game.winType !== g.win_type) fail(`game ${gi}: result ${game.winner}/${game.winType} != ${g.winner}/${g.win_type}`);
});
console.log(`ok ${games.length} games, ${plies} plies`);
