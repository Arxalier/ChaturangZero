/* ChaturangZero — Engine Explorer frontend.
 *
 * This file draws things and nothing else: every rule question (what is legal,
 * what is check, what ended the game, how many perft nodes) is answered by the
 * Python engine over /api/*. If a number appears on screen, chaturanga/ produced it.
 */

const GLYPH = { 6: '♚', 5: '♛', 4: '♜', 3: '♝', 2: '♞', 1: '♟' };
const FILES = 'abcdefgh';

const S = {
  boot: null,
  state: null,
  selected: null,     // {r, c}
  inspect: null,      // last /api/inspect payload
  attacks: null,      // threat map: squares where a Raja would be in check
  attackSide: -1,     // which side's Raja is being probed
  auto: null,         // setInterval handle
  hover: null,        // {from:[r,c], to:[r,c]} for the arrow preview
  aiBusy: false,      // prevents overlapping bot requests while thinking
};

const $ = (id) => document.getElementById(id);

/* ───────────────────────────────── transport ───────────────────────────────── */

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

function toast(msg, bad) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.toggle('bad', !!bad);
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 3200);
}

/* ───────────────────────────────── board drawing ───────────────────────────────── */

const flipped = () => $('tgFlip').checked;
const dispRows = () => (flipped() ? [7, 6, 5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5, 6, 7]);
const dispCols = () => (flipped() ? [7, 6, 5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5, 6, 7]);

function squareAt(r, c) {
  return $('board').querySelector(`.sq[data-r="${r}"][data-c="${c}"]`);
}

/** Legal targets for the currently selected origin, taken from engine output. */
function targetsFor(r, c) {
  if (!S.state) return [];
  const group = S.state.legal_by_origin.find((g) => g.from[0] === r && g.from[1] === c);
  return group ? group.targets : [];
}

function drawLabels() {
  $('rankLabels').innerHTML = dispRows()
    .map((r) => `<span><i>${8 - r}</i><em>r${r}</em></span>`).join('');
  $('fileLabels').innerHTML = dispCols()
    .map((c) => `<span><i>${FILES[c]}</i><em>c${c}</em></span>`).join('');
}

function drawBoard() {
  const st = S.state;
  const board = $('board');
  const cells = [];

  for (const r of dispRows()) {
    for (const c of dispCols()) {
      const sq = st.squares[r * 8 + c];
      const classes = ['sq'];
      if (sq.marked) classes.push('marked');
      if (!$('tgMarks').checked) classes.push('hide-marks');

      const piece = sq.piece;
      const own = piece && piece.player === st.current_player;
      if (own && !st.outcome.over) classes.push('selectable');
      if (S.selected && S.selected.r === r && S.selected.c === c) classes.push('selected');

      // Check highlight on the attacked king.
      const wk = st.white_king, bk = st.black_king;
      if (st.white_in_check && wk && wk[0] === r && wk[1] === c) classes.push('check');
      if (st.black_in_check && bk && bk[0] === r && bk[1] === c) classes.push('check');

      // Last move trail.
      const last = st.history[st.history.length - 1];
      if (last) {
        if (last.from[0] === r && last.from[1] === c) classes.push('lastmove');
        if (last.to[0] === r && last.to[1] === c) classes.push('lastmove-to');
      }

      // Attack map overlay.
      if (S.attacks && S.attacks.attacked[`${r},${c}`]) classes.push('attacked');

      let inner = '';

      // Move markers for the selected piece (engine-filtered legal targets).
      if (S.selected && $('tgLegal').checked) {
        const t = targetsFor(S.selected.r, S.selected.c)
          .find((x) => x.to[0] === r && x.to[1] === c);
        if (t) {
          inner += `<span class="marker ${t.capture ? 'cap' : 'dot'}"></span>`;
          if (t.promotion) inner += '<span class="marker promo"></span>';
        }
      }

      // Pseudo-legal squares that the check filter threw away (or any enemy piece's reach).
      if (S.inspect && !S.inspect.empty && $('tgPseudo').checked) {
        const src = S.inspect.is_own_turn ? S.inspect.filtered : S.inspect.pseudo;
        if (src.some((x) => x.to[0] === r && x.to[1] === c)) {
          inner += '<span class="marker pseudo"></span>';
        }
      }

      if (piece) {
        inner += `<span class="piece ${piece.player === 1 ? 'w' : 'b'}" title="${piece.name}">${GLYPH[piece.kind]}</span>`;
        if ($('tgCodes').checked) inner += `<span class="code-badge">${piece.code}</span>`;
      }

      cells.push(`<div class="${classes.join(' ')}" data-r="${r}" data-c="${c}" title="${sq.alg} · row ${r}, col ${c}">${inner}</div>`);
    }
  }
  board.innerHTML = cells.join('');
  drawLabels();
  drawArrows();
}

/** Arrows: gold for the last played move, violet for a hovered move-list entry. */
function drawArrows() {
  const svg = $('arrows');
  const rowIdx = (r) => dispRows().indexOf(r);
  const colIdx = (c) => dispCols().indexOf(c);
  const pt = (r, c) => [colIdx(c) * 100 + 50, rowIdx(r) * 100 + 50];

  const parts = [`<defs>
    <marker id="ah-gold" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="rgba(217,168,78,.9)"/></marker>
    <marker id="ah-violet" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="rgba(154,140,240,.95)"/></marker>
  </defs>`];

  const last = S.state && S.state.history[S.state.history.length - 1];
  if (last) {
    const [x1, y1] = pt(last.from[0], last.from[1]);
    const [x2, y2] = pt(last.to[0], last.to[1]);
    parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="rgba(217,168,78,.75)"
      stroke-width="7" stroke-linecap="round" marker-end="url(#ah-gold)"/>`);
  }
  if (S.hover) {
    const [x1, y1] = pt(S.hover.from[0], S.hover.from[1]);
    const [x2, y2] = pt(S.hover.to[0], S.hover.to[1]);
    parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="rgba(154,140,240,.95)"
      stroke-width="8" stroke-linecap="round" marker-end="url(#ah-violet)"/>`);
  }
  svg.innerHTML = parts.join('');
}

/* ───────────────────────────────── panels ───────────────────────────────── */

function drawStatus() {
  const st = S.state;
  const turn = $('chipTurn');
  turn.className = 'chip ' + (st.current_player === 1 ? 'white-turn' : 'black-turn');
  turn.querySelector('.chip-val').textContent = st.current_player_name;

  const chk = $('chipCheck');
  const inCheck = st.current_player === 1 ? st.white_in_check : st.black_in_check;
  const other = st.current_player === 1 ? st.black_in_check : st.white_in_check;
  chk.className = 'chip ' + (inCheck || other ? 'danger' : 'ok');
  chk.querySelector('.chip-val').textContent =
    inCheck ? `${st.current_player_name} in check` : other ? 'opponent in check' : 'none';

  $('chipLegal').querySelector('.chip-val').textContent = st.legal_count;
  $('chipLegal').className = 'chip ' + (st.legal_count === 0 ? 'danger' : '');
  $('chipPly').querySelector('.chip-val').textContent = st.history.length;

  const note = $('noteBar');
  if (st.note) {
    note.hidden = false;
    note.innerHTML = `<b>${st.scenario}.</b> ${st.note}`;
  } else {
    note.hidden = true;
  }

  $('btnUndo').disabled = !st.can_undo;
  $('bannerUndo').disabled = !st.can_undo;

  const banner = $('banner');
  if (st.outcome.over) {
    banner.hidden = false;
    $('bannerRule').textContent = st.outcome.rule.replace('_', ' ');
    $('bannerHeadline').textContent = st.outcome.headline;
    $('bannerDetail').textContent = st.outcome.detail;
    stopAuto();
  } else {
    banner.hidden = true;
  }
}

function drawLegalList() {
  const st = S.state;
  $('legalBadge').textContent = st.legal_count;
  const host = $('legalList');
  if (!st.legal_by_origin.length) {
    host.innerHTML = '<p class="empty-state">No legal moves — this position is terminal.</p>';
    return;
  }
  host.innerHTML = st.legal_by_origin.map((g) => `
    <div class="legal-group">
      <header>
        <span class="glyph piece ${g.piece.player === 1 ? 'w' : 'b'}" style="display:inline;font-size:16px">${GLYPH[g.piece.kind]}</span>
        <b>${g.piece.sanskrit}</b> ${g.from_alg}
        <small>(${g.from[0]},${g.from[1]})</small>
      </header>
      <div>${g.targets.map((t) => `
        <button class="mv ${t.capture ? 'cap' : ''} ${t.promotion ? 'promo' : ''}"
          data-from="${g.from.join(',')}" data-to="${t.to.join(',')}"
          title="${t.capture ? 'capture' : 'quiet move'}${t.promotion ? ' + promotion to Mantri' : ''}">
          ${t.to_alg}${t.capture ? '×' : ''}${t.promotion ? '★' : ''}</button>`).join('')}
      </div>
    </div>`).join('');

  host.querySelectorAll('.mv').forEach((btn) => {
    const from = btn.dataset.from.split(',').map(Number);
    const to = btn.dataset.to.split(',').map(Number);
    btn.onclick = () => playMove(from, to);
    btn.onmouseenter = () => { S.hover = { from, to }; drawArrows(); };
    btn.onmouseleave = () => { S.hover = null; drawArrows(); };
  });
}

function drawMaterial() {
  const st = S.state;
  const row = (side, label) => {
    const d = st.counts[side];
    const kinds = Object.entries(d.per_kind).map(([k, n]) => `${k}×${n}`).join(' · ') || '—';
    return `<div class="mat-row ${side}">
        <span class="side">${label}</span>
        <span class="mat-bar"><i style="width:${(d.total / 16) * 100}%"></i></span>
        <span class="mat-count">${d.total}/16</span>
        <span class="mat-kinds">${kinds}</span>
      </div>`;
  };
  let html = row('white', 'White') + row('black', 'Black');
  if (st.captured.length) {
    html += `<div class="captured"><span class="mini-label">Captured</span>
      ${st.captured.map((p) => `<span class="piece ${p.player === 1 ? 'w' : 'b'}"
        style="font-size:19px;display:inline;width:auto;height:auto" title="${p.name}">${GLYPH[p.kind]}</span>`).join('')}</div>`;
  }
  for (const [side, label] of [['white', 'White'], ['black', 'Black']]) {
    if (st.counts[side].non_king === 1) {
      html += `<div class="bare-warn">${label} has one non-Raja piece left — capturing it
        triggers the Bare King rule and ends the game immediately.</div>`;
    }
  }
  $('material').innerHTML = html;
}

function drawHistory() {
  const st = S.state;
  const host = $('history');
  if (!st.history.length) {
    host.innerHTML = '<p class="empty-state">No moves yet.</p>';
    return;
  }
  host.innerHTML = `<table class="hist-table">${st.history.slice().reverse().map((h) => `
    <tr><td class="n">${h.index}</td><td class="p">${h.player}</td>
        <td class="m">${h.text}</td><td class="c">${h.coords}</td></tr>`).join('')}</table>`;
}

function drawStateViews() {
  const st = S.state;
  $('asciiOut').textContent = st.ascii;
  $('matrixOut').innerHTML = st.matrix.flat()
    .map((v) => `<i class="${v > 0 ? 'pos' : v < 0 ? 'neg' : ''}">${v}</i>`).join('');
  $('calls').innerHTML = (st.engine_calls || [])
    .map((c) => `<span class="call">${c}()</span>`).join('') ||
    '<p class="empty-state">—</p>';
}

function drawInspector() {
  const host = $('inspector');
  const ins = S.inspect;
  if (!ins || ins.empty) {
    host.innerHTML = '<p class="empty-state">Click any piece to see which generator produces its moves, and how many of them survive the check filter.</p>';
    return;
  }
  const p = ins.piece;
  const pills = (arr, cls) => arr.length
    ? arr.map((x) => `<span class="pill ${cls} ${x.capture ? 'cap' : ''} ${x.promotion ? 'promo' : ''}">${x.alg}</span>`).join('')
    : '<span class="pill">none</span>';

  host.innerHTML = `
    <div class="ins-head">
      <div class="ins-glyph"><span class="piece ${p.player === 1 ? 'w' : 'b'}" style="font-size:30px">${GLYPH[p.kind]}</span></div>
      <div>
        <div class="ins-name">${p.english} <em>${p.sanskrit}</em></div>
        <div class="ins-sub">${ins.square.alg} · (${ins.square.r},${ins.square.c}) · code ${p.code} · '${p.symbol}'</div>
      </div>
    </div>
    <div class="ins-rows">
      <div class="ins-row"><span>Generator</span><span><code>${ins.generator}</code></span></div>
      <div class="ins-row"><span>Movement</span><span>${ins.mechanics}</span></div>
      <div class="ins-row"><span>Modern kin</span><span>${ins.modern}</span></div>
      <div class="ins-row"><span>Pseudo-legal</span><span class="pillset">${pills(ins.pseudo, '')}</span></div>
      ${ins.is_own_turn ? `
      <div class="ins-row"><span>Legal</span><span class="pillset">${pills(ins.legal, 'legal')}</span></div>` : ''}
      ${ins.is_own_turn && ins.filtered.length ? `
      <div class="ins-row"><span>Rejected</span><span class="pillset">${ins.filtered.map((x) => `<span class="pill cut">${x.alg}</span>`).join('')}</span></div>` : ''}
    </div>
    ${ins.is_own_turn && ins.filtered.length ? `<div class="ins-callout">
      <b>${ins.filtered.length} of ${ins.pseudo.length}</b> pseudo-legal squares were discarded by
      <code>rules.get_all_legal_moves</code>: each one leaves this player's Raja in check
      (a pin, or an illegal king step). Turn on the <b>pseudo-legal</b> overlay to see them on the board.
    </div>` : ''}
    ${!ins.is_own_turn ? `<div class="ins-callout" style="border-color:var(--violet);background:rgba(154,140,240,.14)">
      This piece belongs to the side not on turn, so only its pseudo-legal reach is shown —
      exactly what <code>rules.is_in_check</code> scans when it looks for attacks on the Raja.
    </div>` : ''}`;
}

function drawPerft(res) {
  $('perftOut').innerHTML = `
    <div class="perft-head">
      <div class="perft-stat"><b>${res.nodes.toLocaleString()}</b><span>nodes</span></div>
      <div class="perft-stat"><b>${res.depth}</b><span>depth</span></div>
      <div class="perft-stat"><b>${res.seconds}s</b><span>elapsed</span></div>
      <div class="perft-stat"><b>${res.nps ? res.nps.toLocaleString() : '—'}</b><span>nodes/sec</span></div>
    </div>
    <div class="divide"><table>${res.divide.map((d) => `
      <tr><td>${d.move}</td><td>${d.nodes.toLocaleString()}</td></tr>`).join('')}</table></div>`;
}

/* ───────────────────────── static panels (built once) ───────────────────────── */

function drawCodex() {
  $('codex').innerHTML = S.boot.codex.map((p) => {
    const [orow, ocol] = p.origin;
    const inWindow = (r, c) => Math.abs(r - orow) <= 2 && Math.abs(c - ocol) <= 2;
    const reach = new Set(p.free_moves.filter(([r, c]) => inWindow(r, c)).map(([r, c]) => `${r},${c}`));
    const beyond = p.free_moves.length - reach.size;

    let mini = '';
    for (let r = orow - 2; r <= orow + 2; r++) {
      for (let c = ocol - 2; c <= ocol + 2; c++) {
        if (r === orow && c === ocol) {
          mini += `<i class="origin"><span class="piece w" style="font-size:15px">${GLYPH[p.kind]}</span></i>`;
        } else {
          mini += `<i class="${reach.has(`${r},${c}`) ? 'reach' : ''}"></i>`;
        }
      }
    }
    const leaps = p.surrounded_count === p.free_count;
    return `<div class="codex-card" data-kind="${p.kind}">
      <div class="mini">${mini}</div>
      <div class="codex-info">
        <h4>${p.english} <em>${p.sanskrit}</em> <span class="stat">'${p.symbol}' = ${p.kind}</span></h4>
        <p>${p.mechanics}</p>
        <div class="codex-stats">
          <span class="stat">${p.free_count} moves, open board</span>
          <span class="stat ${leaps ? 'leap' : 'block'}">${p.surrounded_count} when walled in${leaps ? ' — leaps' : ''}</span>
          ${beyond ? `<span class="stat">+${beyond} beyond this window</span>` : ''}
        </div>
        <p style="margin-top:4px"><code style="font-size:10px">${p.generator.replace('chaturanga.', '')}</code> · modern kin: ${p.modern}</p>
      </div>
    </div>`;
  }).join('');
}

function drawComponents() {
  $('components').innerHTML = S.boot.components.map((c, i) => `
    <div class="comp ${i === 3 ? 'open' : ''}">
      <div class="comp-head">
        <span class="dotstatus ${c.status}"></span>
        <span class="comp-title">${c.module}</span>
        <span class="comp-tag">${c.status}</span>
      </div>
      <div class="comp-body">
        <p>${c.summary}</p>
        <ul>${c.items.map((it) => `<li>${it}</li>`).join('')}</ul>
      </div>
    </div>`).join('');
  $('components').querySelectorAll('.comp-head').forEach((h) => {
    h.onclick = () => h.parentElement.classList.toggle('open');
  });
}

function drawRules() {
  const r = S.boot.rules;
  $('rules').innerHTML = r.terminal.map((t) => `
    <div class="rule-row">
      <h4>${t.name} <small>${t.reward}</small></h4>
      <p>${t.detail}</p>
    </div>`).join('') +
    `<ul class="rule-notes">${r.notes.map((n) => `<li>${n}</li>`).join('')}</ul>`;
}

function drawLegendTable() {
  $('legendOut').innerHTML = S.boot.legend.map((l) => `
    <div class="leg-row">
      <span class="g piece w">${GLYPH[l.kind]}</span>
      <span class="g piece b">${GLYPH[l.kind]}</span>
      <span class="nm">${l.english} <em>${l.sanskrit}</em></span>
      <span class="cd">${l.white_code >= 0 ? '+' : ''}${l.white_code} '${l.white_symbol}' / ${l.black_code} '${l.black_symbol}'</span>
    </div>`).join('');
}

function drawScenarios() {
  $('scenarios').innerHTML = S.boot.scenarios.map((s, i) => `
    <button class="scenario-chip" data-id="${s.id}">
      <b>${i + 1}. ${s.label}</b><small>${s.subtitle}</small>
    </button>`).join('');
  $('scenarios').querySelectorAll('.scenario-chip').forEach((b) => {
    b.onclick = () => loadScenario(b.dataset.id);
  });
}

function markActiveScenario() {
  $('scenarios').querySelectorAll('.scenario-chip').forEach((b) => {
    const s = S.boot.scenarios.find((x) => x.id === b.dataset.id);
    b.classList.toggle('active', !!S.state && s && s.label === S.state.scenario);
  });
}

function selectedBotId() {
  const sel = $('botSelect');
  return sel ? sel.value : 'alphazero';
}

function updateBotSubtitle() {
  const id = selectedBotId();
  const bots = (S.boot && S.boot.bots) || [];
  const spec = bots.find((b) => b.id === id);
  const el = $('botSubtitle');
  if (el) el.textContent = spec ? spec.subtitle : '';
}

function drawBots() {
  const sel = $('botSelect');
  const bots = (S.boot && S.boot.bots) || [];
  if (!sel || !bots.length) return;
  sel.innerHTML = bots
    .map((b) => `<option value="${b.id}">${b.name} (${b.family === 'neural' ? 'AlphaZero' : 'Scholar'})</option>`)
    .join('');
  sel.value = 'chanakya';
  updateBotSubtitle();
}

/* ───────────────────────────────── actions ───────────────────────────────── */

function applyState(state) {
  S.state = state;
  drawStatus();
  drawBoard();
  drawLegalList();
  drawMaterial();
  drawHistory();
  drawStateViews();
  markActiveScenario();
}

async function refreshAttacks() {
  if (!$('tgAttack').checked) { S.attacks = null; drawBoard(); return; }
  S.attacks = await apiGet(`/api/threats?player=${S.attackSide}`);
  $('btnAtkWhite').classList.toggle('active', S.attackSide === 1);
  $('btnAtkBlack').classList.toggle('active', S.attackSide === -1);
  drawBoard();
}

async function playMove(from, to) {
  const res = await apiPost('/api/move', { from, to });
  if (!res.result.ok) { toast(res.result.error, true); return; }
  S.selected = null;
  S.inspect = null;
  S.hover = null;
  applyState(res.state);
  drawInspector();
  await refreshAttacks();

  const bits = [];
  if (res.result.captured) bits.push('capture');
  if (res.result.promoted) bits.push('Padati promoted to Mantri (board.make_move)');
  if (bits.length) toast(bits.join(' · '));

  if ($('tgVsAI') && $('tgVsAI').checked && !res.result.done && res.state.current_player === -1) {
    setTimeout(aiMove, 220);
  }
}

async function selectSquare(r, c) {
  const st = S.state;
  const sq = st.squares[r * 8 + c];

  // Clicking a highlighted legal target plays the move.
  if (S.selected) {
    const t = targetsFor(S.selected.r, S.selected.c).find((x) => x.to[0] === r && x.to[1] === c);
    if (t) { await playMove([S.selected.r, S.selected.c], [r, c]); return; }
  }

  if (!sq.piece) { S.selected = null; S.inspect = null; drawBoard(); drawInspector(); return; }

  S.selected = { r, c };
  S.inspect = await apiGet(`/api/inspect?r=${r}&c=${c}`);
  $('calls').innerHTML = (S.inspect.engine_calls || [])
    .map((x) => `<span class="call">${x}()</span>`).join('');
  drawBoard();
  drawInspector();
}

async function newGame() {
  stopAuto();
  const res = await apiPost('/api/new');
  S.selected = null; S.inspect = null;
  applyState(res.state);
  drawInspector();
  await refreshAttacks();
}

async function loadScenario(id) {
  stopAuto();
  const res = await apiPost('/api/scenario', { id });
  S.selected = null; S.inspect = null;
  applyState(res.state);
  drawInspector();
  await refreshAttacks();
}

async function undo() {
  const res = await apiPost('/api/undo');
  if (!res.result.ok) { toast('Nothing to undo', true); return; }
  S.selected = null; S.inspect = null;
  applyState(res.state);
  drawInspector();
  await refreshAttacks();
}

async function randomMove() {
  const res = await apiPost('/api/random');
  if (!res.result.ok) { toast(res.result.error || 'No move available', true); stopAuto(); return false; }
  S.selected = null; S.inspect = null;
  applyState(res.state);
  drawInspector();
  await refreshAttacks();
  return !res.result.done;
}

async function aiMove() {
  if (S.aiBusy || !S.state || S.state.outcome.over) return false;
  const botId = selectedBotId();
  const btn = $('btnAI');
  S.aiBusy = true;
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Thinking…';
  }
  try {
    const res = await apiPost('/api/ai_move', { bot: botId });
    if (!res.result.ok) {
      toast(res.result.error || 'No bot move available', true);
      return false;
    }
    S.selected = null;
    S.inspect = null;
    S.hover = null;
    applyState(res.state);
    drawInspector();
    await refreshAttacks();
    const who = res.result.bot_name || botId;
    const ms = res.result.elapsed_ms != null ? ` (${res.result.elapsed_ms} ms)` : '';
    toast(`${who} moved${ms}`);
    return !res.result.done;
  } catch (err) {
    toast(`Bot move failed: ${err.message}`, true);
    return false;
  } finally {
    S.aiBusy = false;
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = 'Bot Move <kbd>Z</kbd>';
    }
  }
}

function stopAuto() {
  if (S.auto) { clearInterval(S.auto); S.auto = null; }
  $('btnAuto').classList.remove('active');
  $('btnAuto').innerHTML = 'Auto-play <kbd>A</kbd>';
}

function toggleAuto() {
  if (S.auto) { stopAuto(); return; }
  $('btnAuto').classList.add('active');
  $('btnAuto').innerHTML = 'Stop auto-play <kbd>A</kbd>';
  S.auto = setInterval(async () => {
    const cont = await randomMove();
    if (!cont) stopAuto();
  }, 750);
}

async function runPerft() {
  const depth = Number($('perftDepth').value);
  const btn = $('btnPerft');
  btn.disabled = true;
  btn.textContent = `Searching depth ${depth}…`;
  $('perftOut').innerHTML = '<p class="empty-state">Running rules.perft — the engine is pure Python, so deeper searches take a moment.</p>';
  try {
    const res = await apiPost('/api/perft', { depth });
    drawPerft(res.result);
    applyState(res.state);
  } catch (err) {
    toast(`Perft failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run from this position';
  }
}

/* ───────────────────────────────── wiring ───────────────────────────────── */

function wire() {
  $('board').addEventListener('click', (ev) => {
    const sq = ev.target.closest('.sq');
    if (!sq || S.state.outcome.over || S.aiBusy) return;
    selectSquare(Number(sq.dataset.r), Number(sq.dataset.c));
  });

  $('btnNew').onclick = newGame;
  $('bannerNew').onclick = newGame;
  $('btnUndo').onclick = undo;
  $('bannerUndo').onclick = undo;
  $('btnRandom').onclick = randomMove;
  if ($('btnAI')) $('btnAI').onclick = aiMove;
  if ($('botSelect')) $('botSelect').onchange = updateBotSubtitle;
  if ($('tgVsAI')) {
    $('tgVsAI').onchange = () => {
      if ($('tgVsAI').checked && S.state && !S.state.outcome.over && S.state.current_player === -1) {
        aiMove();
      }
    };
  }
  $('btnAuto').onclick = toggleAuto;
  $('btnPerft').onclick = runPerft;

  ['tgLegal', 'tgPseudo', 'tgCodes', 'tgMarks', 'tgFlip'].forEach((id) => {
    $(id).onchange = drawBoard;
  });
  $('tgAttack').onchange = () => {
    $('attackSideWrap').hidden = !$('tgAttack').checked;
    if ($('tgAttack').checked) S.attackSide = S.state.current_player;
    refreshAttacks();
  };
  $('btnAtkWhite').onclick = () => { S.attackSide = 1; refreshAttacks(); };
  $('btnAtkBlack').onclick = () => { S.attackSide = -1; refreshAttacks(); };

  $('stateTabs').querySelectorAll('.tab').forEach((tab) => {
    tab.onclick = () => {
      $('stateTabs').querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      $('asciiOut').hidden = tab.dataset.tab !== 'ascii';
      $('matrixOut').hidden = tab.dataset.tab !== 'matrix';
      $('legendOut').hidden = tab.dataset.tab !== 'legend';
    };
  });

  $('btnTheme').onclick = () => {
    const root = document.documentElement;
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  };
  $('btnHelp').onclick = () => { $('helpModal').hidden = false; };
  $('btnHelpClose').onclick = () => { $('helpModal').hidden = true; };
  $('helpModal').onclick = (ev) => { if (ev.target === $('helpModal')) $('helpModal').hidden = true; };

  document.addEventListener('keydown', (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (ev.target && /^(INPUT|SELECT|TEXTAREA)$/.test(ev.target.tagName)) return;
    const k = ev.key.toLowerCase();
    if (k === 'escape') { S.selected = null; S.inspect = null; $('helpModal').hidden = true; drawBoard(); drawInspector(); }
    else if (k === 'n') newGame();
    else if (k === 'u') undo();
    else if (k === 'r') randomMove();
    else if (k === 'z') aiMove();
    else if (k === 'a') toggleAuto();
    else if (k === 'p') { $('tgPseudo').checked = !$('tgPseudo').checked; drawBoard(); }
    else if (k === 't') { $('tgAttack').checked = !$('tgAttack').checked; $('tgAttack').onchange(); }
    else if (k === 'f') { $('tgFlip').checked = !$('tgFlip').checked; drawBoard(); }
    else if (k >= '1' && k <= '9') {
      const s = S.boot.scenarios[Number(k) - 1];
      if (s) loadScenario(s.id);
    }
  });
}

async function boot() {
  try {
    S.boot = await apiGet('/api/bootstrap');
  } catch (err) {
    document.body.innerHTML = `<pre style="padding:40px;color:#e3636b">Could not reach the engine API.\n${err.message}\n\nStart the server with:  python ui/server.py</pre>`;
    return;
  }
  drawCodex();
  drawComponents();
  drawRules();
  drawLegendTable();
  drawScenarios();
  drawBots();
  wire();
  applyState(S.boot.state);
  await applyDeepLink();
}

/** Restores a position (and optionally a selected square) straight from the URL. */
async function applyDeepLink() {
  const params = new URLSearchParams(location.search);
  if (params.get('theme') === 'light') document.documentElement.dataset.theme = 'light';
  const scenario = params.get('scenario');
  if (scenario && S.boot.scenarios.some((s) => s.id === scenario)) await loadScenario(scenario);
  for (const flag of ['pseudo', 'attack', 'codes', 'flip']) {
    if (params.get(flag) === '1') {
      const box = $('tg' + flag[0].toUpperCase() + flag.slice(1));
      box.checked = true;
      if (flag === 'attack') { $('attackSideWrap').hidden = false; S.attackSide = S.state.current_player; await refreshAttacks(); }
    }
  }
  const sel = params.get('select');
  if (sel) {
    const [r, c] = sel.split(',').map(Number);
    if (Number.isInteger(r) && Number.isInteger(c)) await selectSquare(r, c);
  }
  drawBoard();
}

boot();
