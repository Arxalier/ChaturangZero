# ChaturangZero — Engine Explorer UI

A browser front-end for the Milestone 1 rules engine, built for demos and presentations.

Everything on screen is computed by the `chaturanga` package. The browser never decides
what is legal: it asks the Python engine and draws the answer. The HTTP layer is pure
standard library, so the UI adds **no dependencies** beyond the `numpy` the engine
already needs.

## Run it

```bash
python ui/server.py                  # http://127.0.0.1:8000, opens your browser
python ui/server.py --port 9000      # different port
python ui/server.py --no-browser     # just serve
```

Use an interpreter that has the project requirements installed (`pip install -r requirements.txt`);
the server exits with a clear message if `numpy` is missing.

## What it shows

| Panel | Engine surface |
|---|---|
| Board, click a piece → highlighted targets | `rules.get_all_legal_moves` |
| Square inspector: pseudo-legal vs legal vs rejected | `moves.get_*_moves` then the check filter in `rules` |
| Threat map (shaded squares) | `rules.is_in_check`, probed square by square with the Raja |
| Check highlight on the Raja | `rules.is_in_check` |
| Terminal banner: checkmate / bare king / stalemate-wins | `rules.get_game_outcome` |
| Material bars + bare-king warning | piece counts, the same ones `get_game_outcome` uses |
| Move history | `ChaturangaEnv.move_history` |
| Perft with per-move breakdown | `rules.perft` |
| Board state tabs | `board.render_board`, raw int8 matrix, `pieces` encoding |
| Piece codex diagrams | each generator called on an empty board |
| Engine call trace | which engine functions served the last request |

## Demo positions

Seven preset positions, selectable by chip or by pressing <kbd>1</kbd>–<kbd>7</kbd>:
opening, checkmate in 1, stalemate (Black wins), baring the King, absolute pin,
promotion to Mantri, and leapers in a crowd. Each carries a note explaining what to
look at. Loading one replaces the board directly, so terminal states can be reached in
one move during a talk.

## Deep links

The URL captures a demo state, which is handy for a scripted presentation:

```
http://127.0.0.1:8000/?scenario=pin&select=6,4&pseudo=1
http://127.0.0.1:8000/?scenario=opening&attack=1&theme=light
```

* `scenario` — any preset id (`opening`, `checkmate`, `stalemate`, `bareking`, `pin`, `promotion`, `leapers`)
* `select` — square to pre-select, as `row,col`
* `pseudo`, `attack`, `codes`, `flip` — set to `1` to switch that overlay on
* `theme` — `light` for a bright room or a washed-out projector

## Keyboard

<kbd>N</kbd> new · <kbd>U</kbd> undo · <kbd>R</kbd> random legal move · <kbd>A</kbd> auto-play ·
<kbd>P</kbd> pseudo-legal overlay · <kbd>T</kbd> threat map · <kbd>F</kbd> flip ·
<kbd>1</kbd>–<kbd>7</kbd> demo positions · <kbd>Esc</kbd> clear selection

## Files

```
ui/
├── server.py        # stdlib HTTP server + JSON API, one shared game session
├── engine_api.py    # adapter: engine output -> JSON, plus the call trace
├── scenarios.py     # the seven demo positions
└── static/
    ├── index.html   # layout
    ├── style.css    # dark + light themes
    └── app.js       # rendering and interaction only — no game rules
```

## API

| Route | Purpose |
|---|---|
| `GET /api/bootstrap` | codex, scenarios, component status, rules card, legend, state |
| `GET /api/state` | full board state |
| `GET /api/inspect?r=&c=` | one square: generator, pseudo-legal, legal, rejected |
| `GET /api/threats?player=` | squares where that player's Raja would be in check |
| `POST /api/move` | `{"from":[r,c],"to":[r,c]}`, validated against the legal move list |
| `POST /api/new`, `/api/undo`, `/api/random`, `/api/scenario`, `/api/perft` | session controls |
