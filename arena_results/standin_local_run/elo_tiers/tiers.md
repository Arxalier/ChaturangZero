# Elo tiers (measure)

Checkpoint: `checkpoints/local_run/gen_final.pt`. Ratings are relative to Random = 0.

## Tiers: target vs measured

| Tier | Target | Measured | 95% CI | Gap | Target inside CI? | Settings |
|---|---:|---:|---|---:|---|---|
| Master | 1800 | 31 | [3, 59] | -1769 | no | sims=80, tau=0.05, blunder=0.0 |
| Intermediate | 1300 | 43 | [18, 68] | -1257 | no | sims=15, tau=0.6, blunder=0.0 |
| Novice | 700 | 3 | [-21, 29] | -697 | no | sims=0, tau=1.0, blunder=0.15 |
