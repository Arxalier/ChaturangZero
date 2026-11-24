# Elo tiers (measure)

Checkpoint: `checkpoints/gen_final.pt`. Ratings are relative to Random = 0.

## Tiers: target vs measured

| Tier | Target | Measured | 95% CI | Gap | Target inside CI? | Settings |
|---|---:|---:|---|---:|---|---|
| Master | 1800 | 632 | [528, 754] | -1168 | no | sims=80, tau=0.05, blunder=0.0 |
| Intermediate | 1300 | 285 | [215, 371] | -1015 | no | sims=15, tau=0.6, blunder=0.0 |
| Novice | 700 | 167 | [101, 244] | -533 | no | sims=0, tau=1.0, blunder=0.15 |
