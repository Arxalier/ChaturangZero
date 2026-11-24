# Elo tiers (measure)

Checkpoint: `checkpoints/gen_final.pt`. Ratings are relative to Random = 0.

## Tiers: target vs measured

| Tier | Target | Measured | 95% CI | Gap | Target inside CI? | Settings |
|---|---:|---:|---|---:|---|---|
| Master | 1800 | 540 | [475, 616] | -1260 | no | sims=80, tau=0.05, blunder=0.0 |
| Intermediate | 1300 | 287 | [227, 356] | -1013 | no | sims=15, tau=0.6, blunder=0.0 |
| Novice | 700 | 174 | [118, 237] | -526 | no | sims=0, tau=1.0, blunder=0.15 |
