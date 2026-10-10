# Last interaction pairs; finite development boundary

M3 V3 and all three mechanism controls now have 3 seeds. Learned V3 mean macro F1=.3508 versus fixed=.3505; metadata-off=.3647 and direct=.3377. It does not meet the five-metric promotion rule or establish independent learned-metadata benefit. Its mean AUC=.7493 substantially recovers V2=.6750 while remaining close to primary BP=.7474. No population novelty/effectiveness success is claimed.

There are two remaining, explicit interaction hypotheses after the first two triple tests:
1. M1 makes M2 useful (M1+M2 F1=.3768 versus M2=.3257); test whether adding the repaired reference operator can retain AUC without the old direct-FC shortcut. First run M1_M3v3 and M2_M3v3. The possible third triple is M1_M2_M3v3.
2. M1+M5 raises recall but loses precision; disease-specific pooling could help recover precision. First run M2_M5. The possible fourth triple is M1_M2_M5.

These are the last missing pairs for these hypotheses, not promotion of weak singles. Evaluate all 3 seeds, record both positive and negative deltas. After these results, run at most these two additional triples for complete priority/compatibility ablation. No further architecture/version/hyperparameter search is planned in this study, and no test data have been evaluated. Final model acceptance still requires the predeclared multi-metric and independent contribution evidence; if no qualified Top-3 exists, report that outcome rather than retaining weak modules for appearance.

Overlap checks before experiments:
| Candidate | Core | Closest Plus | Similarity | Reproduces Plus novelty | Decision |
|---|---|---|---|---|---|
| M1_M3v3 | reviewed signed numeric ROI graph + fixed TRAIN-value sparse reference | individual/population generic graph | Medium | NO | ALLOW |
| M2_M3v3 | reviewed disease-token readout + fixed TRAIN-value sparse reference | pooling/population generic components | Medium | NO | ALLOW |
| M2_M5 | disease-token and numeric second-order readout around unchanged BP | ordinary pooling | Low | NO | ALLOW |

All operators are exactly the individually reviewed versions; no Plus spectral/spatial text or LLM edge embeddings are added. M3v3 uses the already frozen Xavier/gain1 initialization, ordinary three-seed training contract, and same PCA/reference bank. The old models and failed configurations stay visible.
