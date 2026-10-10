# Validation-only ablation evidence

Status: COMPLETE; registered matrix complete: True; independent verification current: True.

Mean marginal averages all four backgrounds equally within each seed, then averages the three seeds. Full-minus-one is reported separately. No selection lock or test access occurs.

## M1_M3_M5

| Module | Four-context mean F1 delta | Sample SD | Full-minus-one F1 delta |
|---|---:|---:|---:|
| M1 | +0.010754 | 0.016068 | -0.025238 |
| M3_v2 | +0.011429 | 0.045201 | -0.010906 |
| M5 | +0.017894 | 0.017392 | -0.015503 |

Pending: []
Numeric gate: False

## M1_M4_M5

| Module | Four-context mean F1 delta | Sample SD | Full-minus-one F1 delta |
|---|---:|---:|---:|
| M1 | +0.011783 | 0.017317 | -0.004065 |
| M4 | -0.002979 | 0.008620 | -0.020142 |
| M5 | +0.012481 | 0.025812 | -0.007216 |

Pending: []
Numeric gate: False

## M1_M2_M3v3

| Module | Four-context mean F1 delta | Sample SD | Full-minus-one F1 delta |
|---|---:|---:|---:|
| M1 | +0.030787 | 0.025096 | +0.037976 |
| M2 | -0.001211 | 0.007731 | +0.007780 |
| M3_v3 | +0.010780 | 0.008026 | +0.006690 |

Pending: []
Numeric gate: False

## M1_M2_M5

| Module | Four-context mean F1 delta | Sample SD | Full-minus-one F1 delta |
|---|---:|---:|---:|
| M1 | +0.036513 | 0.007969 | +0.067317 |
| M2 | -0.013332 | 0.015533 | +0.001429 |
| M5 | +0.008713 | 0.020502 | +0.007172 |

Pending: []
Numeric gate: False

Eligible simpler ranking: M1

Illustrative triple ranking (provisional until all complete): M1_M2_M5, M1_M2_M3v3, M1_M3_M5, M1_M4_M5

Mechanism/novelty judgment is not automated. Generic readouts and old M3 current-FC bypass require explicit qualification.
