# Experiment Matrix

| Stage | Benchmark | Models or controls | Purpose | Replication |
|---|---|---|---|---|
| H0 | Heat | classical MLP | derivative and training validation | smoke tests |
| H1 | Heat | bottleneck, no-entanglement, trainable ring, frozen ring | quantum ablation | 3 seeds |
| H2 | Heat | angle and measurement representations | effective rank, CKA, sensitivity | 3 seeds |
| B0 | Burgers | Cole–Hopf reference | nonlinear benchmark validation | quadrature check |
| B1 | Burgers | bottleneck vs trainable QAPINN | integration pilot | 3 seeds |
| B2 | Burgers | stronger IC weighting and longer training | optimization correction | 3 seeds |
| B3 | Burgers | no-entanglement, trainable ring, frozen ring | entanglement and trainability | 3 seeds |
| B4 | Burgers | local/ring classical trig controls | test uniquely quantum explanation | 3 seeds |
| B5 | Burgers | classical and quantum representation geometry | effective rank, CKA, information | 3 seeds |
| B6 | Burgers | five fixed phase seeds | phase sensitivity | 5 paired phase runs |
| B7 | Burgers | 2×2×2 factorial | data/model/phase sensitivity | 8 matched cells |
| B8 | Burgers | 2×4×2 extension | model-initialization robustness | 16 matched cells, 4 model blocks |

## Primary metrics

- relative \(L_2\), MAE, maximum error;
- PDE, initial, and boundary losses;
- low-, middle-, and high-frequency errors;
- gradient connectivity;
- effective rank and feature correlation;
- linear CKA and input sensitivity;
- trainable parameters and runtime.
