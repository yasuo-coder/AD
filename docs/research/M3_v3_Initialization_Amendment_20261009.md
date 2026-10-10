# M3 V3 initialization amendment before formal training

The population operator, fixed TRAIN PCA, metadata features, top32 support, output rank8, norm bound<2, controls, loss and training schedule remain as reviewed in M3_v3_Review_20261009.md. No V3 epoch has been trained and no validation performance was used for this amendment.

A fixed first20 TRAIN probe found the proposed std=.02/gain=.1 initialization produced metadata-scorer classification gradient norm 2.92e-9 against weight-decay gradient 9.53e-4. Merely checking nonzero gradients was insufficient. A Xavier-uniform/gain=1 initialization increased the same classification gradient to 4.51e-5 and mean residual norm to .4395, still far below the unchanged <2 norm bound. This does not establish that gradients will dominate decay or that validation improves; actual learned relations must be checked after training.

Use Xavier-uniform for both metadata MLP Linear weights and both output-head matrices, zero biases, and gamma_raw=0 (gamma=1). Keep theta=0, temperature initialization and fusion as reviewed. Initialize all four controls with the same random draws for the output head, then zero/freeze scorers for fixed/no_metadata/direct exactly as their definitions require. Freeze this one initialization correction now; no validation-based search over scales.

The original small-initialization code and contract proof remain archived. Use candidate_v3_initialized_runner to apply the amendment, avoiding edits to sources already captured by running experiments.

Candidate: M3 V3 initialized operator and controls. Problem: practically negligible classification gradients at initial signal scale. Core mechanism: unchanged reviewed reference-only relation readout; only initial parameter scale differs. Closest Plus component: population graph, same as prior review. Similarity: Medium. Reproduces a BrainPromptPlus novelty: NO. Decision: ALLOW. Initialization itself is not a scientific innovation.

Evidence: verification/m3v3_initialization_scale_probe.json (TRAIN only); original full design: audits/M3_v3_Review_20261009.md. Main implementation/source and PCA provenance remain separately hashed.
