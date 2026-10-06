# Critic Data Coverage Minimal Experiment

Result: **INCONCLUSIVE**  
Next stage: `CRITIC_DATA_COVERAGE_EXPERIMENT_FOLLOWUP`  
Failure stage: `DIAGNOSTIC_ROLLOUT`

Failure: AssertionError: fixed diagnostic rollout SHA reproduction failed; persisted rollout retained

TRAIN seeds: `[20001, 20002, 20003, 20004]`; rollouts: `4`; unique transitions: `1200`.

Episode 20001 SHA gate: **PASS** (`46b2e837438885dc1a61bb45fae35d5ee490ab335b1daa91a35ddcf6ff46ec2a`).

Coverage target/count: `FULL_EPISODE_MC_RETURN` / `1200`; Welford count: `1200`.

Critic steps: `40`; unique minibatch indices: `1200`; duplicates: `0`.

Coverage checkpoint reload: **PASS**; final Critic digest: `fb32158378cd19271d3fb09544e641b9d67651d470252eb4b2e8e885b971f7d6`.

Diagnostic realization: LOAD `12184115623283823756`; rollout SHA: `0d9f992fbee131b23fa8d3f9227380795a82ad2a72628baf5e02370dd60ec778`; expected `3fa12013dbd9d17df6851a98fc1462db599d46d5bc7c0c0db10128c987b39c20`.

Diagnostic SHA gate failed, so baseline metric reproduction, coverage metrics, and UAV_1 branch evaluation were not run. All recovery NPZs and metadata were retained. Formal PPO update, Epoch 1, and production changes were not executed.\n\nFinal validation: GIT_DIFF_CHECK = PASS.\n

PopArt raw-value check: finite before/after; max absolute drift 0.0001080334; 1190/1200 values meet the old 1e-4 allclose threshold.
