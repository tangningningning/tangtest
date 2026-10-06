# Formal Macro N2 Target Short10

**Result: FORMAL_N2_SHORT10_GATE = PARTIAL; SELECTED_FORMAL_TARGET = N2; TARGET_SEARCH_CLOSED = YES.**

## Run and configuration

- Run: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\formal_macro_n2_target_ckptv31_archv9_short10`.
- Fresh initialization seed 27005; episode seeds 20001–20010; resume false; 10/10 episodes completed.
- Macro Actor DURATION_AWARE_SMDP_GAE; Macro Critic TWO_STEP_SMDP_TD; architecture v9; checkpoint v31.
- Macro Critic steps 50 per episode; replay history 0; Exact-Q disabled; algorithm and configuration were unchanged for this run.
- Persisted 1017 real Macro transitions; trainer exited with code 0.

## Production trace and frozen-target checks

- N2_TARGET_RECOMPUTE_MAX_ABS_ERROR = 0.0; REALIZED_RETURN_RECOMPUTE_MAX_ABS_ERROR = 0.0.
- n2_target dtype: `float32`; critic training target dtype: `float32`; realized-return dtype: `float32`.
- Target detached: True; target frozen for all 10 critic updates: True; before/after target hashes identical in all 10 episodes: True.
- These are the trainer production-path per-episode lightweight trace checks; no separate preflight was run.
- EXACT_Q_ENV_EXECUTIONS = 0.

## Per-episode target metrics

| Episode | PRE target EV | POST target EV | PRE target corr | POST target corr | PRE target std ratio | POST target std ratio |
|---|---:|---:|---:|---:|---:|---:|
| E1 | -0.000089 | 0.548632 | -0.205067 | 0.741974 | 0.000217 | 0.698468 |
| E2 | 0.083406 | 0.473639 | 0.383674 | 0.688560 | 0.131088 | 0.666748 |
| E3 | -0.041997 | 0.516663 | -0.003491 | 0.720905 | 0.201471 | 0.665762 |
| E4 | 0.396722 | 0.828469 | 0.629930 | 0.910933 | 0.639408 | 0.874475 |
| E5 | 0.695078 | 0.893226 | 0.834099 | 0.945229 | 0.808736 | 0.930008 |
| E6 | 0.491231 | 0.774800 | 0.701215 | 0.880260 | 0.722920 | 0.872708 |
| E7 | 0.299591 | 0.649328 | 0.626977 | 0.806177 | 0.321184 | 0.781830 |
| E8 | 0.627264 | 0.903305 | 0.796719 | 0.950475 | 0.883309 | 0.940581 |
| E9 | 0.372842 | 0.735931 | 0.614191 | 0.858600 | 0.680436 | 0.823060 |
| E10 | 0.180987 | 0.703423 | 0.443747 | 0.843041 | 0.317555 | 0.757634 |

## Realized-return reference metrics

| Episode | PRE return EV | POST return EV | PRE return corr | POST return corr | PRE return std ratio | POST return std ratio |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 0.000001 | 0.013936 | 0.017274 | 0.118763 | 0.000041 | 0.131753 |
| E2 | 0.006087 | 0.064570 | 0.102346 | 0.267620 | 0.036108 | 0.183654 |
| E3 | 0.018925 | -0.154540 | 0.184715 | -0.279007 | 0.061448 | 0.203056 |
| E4 | -0.367394 | -0.379144 | -0.430159 | -0.228614 | 0.313098 | 0.428203 |
| E5 | 0.292935 | 0.369637 | 0.675907 | 0.748808 | 0.271042 | 0.311685 |
| E6 | 0.051154 | 0.082728 | 0.284439 | 0.373635 | 0.111953 | 0.135150 |
| E7 | -0.059681 | -0.276800 | -0.315492 | -0.579036 | 0.083526 | 0.203321 |
| E8 | -0.297067 | -0.286082 | -0.010573 | 0.033324 | 0.534568 | 0.569228 |
| E9 | 0.163768 | 0.143826 | 0.448278 | 0.387232 | 0.255443 | 0.308986 |
| E10 | -0.048467 | -0.081842 | -0.489183 | -0.306574 | 0.047257 | 0.112746 |

## Macro Critic fitting snapshots

| Episode | Step | Target EV | Target corr | Target std ratio | Target MSE | Return EV | Return corr | Return std ratio | Return MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 0 | -0.000089 | -0.205067 | 0.000217 | 4364.423373 | 0.000001 | 0.017274 | 0.000041 | 391295.807226 |
| E1 | 10 | 0.053694 | 0.675649 | 0.040978 | 4134.398232 | -0.001505 | -0.093464 | 0.007730 | 390889.631672 |
| E1 | 25 | 0.313689 | 0.669588 | 0.302629 | 3087.756478 | -0.009379 | -0.053609 | 0.057085 | 385539.801960 |
| E1 | 50 | 0.548632 | 0.741974 | 0.698468 | 1964.885289 | 0.013936 | 0.118763 | 0.131753 | 393259.883024 |
| E2 | 0 | 0.083406 | 0.383674 | 0.131088 | 6730.692764 | 0.006087 | 0.102346 | 0.036108 | 101049.716414 |
| E2 | 10 | 0.181449 | 0.468815 | 0.273012 | 5965.878633 | 0.039918 | 0.303011 | 0.075200 | 99078.387788 |
| E2 | 25 | 0.339745 | 0.590294 | 0.497013 | 4816.443352 | 0.084740 | 0.377946 | 0.136901 | 94255.095039 |
| E2 | 50 | 0.473639 | 0.688560 | 0.666748 | 3836.098167 | 0.064570 | 0.267620 | 0.183654 | 96452.820442 |
| E3 | 0 | -0.041997 | -0.003491 | 0.201471 | 15465.807548 | 0.018925 | 0.184715 | 0.061448 | 1774879.286558 |
| E3 | 10 | 0.094164 | 0.312930 | 0.251599 | 13042.948722 | -0.051155 | -0.294946 | 0.076737 | 1707529.151220 |
| E3 | 25 | 0.265426 | 0.525384 | 0.422420 | 10557.627500 | -0.145220 | -0.499162 | 0.128837 | 1755864.963003 |
| E3 | 50 | 0.516663 | 0.720905 | 0.665762 | 6931.039118 | -0.154540 | -0.279007 | 0.203056 | 1748847.504414 |
| E4 | 0 | 0.396722 | 0.629930 | 0.639408 | 2171.898309 | -0.367394 | -0.430159 | 0.313098 | 66735.237755 |
| E4 | 10 | 0.641387 | 0.809199 | 0.925027 | 1287.206634 | -0.410290 | -0.226423 | 0.452957 | 66186.759716 |
| E4 | 25 | 0.769198 | 0.877059 | 0.882975 | 828.767422 | -0.417413 | -0.266527 | 0.432365 | 66132.895025 |
| E4 | 50 | 0.828469 | 0.910933 | 0.874475 | 645.103629 | -0.379144 | -0.228614 | 0.428203 | 63537.809874 |
| E5 | 0 | 0.695078 | 0.834099 | 0.808736 | 2927.760503 | 0.292935 | 0.675907 | 0.271042 | 75013.362348 |
| E5 | 10 | 0.784426 | 0.902824 | 1.077936 | 2030.849187 | 0.397070 | 0.730190 | 0.361262 | 67614.194346 |
| E5 | 25 | 0.851309 | 0.922788 | 0.937913 | 1393.127408 | 0.382867 | 0.766180 | 0.314334 | 70163.055863 |
| E5 | 50 | 0.893226 | 0.945229 | 0.930008 | 999.356478 | 0.369637 | 0.748808 | 0.311685 | 70743.526123 |
| E6 | 0 | 0.491231 | 0.701215 | 0.722920 | 1646.853519 | 0.051154 | 0.284439 | 0.111953 | 297748.560554 |
| E6 | 10 | 0.537798 | 0.733348 | 0.732174 | 1128.174263 | 0.083039 | 0.422871 | 0.113386 | 280085.302498 |
| E6 | 25 | 0.651452 | 0.807454 | 0.830470 | 842.498516 | 0.084630 | 0.393327 | 0.128609 | 274280.737855 |
| E6 | 50 | 0.774800 | 0.880260 | 0.872708 | 540.964250 | 0.082728 | 0.373635 | 0.135150 | 276668.799789 |
| E7 | 0 | 0.299591 | 0.626977 | 0.321184 | 16244.635359 | -0.059681 | -0.315492 | 0.083526 | 3628797.550125 |
| E7 | 10 | 0.410499 | 0.661305 | 0.497518 | 12659.001804 | -0.111450 | -0.366005 | 0.129383 | 3423706.303918 |
| E7 | 25 | 0.526696 | 0.725918 | 0.742071 | 9888.756605 | -0.272902 | -0.610578 | 0.192981 | 3517494.133590 |
| E7 | 50 | 0.649328 | 0.806177 | 0.781830 | 7284.548451 | -0.276800 | -0.579036 | 0.203321 | 3555184.211362 |
| E8 | 0 | 0.627264 | 0.796719 | 0.883309 | 5251.541107 | -0.297067 | -0.010573 | 0.534568 | 123585.064574 |
| E8 | 10 | 0.732103 | 0.855983 | 0.880550 | 3773.357442 | -0.344883 | -0.057143 | 0.532898 | 124955.228893 |
| E8 | 25 | 0.830843 | 0.911549 | 0.920401 | 2382.540157 | -0.279350 | 0.027751 | 0.557015 | 122338.276375 |
| E8 | 50 | 0.903305 | 0.950475 | 0.940581 | 1362.410724 | -0.286082 | 0.033324 | 0.569228 | 122822.793093 |
| E9 | 0 | 0.372842 | 0.614191 | 0.680436 | 16535.564228 | 0.163768 | 0.448278 | 0.255443 | 563964.630014 |
| E9 | 10 | 0.451075 | 0.672536 | 0.637477 | 14181.471043 | 0.137464 | 0.406860 | 0.239316 | 550247.899889 |
| E9 | 25 | 0.595353 | 0.772142 | 0.742977 | 10448.947054 | 0.127595 | 0.368190 | 0.278922 | 543434.657905 |
| E9 | 50 | 0.735931 | 0.858600 | 0.823060 | 6814.194670 | 0.143826 | 0.387232 | 0.308986 | 543240.573729 |
| E10 | 0 | 0.180987 | 0.443747 | 0.317555 | 52333.458470 | -0.048467 | -0.489183 | 0.047257 | 10333605.476568 |
| E10 | 10 | 0.312878 | 0.568609 | 0.466440 | 39881.918604 | -0.084848 | -0.576477 | 0.069413 | 9941786.352897 |
| E10 | 25 | 0.497452 | 0.717967 | 0.583713 | 29191.766143 | -0.084542 | -0.443201 | 0.086865 | 10121571.885368 |
| E10 | 50 | 0.703423 | 0.843041 | 0.757634 | 17104.384569 | -0.081842 | -0.306574 | 0.112746 | 10001468.314433 |

## Macro Actor and Micro stability

| Episode | Macro KL | Macro clip frac | Macro entropy | Advantage mean / std | Micro EV | Micro KL | Micro clip frac | Micro entropy | Micro value RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 0.000049 | 0.000000 | 4.721355 | 4.492 / 149.000 | 0.855633 | 0.000567 | 0.000000 | 4.243611 | 53.583107 |
| E2 | 0.000020 | 0.000000 | 4.726707 | -31.554 / 158.777 | 0.905403 | 0.001432 | 0.002500 | 4.225111 | 48.994080 |
| E3 | 0.000023 | 0.000000 | 4.715298 | -101.770 / 199.646 | 0.734259 | 0.004042 | 0.056667 | 4.251495 | 105.875740 |
| E4 | 0.000011 | 0.000000 | 4.717597 | 18.016 / 102.568 | 0.902758 | 0.000604 | 0.000000 | 4.191922 | 44.549519 |
| E5 | 0.000009 | 0.000000 | 4.724061 | 23.752 / 102.568 | 0.920145 | 0.003284 | 0.047500 | 4.104445 | 24.528156 |
| E6 | 0.000012 | 0.000000 | 4.683344 | 69.219 / 52.360 | 0.932452 | 0.004566 | 0.053333 | 4.267605 | 15.672721 |
| E7 | 0.000011 | 0.000000 | 4.706271 | -146.554 / 197.963 | 0.862378 | 0.000164 | 0.000000 | 4.343196 | 78.013702 |
| E8 | 0.000017 | 0.000000 | 4.715958 | -11.121 / 126.142 | 0.864552 | 0.003934 | 0.039167 | 4.180675 | 49.492363 |
| E9 | 0.000026 | 0.000000 | 4.699792 | -61.254 / 227.857 | 0.704483 | 0.003353 | 0.020833 | 4.009983 | 111.972412 |
| E10 | 0.000026 | 0.000000 | 4.694562 | -290.699 / 486.056 | 0.860323 | 0.005895 | 0.070000 | 4.071353 | 164.676346 |

No Macro or Micro early-stop occurred. Macro KL ranged from 0.000009 to 0.000049; Macro clip fraction was 0 throughout. Micro EV ranged from 0.704483 to 0.932452, Micro KL stayed below 0.006, and Micro clip fraction stayed below 0.07.

## Cache, duration, and UAV action distributions

| Episode | KEEP / UPDATE | Requested duration histogram | Actual duration histogram (trace) | UAV0 STAY / MOVE | UAV1 STAY / MOVE |
|---|---:|---|---|---:|---:|
| E1 | 52 / 56 | `{"1":40,"3":40,"5":28}` | `{"1":40,"3":40,"5":28}` | 143 / 157 | 150 / 150 |
| E2 | 55 / 51 | `{"1":38,"3":39,"5":29}` | `{"1":38,"3":39,"5":29}` | 153 / 147 | 156 / 144 |
| E3 | 54 / 48 | `{"1":37,"3":30,"5":35}` | `{"1":38,"3":29,"5":35}` | 139 / 161 | 128 / 172 |
| E4 | 60 / 45 | `{"1":40,"3":32,"5":33}` | `{"1":40,"2":1,"3":31,"5":33}` | 155 / 145 | 148 / 152 |
| E5 | 53 / 49 | `{"1":34,"3":36,"5":32}` | `{"1":35,"3":35,"5":32}` | 170 / 130 | 145 / 155 |
| E6 | 53 / 45 | `{"1":32,"3":31,"5":35}` | `{"1":32,"3":31,"5":35}` | 154 / 146 | 152 / 148 |
| E7 | 42 / 49 | `{"1":23,"3":31,"5":37}` | `{"1":23,"2":1,"3":30,"5":37}` | 159 / 141 | 146 / 154 |
| E8 | 41 / 55 | `{"1":32,"3":24,"5":40}` | `{"1":33,"3":24,"5":39}` | 143 / 157 | 139 / 161 |
| E9 | 48 / 57 | `{"1":38,"3":36,"5":31}` | `{"1":38,"2":1,"3":35,"5":31}` | 163 / 137 | 168 / 132 |
| E10 | 47 / 57 | `{"1":40,"3":30,"5":34}` | `{"1":40,"3":30,"5":34}` | 175 / 125 | 153 / 147 |

- Aggregate Cache counts: KEEP 505, UPDATE 512.
- Requested-duration histogram from persisted trace: `{"1":354,"3":329,"5":334}`.
- Actual-duration histogram from persisted trace: `{"1":357,"2":3,"3":324,"5":333}`.
- Aggregate UAV counts: UAV0 STAY/MOVE 1554/1446; UAV1 STAY/MOVE 1485/1515.
- Tail-truncated durations recorded by trainer: `{"3->1":2,"3->2":3,"5->1":1}`.

## E1–E5 versus E6–E10 medians

| Metric | E1–E5 median | E6–E10 median | Last5 − First5 |
|---|---:|---:|---:|
| PRE_TARGET_EV | 0.083406 | 0.372842 | 0.289436 |
| PRE_TARGET_CORRELATION | 0.383674 | 0.626977 | 0.243303 |
| PRE_TARGET_STD_RATIO | 0.201471 | 0.680436 | 0.478964 |
| POST_RETURN_EV | 0.013936 | -0.081842 | -0.095778 |
| POST_RETURN_CORRELATION | 0.118763 | 0.033324 | -0.085438 |
| POST_RETURN_STD_RATIO | 0.203056 | 0.203321 | 0.000265 |

## Short10 gate

| Criterion | Result | Evidence |
|---|---|---|
| A. E2–E10 pre-target std ratio < 3 | PASS | max = 0.883309 |
| B. Median E2–E10 pre-target EV > 0 | PASS | median = 0.372842 |
| C. Median E2–E10 pre-target correlation > 0 | PASS | median = 0.626977 |
| D. Post-update target EV positive in majority | PASS | 10/10 episodes |
| E. E2–E10 post-return correlation vs fresh5 baseline | PASS threshold; weak trend | median = 0.033324 vs baseline 0.019503; E6–E10 median = 0.033324, E1–E5 = 0.118763 |
| F. No 3 consecutive post-return correlations < -0.5 | PASS | longest run = 1 |
| G. Macro Actor stable | PASS | no early-stop; finite low KL; zero clip fraction; entropy stable |
| H. Micro stable | PASS | positive EV throughout; finite KL/clip/entropy/RMSE; no early-stop |
| I. No Cache/duration/UAV action collapse | PASS | both Cache actions, all requested durations, and both UAV actions represented |
| J. Exact-Q environment executions = 0 | PASS | 0 |

Target behavior and Actor/Micro/action checks pass. The E2–E10 return-correlation median is slightly above the fresh5 reference, but late-window correlation remains close to zero and the E6–E10 median is below E1–E5. This is PARTIAL under the requested no-improvement-trend branch; do not continue searching N-step targets.

## Final decision

- FORMAL_N2_SHORT10_GATE = PARTIAL
- SELECTED_FORMAL_TARGET = N2
- TARGET_SEARCH_CLOSED = YES
- NEXT_DECISION = 不再继续搜索 N-step target；后续单独研究长期 value calibration 问题。

## Artifacts

- JSON report: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\formal_macro_n2_target_short10.json`
- Markdown report: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\formal_macro_n2_target_short10.md`
- Formal run, checkpoints, and complete transition trace retained at `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\formal_macro_n2_target_ckptv31_archv9_short10`.
