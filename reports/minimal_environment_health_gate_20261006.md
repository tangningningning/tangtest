# Minimal Environment Health Gate

- ENVIRONMENT_HEALTHY = YES
- EPISODES_COMPLETED = 4 total environment completions (3 fully instrumented gate episodes plus 1 preflight)
- EARLY_STOP_AFTER_3 = NO for total execution: 1 preflight plus 3 fully instrumented captures gave 4; the corrected batch itself stopped after 3
- Environment exception / timeout count = 0 / 0
- Diagnostic recorder exception count = 1; occurred after the preflight environment rollout returned normally and was fixed before the audited episodes

## Static implementation facts

No Gym-style reset()/step() API. Each collect_dual_timescale_episode initializes cache from context.initial_cache and UAV positions from context.initial_uav (reset-equivalent), then calls decision.slot_environment.execute_configured_slot once per aligned slot.

300 aligned slots; origins 0..299; terminal_origin=299

Micro trainer passes bootstrap_value=0 to compute_gae; TD bootstrap and recursive trace are masked by not done. Macro stores old_next_value=0 at done and compute_macro_smdp_gae masks gamma**actual_duration bootstrap and (gamma*lambda)**actual_duration trace by not done. True terminal bootstrap=0. Current fixed 300-slot protocol defines origin 299 as terminal, not truncation.

actual_duration_for_frame(requested,start,total)=min(requested,total-start); tail mismatch is explained when actual equals remaining slots and requested exceeds remaining.

Per task utility=omega_delay*(1-delay/T_ref)+omega_energy*(1-energy/E_ref); Q=sum(task utility); G=Q-cache_cost-movement_cost.

No independent terminated/truncated fields exist. The configured finite-horizon boundary is treated as true terminal; a hypothetical separate truncation would currently also be masked as done.

## Observation

- Macro env/network shape: [400] / [400]; match=True
- Micro env/network shape: [400] / [400]; match=True
- dtype(s)=['float32']; NaN=0; Inf=0; abs max=1.0
- Checked 3166 vectors (1266400 scalar values), including the preflight observation checks.

## Reward

- Slot mean/std/min/max over all 1200 tested slots: -3.197843252119798 / 35.46620355856449 / -364.80090328883915 / 13.888292114824436
- Episode return mean/std/min/max over 4 environment completions: -959.3529756359392 / 1610.916614110401 / -2542.8014940107546 / 1043.006338191848
- Finite=True; all zero=False; reconstruction pass=True; max error=1.1368683772161603e-13
- Reconstruction scope: 900 slots from the three fully instrumented episodes; this exceeds the required 10 witness slots. The earlier completed preflight did not retain per-slot delay/energy components.

| Ep | Slot | Tasks | Delay s | Energy J | Delay term | Energy term | Cache cost | Move cost | Reward | Rebuilt | Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 12 | 186.96359 | 13.697879 | 3.5375165 | 3.0111221 | 0.27362621 | 0 | 6.2750124 | 6.2750124 | 8.9e-16 |
| 1 | 64 | 14 | 317.94992 | 30.270646 | 2.1715876 | 1.6467974 | 0 | 0 | 3.818385 | 3.818385 | 8.9e-16 |
| 1 | 128 | 18 | 220.39421 | 10.195642 | 6.4826345 | 5.8684976 | 0 | 0.00187 | 12.349262 | 12.349262 | 1.8e-15 |
| 1 | 192 | 12 | 79.847476 | 2.4473348 | 5.6358448 | 4.4803897 | 0 | 0.00473 | 10.111505 | 10.111505 | 0 |
| 1 | 256 | 10 | 219.98443 | 21.809921 | 1.6906617 | 1.1517279 | 0 | 0.00374 | 2.8386496 | 2.8386496 | 1.3e-15 |
| 2 | 21 | 13 | 270.26942 | 15.935134 | 2.5056146 | 3.1189471 | 0 | 0.00286 | 5.6217017 | 5.6217017 | 8.9e-16 |
| 2 | 85 | 13 | 414.01646 | 40.142223 | -0.31028744 | -0.042383726 | 0 | 0.00187 | -0.35454117 | -0.35454117 | 1.9e-15 |
| 2 | 149 | 16 | 180.49767 | 19.483822 | 6.0641792 | 3.8555054 | 0 | 0.00187 | 9.9178147 | 9.9178147 | 0 |
| 2 | 213 | 14 | 440.19773 | 42.949106 | -0.22316004 | -0.0089493188 | 0 | 0.00473 | -0.23683935 | -0.23683935 | 4.2e-16 |
| 2 | 277 | 16 | 254.64303 | 16.634228 | 4.6117245 | 4.2276488 | 0 | 0.00187 | 8.8375033 | 8.8375033 | 0 |
| 3 | 42 | 16 | 947.13084 | 146.2999 | -8.9536185 | -12.706073 | 0 | 0.00187 | -21.661561 | -21.661561 | 7.1e-15 |
| 3 | 106 | 10 | 57.642227 | 3.1042601 | 4.8708298 | 3.5945984 | 0 | 0.00473 | 8.4606982 | 8.4606982 | 0 |
| 3 | 170 | 10 | 211.8036 | 32.897369 | 1.8509184 | -0.2962401 | 0 | 0 | 1.5546783 | 1.5546783 | 6.7e-16 |
| 3 | 234 | 15 | 702.87184 | 73.63639 | -4.7687587 | -3.6165628 | 0 | 0.00473 | -8.3900514 | -8.3900514 | 1.8e-15 |
| 3 | 299 | 13 | 223.96268 | 8.3754544 | 3.4127306 | 4.1062054 | 0 | 0 | 7.518936 | 7.518936 | 0 |

## State/action and duration

- KEEP exact / cost zero: True / True; no-effect UPDATE count=0
- Requested/actual mismatch count=3; all tail-explained=True
- Mismatch scope: Three fully instrumented episodes; all three recorded mismatches were fully explained by the episode-horizon tail.

### Intermediate transition

{
  "episode": 1,
  "level": "Macro frame plus contained Micro slot",
  "macro_frame_start_slot": 0,
  "micro_slot": 0,
  "terminated": false,
  "truncated": false,
  "done": false,
  "remaining_slots_before": 300,
  "remaining_slots_after_macro": 299,
  "requested_duration": 1,
  "actual_duration": 1,
  "next_state_valid": true,
  "bootstrap_mask": 1,
  "macro_old_next_value": 0.0888698548078537,
  "micro_next_observation_shape": [
    400
  ],
  "micro_next_observation_is_zero_placeholder": false,
  "terminated_truncated_fields_exist_in_dataclasses": false
}

### Final transition

{
  "episode": 1,
  "level": "Macro terminal frame and last Micro slot",
  "macro_frame_start_slot": 299,
  "micro_slot": 299,
  "terminated": true,
  "truncated": false,
  "done": true,
  "remaining_slots_before": 1,
  "remaining_slots_after": 0,
  "requested_duration": 5,
  "actual_duration": 1,
  "micro_requested_duration": 5,
  "micro_actual_duration": 1,
  "next_state_valid": false,
  "bootstrap_mask": 0,
  "macro_old_next_value": 0.0,
  "micro_next_observation_shape": [
    400
  ],
  "micro_next_observation_is_zero_placeholder": true,
  "macro_next_observation_is_zero_placeholder": true,
  "terminated_truncated_fields_exist_in_dataclasses": false
}

| Ep | Frame start | Requested | Actual | Remaining before | Explained by horizon |
|---:|---:|---:|---:|---:|---|
| 1 | 299 | 5 | 1 | 1 | True |
| 2 | 299 | 3 | 1 | 1 | True |
| 3 | 298 | 3 | 2 | 2 | True |

## Random action coverage

- Cache KEEP/UPDATE rates: 0.5050847457627119 / 0.49491525423728816
- Duration 1/3/5 rates: 0.30847457627118646 / 0.34576271186440677 / 0.34576271186440677
- UAV STAY/MOVE rates: 0.48444444444444446 / 0.5155555555555555
- Rate scope: The three fully instrumented episodes (900 slots, 295 Macro decisions, 1800 UAV-head actions); the earlier preflight action counts were not retained.

Actor/Critic parameters were frozen; no trainer or optimizer was constructed and optimizer.step count was zero.

- First environment failure: NONE
- Next step: STOP; do not enter data layer or Layer 2.
