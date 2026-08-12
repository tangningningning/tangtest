# Third-party notices

This project adapts the following upstream implementations for the PPO-core
rebuild. The temporary clones used for source inspection were not added to the
project tree.

## CleanRL

- Repository: [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)
- Commit: `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`
- Source used: [`cleanrl/ppo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_continuous_action.py)
- License: MIT; see [`LICENSE`](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/LICENSE)
- Use: directly adapted standard GAE recursion, rollout flatten/minibatch flow,
  advantage normalization, clipped surrogate, entropy/value loss, approximate
  KL, clip fraction, gradient clipping, and optimizer sequencing.

## HARL

- Repository: [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL)
- Commit: `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`
- Source used: [`harl/algorithms/actors/happo.py`](https://github.com/PKU-MARL/HARL/blob/b1af98b0dbab72a2eee9d160751cd09aedbb8ce2/harl/algorithms/actors/happo.py), [`harl/runners/on_policy_ha_runner.py`](https://github.com/PKU-MARL/HARL/blob/b1af98b0dbab72a2eee9d160751cd09aedbb8ce2/harl/runners/on_policy_ha_runner.py), and [`harl/common/buffers/on_policy_actor_buffer.py`](https://github.com/PKU-MARL/HARL/blob/b1af98b0dbab72a2eee9d160751cd09aedbb8ce2/harl/common/buffers/on_policy_actor_buffer.py)
- License: MIT as declared by HARL `setup.py`; the inspected repository root
  did not contain a separate `LICENSE` file.
- Use: directly adapted HAPPO clipped actor loss, the detached importance
  factor, one actor order sampled per update, sequential actor updates, and
  factor refresh after each completed actor update.

## PPOxFamily

- Repository: [opendilab/PPOxFamily](https://github.com/opendilab/PPOxFamily)
- Commit: `78f781115681ebb245b7c56675d64db7cd732323`
- Source used: [`chapter2_action/hybrid_tutorial.py`](https://github.com/opendilab/PPOxFamily/blob/78f781115681ebb245b7c56675d64db7cd732323/chapter2_action/hybrid_tutorial.py)
- License: Apache-2.0; see [`LICENSE`](https://github.com/opendilab/PPOxFamily/blob/78f781115681ebb245b7c56675d64db7cd732323/LICENSE)
- Use: adapted hybrid-action conditional masking: a discrete action choice
  selects whether the associated continuous branch contributes to the action
  probability/entropy calculation.

## Project adaptations

The project-specific adaptations are two full-observation Cache/UAV actors,
conditional Bernoulli-plus-Beta/Normal masks, the unchanged environment action
container, one shared centralized PopArt critic, and project metrics/checkpoint
metadata. The formal objective is HAPPO branch-joint clipping with sequential
actor updates; there is no cross-branch joint ratio and no per-dimension
factorized clipping. The rebuild provides the requested update semantics but
does not claim an automatic guarantee of stable learning or optimality.
