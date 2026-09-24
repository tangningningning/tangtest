# Formal episode generation permission fix

## Task

`FIX_FORMAL_EPISODE_GENERATION_PERMISSION`

Repository: `C:/Users/tangning/Desktop/code/adaptive_cache_mec`

Formal Python: `C:/software/Anaconda_envs/envs/adaptive_cache_mec/python.exe`

## Root cause

The formal lazy episode generator called SUMO `randomTrips.py` with its route
output in `tempfile.TemporaryDirectory()`, then finalized the route with
`generated_route.replace(route_file)`. On Windows, the cross-directory move
preserved the temporary file's ACL instead of creating the canonical file with
the inheritance expected from
`data/episode_pools/proposed_joint_training`. This explains the observed
`inheritance disabled` state and the formal user's missing read permission on
`sumo_demand.rou.xml`.

## Implemented fix

`src/scenario/sumo_scenario.py` now treats the SUMO temporary route as a byte
source only. The destination is opened and created in the canonical episode
directory, the bytes are copied and flushed with `fsync`, and the destination
SHA-256 is checked against the source before generation continues. No
Windows-only ACL API was added, so the code remains usable on Linux/server
systems and the route bytes are unchanged.

`src/ppo_simple/trainer.py` now reads `episode_manifest.json` and every
manifest-declared artifact with the formal Python process before output
validation and provenance validation. Permission/stat/read-only-filesystem
failures are surfaced as `EPISODE_FILESYSTEM_PERMISSION_FAILURE`; provenance
failures retain their existing fail-closed `REALIZATION_PROVENANCE_MISMATCH`
classification. Existing episodes are never overwritten by this gate.

## Verification boundary

The canonical pool ACL normalization completed before this source fix. The
pool audit found 131 objects, zero protected objects, and zero deny rules;
`episode_20001` through `episode_20010` had readable SUMO demand files and
manifest hashes matched under the formal Python read checks. The source was
syntax-checked, imported, and the readability gate passed against the existing
`episode_20003` without running PPO.

The requested generation-only check for seed `20003` was not run because
`data/episode_pools/proposed_joint_training/episode_20003` already exists.
It was not deleted or overwritten. The user must remove that old realization
before a separate generation-only gate can establish new-file ACL inheritance,
post-generation readability, hash match, and provenance for a newly generated
`episode_20003`.

Formal run state remains:

```text
FORMAL_COMPLETED_EPOCHS = 2
NEXT_FORMAL_EPOCH = 3
NEXT_FORMAL_SEED = 20003
FORMAL_TRAINING_RESTARTED = NO
FORMAL_RESUME_SAFE = NOT_ESTABLISHED
```

No PPO epoch, resume, benchmark, or environment generation was run by this
task. Existing realization bytes and manifests were not modified. No commit or
push was performed.

## Source freeze artifacts

- Patch: `reports/formal_training_filesystem_permission_fix.patch`
- Manifest: `reports/formal_training_filesystem_permission_fix_manifest.json`
- `ALGORITHM_SEMANTICS_CHANGED = NO`
- `ENVIRONMENT_CONTENT_SEMANTICS_CHANGED = NO`
- `FILESYSTEM_PERMISSION_HANDLING_CHANGED = YES`
