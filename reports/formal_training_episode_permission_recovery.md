# FORMAL_TRAINING_EPISODE_PERMISSION_RECOVERY_ONLY

## Final status

    TASK = FORMAL_TRAINING_EPISODE_PERMISSION_RECOVERY_ONLY
    FORMAL_TRAINING_RESTARTED = NO
    RESUME_EXECUTED = NO
    REALIZATION_REGENERATED = NO
    FILES_MODIFIED_CONTENT = NO
    MANIFEST_MODIFIED = NO
    COMMITTED = NO
    PUSHED = NO

The formal run had already completed epoch 1 before this task. This task only
repaired the Windows ACL on one existing artifact and performed read-only
verification.

## Root cause

The failing object was:

    data/episode_pools/proposed_joint_training/episode_20002/sumo_demand.rou.xml

It was a normal file, not a directory, junction, symlink, or reparse point:

    EXISTS = YES
    IS_FILE = YES
    IS_DIR = NO
    REPARSE_POINT = NO
    SIZE_BYTES = 25965

Before repair, its ACL was protected (D:P) and contained only SYSTEM,
Administrators, and OWNER RIGHTS. It did not inherit the normal ACL from the
episode directory, so the formal training identity LAPTOP-TDK6NJKI\tangning
had no effective read permission. The user-reported Path.open("rb")
PermissionError is therefore a Windows ACL/inheritance issue, not a trainer,
generator, or provenance implementation bug.

The Codex sandbox owner could read the file before repair through OWNER RIGHTS;
that identity-specific result does not contradict the formal user's failure.

## Repair

Before changing ACL metadata, the artifact hash already matched its manifest:

    EXPECTED_SHA256 = 3a580ac783332b15defaf298568fbefa473db468a9cea11b89df0959bfd8b883
    PRE_REPAIR_SHA256 = 3a580ac783332b15defaf298568fbefa473db468a9cea11b89df0959bfd8b883

The only repair executed was:

    icacls episode_20002\sumo_demand.rou.xml /inheritance:e

It succeeded without elevation. The file now inherits the canonical episode
directory ACL, including the tangning access rule. No ownership, content,
manifest, hash, or generator behavior was changed.

After repair:

    EPISODE_20002_READABLE = YES
    EPISODE_20002_POWERSHELL_READ = PASS
    EPISODE_20002_PYTHON_RB_READ = PASS
    EPISODE_20002_HASH_MATCH = YES
    POST_REPAIR_SHA256 = 3a580ac783332b15defaf298568fbefa473db468a9cea11b89df0959bfd8b883

## Pool verification

The exact environment interpreter was used for read-only Python checks:
C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe.

Every manifest-declared generated artifact in episodes 20001 through 20010 was
opened in binary mode and hashed. The repository's existing pool validator then
validated episode seed, generation fingerprint, source identity, artifact set,
and every artifact hash. All ten existing episodes were skipped as valid; no
episode was generated or overwritten.

    EPISODE_20001_READABLE = YES    EPISODE_20001_PROVENANCE = PASS
    EPISODE_20002_READABLE = YES    EPISODE_20002_PROVENANCE = PASS
    EPISODE_20003_READABLE = YES    EPISODE_20003_PROVENANCE = PASS
    EPISODE_20004_READABLE = YES    EPISODE_20004_PROVENANCE = PASS
    EPISODE_20005_READABLE = YES    EPISODE_20005_PROVENANCE = PASS
    EPISODE_20006_READABLE = YES    EPISODE_20006_PROVENANCE = PASS
    EPISODE_20007_READABLE = YES    EPISODE_20007_PROVENANCE = PASS
    EPISODE_20008_READABLE = YES    EPISODE_20008_PROVENANCE = PASS
    EPISODE_20009_READABLE = YES    EPISODE_20009_PROVENANCE = PASS
    EPISODE_20010_READABLE = YES    EPISODE_20010_PROVENANCE = PASS

    EPISODE_20001_20010_ALL_READABLE = YES
    EPISODE_20001_20010_ALL_PROVENANCE_PASS = YES

## Formal resume state

The run directory is runs/proposed_joint_formal_300_epoch. Read-only checks
found run_summary.json, run_manifest.json, latest_checkpoint.pt,
episode_metrics.jsonl, and update_metrics.jsonl. The summary records a
complete 300-slot epoch:

    completed_epochs = 1
    target_epochs = 300
    status = running
    episode_metrics = 1 complete row
    num_transitions = 300
    terminal_origin = 299
    nan_inf_detected = false

The checkpoint independently records:

    completed_epochs = 1
    completed_episodes = 1
    next_pool_index = 1
    episode_seed_start = 20001
    NEXT_FORMAL_EPOCH = 2
    NEXT_FORMAL_SEED = 20002

Therefore:

    FORMAL_COMPLETED_EPOCHS = 1
    NEXT_FORMAL_EPOCH = 2
    NEXT_FORMAL_SEED = 20002
    RESUME_STATE_VALID = YES
    FRESH_COMMAND_MUST_NOT_BE_USED = YES
    RESUME_REQUIRED = YES

## Final user command

Do not run the fresh command. The user should execute only this resume command:

    Set-Location 'C:\Users\tangning\Desktop\code\adaptive_cache_mec'
    & 'C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe' -m ppo_simple.trainer --config configs/proposed/system_config.toml --epochs 300 --run-dir runs/proposed_joint_formal_300_epoch --episode-pool data/episode_pools/proposed_joint_training --episode-seed-start 20001 --generate-missing-episodes --episode-generation-device cpu --resume

    ADMIN_PERMISSION_REQUIRED = NO
