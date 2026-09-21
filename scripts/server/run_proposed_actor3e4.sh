#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/9900299/tn/adaptive_cache_mec_server
PYTHON=python

RUN_NAME=proposed_actor3e4_300_epoch
RUN_DIR=runs/proposed_actor3e4_300_epoch
BASE_CONFIG=configs/proposed/system_config.toml
CONFIG_PATH="$RUN_DIR/run_config.toml"
EPISODE_POOL=environment_data/proposed_joint_training
EPISODE_SEED_START=20001
EPISODE_SEED_END=20300
EPOCHS=300
TARGET_ACTOR_LR=0.0003
TARGET_CRITIC_LR=0.001

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -d "$PROJECT_ROOT" ]] || die "PROJECT_ROOT does not exist: $PROJECT_ROOT"
command -v "$PYTHON" >/dev/null 2>&1 || die "Python executable is not available: $PYTHON"

cd "$PROJECT_ROOT"

BASE_CONFIG_ABS="$PROJECT_ROOT/$BASE_CONFIG"
EPISODE_POOL_ABS="$PROJECT_ROOT/$EPISODE_POOL"
[[ -f "$BASE_CONFIG_ABS" ]] || die "base config does not exist: $BASE_CONFIG_ABS"
[[ -d "$EPISODE_POOL_ABS" ]] || die "episode pool does not exist: $EPISODE_POOL_ABS"
[[ -d "$EPISODE_POOL_ABS/episode_20001" ]] || die "episode_20001 does not exist"
[[ -d "$EPISODE_POOL_ABS/episode_20300" ]] || die "episode_20300 does not exist"

if [[ -e "$RUN_DIR" || -L "$RUN_DIR" ]]; then
    die "run directory already exists; refusing to overwrite: $RUN_DIR"
fi

mkdir -p -- "$(dirname "$RUN_DIR")"
CONFIG_TMP="$PROJECT_ROOT/${RUN_DIR}.run_config.tmp.$$"
if [[ -e "$CONFIG_TMP" || -L "$CONFIG_TMP" ]]; then
    die "temporary config path already exists; refusing to overwrite: $CONFIG_TMP"
fi
cleanup() {
    rm -f -- "$CONFIG_TMP"
}
trap cleanup EXIT

cp -- "$BASE_CONFIG_ABS" "$CONFIG_TMP"

"$PYTHON" - "$BASE_CONFIG_ABS" "$CONFIG_TMP" "$TARGET_ACTOR_LR" "$TARGET_CRITIC_LR" <<'PY'
import copy
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib  # type: ignore[no-redef]


def fail(message: str) -> None:
    raise SystemExit(f"CONFIG_VALIDATION_ERROR: {message}")


def load(path: Path) -> dict:
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    if not isinstance(value, dict):
        fail(f"TOML root is not a table: {path}")
    return value


source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
target_actor = float(sys.argv[3])
target_critic = float(sys.argv[4])
base_config = load(source_path)
base_ppo = base_config.get("algorithm", {}).get("ppo")
if not isinstance(base_ppo, dict):
    fail("missing [algorithm.ppo] table")
if "actor_learning_rate" not in base_ppo:
    fail("missing algorithm.ppo.actor_learning_rate")
if "critic_learning_rate" not in base_ppo:
    fail("missing algorithm.ppo.critic_learning_rate")
if float(base_ppo["actor_learning_rate"]) != 0.0003:
    fail("shared base actor learning rate is not 0.0003")
if float(base_ppo["critic_learning_rate"]) != target_critic:
    fail("shared base critic learning rate is not 0.001")

text = source_path.read_text(encoding="utf-8")
table_pattern = re.compile(
    r"(?ms)(^\[algorithm\.ppo\][ \t]*\r?\n)(.*?)(?=^\[[^\]\r\n]+\][ \t]*\r?\n|\Z)"
)
table_match = table_pattern.search(text)
if table_match is None:
    fail("could not locate [algorithm.ppo] table text")
body = table_match.group(2)
actor_pattern = re.compile(
    r"(?m)^(?P<prefix>[ \t]*actor_learning_rate[ \t]*=[ \t]*)"
    r"(?P<value>[^#\r\n]*?)(?P<suffix>[ \t]*(?:#[^\r\n]*)?)"
    r"(?P<newline>\r?)$"
)
actor_matches = list(actor_pattern.finditer(body))
if len(actor_matches) != 1:
    fail(f"algorithm.ppo.actor_learning_rate is ambiguous: {len(actor_matches)} matches")

target_literal = f"{target_actor:.4f}"
new_body = actor_pattern.sub(
    lambda match: (
        f"{match.group('prefix')}{target_literal}{match.group('suffix')}"
        f"{match.group('newline')}"
    ),
    body,
    count=1,
)
new_text = text[: table_match.start(2)] + new_body + text[table_match.end(2) :]
with target_path.open("w", encoding="utf-8", newline="") as stream:
    stream.write(new_text)

new_config = load(target_path)
new_ppo = new_config.get("algorithm", {}).get("ppo")
if not isinstance(new_ppo, dict):
    fail("generated config is missing [algorithm.ppo]")
if float(new_ppo.get("actor_learning_rate")) != target_actor:
    fail("generated actor learning rate does not match target")
if float(new_ppo.get("critic_learning_rate")) != target_critic:
    fail("generated critic learning rate does not match target")

base_without_actor = copy.deepcopy(base_config)
new_without_actor = copy.deepcopy(new_config)
del base_without_actor["algorithm"]["ppo"]["actor_learning_rate"]
del new_without_actor["algorithm"]["ppo"]["actor_learning_rate"]
if base_without_actor != new_without_actor:
    fail("generated config differs from shared config outside actor learning rate")
PY

mkdir "$RUN_DIR"
mv -- "$CONFIG_TMP" "$CONFIG_PATH"
trap - EXIT

: > "$RUN_DIR/server_stdout.log"
: > "$RUN_DIR/server_stderr.log"
{
    echo "RUN_NAME=$RUN_NAME"
    echo "CONFIG_PATH=$CONFIG_PATH"
    echo "ACTOR_LR=$TARGET_ACTOR_LR"
    echo "CRITIC_LR=$TARGET_CRITIC_LR"
    echo "EPOCHS=$EPOCHS"
    echo "EPISODE_POOL=$EPISODE_POOL"
    echo "EPISODE_SEED_START=$EPISODE_SEED_START"
    echo "EPISODE_SEED_END=$EPISODE_SEED_END"
    echo "RUN_DIR=$RUN_DIR"
} | tee -a "$RUN_DIR/server_stdout.log"

PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m ppo_simple.trainer \
    --config "$CONFIG_PATH" \
    --epochs "$EPOCHS" \
    --run-dir "$RUN_DIR" \
    --episode-pool "$EPISODE_POOL" \
    --episode-seed-start "$EPISODE_SEED_START" \
    --exact-q-parallel-workers 4 \
    2> >(tee -a "$RUN_DIR/server_stderr.log" >&2) \
    | tee -a "$RUN_DIR/server_stdout.log"
