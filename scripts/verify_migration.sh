#!/usr/bin/env bash
# Checks the Agent-separation migration on a machine that has a GPU and the
# submodules -- the things a laptop cannot answer.
#
#   ./scripts/verify_migration.sh
#
# Prints PASS/FAIL per check and a summary. Paste the summary back into the
# migration discussion. Nothing here writes to the repo except uv.lock
# (step 2), which is expected to change and should be committed.
#
# Prerequisites: submodules checked out, CUDA available, and for step 5 the
# perception servers running at whatever *_SERVICE_URL points to (.envrc on
# this deployment). Steps 1-4 need no servers.
#
# Start the servers with:
#     uv run --no-sync --active capx/serving/launch_servers.py --profile default
#
# Run this on a clean tree. Step 2 regenerates uv.lock, so discard any local
# edit to it first:
#     git checkout uv.lock && git pull
#
# Remove this script once the migration lands.

set -uo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
PASS=0; FAIL=0; SKIP=0
RESULTS=()

run_check() {
  local name="$1"; shift
  printf "\n${BOLD}== %s${NC}\n" "$name"
  if "$@"; then
    printf "${GREEN}PASS${NC} %s\n" "$name"
    RESULTS+=("PASS  $name"); PASS=$((PASS + 1))
  else
    printf "${RED}FAIL${NC} %s\n" "$name"
    RESULTS+=("FAIL  $name"); FAIL=$((FAIL + 1))
  fi
}

skip_check() {
  printf "\n${YELLOW}SKIP${NC} %s -- %s\n" "$1" "$2"
  RESULTS+=("SKIP  $1 ($2)"); SKIP=$((SKIP + 1))
}

# ---------------------------------------------------------------------------
# 1. Submodules present
#
# Everything else assumes them. A missing submodule makes step 2 fail in a way
# that looks like a pyproject bug but is not.
# ---------------------------------------------------------------------------
check_submodules() {
  local missing=0
  for d in capx/third_party/sam3 capx/third_party/robosuite capx/third_party/curobo; do
    if [ ! -f "$d/pyproject.toml" ] && [ ! -f "$d/setup.py" ]; then
      echo "  missing: $d"
      missing=1
    fi
  done
  [ "$missing" -eq 0 ] || echo "  run: git submodule update --init --recursive"
  return "$missing"
}

# ---------------------------------------------------------------------------
# 2. Root lock resolves
#
# The base demotion moved 19 packages into an `env` extra and scoped the sam3 /
# nvidia-curobo / contact_graspnet_pytorch / verl path sources to their extras.
# Before that, `uv lock` built curobo's metadata on every resolution and died
# on CUDA_HOME. This regenerates uv.lock -- commit the result.
# ---------------------------------------------------------------------------
check_root_lock() {
  uv lock 2>&1 | tail -20
  return "${PIPESTATUS[0]}"
}

# ---------------------------------------------------------------------------
# 3. Each extra still resolves
#
# The `extra` markers narrow what uv evaluates, so these should be unaffected,
# but narrowing is exactly the kind of change that silently drops a package.
# --dry-run resolves and reports without installing.
# ---------------------------------------------------------------------------
check_extras() {
  local bad=0
  for extra in robosuite libero verl; do
    printf -- "-- extra: %s\n" "$extra"
    if ! uv sync --extra "$extra" --dry-run 2>&1 | tail -5; then
      echo "  FAILED: $extra"
      bad=1
    fi
  done
  return "$bad"
}

# ---------------------------------------------------------------------------
# 4. launch.py still starts
#
# --web-ui and the nodeenv frontend build are gone; this catches an import or
# arg-parsing break without running a trial.
# ---------------------------------------------------------------------------
check_launch_help() {
  uv run --no-sync --active capx/envs/launch.py --help >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# 5. expose_env=False does not break oracle code -- THE IMPORTANT ONE
#
# `env` and `APIS` are no longer bound into the generated code's namespace by
# default. Oracle code was checked and uses neither, going through the bound
# API helpers instead, so reward should be unchanged. But the default changed,
# so this is a behaviour change until something runs it.
#
# Expect reward 1.0. Needs the perception servers.
# ---------------------------------------------------------------------------
check_oracle() {
  local out
  out=$(uv run --no-sync --active tests/test_environments.py \
          --env_name franka_pick_place_code_env 2>&1)
  echo "$out" | tail -25
  echo "$out" | grep -qiE "NameError.*(env|APIS)" && {
    echo "  >> oracle code reached for env/APIS -- expose_env broke it"
    return 1
  }
  echo "$out" | grep -qE "[Rr]eward.*1\.0|reward: 1" && return 0
  echo "  >> reward was not 1.0; compare against main before this branch"
  return 1
}

# ---------------------------------------------------------------------------

printf "${BOLD}CaP-X migration verification${NC}\n"
printf "branch: %s\n" "$(git rev-parse --abbrev-ref HEAD)"
printf "commit: %s\n" "$(git rev-parse --short HEAD)"

run_check "1. submodules present" check_submodules
run_check "2. root uv lock resolves" check_root_lock
run_check "3. robosuite/libero/verl extras resolve" check_extras
run_check "4. launch.py --help works" check_launch_help

# The perception servers are wherever *_SERVICE_URL says -- on this
# deployment that is 192.168.0.200, not localhost. resolve_endpoint() in
# capx/serving/launch_servers.py reads the same vars and falls back to
# 127.0.0.1, so mirror that rather than assuming either.
SAM3_URL="${SAM3_SERVICE_URL:-http://127.0.0.1:8114}"
PYROKI_URL="${PYROKI_SERVICE_URL:-http://127.0.0.1:8116}"

if curl -s --max-time 3 "$SAM3_URL/" >/dev/null 2>&1 \
   || curl -s --max-time 3 "$PYROKI_URL/" >/dev/null 2>&1; then
  printf "  perception API reachable (%s)\n" "$SAM3_URL"
  run_check "5. oracle reward unchanged with expose_env=False" check_oracle
else
  skip_check "5. oracle reward with expose_env=False" \
    "no response from $SAM3_URL or $PYROKI_URL -- start them with: uv run --no-sync --active capx/serving/launch_servers.py --profile default"
fi

printf "\n${BOLD}== summary${NC}\n"
for r in "${RESULTS[@]}"; do printf "  %s\n" "$r"; done
printf "\n  %d passed, %d failed, %d skipped\n" "$PASS" "$FAIL" "$SKIP"

if [ -n "$(git status --porcelain uv.lock)" ]; then
  printf "\n  ${YELLOW}uv.lock changed -- commit it${NC}\n"
fi

[ "$FAIL" -eq 0 ]
