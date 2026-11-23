#!/usr/bin/env bash
# CONTEXT=auto must not replay a shrunk fit from context-fit.tsv across reboots or from a legacy bootless row (issue #56).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

extract_fns() {
  awk '/^(fit_boot_id|remembered_context|remember_context)\(\) \{/,/^\}/' "$1"
}

check_script() {
  local script=$1
  local fns="$tmp/fns.sh"
  extract_fns "$script" > "$fns"
  [[ -s "$fns" ]] || { echo "$script: missing context-fit functions" >&2; exit 1; }

  # shellcheck disable=SC1090
  source "$fns"

  STATE_DIR="$tmp/state"
  FIT_FILE="$STATE_DIR/context-fit.tsv"
  CONTEXT_MAX=1048576
  rm -rf "$STATE_DIR"

  # Same boot: a shrunk fit is replayed.
  FIT_BOOT_ID="boot-aaa" remember_context "key1" 498688
  got=$(FIT_BOOT_ID="boot-aaa" remembered_context "key1")
  [[ "$got" == "498688" ]] || { echo "$script: expected same-boot shrunk fit 498688, got '$got'" >&2; exit 1; }

  # Different boot: the shrunk fit is stale and must not be replayed.
  got=$(FIT_BOOT_ID="boot-bbb" remembered_context "key1")
  [[ -z "$got" ]] || { echo "$script: replayed shrunk fit '$got' across reboot" >&2; exit 1; }

  # Legacy 3-column shrunk entry (no boot_id): treated as stale.
  printf 'key2\t498688\t2026-11-12T00:00:00Z\n' > "$FIT_FILE"
  got=$(FIT_BOOT_ID="boot-bbb" remembered_context "key2")
  [[ -z "$got" ]] || { echo "$script: replayed legacy shrunk fit '$got'" >&2; exit 1; }

  # Full CONTEXT_MAX entry: replayed across boots.
  FIT_BOOT_ID="boot-aaa" remember_context "key3" 1048576
  got=$(FIT_BOOT_ID="boot-bbb" remembered_context "key3")
  [[ "$got" == "1048576" ]] || { echo "$script: expected full-window fit 1048576, got '$got'" >&2; exit 1; }

  echo "context-fit ($script): pass"
}

check_script start.sh
check_script start-tp4-switchless.sh
