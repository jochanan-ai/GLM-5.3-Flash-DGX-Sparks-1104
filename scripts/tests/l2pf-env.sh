#!/usr/bin/env bash
# On TensorFold 1.3.0 an unset TF_GLM_L2PF means on for a GB10 (zig/src/families/glm/cuda_l2pf.zig), so the launchers
# must name the setting either way: L2PF=0 has to reach the ranks as TF_GLM_L2PF=0, or it cannot turn prefetch off.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
fails=0
check() { # check <launcher> <L2PF> <want TF_GLM_L2PF value>
  local script=$1 l2pf=$2 want=$3 got
  got=$(env -i PATH="$PATH" L2PF="$l2pf" bash -c '
    eval "$(awk "/^bits\\(\\) \\{/" "$1")"
    eval "$(awk "/^build_env_args\\(\\) \\{/,/^\\}/" "$1")"
    MANAGED_ENV=" TF_GLM_L2PF " EXTRA_EOS=none THINK_OFF_SPLIT=0 DENSE=q8 HEAD=bf16
    build_env_args
    for a in "${ENV_ARGS[@]}"; do [[ "$a" != TF_GLM_L2PF=* ]] || echo "${a#TF_GLM_L2PF=}"; done' _ "$script")
  if [[ "$got" == "$want" ]]; then echo "l2pf-env ($script): L2PF=$l2pf: pass"
  else echo "l2pf-env ($script): L2PF=$l2pf: FAIL (got '${got}', want '$want')" >&2; fails=1; fi
}
for s in start.sh start-tp4-switchless.sh; do
  check "$s" 0 0
  check "$s" 1 1
done
exit $fails
