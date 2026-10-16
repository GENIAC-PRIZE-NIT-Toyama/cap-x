#!/usr/bin/env bash
# Build the BEHAVIOR (Isaac Sim 5.1) venv and add what CaP-X needs to run in it.
#
# Usage:
#   ./scripts/setup_behavior.sh <BEHAVIOR-1K checkout> [uv_install.sh options...]
#
# Example:
#   git clone -b capx/isaac-sim-5.1 https://github.com/GENIAC-PRIZE-NIT-Toyama/BEHAVIOR-1K.git ~/BEHAVIOR-1K
#   ./scripts/setup_behavior.sh ~/BEHAVIOR-1K --dataset
#
# The venv ends up at <BEHAVIOR-1K checkout>/.venv. Run BEHAVIOR tasks with it active:
#   source ~/BEHAVIOR-1K/.venv/bin/activate
#   uv run --no-sync --active capx/envs/launch.py --config-path env_configs/r1pro/...
#
# The perception and PyRoKi servers (SAM3, GraspNet, PyRoKi) do not run in this
# venv; start them from CaP-X's own environment (capx/serving/launch_servers.py).

set -euo pipefail

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

B1K_DIR=$(cd "$1" && pwd)
shift
CAPX_DIR=$(cd "$(dirname "$0")/.." && pwd)

[ -x "$B1K_DIR/uv_install.sh" ] || {
    echo "ERROR: $B1K_DIR/uv_install.sh not found. Clone the capx/isaac-sim-5.1 branch of the BEHAVIOR-1K fork."
    exit 1
}

# 1. OmniGibson + cuRobo + Isaac Sim (+ datasets) into $B1K_DIR/.venv
(cd "$B1K_DIR" && ./uv_install.sh "$@")

# shellcheck disable=SC1091
source "$B1K_DIR/.venv/bin/activate"

# 2. CaP-X itself (base dependencies only; its torch>=2.8 extras would replace torch 2.7.0)
uv pip install -e "$CAPX_DIR"

# 3. What the R1Pro code path imports beyond OmniGibson and the CaP-X base.
#    capx.integrations imports the Franka APIs, which import pyroki / jax at module
#    level, so they are needed even though R1Pro's IK runs on the PyRoKi server.
#    Pins follow CaP-X's uv.lock (pyroki commit, jax<0.4.30 override).
OVERRIDES=$(mktemp)
trap 'rm -f "$OVERRIDES"' EXIT
printf '%s\n' "jax<0.4.30" "jaxlib<0.4.30" "numpy<2" >"$OVERRIDES"
uv pip install --override "$OVERRIDES" \
    open3d \
    viser \
    pyrender \
    "robot_descriptions>=1.16.0" \
    "yourdfpy>=0.0.56" \
    "pyroki @ git+https://github.com/chungmin99/pyroki.git@95afccc22658c461ab1042a048ae4e9c24bc2a47"

# 4. Check that the BEHAVIOR stack imports from CaP-X
OMNI_KIT_ACCEPT_EULA=YES python -c "
import torch, omnigibson, curobo
from capx.envs.simulators.r1pro_b1k import R1ProBehaviourLowLevel
from capx.integrations.r1pro.control import R1ProControlApi
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
"

echo ""
echo "BEHAVIOR venv ready: source $B1K_DIR/.venv/bin/activate"
