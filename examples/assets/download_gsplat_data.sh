set -e -x

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

wget https://github.com/nerfstudio-project/gsplat/blob/v1.0/assets/test_garden.npz -O "${SCRIPT_DIR}"/test_garden.npz
