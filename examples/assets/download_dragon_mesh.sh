set -e -x

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

gdown "https://drive.google.com/uc?id=1uRDvoS_l2Or8g8YDDPYV79K6_RfFYBeF" -O "${SCRIPT_DIR}"/dragon.obj
