set -e -x

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

gdown "https://drive.google.com/uc?id=137hJErx-gDd_2kkhKJ4NEM75-DiEtuhF" -O "${SCRIPT_DIR}"/ckpt_6999_crop.pt
