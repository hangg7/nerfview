# This downloads the COLMAP model for the mip-NeRF garden dataset
# with the images that are downscaled by a factor of 8.
# The full dataset is available at https://jonbarron.info/mipnerf360/.

set -e -x

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

pushd "${SCRIPT_DIR}"
gdown "https://drive.google.com/uc?id=1wYHdrgwXPHtREdCjItvt4gqRQGISMade"
mkdir -p colmap_garden
# shellcheck disable=SC2035
unzip *.zip && rm *.zip
ln -sf "$(realpath colmap_garden/images_8)" colmap_garden/images
popd
