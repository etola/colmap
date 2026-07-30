#!/bin/bash
# Run SfM on 360-degree panoramas using the native EQUIRECTANGULAR camera
# model, with per-image masks and GPS EXIF pose priors.
#
# Meant to be run inside the COLMAP Docker container (see run.sh), where your
# host working directory is mounted at /working. GPS EXIF tags are read
# automatically during feature extraction - no extra input is needed for that.
#
# Usage (from inside the container, after ./run.sh <host_dir>):
#   ./run_panorama_sfm.sh [main_path]
#
# Expects (relative to main_path, default "."):
#   images/       panorama images (--image_dir)
#   masks/        optional per-image masks (--mask_dir)
#   colmap_out/   output directory, created if missing (--output_dir)

set -euo pipefail

MAIN_PATH="."
IMAGE_DIR="images"
MASK_DIR="masks"
OUTPUT_DIR="colmap_out"
MATCHER="sequential"
PRIOR_POSITION_STD="1.0"
NUM_THREADS="-1"
USE_GPU=1

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | tail -n +2
    exit 1
}

if [[ $# -gt 0 && "$1" != --* ]]; then
    MAIN_PATH="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image_dir) IMAGE_DIR="$2"; shift 2 ;;
        --mask_dir) MASK_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --matcher) MATCHER="$2"; shift 2 ;;
        --prior_position_std) PRIOR_POSITION_STD="$2"; shift 2 ;;
        --num_threads) NUM_THREADS="$2"; shift 2 ;;
        --use_cpu) USE_GPU=0; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ "$MATCHER" != "sequential" && "$MATCHER" != "exhaustive" ]]; then
    echo "--matcher must be 'sequential' or 'exhaustive', got '$MATCHER'" >&2
    exit 1
fi

IMAGE_PATH="${MAIN_PATH}/${IMAGE_DIR}"
MASK_PATH="${MAIN_PATH}/${MASK_DIR}"
OUTPUT_PATH="${MAIN_PATH}/${OUTPUT_DIR}"

if [[ ! -d "$IMAGE_PATH" ]]; then
    echo "Image folder not found: $IMAGE_PATH" >&2
    exit 1
fi

MASK_ARGS=()
if [[ -d "$MASK_PATH" ]]; then
    MASK_ARGS=(--ImageReader.mask_path "$MASK_PATH")
else
    echo "No mask folder at $MASK_PATH, proceeding without masks."
fi

mkdir -p "$OUTPUT_PATH"
DATABASE_PATH="${OUTPUT_PATH}/database.db"
SPARSE_PATH="${OUTPUT_PATH}/sparse"
if [[ -f "$DATABASE_PATH" ]]; then
    echo "Removing existing database at $DATABASE_PATH"
    rm -f "$DATABASE_PATH"
fi
if [[ -d "$SPARSE_PATH" ]]; then
    echo "Removing existing reconstruction at $SPARSE_PATH"
    rm -rf "$SPARSE_PATH"
fi
mkdir -p "$SPARSE_PATH"

echo "=== Extracting features (EQUIRECTANGULAR) ==="
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model EQUIRECTANGULAR \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu "$USE_GPU" \
    --FeatureExtraction.num_threads "$NUM_THREADS" \
    "${MASK_ARGS[@]}"

echo "=== Matching features (${MATCHER}) ==="
colmap "${MATCHER}_matcher" \
    --database_path "$DATABASE_PATH" \
    --FeatureMatching.use_gpu "$USE_GPU" \
    --FeatureMatching.num_threads "$NUM_THREADS"

echo "=== Mapping with GPS pose priors ==="
colmap pose_prior_mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$SPARSE_PATH" \
    --prior_position_std_x "$PRIOR_POSITION_STD" \
    --prior_position_std_y "$PRIOR_POSITION_STD" \
    --prior_position_std_z "$PRIOR_POSITION_STD" \
    --overwrite_priors_covariance 1

echo "=== Done. Reconstruction written under ${SPARSE_PATH} ==="
