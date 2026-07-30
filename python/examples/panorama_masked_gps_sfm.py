"""Run SfM on 360-degree panoramas using the native EQUIRECTANGULAR camera
model, with per-image masks and GPS EXIF priors.

Unlike panorama_sfm.py's SPHERICAL mode, this script wires `mask_path`
through to feature extraction, and runs the mapper with GPS pose priors
enabled (equivalent to the `colmap pose_prior_mapper` CLI command). GPS
EXIF tags are read automatically during feature extraction - no extra
input is needed for that part.

Usage:
    python panorama_masked_gps_sfm.py /path/to/main_folder

This expects (relative to main_folder, unless overridden):
    images/       panorama images (--image_dir)
    masks/        optional per-image masks (--mask_dir)
    colmap_out/   output directory, created if missing (--output_dir)
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pycolmap


def reconstruct(
    image_path: Path,
    mask_path: Path | None,
    output_path: Path,
    matcher: str,
    prior_position_std: float,
    num_threads: int,
    use_gpu: bool,
    gpu_index: str,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    database_path = output_path / "database.db"
    if database_path.exists():
        database_path.unlink()

    rec_path = output_path / "sparse"
    if rec_path.exists():
        shutil.rmtree(rec_path)
    rec_path.mkdir(parents=True)

    reader_options = pycolmap.ImageReaderOptions(camera_model="EQUIRECTANGULAR")
    if mask_path is not None:
        reader_options.mask_path = mask_path

    pycolmap.extract_features(
        database_path,
        image_path,
        reader_options=reader_options,
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=pycolmap.FeatureExtractionOptions(
            use_gpu=use_gpu, gpu_index=gpu_index, num_threads=num_threads
        ),
    )

    matching_options = pycolmap.FeatureMatchingOptions(
        use_gpu=use_gpu, gpu_index=gpu_index, num_threads=num_threads
    )
    if matcher == "sequential":
        pycolmap.match_sequential(database_path, matching_options=matching_options)
    elif matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path, matching_options=matching_options)
    else:
        raise ValueError(f"Unknown matcher: {matcher}")

    # Overwrite the GPS position uncertainty read into the database during
    # feature extraction, mirroring `colmap pose_prior_mapper
    # --prior_position_std_{x,y,z}`.
    covariance = np.diag([prior_position_std**2] * 3)
    with pycolmap.Database.open(database_path) as db:
        for prior in db.read_all_pose_priors():
            prior.position_covariance = covariance
            db.update_pose_prior(prior)

    # Enable GPS pose priors, mirroring `colmap pose_prior_mapper`.
    mapper_options = pycolmap.IncrementalPipelineOptions(
        num_threads=num_threads,
        use_prior_position=True,
    )

    recs = pycolmap.incremental_mapping(
        database_path, image_path, rec_path, mapper_options
    )
    for idx, rec in recs.items():
        print(f"#{idx} {rec.summary()}")
        rec.write(rec_path / str(idx))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "main_path",
        type=Path,
        help="Folder containing the image_dir/mask_dir subfolders. The "
        "output_dir is also created relative to this folder.",
    )
    parser.add_argument(
        "--image_dir",
        default="images",
        help="Subfolder of main_path with the panorama images.",
    )
    parser.add_argument(
        "--mask_dir",
        default="masks",
        help="Subfolder of main_path with masks mirroring image_dir "
        "(mask for image_dir/abc/012.jpg must be at "
        "mask_dir/abc/012.jpg.png). If it doesn't exist, no masks are used.",
    )
    parser.add_argument(
        "--output_dir",
        default="colmap_out",
        help="Subfolder of main_path to write the database and sparse "
        "reconstruction to.",
    )
    parser.add_argument(
        "--matcher",
        choices=["sequential", "exhaustive"],
        default="sequential",
        help="Use 'sequential' if panoramas were captured along a path/video, "
        "'exhaustive' for unordered small collections.",
    )
    parser.add_argument(
        "--prior_position_std",
        type=float,
        default=1.0,
        help="Assumed GPS position uncertainty in meters.",
    )
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--gpu_index", default="-1")
    parser.add_argument("--use_gpu", default=True, action="store_true")
    parser.add_argument("--use_cpu", dest="use_gpu", action="store_false")
    args = parser.parse_args()

    image_path = args.main_path / args.image_dir
    mask_path = args.main_path / args.mask_dir
    if not mask_path.is_dir():
        print(f"No mask folder at {mask_path}, proceeding without masks.")
        mask_path = None
    output_path = args.main_path / args.output_dir

    reconstruct(
        image_path,
        mask_path,
        output_path,
        args.matcher,
        args.prior_position_std,
        args.num_threads,
        args.use_gpu,
        args.gpu_index,
    )


if __name__ == "__main__":
    main()
