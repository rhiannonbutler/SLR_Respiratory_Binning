#!/usr/bin/env python3
"""Combine repeated DICOM acquisitions into RMS-averaged DICOM series."""
import argparse
import copy
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pydicom


def get_match_key(dcm, position_tol=0.5):
    if hasattr(dcm, "ImagePositionPatient"):
        z = round(float(dcm.ImagePositionPatient[2]) / position_tol) * position_tol
    elif hasattr(dcm, "SliceLocation"):
        z = round(float(dcm.SliceLocation) / position_tol) * position_tol
    else:
        z = None

    echo = int(dcm.EchoNumbers) if hasattr(dcm, "EchoNumbers") else None
    rep = int(dcm.AcquisitionNumber) if hasattr(dcm, "AcquisitionNumber") else None

    return (z, echo, rep)


def load_indexed(folder, pattern="*.dcm"):
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if len(files) == 0:
        files = sorted([f for f in folder.iterdir() if f.is_file()])

    indexed = {}
    collisions = 0
    for f in files:
        dcm = pydicom.dcmread(f)
        key = get_match_key(dcm)
        if key in indexed:
            collisions += 1
            print(f"WARNING: duplicate key {key} -> {indexed[key][0].name} and {f.name}")
        indexed[key] = (f, dcm)

    print(f"Loaded {len(indexed)} unique keys from {folder} ({collisions} collisions)")
    return indexed


def check_acquisition_distribution(folder):
    folder = Path(folder)
    files = sorted(folder.glob("*.dcm"))
    acq_numbers = []
    for f in files:
        dcm = pydicom.dcmread(f)
        acq_numbers.append(getattr(dcm, "AcquisitionNumber", None))
    print(f"AcquisitionNumber distribution: {Counter(acq_numbers)}")
    print(f"Total DICOM files: {len(files)}")


def rms_combine_reps(indexed, output_dir, position_tol=0.5, series_suffix="_RMS"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for key in indexed:
        z, echo, rep = key
        groups[(z, echo)].append(key)

    keys = sorted(groups.keys(), key=lambda x: (x[0] or 0, x[1] or 0))

    for z, echo in keys:
        rep_keys = sorted(groups[(z, echo)], key=lambda k: k[2] if k[2] is not None else 0)
        imgs = []
        for key in rep_keys:
            _, dcm = indexed[key]
            imgs.append(dcm.pixel_array.astype(np.float32))

        if len(imgs) == 0:
            continue

        rms = np.sqrt(np.mean(np.stack(imgs, axis=0) ** 2, axis=0))

        _, template = indexed[rep_keys[0]]
        out_dcm = copy.deepcopy(template)
        out_dtype = template.pixel_array.dtype

        out_pixel = np.round(rms).astype(out_dtype)
        out_dcm.PixelData = out_pixel.tobytes()

        if hasattr(out_dcm, "SeriesDescription"):
            out_dcm.SeriesDescription = str(out_dcm.SeriesDescription) + series_suffix
        else:
            out_dcm.SeriesDescription = f"RMS combined{series_suffix}"

        out_dcm.AcquisitionNumber = 1
        out_dcm.InstanceNumber = 1

        echo_str = f"echo{echo}" if echo is not None else "echo_unknown"
        z_str = f"z{z:.1f}" if z is not None else "z_unknown"
        out_name = f"{echo_str}_{z_str}_rms.dcm"
        out_path = output_dir / out_name
        out_dcm.save_as(str(out_path))
        print(f"Saved {out_path} ({len(rep_keys)} reps)")


def combine_folder(folder, output_root, position_tol=0.5):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)

    out_dir = Path(output_root) / f"{folder.name}_rms"
    indexed = load_indexed(folder)
    rms_combine_reps(indexed, out_dir, position_tol=position_tol)


def main():
    parser = argparse.ArgumentParser(description="Combine repeated DICOM acquisitions into RMS-averaged DICOM series")
    parser.add_argument("--folders", nargs="+", required=True, help="One or more DICOM folders to combine")
    parser.add_argument("--output_root", required=True, help="Root output directory")
    parser.add_argument("--position_tol", type=float, default=0.5, help="Position tolerance in mm for matching slices")
    args = parser.parse_args()

    for folder in args.folders:
        print(f"Processing {folder}")
        check_acquisition_distribution(folder)
        combine_folder(folder, args.output_root, position_tol=args.position_tol)


if __name__ == "__main__":
    main()
