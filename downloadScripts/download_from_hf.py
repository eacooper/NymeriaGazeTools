#!/usr/bin/env python3
"""
Download eye gaze data from the Nymeria EyeGaze HuggingFace dataset.

Downloads processed/ and/or raw/ folders to a local output directory,
mirroring the HF repo structure. Skips files that already exist (resume-safe).

Usage:
    python download_from_hf.py --output /path/to/data
    python download_from_hf.py --output /path/to/data --folder processed
    python download_from_hf.py --output /path/to/data --folder raw
    python download_from_hf.py --output /path/to/data --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

REPO_ID = "nymeriagazedata/eye-gaze-data"
REPO_TYPE = "dataset"
HF_TOKEN = os.environ.get("HF_TOKEN")


def get_repo_files(folder: str) -> list[str]:
    """List all files in the HF repo under the given folder prefix."""
    all_files = list(list_repo_files(REPO_ID, repo_type=REPO_TYPE, token=HF_TOKEN))
    return [f for f in all_files if f.startswith(folder + "/")]


def download_folder(folder: str, output_root: Path, dry_run: bool = False):
    """Download all files under folder from HF repo to output_root."""
    print(f"\nFetching file list for '{folder}/'...")
    files = get_repo_files(folder)

    if not files:
        print(f"  No files found under '{folder}/'.")
        return

    existing = [f for f in files if (output_root / f).exists()]
    to_download = [f for f in files if not (output_root / f).exists()]

    print(f"  Total:    {len(files)} files")
    print(f"  Existing: {len(existing)} files (will skip)")
    print(f"  To fetch: {len(to_download)} files")

    if dry_run:
        print("\n  [dry-run] Files that would be downloaded:")
        for f in to_download[:20]:
            print(f"    {f}")
        if len(to_download) > 20:
            print(f"    ... and {len(to_download) - 20} more")
        return

    failed = []
    for filename in tqdm(to_download, desc=folder, unit="file"):
        try:
            cached_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=filename,
                token=HF_TOKEN,
            )
            dest = output_root / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(cached_path).read_bytes())
        except Exception as e:
            print(f"\n  Error: {filename} — {e}")
            failed.append(filename)

    print(f"  Done. {len(to_download) - len(failed)} downloaded, {len(failed)} failed.")
    if failed:
        print("  Failed files:")
        for f in failed:
            print(f"    {f}")


def main():
    parser = argparse.ArgumentParser(description="Download Nymeria eye gaze data from HuggingFace.")
    parser.add_argument("--output", required=True, help="Local directory to save data into.")
    parser.add_argument(
        "--folder",
        choices=["processed", "raw", "both"],
        default="both",
        help="Which folder to download (default: both).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without fetching them.",
    )
    args = parser.parse_args()

    if not HF_TOKEN:
        print("Warning: HF_TOKEN not set. Access may fail for private repos.")
        print("  Set it with: export HF_TOKEN=your_token\n")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    folders = ["processed", "raw"] if args.folder == "both" else [args.folder]

    print("=" * 60)
    print("NYMERIA EYE GAZE — HUGGINGFACE DOWNLOAD")
    print("=" * 60)
    print(f"Repo:    {REPO_ID}")
    print(f"Output:  {output_root.absolute()}")
    print(f"Folders: {', '.join(folders)}")
    if args.dry_run:
        print("Mode:    DRY RUN (no files will be downloaded)")
    print("=" * 60)

    for folder in folders:
        download_folder(folder, output_root, dry_run=args.dry_run)

    print("\nComplete.")
    if not args.dry_run:
        print(f"Data saved to: {output_root.absolute()}")


if __name__ == "__main__":
    main()
