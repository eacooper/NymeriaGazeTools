#!/usr/bin/env python3
"""
Unified download script for Nymeria eye gaze dataset.
Downloads eye gaze data and metadata for all sequences, producing:
  - Raw per-sequence files (metadata.json + eye gaze CSV)
  - Processed normalized output (metadata.csv + individual eye_gaze CSVs)

Tries personalized_eye_gaze.csv first; falls back to general_eye_gaze.csv.
"""

import csv
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm


METADATA_FIELDS = [
    "date", "session_id", "fake_name", "act_id", "location", "script",
    "participant_gender", "participant_height_cm", "participant_weight_kg",
    "participant_bmi", "participant_age_group", "participant_ethnicity",
]


class EyegazeDownloader:
    def __init__(self, url_json_path: str, raw_output_root: str, processed_output_root: str):
        """
        Args:
            url_json_path: Path to a nymeria_download_urls*.json file
            raw_output_root: Root directory for raw per-sequence downloads
            processed_output_root: Root directory for processed normalized output
        """
        self.url_json_path = Path(url_json_path)
        self.raw_output_root = Path(raw_output_root)
        self.processed_output_root = Path(processed_output_root)

        print(f"Loading URLs from: {self.url_json_path}")
        with open(self.url_json_path, 'r') as f:
            data = json.load(f)
            self.sequences = data['sequences']

        self.stats = {
            'sequences_processed': 0,
            'sequences_successful': 0,
            'sequences_skipped': 0,
            'sequences_failed': 0,
            'personalized_count': 0,
            'general_count': 0,
            'metadata_downloaded': 0,
            'total_bytes_downloaded': 0,
            'total_bytes_saved': 0,
        }

    def get_all_participants(self) -> Dict[str, List[str]]:
        """Get all participants and their sequences."""
        participant_seqs = {}
        for seq_id in self.sequences.keys():
            parts = seq_id.split('_')
            if len(parts) >= 4:
                participant = '_'.join(parts[2:-2])
                if participant not in participant_seqs:
                    participant_seqs[participant] = []
                participant_seqs[participant].append(seq_id)
        return participant_seqs

    def download_file(self, url: str, dest_path: Path, description: str = "") -> bool:
        """Download a file from URL with progress bar."""
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(dest_path, 'wb') as f, tqdm(
                desc=description,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        self.stats['total_bytes_downloaded'] += len(chunk)

            return True
        except Exception as e:
            print(f"  Error downloading {description}: {e}")
            return False

    def extract_eye_gaze_from_zip(self, zip_path: Path, output_dir: Path) -> Tuple[bool, str]:
        """Extract eye gaze CSV from recording_head.zip.

        Tries personalized_eye_gaze.csv first, then general_eye_gaze.csv.

        Returns:
            (success, gaze_type) where gaze_type is "personalized" or "general"
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                names = zip_ref.namelist()

                # Try personalized first, then general
                for gaze_type, filename in [
                    ("personalized", "personalized_eye_gaze.csv"),
                    ("general", "general_eye_gaze.csv"),
                ]:
                    target = None
                    for name in names:
                        if filename in name:
                            target = name
                            break

                    if target:
                        output_path = output_dir / filename
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        with zip_ref.open(target) as source, open(output_path, 'wb') as dest:
                            data = source.read()
                            dest.write(data)
                            self.stats['total_bytes_saved'] += len(data)

                        print(f"    Extracted {filename} ({len(data) / (1024**2):.1f} MB)")
                        return True, gaze_type

                print(f"    No eye gaze CSV found in zip")
                return False, ""

        except Exception as e:
            print(f"    Error extracting from zip: {e}")
            return False, ""

    def _get_raw_gaze_path(self, seq_dir: Path) -> Tuple[Path, str]:
        """Find which gaze CSV exists in a raw sequence directory.

        Returns:
            (path, gaze_type) or (None, "") if neither exists
        """
        for gaze_type, filename in [
            ("personalized", "personalized_eye_gaze.csv"),
            ("general", "general_eye_gaze.csv"),
        ]:
            path = seq_dir / filename
            if path.exists():
                return path, gaze_type
        return None, ""

    def download_sequence(self, participant: str, seq_id: str) -> bool:
        """Download eye gaze data and metadata for one sequence."""
        self.stats['sequences_processed'] += 1

        seq_dir = self.raw_output_root / participant / seq_id
        metadata_path = seq_dir / 'metadata.json'
        existing_gaze, _ = self._get_raw_gaze_path(seq_dir)

        # Resume: skip if both files already exist
        if metadata_path.exists() and existing_gaze is not None:
            print(f"  [skip] {seq_id} (already downloaded)")
            self.stats['sequences_skipped'] += 1
            self.stats['sequences_successful'] += 1
            return True

        print(f"\n  {seq_id}")
        seq_dir.mkdir(parents=True, exist_ok=True)
        seq_data = self.sequences[seq_id]
        success = True

        # 1. Download metadata.json
        if not metadata_path.exists() and 'metadata_json' in seq_data:
            url = seq_data['metadata_json']['download_url']
            print(f"    Downloading metadata...")
            if self.download_file(url, metadata_path, "metadata"):
                self.stats['metadata_downloaded'] += 1
                self.stats['total_bytes_saved'] += metadata_path.stat().st_size
            else:
                success = False

        # 2. Download recording_head.zip, extract gaze CSV, delete zip
        if existing_gaze is None and 'recording_head' in seq_data:
            print(f"    Downloading recording_head.zip...")
            zip_url = seq_data['recording_head']['download_url']
            zip_path = seq_dir / 'recording_head_temp.zip'

            if self.download_file(zip_url, zip_path, "recording_head"):
                extracted, gaze_type = self.extract_eye_gaze_from_zip(zip_path, seq_dir)
                if extracted:
                    if gaze_type == "personalized":
                        self.stats['personalized_count'] += 1
                    else:
                        self.stats['general_count'] += 1
                else:
                    success = False

                # Delete zip immediately
                try:
                    zip_size = zip_path.stat().st_size
                    zip_path.unlink()
                    print(f"    Deleted zip ({zip_size / (1024**2):.1f} MB freed)")
                except Exception as e:
                    print(f"    Could not delete zip: {e}")
            else:
                success = False
        elif existing_gaze is None:
            print(f"    No recording_head data available for {seq_id}")
            success = False

        if success:
            self.stats['sequences_successful'] += 1
        else:
            self.stats['sequences_failed'] += 1

        return success

    def download_all(self):
        """Download all sequences for all participants."""
        participant_seqs = self.get_all_participants()
        total_seqs = sum(len(seqs) for seqs in participant_seqs.values())

        print("=" * 70)
        print("NYMERIA EYE GAZE DATASET DOWNLOAD")
        print("=" * 70)
        print(f"URL source: {self.url_json_path.name}")
        print(f"Raw output: {self.raw_output_root.absolute()}")
        print(f"Processed output: {self.processed_output_root.absolute()}")
        print(f"Participants: {len(participant_seqs)}")
        print(f"Sequences: {total_seqs}")
        print("=" * 70)

        for participant in sorted(participant_seqs.keys()):
            sequences = participant_seqs[participant]
            print(f"\n--- {participant.upper()} ({len(sequences)} sequences) ---")

            for seq_id in sequences:
                self.download_sequence(participant, seq_id)

        self._print_download_summary()

    def build_processed_files(self):
        """Build normalized metadata.csv and individual eye_gaze CSVs from raw data."""
        print("\n" + "=" * 70)
        print("BUILDING PROCESSED FILES")
        print("=" * 70)

        processed_gaze_dir = self.processed_output_root / 'eye_gaze'
        processed_gaze_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows = []
        processed_count = 0
        skipped_count = 0

        # Walk raw directories
        participant_dirs = sorted(
            [d for d in self.raw_output_root.iterdir() if d.is_dir()]
        )

        for participant_dir in participant_dirs:
            participant = participant_dir.name
            seq_dirs = sorted(
                [d for d in participant_dir.iterdir() if d.is_dir()]
            )

            for seq_dir in seq_dirs:
                seq_id = seq_dir.name
                metadata_path = seq_dir / 'metadata.json'
                gaze_path, gaze_type = self._get_raw_gaze_path(seq_dir)

                # Read metadata
                meta_row = {"sequence_uid": seq_id}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                    for field in METADATA_FIELDS:
                        meta_row[field] = meta.get(field, "")
                else:
                    for field in METADATA_FIELDS:
                        meta_row[field] = ""

                meta_row["gaze_type"] = gaze_type if gaze_type else ""
                meta_row["has_gaze_data"] = gaze_path is not None
                metadata_rows.append(meta_row)

                # Build processed eye_gaze CSV
                if gaze_path is None:
                    continue

                output_path = processed_gaze_dir / f"{seq_id}.csv"
                if output_path.exists():
                    skipped_count += 1
                    continue

                # Read raw CSV, add gaze_type and sequence_uid columns, write processed
                with open(gaze_path, 'r', newline='') as infile:
                    reader = csv.DictReader(infile)
                    fieldnames = reader.fieldnames + ['gaze_type', 'sequence_uid']

                    with open(output_path, 'w', newline='') as outfile:
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in reader:
                            row['gaze_type'] = gaze_type
                            row['sequence_uid'] = seq_id
                            writer.writerow(row)

                processed_count += 1
                print(f"  Processed: {seq_id} ({gaze_type})")

        # Write metadata.csv
        metadata_path = self.processed_output_root / 'metadata.csv'
        if metadata_rows:
            fieldnames = ["sequence_uid"] + METADATA_FIELDS + ["gaze_type", "has_gaze_data"]
            with open(metadata_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata_rows)

        print(f"\nProcessed files summary:")
        print(f"  metadata.csv: {len(metadata_rows)} sequences -> {metadata_path}")
        print(f"  eye_gaze CSVs: {processed_count} new, {skipped_count} skipped (already exist)")

    def _print_download_summary(self):
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)
        print(f"Sequences processed: {self.stats['sequences_processed']}")
        print(f"  Successful: {self.stats['sequences_successful']}")
        print(f"  Skipped (resume): {self.stats['sequences_skipped']}")
        print(f"  Failed: {self.stats['sequences_failed']}")
        print(f"Gaze type breakdown:")
        print(f"  Personalized: {self.stats['personalized_count']}")
        print(f"  General (fallback): {self.stats['general_count']}")
        print(f"Data downloaded: {self.stats['total_bytes_downloaded'] / (1024**3):.2f} GB")
        print(f"Disk space used: {self.stats['total_bytes_saved'] / (1024**3):.2f} GB")
        print("=" * 70)


def main():
    REPO_ROOT = Path(__file__).parent.parent.parent
    URL_JSON = REPO_ROOT / 'data' / 'nymeria_download_urls.json'
    RAW_DIR = REPO_ROOT / 'data' / 'raw'
    PROCESSED_DIR = REPO_ROOT / 'data' / 'processed'

    if not URL_JSON.exists():
        print(f"Error: URL JSON file not found at {URL_JSON}")
        print("Please ensure data/nymeria_download_urls.json exists.")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    downloader = EyegazeDownloader(
        url_json_path=str(URL_JSON),
        raw_output_root=str(RAW_DIR),
        processed_output_root=str(PROCESSED_DIR),
    )

    try:
        downloader.download_all()
        downloader.build_processed_files()
        print(f"\nDone! Data saved to:")
        print(f"  Raw: {RAW_DIR.absolute()}")
        print(f"  Processed: {PROCESSED_DIR.absolute()}")
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        print(f"Partial data saved to: {RAW_DIR.absolute()}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during download: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
