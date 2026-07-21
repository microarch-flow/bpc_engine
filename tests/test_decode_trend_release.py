from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts" / "freeze_decode_trend_release.py"
BUNDLED_RELEASE = (
    ROOT / "studies" / "decode_trend" / "releases" / "v1.0.0"
)


class DecodeTrendFrozenReleaseTests(unittest.TestCase):
    def test_default_v1_release_is_bundled_and_checksummed(self):
        manifest_path = BUNDLED_RELEASE / "release_manifest.json"
        self.assertTrue(
            manifest_path.is_file(),
            "the default v1.0.0 release must be present in a fresh clone",
        )
        release_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        self.assertEqual(release_manifest["dataset_version"], "v1.0.0")
        self.assertEqual(release_manifest["model_count"], 20)
        self.assertEqual(release_manifest["row_count"], 3357)

        checksum_lines = (BUNDLED_RELEASE / "SHA256SUMS").read_text(
            encoding="utf-8"
        ).splitlines()
        checked_paths: set[str] = set()
        for line in checksum_lines:
            expected, relative = line.split("  ", 1)
            path = BUNDLED_RELEASE / relative
            self.assertTrue(path.is_file(), relative)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected, relative)
            checked_paths.add(relative)

        expected_paths = {
            str(path.relative_to(BUNDLED_RELEASE))
            for path in BUNDLED_RELEASE.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        }
        self.assertEqual(checked_paths, expected_paths)

    def test_release_is_generated_and_checksummed_from_clean_clone_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            release_root = Path(temp_dir) / "releases"
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(FREEZER),
                    "--version",
                    "test-v1",
                    "--release-root",
                    str(release_root),
                ],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            release = release_root / "test-v1"
            release_manifest = json.loads(
                (release / "release_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            validation = json.loads(
                (release / "data" / "validation_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                release_manifest["dataset_version"], "test-v1"
            )
            self.assertEqual(release_manifest["model_count"], 20)
            self.assertEqual(release_manifest["row_count"], 3357)
            self.assertEqual(validation["status"], "pass")

            with (release / "data" / "decode_results.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3357)
            self.assertIn("p3_flops_support", rows[0])

            checksum_lines = (
                release / "SHA256SUMS"
            ).read_text(encoding="utf-8").splitlines()
            checked_paths: set[str] = set()
            for line in checksum_lines:
                expected, relative = line.split("  ", 1)
                path = release / relative
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, expected, relative)
                checked_paths.add(relative)

            expected_paths = {
                str(path.relative_to(release))
                for path in release.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS"
            }
            self.assertEqual(checked_paths, expected_paths)


if __name__ == "__main__":
    unittest.main()
