from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DecodeTrendRunnerTests(unittest.TestCase):
    def test_two_model_pilot_writes_auditable_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "run_decode_trend.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "trend"
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(script),
                    "--model",
                    "meta-llama:llama-2-70b-chat-hf:2023-07-18",
                    "--model",
                    "zai-org:glm-5.2-fp8:2026-06-16",
                    "--contexts",
                    "128",
                    "2048",
                    "8192",
                    "--batches",
                    "1",
                    "4",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("wrote 12 rows for 2 models", completed.stdout)

            expected_files = {
                "run_manifest.json",
                "model_profiles.jsonl",
                "decode_results.jsonl",
                "decode_results.csv",
                "validation_report.json",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                expected_files,
            )

            with (output_dir / "decode_results.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            llama_extrapolated = [
                row
                for row in rows
                if row["model_release_id"].startswith("meta-llama:")
                and row["context_C"] == "8192"
            ]
            self.assertEqual(len(llama_extrapolated), 2)
            self.assertTrue(
                all(row["is_extrapolated"] == "True" for row in llama_extrapolated)
            )
            for row in rows:
                self.assertEqual(
                    row["per_token_engine_total_bytes"],
                    row["per_token_logical_hbm_bytes"],
                )
                self.assertEqual(row["per_token_activation_bytes"], "0.0")
                self.assertIn(
                    row["p3_flops_support"],
                    {"supported", "partially_supported"},
                )

            glm_rows = [
                row
                for row in rows
                if row["model_release_id"].startswith("zai-org:")
            ]
            self.assertTrue(glm_rows)
            self.assertTrue(
                all(
                    row["decode_profile_weight_capacity_bytes"]
                    == "745584507456"
                    for row in glm_rows
                )
            )
            self.assertTrue(
                all(
                    row["full_checkpoint_capacity_bytes"]
                    == "755617140416"
                    for row in glm_rows
                )
            )

            report = json.loads(
                (output_dir / "validation_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["row_count"], 12)
            self.assertEqual(report["checks"]["extrapolated_rows"], 2)

            profiles = [
                json.loads(line)
                for line in (
                    output_dir / "model_profiles.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(profiles), 2)
            glm_profile = next(
                profile
                for profile in profiles
                if profile["model_release_id"].startswith("zai-org:")
            )
            self.assertEqual(
                glm_profile["mechanism_layer_counts"],
                {
                    "softmax_attention:mla:dsa": 21,
                    "softmax_attention:mla:fixed_topk": 57,
                },
            )
            self.assertEqual(
                glm_profile["p3_audit"]["overall"]["flops"],
                "partially_supported",
            )

    def test_unknown_full_checkpoint_capacity_remains_unknown(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "run_decode_trend.py"
        source_manifest_path = (
            root / "studies" / "decode_trend" / "models.json"
        )
        manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        source_audit_path = source_manifest_path.parent / manifest[
            "mechanism_audit_path"
        ]
        audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
        model = manifest["models"][0]
        model["capacity"]["full_checkpoint_capacity_bytes"] = None
        model["config_path"] = str(
            (
                source_manifest_path.parent / model["config_path"]
            ).resolve()
        )
        manifest["models"] = [model]
        audit["models"] = [
            entry
            for entry in audit["models"]
            if entry["model_release_id"] == model["model_release_id"]
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "models.json"
            audit_path = temp_path / "mechanism_audit.json"
            output_dir = temp_path / "trend"
            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            audit_path.write_text(
                json.dumps(audit),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(script),
                    "--manifest",
                    str(manifest_path),
                    "--contexts",
                    "128",
                    "--batches",
                    "1",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(
                (output_dir / "decode_results.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIsNone(
                result["capacity"]["full_checkpoint_capacity_bytes"]
            )
            self.assertIsNone(
                result["capacity"]["persistent_full_checkpoint_bytes"]
            )


if __name__ == "__main__":
    unittest.main()
