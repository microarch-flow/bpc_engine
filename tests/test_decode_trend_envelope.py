from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None
SELECTED_BATCHES = {1, 32, 256}
EXPECTED_FIGURE_STEMS = {
    "weight_capacity_by_release",
    "active_parameter_ratio_by_release",
    "cache_capacity_per_request_by_context",
    "fixed_context_C2048_flops_by_year",
    "fixed_context_C2048_logical_hbm_B1_by_year",
    "fixed_context_C2048_logical_hbm_B32_by_year",
    "fixed_context_C2048_logical_hbm_B256_by_year",
    "advertised_ceiling_flops_by_model",
    "advertised_ceiling_logical_hbm_B1_by_model",
    "advertised_ceiling_logical_hbm_B32_by_model",
    "advertised_ceiling_logical_hbm_B256_by_model",
}
for _batch in SELECTED_BATCHES:
    EXPECTED_FIGURE_STEMS.update(
        {
            f"flops_per_token_by_context_B{_batch}",
            f"logical_hbm_bytes_per_token_by_context_B{_batch}",
            f"persistent_memory_by_context_B{_batch}",
            f"tbps_per_pflops_by_context_B{_batch}",
            f"annual_envelope_flops_B{_batch}",
            f"annual_envelope_logical_hbm_B{_batch}",
        }
    )


class DecodeTrendEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.runner = cls.root / "scripts" / "run_decode_trend.py"
        cls.analyzer = (
            cls.root / "scripts" / "analyze_decode_trend_envelope.py"
        )

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(
            prefix="decode-trend-envelope-"
        )
        self.temp_path = Path(self.tempdir.name)
        self.release_dir = self.temp_path / "release"
        self.output_dir = self.temp_path / "analysis"
        self.environment = os.environ.copy()
        self.environment["PYTHONDONTWRITEBYTECODE"] = "1"
        self.environment["MPLCONFIGDIR"] = str(
            self.temp_path / "mplconfig"
        )
        self._build_release()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _run_command(self, arguments: list[str], timeout: int = 120) -> str:
        completed = subprocess.run(
            arguments,
            cwd=self.root,
            env=self.environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        self.assertEqual(
            completed.returncode,
            0,
            "command failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )
        return completed.stdout

    def _build_release(self) -> None:
        data_dir = self.release_dir / "data"
        self._run_command(
            [
                sys.executable,
                "-B",
                str(self.runner),
                "--model",
                "google:palm-540b:2022-04-05",
                "--model",
                "moonshotai:kimi-k2.6:2026-04-14",
                "--contexts",
                "128",
                "2048",
                "262144",
                "--batches",
                "1",
                "32",
                "256",
                "--output-dir",
                str(data_dir),
            ]
        )
        run_manifest = json.loads(
            (data_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        validation = json.loads(
            (data_dir / "validation_report.json").read_text(
                encoding="utf-8"
            )
        )
        release_manifest = {
            "dataset_release_schema_version": 1,
            "dataset_version": "test-v1",
            "study_id": run_manifest["study_id"],
            "model_count": validation["model_count"],
            "row_count": validation["row_count"],
            "source_repository": {
                "git_commit": run_manifest["git_commit"] or "test",
                "git_worktree_dirty": run_manifest[
                    "git_worktree_dirty"
                ],
            },
        }
        self.release_dir.mkdir(parents=True, exist_ok=True)
        (self.release_dir / "release_manifest.json").write_text(
            json.dumps(release_manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_analysis_tables_are_scoped_and_closed(self) -> None:
        stdout = self._run_command(
            [
                sys.executable,
                "-B",
                str(self.analyzer),
                "--release-dir",
                str(self.release_dir),
                "--output-dir",
                str(self.output_dir),
                "--no-plots",
            ]
        )
        self.assertIn("analyzed 18 source rows", stdout)
        expected_files = {
            "analysis_manifest.json",
            "quality_summary.json",
            "model_summary.csv",
            "annual_envelope.csv",
            "fixed_context_comparison.csv",
            "native_max_points.csv",
            "context_boundary_points.csv",
            "component_shares.csv",
            "crossover_points.csv",
        }
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()},
            expected_files,
        )

        manifest = json.loads(
            (self.output_dir / "analysis_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["row_counts"],
            {
                "annual_envelope": 30,
                "component_shares": 6,
                "context_boundary_points": 6,
                "crossover_points": 6,
                "fixed_context_comparison": 6,
                "model_summary": 2,
                "native_main_results": 15,
                "native_max_points": 6,
                "source_results": 18,
            },
        )

        quality = json.loads(
            (self.output_dir / "quality_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(quality["status"], "pass")
        self.assertEqual(quality["source_extrapolated_rows"], 3)
        self.assertEqual(quality["native_main_rows"], 15)
        self.assertEqual(
            quality["native_main_rows_by_batch"],
            {"1": 5, "32": 5, "256": 5},
        )

        model_summary_rows = self._read_csv(
            self.output_dir / "model_summary.csv"
        )
        self.assertEqual(len(model_summary_rows), 2)
        self.assertTrue(
            all(
                "frontier_resource_envelope"
                in json.loads(row["sample_roles"])
                for row in model_summary_rows
            )
        )

        envelope_rows = self._read_csv(
            self.output_dir / "annual_envelope.csv"
        )
        self.assertEqual(len(envelope_rows), 30)
        self.assertEqual(
            {row["envelope_scope"] for row in envelope_rows},
            {"all_representative_models", "designated_frontier"},
        )
        self.assertTrue(
            all(row["grid_kind"] == "common_power_of_two" for row in envelope_rows)
        )
        envelope_keys = {
            (
                row["envelope_scope"],
                int(row["year"]),
                int(row["context_C"]),
                int(row["concurrency_B"]),
            )
            for row in envelope_rows
        }
        self.assertEqual(len(envelope_keys), len(envelope_rows))
        for row in envelope_rows:
            eligible_ids = set(
                json.loads(row["eligible_model_release_ids"])
            )
            eligible_names = set(
                json.loads(row["eligible_model_short_names"])
            )
            self.assertEqual(
                int(row["eligible_model_count"]), len(eligible_ids)
            )
            self.assertEqual(len(eligible_names), len(eligible_ids))
            for metric in (
                "per_token_total_flops",
                "per_token_logical_hbm_bytes",
            ):
                winner_ids = set(
                    json.loads(
                        row[f"max_{metric}_model_release_ids"]
                    )
                )
                self.assertTrue(winner_ids)
                self.assertLessEqual(winner_ids, eligible_ids)

        fixed_rows = self._read_csv(
            self.output_dir / "fixed_context_comparison.csv"
        )
        self.assertEqual(len(fixed_rows), 6)
        self.assertEqual(
            {int(row["context_C"]) for row in fixed_rows}, {2048}
        )
        self.assertEqual(
            {int(row["concurrency_B"]) for row in fixed_rows},
            SELECTED_BATCHES,
        )
        fixed_keys = {
            (
                row["model_release_id"],
                row["deployment_profile_id"],
                int(row["concurrency_B"]),
            )
            for row in fixed_rows
        }
        self.assertEqual(len(fixed_keys), len(fixed_rows))
        fixed_models = {
            (row["model_release_id"], row["deployment_profile_id"])
            for row in fixed_rows
        }
        self.assertEqual(len(fixed_models), 2)
        for model_key in fixed_models:
            fixed_model_rows = [
                row
                for row in fixed_rows
                if (
                    row["model_release_id"],
                    row["deployment_profile_id"],
                )
                == model_key
            ]
            self.assertEqual(
                {
                    int(row["concurrency_B"])
                    for row in fixed_model_rows
                },
                SELECTED_BATCHES,
            )
            self.assertEqual(
                len(
                    {
                        float(row["per_token_total_flops"])
                        for row in fixed_model_rows
                    }
                ),
                1,
            )

        native_max_rows = self._read_csv(
            self.output_dir / "native_max_points.csv"
        )
        self.assertEqual(len(native_max_rows), 6)
        advertised_by_model = {
            (
                row["model_release_id"],
                row["deployment_profile_id"],
            ): int(row["advertised_max_context_tokens_at_release"])
            for row in model_summary_rows
        }
        for row in native_max_rows:
            key = (
                row["model_release_id"],
                row["deployment_profile_id"],
            )
            self.assertEqual(
                int(row["context_C"]), advertised_by_model[key]
            )

        component_rows = self._read_csv(
            self.output_dir / "component_shares.csv"
        )
        self.assertEqual(len(component_rows), 6)
        for row in component_rows:
            self.assertTrue(
                math.isclose(
                    sum(
                        float(row[field])
                        for field in (
                            "parameter_flops_share",
                            "attention_flops_share",
                            "index_flops_share",
                            "state_flops_share",
                            "extra_flops_share",
                        )
                    ),
                    1.0,
                    rel_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    sum(
                        float(row[field])
                        for field in (
                            "weight_capacity_share",
                            "kv_cache_capacity_share",
                            "index_cache_capacity_share",
                            "state_cache_capacity_share",
                        )
                    ),
                    1.0,
                    rel_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    sum(
                        float(row[field])
                        for field in (
                            "kv_cache_capacity_share",
                            "index_cache_capacity_share",
                            "state_cache_capacity_share",
                        )
                    ),
                    float(row["batch_cache_capacity_share"]),
                    rel_tol=1e-12,
                )
            )

        crossover_rows = self._read_csv(
            self.output_dir / "crossover_points.csv"
        )
        self.assertEqual(len(crossover_rows), 6)
        allowed_statuses = {
            "crossed_in_grid",
            "at_or_below_min_scanned",
            "not_reached_within_advertised",
        }
        self.assertTrue(
            all(
                row["batch_cache_vs_weight_capacity_status"]
                in allowed_statuses
                for row in crossover_rows
            )
        )
        self.assertTrue(
            all(
                row["batch_cache_vs_weight_capacity_stable_status"]
                in allowed_statuses
                for row in crossover_rows
            )
        )

    def test_context_boundary_tags_are_derived_from_profile_facts(
        self,
    ) -> None:
        spec = importlib.util.spec_from_file_location(
            "decode_trend_envelope_analyzer", self.analyzer
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        analyzer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyzer)

        model_id = "example:model:2025-01-01"
        profile_id = "example-profile"
        profile = {
            "model_release_id": model_id,
            "deployment_profile_id": profile_id,
            "checkpoint": "example/model",
            "context": {
                "advertised_max_context_tokens_at_release": 262144,
                "trained_max_context_tokens": None,
                "evaluated_max_context_tokens": 131072,
                "deployed_max_context_tokens": None,
            },
        }
        row = {
            "year": 2025,
            "release_date": "2025-01-01",
            "model_release_id": model_id,
            "deployment_profile_id": profile_id,
            "context_C": 131072,
            "concurrency_B": 1,
            "context_anchor_tags": ["common_power_of_two"],
            "per_token_total_flops": 1.0,
            "per_token_logical_hbm_bytes": 1.0,
            "decode_profile_weight_capacity_bytes": 1,
            "cache_bytes_per_request": 1.0,
            "batch_cache_bytes": 1.0,
            "persistent_decode_profile_bytes": 2.0,
            "tbps_per_pflops": 1000.0,
            "p3_flops_support": "supported",
            "p3_logical_hbm_traffic_support": "supported",
            "p3_cache_capacity_support": "supported",
            "p3_weight_capacity_support": "supported",
        }
        advertised_row = {**row, "context_C": 262144}
        boundary_rows = analyzer._context_boundary_rows(
            [row, advertised_row], {(model_id, profile_id): profile}
        )
        self.assertEqual(len(boundary_rows), 2)
        evaluated_row = next(
            value
            for value in boundary_rows
            if value["context_C"] == 131072
        )
        self.assertEqual(
            json.loads(evaluated_row["boundary_tags"]),
            ["evaluated_max"],
        )

    def test_annual_series_breaks_only_when_eligible_cohort_changes(
        self,
    ) -> None:
        spec = importlib.util.spec_from_file_location(
            "decode_trend_envelope_segmentation", self.analyzer
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        analyzer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyzer)

        rows = [
            {
                "context_C": context,
                "eligible_model_release_ids": json.dumps(cohort),
                "winner": winner,
            }
            for context, cohort, winner in (
                (128, ["B", "A"], "A"),
                (256, ["A", "B"], "B"),
                (512, ["B", "C"], "B"),
                (1024, ["C", "B"], "C"),
                (2048, ["C"], "C"),
            )
        ]
        segments = analyzer._split_annual_series_by_eligible_cohort(rows)
        self.assertEqual(
            [
                [row["context_C"] for row in segment]
                for segment in segments
            ],
            [[128, 256], [512, 1024], [2048]],
        )

    def test_crossover_records_reversal_and_stable_interval(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "decode_trend_envelope_crossover", self.analyzer
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        analyzer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyzer)

        rows = [
            {
                "context_C": context,
                "left": left,
                "right": 1.0,
                "context_anchor_tags": [],
            }
            for context, left in (
                (128, 0.5),
                (256, 1.2),
                (512, 0.8),
                (1024, 1.5),
            )
        ]
        result = analyzer._first_crossover(
            rows,
            lambda row: row["left"],
            lambda row: row["right"],
        )
        self.assertEqual(result["upper_context_C"], 256)
        self.assertFalse(result["remains_dominant_after_first_reach"])
        self.assertEqual(result["dominance_reversal_count"], 1)
        self.assertEqual(result["stable_lower_context_C"], 512)
        self.assertEqual(result["stable_upper_context_C"], 1024)

    def test_analysis_rejects_profile_fact_mismatch(self) -> None:
        results_path = self.release_dir / "data" / "decode_results.csv"
        with results_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        rows[0]["year"] = "1999"
        with results_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(self.analyzer),
                "--release-dir",
                str(self.release_dir),
                "--output-dir",
                str(self.output_dir),
                "--no-plots",
            ],
            cwd=self.root,
            env=self.environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "year disagrees with model profile", completed.stderr
        )

    @unittest.skipUnless(
        MATPLOTLIB_AVAILABLE,
        "matplotlib is optional; install requirements-plot.txt to test plots",
    )
    def test_analysis_renders_png_and_svg(self) -> None:
        self._run_command(
            [
                sys.executable,
                "-B",
                str(self.analyzer),
                "--release-dir",
                str(self.release_dir),
                "--output-dir",
                str(self.output_dir),
                "--dpi",
                "40",
            ],
            timeout=180,
        )
        figure_dir = self.output_dir / "figures"
        png_paths = sorted(figure_dir.glob("*.png"))
        svg_paths = sorted(figure_dir.glob("*.svg"))
        self.assertEqual(
            {path.stem for path in png_paths}, EXPECTED_FIGURE_STEMS
        )
        self.assertEqual(
            {path.stem for path in svg_paths}, EXPECTED_FIGURE_STEMS
        )
        self.assertEqual(len(png_paths), 29)
        self.assertEqual(len(svg_paths), 29)
        manifest = json.loads(
            (self.output_dir / "analysis_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected_figure_artifacts = {
            f"figures/{stem}.{suffix}"
            for stem in EXPECTED_FIGURE_STEMS
            for suffix in ("png", "svg")
        }
        self.assertEqual(
            set(manifest["artifacts"]["figures"]),
            expected_figure_artifacts,
        )
        self.assertEqual(
            len(manifest["artifacts"]["figures"]),
            len(expected_figure_artifacts),
        )
        for path in png_paths:
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(path.stat().st_size, 100)
        for path in svg_paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("<svg", text[:2048])
            self.assertGreater(path.stat().st_size, 100)
            self.assertTrue(
                all(line == line.rstrip() for line in text.splitlines())
            )


if __name__ == "__main__":
    unittest.main()
