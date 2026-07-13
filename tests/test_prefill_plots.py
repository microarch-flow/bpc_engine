"""End-to-end smoke tests for the optional Prefill plotting workflow.

The plotting dependency is intentionally not part of the calculator's core
installation.  These tests therefore skip cleanly when matplotlib is absent,
while still exercising the real command-line script when it is installed.
"""

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
from typing import Any


MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None


@unittest.skipUnless(
    MATPLOTLIB_AVAILABLE,
    "matplotlib is optional; install requirements-plot.txt to test plots",
)
class PrefillPlotScriptTests(unittest.TestCase):
    """Use one model and tiny shapes to keep the plotting smoke test fast."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.script = cls.project_root / "scripts" / "generate_prefill_plots.py"

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="prefill-plots-")
        self.temp_path = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_script(self, output_dir: Path, *arguments: str) -> None:
        self.assertTrue(self.script.is_file(), f"missing plotting script: {self.script}")
        environment = os.environ.copy()
        # The test and common headless/CI environments may have a read-only
        # HOME.  Giving matplotlib an explicit writable config directory also
        # avoids its slow temporary-directory fallback warning.
        environment["MPLCONFIGDIR"] = str(self.temp_path / "mplconfig")
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(self.script),
                "--precision",
                "16",
                "--models",
                "qwen3_8b",
                "--output-dir",
                str(output_dir),
                "--dpi",
                "48",
                *arguments,
            ],
            cwd=self.project_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            "prefill plotting command failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )

    def read_csv(self, path: Path) -> list[dict[str, str]]:
        self.assertTrue(path.is_file(), f"missing CSV: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def only_matching_file(self, directory: Path, pattern: str) -> Path:
        matches = sorted(directory.glob(pattern))
        self.assertEqual(
            len(matches),
            1,
            f"expected one {pattern!r} under {directory}, found {matches}",
        )
        return matches[0]

    def assert_rendered_images(self, directory: Path) -> None:
        png_paths = sorted(directory.glob("*.png"))
        svg_paths = sorted(directory.glob("*.svg"))
        self.assertTrue(png_paths, f"no PNG generated under {directory}")
        self.assertTrue(svg_paths, f"no SVG generated under {directory}")

        for path in png_paths:
            with path.open("rb") as handle:
                self.assertEqual(handle.read(8), b"\x89PNG\r\n\x1a\n")
            self.assertGreater(path.stat().st_size, 100)
        for path in svg_paths:
            text = path.read_text(encoding="utf-8")
            prefix = text[:2048]
            self.assertIn("<svg", prefix)
            self.assertGreater(path.stat().st_size, 100)
            self.assertTrue(
                all(line == line.rstrip() for line in text.splitlines()),
                f"SVG contains trailing whitespace: {path}",
            )

    @staticmethod
    def json_rows(payload: Any) -> list[dict[str, Any]]:
        """Accept the documented result list and a metadata wrapper if added."""

        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("results", "rows", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise AssertionError("plot JSON must contain a list of result objects")

    def test_all_experiments_data_and_images_are_consistent(self) -> None:
        output_dir = self.temp_path / "all"
        self.run_script(
            output_dir,
            "--experiments",
            "equal",
            "token-budget",
            "ragged",
            "--prompt-lengths",
            "2",
            "4",
            "--equal-batches",
            "1",
            "2",
            "--token-budgets",
            "4",
            "--token-budget-batches",
            "1",
            "2",
            "--ragged-lengths",
            "1",
            "3",
            "--ragged-execution-modes",
            "varlen",
            "padded",
            "--execution-mode",
            "varlen",
        )

        all_detail = self.read_csv(output_dir / "prefill_all_detail.csv")
        all_summary = self.read_csv(output_dir / "prefill_all_summary.csv")
        self.assertEqual(len(all_detail), 8)
        self.assertEqual(len(all_summary), len(all_detail))

        payload = json.loads(
            (output_dir / "prefill_all.json").read_text(encoding="utf-8")
        )
        all_json = self.json_rows(payload)
        self.assertEqual(len(all_json), len(all_detail))
        first_json = all_json[0]
        first_detail = next(
            row
            for row in all_detail
            if row["model_id"] == first_json["model_id"]
            and row["trace_id"] == first_json["trace_id"]
            and row["execution_mode"]
            == first_json["result"]["execution_mode"]
        )
        self.assertEqual(
            json.loads(first_detail["prompt_lengths"]),
            first_json["result"]["prompt_tokens"],
        )
        self.assertTrue(
            math.isclose(
                float(first_detail["total_flops"]),
                first_json["result"]["batch_work"]["total_flops"],
            )
        )
        self.assertTrue(
            math.isclose(
                float(first_detail["total_operand_bytes"]),
                first_json["result"]["batch_operand_work"]["total_bytes"],
            )
        )

        expected_counts = {"equal": 4, "token-budget": 2, "ragged": 2}
        for experiment, expected_count in expected_counts.items():
            directory_name = experiment.replace("-", "_")
            experiment_dir = output_dir / directory_name
            detail_path = self.only_matching_file(
                experiment_dir, "*detail.csv"
            )
            summary_path = self.only_matching_file(
                experiment_dir, "*summary.csv"
            )
            json_path = self.only_matching_file(experiment_dir, "*.json")
            detail_rows = self.read_csv(detail_path)
            summary_rows = self.read_csv(summary_path)
            json_result_rows = self.json_rows(
                json.loads(json_path.read_text(encoding="utf-8"))
            )
            self.assertEqual(len(detail_rows), expected_count)
            self.assertEqual(len(summary_rows), expected_count)
            self.assertEqual(len(json_result_rows), expected_count)
            self.assertEqual(
                {row["experiment"] for row in detail_rows}, {experiment}
            )
            self.assert_rendered_images(experiment_dir)

        equal_rows = [
            row for row in all_detail if row["experiment"] == "equal"
        ]
        self.assertEqual(
            sorted(int(row["valid_input_tokens"]) for row in equal_rows),
            [2, 4, 4, 8],
        )
        for row in equal_rows:
            lengths = json.loads(row["prompt_lengths"])
            self.assertEqual(len(set(lengths)), 1)
            self.assertEqual(sum(lengths), int(row["valid_input_tokens"]))

        budget_rows = [
            row
            for row in all_detail
            if row["experiment"] == "token-budget"
        ]
        self.assertEqual(len(budget_rows), 2)
        for row in budget_rows:
            lengths = json.loads(row["prompt_lengths"])
            self.assertEqual(sum(lengths), 4)
            self.assertLessEqual(max(lengths) - min(lengths), 1)
            self.assertEqual(int(row["valid_input_tokens"]), 4)

        ragged_rows = {
            row["execution_mode"]: row
            for row in all_detail
            if row["experiment"] == "ragged"
        }
        self.assertEqual(set(ragged_rows), {"varlen", "padded"})
        varlen = ragged_rows["varlen"]
        padded = ragged_rows["padded"]
        self.assertEqual(json.loads(varlen["prompt_lengths"]), [1, 3])
        self.assertEqual(json.loads(padded["prompt_lengths"]), [1, 3])
        self.assertEqual(int(varlen["valid_input_tokens"]), 4)
        self.assertEqual(int(varlen["executed_input_tokens"]), 4)
        self.assertEqual(int(padded["valid_input_tokens"]), 4)
        self.assertEqual(int(padded["executed_input_tokens"]), 6)
        self.assertTrue(math.isclose(float(varlen["token_efficiency"]), 1.0))
        self.assertTrue(
            math.isclose(float(padded["token_efficiency"]), 2.0 / 3.0)
        )
        self.assertTrue(
            math.isclose(float(padded["causal_pair_efficiency"]), 7.0 / 12.0)
        )
        self.assertGreater(
            float(padded["batch_total_flops"]),
            float(varlen["batch_total_flops"]),
        )
        self.assertTrue(
            math.isclose(
                float(padded["cache_bytes_total"]),
                float(varlen["cache_bytes_total"]),
            )
        )

        # Flattened CSV aliases must agree with the auditable work components.
        for row in all_detail:
            valid_tokens = int(row["valid_input_tokens"])
            self.assertTrue(
                math.isclose(
                    float(row["total_flops"]),
                    float(row["batch_total_flops"]),
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["total_compulsory_bytes"]),
                    float(row["batch_total_bytes"]),
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["total_operand_bytes"]),
                    float(row["batch_operand_total_bytes"]),
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["per_input_total_flops"]) * valid_tokens,
                    float(row["batch_total_flops"]),
                    rel_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["tbps_per_pflops"]),
                    1000.0 * float(row["bytes_per_flop"]),
                    rel_tol=1e-12,
                )
            )

    def test_unselected_experiments_do_not_leave_artifacts(self) -> None:
        output_dir = self.temp_path / "equal-only"
        stale_directory = output_dir / "ragged"
        stale_directory.mkdir(parents=True)
        (stale_directory / "prefill_ragged_mode_padded.png").write_bytes(
            b"stale"
        )
        self.run_script(
            output_dir,
            "--experiments",
            "equal",
            "--prompt-lengths",
            "2",
            "--equal-batches",
            "1",
        )

        detail_rows = self.read_csv(output_dir / "prefill_all_detail.csv")
        self.assertEqual(len(detail_rows), 1)
        self.assertEqual(detail_rows[0]["experiment"], "equal")
        self.assertTrue((output_dir / "equal").is_dir())
        self.assertFalse((output_dir / "token_budget").exists())
        self.assertFalse((output_dir / "ragged").exists())
        self.assert_rendered_images(output_dir / "equal")


if __name__ == "__main__":
    unittest.main()
