from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

MODEL_IDS = (
    "bigscience:bloom-176b:2022-07-12",
    "google:palm-540b:2022-04-05",
    "zai-org:glm-130b-int4:2022-08-24",
    "meta-llama:llama-2-70b-chat-hf:2023-07-18",
    "tiiuae:falcon-180b:2023-09-06",
    "mistralai:mistral-7b-instruct-v0.1:2023-09-27",
    "01-ai:yi-34b-200k:2023-11-05",
    "mistralai:mixtral-8x7b-instruct-v0.1:2023-12-11",
    "deepseek-ai:deepseek-v2-chat:2024-05-06",
    "meta-llama:llama-3.1-405b-instruct-fp8:2024-07-23",
    "nvidia:llama-3.1-70b-instruct-fp8:2024-07-23",
    "ai21labs:jamba-1.5-large:2024-08-22",
    "deepseek-ai:deepseek-v3:2024-12-26",
    "meta-llama:llama-4-scout-17b-16e-instruct:2025-04-05",
    "qwen:qwen3-32b-awq:2025-04-29",
    "moonshotai:kimi-linear-48b-a3b-instruct:2025-10-30",
    "moonshotai:kimi-k2-thinking:2025-11-06",
    "moonshotai:kimi-k2.6:2026-04-14",
    "qwen:qwen3.6-35b-a3b-fp8:2026-04-15",
    "zai-org:glm-5.2-fp8:2026-06-16",
)

EXPECTED_TABLE_FILES = {
    "analysis_manifest.json",
    "quality_summary.json",
    "technology_observations.csv",
    "annual_sample_summary.csv",
    "trend_candidates.csv",
    "selected_trend_functions.csv",
    "selected_trend_functions.json",
    "composition_group_summary.csv",
    "trend_backtests.csv",
    "trend_sensitivity.csv",
    "fitted_observations.csv",
    "trend_projection_grid.csv",
    "technology_milestones.csv",
    "technology_cooccurrence.csv",
}

EXPECTED_FIGURE_STEMS = {
    "parameter_scale_and_active_trends",
    "parameter_sparsity_decoupling",
    "context_boundary_trends",
    "token_mixer_composition",
    "kv_layout_composition",
    "attention_access_composition",
    "moe_presence_and_intensity",
    "deployment_profile_precision",
    "technology_presence_timeline",
    "trend_evidence_summary",
}


def _load_analyzer() -> object:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "analyze_decode_technology_trends.py"
    module_name = "decode_technology_trends_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import analyzer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ANALYZER = _load_analyzer()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_release_checksums(release_dir: Path) -> None:
    paths = sorted(
        path
        for path in release_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (release_dir / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(release_dir)}\n"
            for path in paths
        ),
        encoding="utf-8",
    )


class TechnologyTrendAlgorithmTests(unittest.TestCase):
    def test_theil_sen_resists_one_large_outlier(self) -> None:
        fit = ANALYZER._theil_sen_fit(
            [
                (2020.0, 1.0),
                (2021.0, 2.0),
                (2022.0, 4.0),
                (2023.0, 8.0),
                (2024.0, 2.0**40),
            ],
            reference_year=2024.0,
        )
        self.assertEqual(fit["status"], "fit")
        self.assertAlmostEqual(fit["beta_per_year"], 1.0)
        self.assertAlmostEqual(fit["alpha"], 4.0)
        prediction = ANALYZER._predict_fit(
            fit, "positive_log", 2025.0, reference_year=2024.0
        )
        self.assertAlmostEqual(prediction, 32.0)

    def test_theil_sen_rejects_an_all_same_date_sample(self) -> None:
        fit = ANALYZER._theil_sen_fit(
            [(2024.5, 1.0), (2024.5, 2.0), (2024.5, 4.0)]
        )
        self.assertEqual(fit["status"], "insufficient")
        self.assertIsNone(fit["alpha"])
        self.assertIsNone(fit["beta_per_year"])
        self.assertIn("same release time", fit["reason"])

    def test_ridge_logistic_is_finite_under_complete_separation(self) -> None:
        fit = ANALYZER._ridge_logistic_fit(
            [
                (2020.0, 0.0),
                (2021.0, 0.0),
                (2022.0, 0.0),
                (2023.0, 1.0),
                (2024.0, 1.0),
                (2025.0, 1.0),
            ]
        )
        self.assertEqual(fit["status"], "fit")
        self.assertTrue(math.isfinite(fit["alpha"]))
        self.assertTrue(math.isfinite(fit["beta_per_year"]))
        self.assertGreater(fit["beta_per_year"], 0.0)
        early = ANALYZER._predict_fit(fit, "binary", 2020.0)
        late = ANALYZER._predict_fit(fit, "binary", 2025.0)
        self.assertIsNotNone(early)
        self.assertIsNotNone(late)
        self.assertTrue(math.isfinite(early))
        self.assertTrue(math.isfinite(late))
        self.assertGreater(late, early)

    def test_constant_probability_is_a_finite_same_date_fallback(self) -> None:
        points = [(2024.5, 1.0)] * 4
        trend = ANALYZER._ridge_logistic_fit(points)
        fallback = ANALYZER._fit_candidate(points, "binary", "constant")
        self.assertEqual(trend["status"], "insufficient")
        self.assertEqual(fallback["status"], "fit")
        self.assertTrue(math.isfinite(fallback["alpha"]))
        prediction = ANALYZER._predict_fit(
            fallback, "binary", 2030.0
        )
        self.assertAlmostEqual(prediction, 0.9)

    def test_single_class_logistic_trend_is_not_a_finite_fit(self) -> None:
        for observed in (0.0, 1.0):
            with self.subTest(observed=observed):
                fit = ANALYZER._ridge_logistic_fit(
                    [
                        (2020.0, observed),
                        (2021.0, observed),
                        (2022.0, observed),
                        (2023.0, observed),
                    ]
                )
                self.assertIn(
                    fit["status"], {"insufficient", "failed"}
                )
                self.assertIsNone(fit["alpha"])
                self.assertIsNone(fit["beta_per_year"])
                self.assertRegex(
                    str(fit["reason"]).lower(),
                    r"class|variation|constant",
                )

    def test_pure_linear_and_ssm_architectures_need_no_softmax_layers(
        self,
    ) -> None:
        cases = {
            "linear": (
                {
                    "model": {
                        "layer_groups": [
                            {
                                "layers": 6,
                                "mixers": [
                                    {"kind": "linear_attention"},
                                    {"kind": "recurrent_state"},
                                ],
                            }
                        ]
                    }
                },
                (6, 0, 6, 0),
            ),
            "ssm": (
                {
                    "model": {
                        "layer_groups": [
                            {
                                "layers": 8,
                                "mixers": [{"kind": "mamba"}],
                            }
                        ]
                    }
                },
                (8, 0, 0, 8),
            ),
        }
        for label, (config, expected_counts) in cases.items():
            with self.subTest(label=label):
                features = ANALYZER._canonical_architecture(config)
                self.assertEqual(
                    (
                        features["physical_sequence_layer_count"],
                        features["softmax_layer_count"],
                        features["linear_recurrent_layer_count"],
                        features["ssm_layer_count"],
                    ),
                    expected_counts,
                )
                for layout in ANALYZER.KV_LAYOUTS:
                    self.assertEqual(
                        features[f"kv_{layout}_layer_count"], 0
                    )
                    self.assertIsNone(
                        features[f"kv_{layout}_share_of_softmax"]
                    )
                    self.assertEqual(features[f"has_{layout}"], 0)
                for access in ANALYZER.ACCESS_CATEGORIES:
                    self.assertEqual(
                        features[f"access_{access}_layer_count"], 0
                    )
                    self.assertIsNone(
                        features[f"access_{access}_share_of_softmax"]
                    )
                    self.assertEqual(
                        features[f"has_{access}_access"], 0
                    )

    def test_metric_roles_and_conditional_precision_are_explicit(
        self,
    ) -> None:
        roles = {
            spec.metric_id: spec.projection_role
            for spec in ANALYZER.METRIC_SPECS
        }
        for metric_id in (
            "parameters.resident_elements",
            "parameters.active_ratio",
            "context.advertised_max",
        ):
            self.assertEqual(roles[metric_id], "independent_marginal")
        for metric_id in (
            "parameters.active_elements",
            "parameters.sparsity_multiplier",
            "parameters.sparsity_gap_bits",
        ):
            self.assertEqual(roles[metric_id], "derived_identity")
        for metric_id in (
            "context.trained_max",
            "context.evaluated_max",
            "context.deployed_max",
            "context.trained_to_advertised_ratio",
            "context.evaluated_to_advertised_ratio",
            "context.deployed_to_advertised_ratio",
        ):
            self.assertEqual(roles[metric_id], "marginal_diagnostic")
        for metric_ids in ANALYZER.COMPOSITION_GROUPS.values():
            for metric_id in metric_ids:
                self.assertEqual(
                    roles[metric_id], "composition_component"
                )

        profile = {
            "deployment_defaults": {
                "weight_bits": 8,
                "expert_weight_bits": 8,
                "kv_bits": 16,
                "index_bits": 16,
                "state_bits": 16,
            },
            "always_active_weight_groups": [
                {"parameters": 100, "weight_bits": 8}
            ],
            "routed_expert_groups": [],
            "capacity": {
                "decode_profile_weight_capacity_bytes": 100
            },
        }
        pure_state_config = {
            "model": {
                "layer_groups": [
                    {
                        "layers": 4,
                        "mixers": [
                            {
                                "kind": "recurrent_state",
                                "state_elements": 32,
                                "state_bits": 8,
                            }
                        ],
                    }
                ]
            }
        }
        precision = ANALYZER._precision_features(
            profile, pure_state_config, 100
        )
        self.assertEqual(json.loads(precision["kv_bits_used_values"]), [])
        self.assertEqual(
            json.loads(precision["index_bits_used_values"]), []
        )
        self.assertEqual(
            json.loads(precision["state_bits_used_values"]), [8.0]
        )
        self.assertIsNone(precision["kv_effective_bits_used"])
        self.assertIsNone(precision["index_effective_bits_used"])
        self.assertAlmostEqual(
            precision["state_effective_bits_used"], 8.0
        )


class DecodeTechnologyTrendEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.runner = cls.root / "scripts" / "run_decode_trend.py"
        cls.analyzer = (
            cls.root / "scripts" / "analyze_decode_technology_trends.py"
        )
        cls.class_tempdir = tempfile.TemporaryDirectory(
            prefix="decode-technology-trends-fixture-"
        )
        cls.class_temp_path = Path(cls.class_tempdir.name)
        cls.release_dir = cls.class_temp_path / "release"
        cls.environment = os.environ.copy()
        cls.environment["PYTHONDONTWRITEBYTECODE"] = "1"
        cls.environment["MPLCONFIGDIR"] = str(
            cls.class_temp_path / "mplconfig"
        )
        cls._build_release()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.class_tempdir.cleanup()

    def setUp(self) -> None:
        self.test_tempdir = tempfile.TemporaryDirectory(
            prefix="case-", dir=self.class_temp_path
        )
        self.output_dir = Path(self.test_tempdir.name) / "analysis"

    def tearDown(self) -> None:
        self.test_tempdir.cleanup()

    @classmethod
    def _checked_subprocess(
        cls, arguments: list[str], *, timeout: int = 180
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            arguments,
            cwd=cls.root,
            env=cls.environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "command failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return completed

    @classmethod
    def _build_release(cls) -> None:
        data_dir = cls.release_dir / "data"
        arguments = [
            sys.executable,
            "-B",
            str(cls.runner),
        ]
        for model_id in MODEL_IDS:
            arguments.extend(["--model", model_id])
        arguments.extend(
            [
                "--contexts",
                "128",
                "--batches",
                "1",
                "--output-dir",
                str(data_dir),
            ]
        )
        cls._checked_subprocess(arguments)
        run_manifest = json.loads(
            (data_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        validation = json.loads(
            (data_dir / "validation_report.json").read_text(
                encoding="utf-8"
            )
        )
        if validation["status"] != "pass":
            raise RuntimeError(f"temporary release did not validate: {validation}")
        release_manifest = {
            "dataset_release_schema_version": 1,
            "dataset_version": "p9a-test-v1",
            "study_id": run_manifest["study_id"],
            "model_count": validation["model_count"],
            "row_count": validation["row_count"],
            "source_repository": {
                "git_commit": run_manifest["git_commit"] or "test",
                "git_worktree_dirty": run_manifest["git_worktree_dirty"],
            },
            "frozen_inputs": {
                "model_manifest": (
                    "source_snapshot/studies/decode_trend/models.json"
                ),
                "mechanism_audit": (
                    "source_snapshot/studies/decode_trend/"
                    "mechanism_audit.json"
                ),
                "runner": "source_snapshot/scripts/run_decode_trend.py",
                "engine": "source_snapshot/decode_engine/",
            },
            "canonical_data": {
                "csv": "data/decode_results.csv",
                "jsonl": "data/decode_results.jsonl",
                "model_profiles": "data/model_profiles.jsonl",
                "run_manifest": "data/run_manifest.json",
                "validation_report": "data/validation_report.json",
            },
        }
        cls.release_dir.mkdir(parents=True, exist_ok=True)
        (cls.release_dir / "release_manifest.json").write_text(
            json.dumps(release_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        profiles = [
            json.loads(line)
            for line in (data_dir / "model_profiles.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for profile in profiles:
            relative_config = Path(profile["config_path"])
            source = cls.root / relative_config
            target = (
                cls.release_dir / "source_snapshot" / relative_config
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        _write_release_checksums(cls.release_dir)

    def _run_analysis(
        self, *, plots: bool = False, timeout: int = 180
    ) -> subprocess.CompletedProcess[str]:
        arguments = [
            sys.executable,
            "-B",
            str(self.analyzer),
            "--release-dir",
            str(self.release_dir),
            "--output-dir",
            str(self.output_dir),
            "--projection-through-year",
            "2030",
        ]
        if plots:
            arguments.extend(["--dpi", "40"])
        else:
            arguments.append("--no-plots")
        return self._checked_subprocess(arguments, timeout=timeout)

    def _run_analysis_for_release(
        self,
        release_dir: Path,
        output_dir: Path,
        *,
        timeout: int = 180,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(self.analyzer),
                "--release-dir",
                str(release_dir),
                "--output-dir",
                str(output_dir),
                "--no-plots",
            ],
            cwd=self.root,
            env=self.environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _copy_release(self, label: str) -> Path:
        target = Path(self.test_tempdir.name) / label
        shutil.copytree(self.release_dir, target)
        return target

    def test_release_level_analysis_manifest_and_output_isolation(
        self,
    ) -> None:
        source_hashes_before = {
            str(path.relative_to(self.release_dir)): _sha256(path)
            for path in self.release_dir.rglob("*")
            if path.is_file()
        }
        completed = self._run_analysis()
        self.assertIn(
            f"analyzed {len(MODEL_IDS)} release observations",
            completed.stdout,
        )
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()},
            EXPECTED_TABLE_FILES,
        )
        self.assertFalse((self.output_dir / "figures").exists())
        source_hashes_after = {
            str(path.relative_to(self.release_dir)): _sha256(path)
            for path in self.release_dir.rglob("*")
            if path.is_file()
        }
        self.assertEqual(source_hashes_after, source_hashes_before)

        manifest = json.loads(
            (self.output_dir / "analysis_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["analysis_manifest_schema_version"], 2)
        metric_count = len(ANALYZER.METRIC_SPECS)
        self.assertEqual(manifest["source_release_dir"], str(self.release_dir))
        self.assertEqual(manifest["source_dataset_version"], "p9a-test-v1")
        self.assertEqual(
            manifest["statistical_unit"],
            "one model_release_id + deployment_profile_id",
        )
        self.assertFalse(manifest["decode_result_grid_used_as_fit_observations"])
        self.assertEqual(
            manifest["row_counts"]["technology_observations"],
            len(MODEL_IDS),
        )
        self.assertEqual(
            manifest["row_counts"]["selected_trend_functions"],
            metric_count,
        )
        self.assertEqual(
            manifest["row_counts"]["trend_candidates"], 2 * metric_count
        )
        self.assertEqual(
            manifest["row_counts"]["trend_projection_grid"],
            9 * metric_count,
        )
        self.assertEqual(
            manifest["row_counts"]["technology_milestones"],
            len(ANALYZER.BINARY_TECHNOLOGIES),
        )
        self.assertEqual(
            manifest["row_counts"]["composition_group_summary"],
            len(ANALYZER.COMPOSITION_GROUPS),
        )
        self.assertEqual(
            manifest["row_counts"]["technology_cooccurrence"],
            math.comb(len(ANALYZER.BINARY_TECHNOLOGIES), 2),
        )
        self.assertEqual(
            set(manifest["artifacts"]["tables"]),
            EXPECTED_TABLE_FILES - {"analysis_manifest.json"},
        )
        self.assertEqual(manifest["artifacts"]["figures"], [])
        expected_hashed_artifacts = (
            EXPECTED_TABLE_FILES - {"analysis_manifest.json"}
        )
        self.assertEqual(
            set(manifest["artifacts"]["sha256"]),
            expected_hashed_artifacts,
        )
        for relative_path, expected_hash in manifest["artifacts"][
            "sha256"
        ].items():
            self.assertEqual(
                _sha256(self.output_dir / relative_path), expected_hash
            )
        for relative_path, expected_hash in manifest["input_sha256"].items():
            self.assertEqual(
                _sha256(self.release_dir / relative_path), expected_hash
            )

        quality = json.loads(
            (self.output_dir / "quality_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(quality["status"], "pass")
        self.assertEqual(
            quality["release_observation_count"], len(MODEL_IDS)
        )
        self.assertEqual(
            quality["year_counts"],
            {"2022": 3, "2023": 5, "2024": 5, "2025": 4, "2026": 3},
        )
        self.assertTrue(
            quality["curated_sample_prevalence_not_industry_adoption"]
        )

        functions = json.loads(
            (self.output_dir / "selected_trend_functions.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            functions["technology_trend_function_schema_version"], 2
        )
        self.assertEqual(len(functions["functions"]), metric_count)
        self.assertEqual(
            set(functions["composition_groups"]),
            set(ANALYZER.COMPOSITION_GROUPS),
        )
        for group_name, metric_ids in ANALYZER.COMPOSITION_GROUPS.items():
            metadata = functions["composition_groups"][group_name]
            self.assertEqual(metadata["metric_ids"], list(metric_ids))
            self.assertEqual(
                metadata["projection_formula"],
                "p_k(t)=sigmoid(alpha_k+beta_k*(t-t0))/"
                "sum_j(sigmoid(alpha_j+beta_j*(t-t0)))",
            )

    def test_source_snapshot_config_is_authoritative(self) -> None:
        release_dir = self._copy_release("snapshot-authority-release")
        output_dir = Path(self.test_tempdir.name) / "snapshot-authority-output"
        profiles_path = release_dir / "data" / "model_profiles.jsonl"
        profiles = [
            json.loads(line)
            for line in profiles_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        palm_profile = next(
            profile
            for profile in profiles
            if profile["model_release_id"]
            == "google:palm-540b:2022-04-05"
        )
        snapshot_config = (
            release_dir
            / "source_snapshot"
            / str(palm_profile["config_path"])
        )
        config = json.loads(snapshot_config.read_text(encoding="utf-8"))
        config["model"]["metadata"][
            "p9a_snapshot_authority_test_marker"
        ] = True
        snapshot_config.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        snapshot_hash = _sha256(snapshot_config)
        palm_profile["config_sha256"] = snapshot_hash
        profiles_path.write_text(
            "".join(
                json.dumps(profile, ensure_ascii=False) + "\n"
                for profile in profiles
            ),
            encoding="utf-8",
        )

        results_csv = release_dir / "data" / "decode_results.csv"
        result_rows = _read_csv(results_csv)
        fieldnames = list(result_rows[0])
        for row in result_rows:
            if (
                row["model_release_id"]
                == "google:palm-540b:2022-04-05"
            ):
                row["config_sha256"] = snapshot_hash
        with results_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(result_rows)

        results_jsonl = release_dir / "data" / "decode_results.jsonl"
        json_records = [
            json.loads(line)
            for line in results_jsonl.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        for record in json_records:
            if (
                record["model_release_id"]
                == "google:palm-540b:2022-04-05"
            ):
                record["config_sha256"] = snapshot_hash
        results_jsonl.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in json_records
            ),
            encoding="utf-8",
        )
        _write_release_checksums(release_dir)

        completed = self._run_analysis_for_release(
            release_dir, output_dir
        )
        self.assertEqual(
            completed.returncode,
            0,
            "the frozen source_snapshot config must be authoritative:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )
        manifest = json.loads(
            (output_dir / "analysis_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn(
            snapshot_hash,
            set(manifest["verified_config_sha256"].values()),
        )

    def test_snapshot_or_checksum_tampering_is_rejected(self) -> None:
        snapshot_tampered = self._copy_release(
            "snapshot-tampered-release"
        )
        snapshot_path = next(
            (
                snapshot_tampered / "source_snapshot" / "configs"
            ).rglob("*.json")
        )
        snapshot_path.write_text(
            snapshot_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        snapshot_result = self._run_analysis_for_release(
            snapshot_tampered,
            Path(self.test_tempdir.name) / "snapshot-tampered-output",
        )
        self.assertEqual(snapshot_result.returncode, 2)
        self.assertRegex(
            snapshot_result.stderr.lower(), r"sha-?256|checksum"
        )

        checksum_tampered = self._copy_release(
            "checksum-tampered-release"
        )
        sums_path = checksum_tampered / "SHA256SUMS"
        lines = sums_path.read_text(encoding="utf-8").splitlines()
        original_digest, separator, relative_path = lines[0].partition("  ")
        self.assertEqual(len(original_digest), 64)
        lines[0] = (
            ("0" if original_digest[0] != "0" else "1")
            + original_digest[1:]
            + separator
            + relative_path
        )
        sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        checksum_result = self._run_analysis_for_release(
            checksum_tampered,
            Path(self.test_tempdir.name) / "checksum-tampered-output",
        )
        self.assertEqual(checksum_result.returncode, 2)
        self.assertRegex(
            checksum_result.stderr.lower(), r"sha-?256|checksum"
        )

    def test_physical_layers_context_moe_and_precision_anchors(
        self,
    ) -> None:
        self._run_analysis()
        observations = {
            row["model_release_id"]: row
            for row in _read_csv(
                self.output_dir / "technology_observations.csv"
            )
        }
        self.assertEqual(set(observations), set(MODEL_IDS))

        for row in observations.values():
            physical = int(row["physical_sequence_layer_count"])
            broad_mixer_total = sum(
                int(row[field])
                for field in (
                    "softmax_layer_count",
                    "linear_recurrent_layer_count",
                    "ssm_layer_count",
                )
            )
            self.assertEqual(physical, broad_mixer_total)
            self.assertTrue(
                math.isclose(
                    sum(
                        float(row[field])
                        for field in (
                            "softmax_layer_share",
                            "linear_recurrent_layer_share",
                            "ssm_layer_share",
                        )
                    ),
                    1.0,
                    rel_tol=1e-12,
                )
            )
            softmax = int(row["softmax_layer_count"])
            self.assertEqual(
                softmax,
                sum(
                    int(row[f"kv_{layout}_layer_count"])
                    for layout in ANALYZER.KV_LAYOUTS
                ),
            )
            self.assertEqual(
                softmax,
                sum(
                    int(row[f"access_{access}_layer_count"])
                    for access in ANALYZER.ACCESS_CATEGORIES
                ),
            )

        kimi_linear = observations[
            "moonshotai:kimi-linear-48b-a3b-instruct:2025-10-30"
        ]
        self.assertEqual(
            (
                int(kimi_linear["physical_sequence_layer_count"]),
                int(kimi_linear["softmax_layer_count"]),
                int(kimi_linear["linear_recurrent_layer_count"]),
                int(kimi_linear["recurrent_layer_count"]),
            ),
            (27, 7, 20, 20),
        )
        qwen36 = observations[
            "qwen:qwen3.6-35b-a3b-fp8:2026-04-15"
        ]
        self.assertEqual(
            (
                int(qwen36["physical_sequence_layer_count"]),
                int(qwen36["softmax_layer_count"]),
                int(qwen36["linear_recurrent_layer_count"]),
                int(qwen36["linear_attention_layer_count"]),
                int(qwen36["recurrent_layer_count"]),
            ),
            (40, 10, 30, 30, 0),
        )
        jamba = observations[
            "ai21labs:jamba-1.5-large:2024-08-22"
        ]
        self.assertEqual(
            (
                int(jamba["physical_sequence_layer_count"]),
                int(jamba["softmax_layer_count"]),
                int(jamba["ssm_layer_count"]),
                int(jamba["mamba_layer_count"]),
            ),
            (72, 9, 63, 63),
        )
        for model_id in (
            "moonshotai:kimi-k2-thinking:2025-11-06",
            "moonshotai:kimi-k2.6:2026-04-14",
        ):
            row = observations[model_id]
            self.assertEqual(int(row["physical_sequence_layer_count"]), 61)
            self.assertEqual(int(row["kv_mla_layer_count"]), 61)

        mixtral = observations[
            "mistralai:mixtral-8x7b-instruct-v0.1:2023-12-11"
        ]
        self.assertEqual(int(mixtral["has_moe"]), 1)
        self.assertEqual(int(mixtral["moe_layer_count"]), 32)
        self.assertAlmostEqual(float(mixtral["moe_layer_share"]), 1.0)
        self.assertEqual(int(mixtral["moe_expert_count"]), 8)
        self.assertEqual(int(mixtral["moe_selected_per_token"]), 2)
        self.assertAlmostEqual(float(mixtral["moe_routing_density"]), 0.25)
        self.assertEqual(int(jamba["moe_layer_count"]), 36)
        self.assertAlmostEqual(float(jamba["moe_layer_share"]), 0.5)

        llama4 = observations[
            "meta-llama:llama-4-scout-17b-16e-instruct:2025-04-05"
        ]
        self.assertEqual(
            int(llama4["advertised_max_context_tokens_at_release"]),
            10_485_760,
        )
        self.assertEqual(int(llama4["trained_max_context_tokens"]), 262_144)
        self.assertEqual(int(llama4["evaluated_max_context_tokens"]), 131_072)
        self.assertEqual(int(llama4["deployed_max_context_tokens"]), 3_600_000)
        self.assertAlmostEqual(
            float(llama4["evaluated_to_advertised_context_ratio"]),
            0.0125,
        )
        self.assertAlmostEqual(
            float(llama4["deployed_to_advertised_context_ratio"]),
            3_600_000 / 10_485_760,
        )
        self.assertEqual(jamba["trained_max_context_tokens"], "")

        palm = observations["google:palm-540b:2022-04-05"]
        self.assertAlmostEqual(
            float(palm["matrix_effective_weight_bits"]), 8.0
        )
        self.assertAlmostEqual(
            float(palm["explicit_weight_parameter_share_le8"]), 1.0
        )
        self.assertEqual(int(palm["has_majority_explicit_weight_le4"]), 0)
        self.assertAlmostEqual(
            float(kimi_linear["matrix_effective_weight_bits"]), 16.0
        )
        self.assertEqual(
            int(kimi_linear["has_majority_explicit_weight_le8"]), 0
        )
        qwen3 = observations["qwen:qwen3-32b-awq:2025-04-29"]
        self.assertGreater(
            float(qwen3["explicit_weight_parameter_share_le4"]), 0.9
        )
        self.assertEqual(
            int(qwen3["has_majority_explicit_weight_le4"]), 1
        )
        self.assertGreater(
            float(qwen36["matrix_effective_weight_bits"]), 8.0
        )
        self.assertLess(
            float(qwen36["matrix_effective_weight_bits"]), 9.0
        )
        self.assertEqual(
            int(qwen36["has_majority_explicit_weight_le8"]), 1
        )
        self.assertEqual(
            int(qwen36["has_majority_explicit_weight_le4"]), 0
        )

    def test_derived_identities_and_conditional_metrics_close(
        self,
    ) -> None:
        self._run_analysis()
        observation_rows = _read_csv(
            self.output_dir / "technology_observations.csv"
        )
        observations = {
            row["model_release_id"]: row for row in observation_rows
        }
        selected_rows = {
            row["metric_id"]: row
            for row in _read_csv(
                self.output_dir / "selected_trend_functions.csv"
            )
        }
        composition_summaries = {
            row["composition_group"]: row
            for row in _read_csv(
                self.output_dir / "composition_group_summary.csv"
            )
        }
        self.assertEqual(
            set(composition_summaries), set(ANALYZER.COMPOSITION_GROUPS)
        )
        kv_summary = composition_summaries["kv_layout"]
        self.assertEqual(kv_summary["selected_candidate_id"], "trend")
        self.assertAlmostEqual(
            float(kv_summary["trend_relative_cv_improvement"]),
            0.1356,
            places=3,
        )
        self.assertEqual(int(kv_summary["valid_constant_fold_count"]), 2)
        self.assertEqual(int(kv_summary["valid_trend_fold_count"]), 2)
        self.assertEqual(
            kv_summary["applied_formula"],
            "p_k(t)=sigmoid(alpha_k+beta_k*(t-t0))/"
            "sum_j(sigmoid(alpha_j+beta_j*(t-t0)))",
        )
        kv_deltas: list[float] = []
        for metric_id in ANALYZER.COMPOSITION_GROUPS["kv_layout"]:
            row = selected_rows[metric_id]
            self.assertEqual(row["selected_candidate_id"], "trend")
            self.assertEqual(
                row["composition_group_evidence_grade"],
                kv_summary["evidence_grade"],
            )
            self.assertAlmostEqual(
                float(
                    row[
                        "composition_group_trend_relative_cv_improvement"
                    ]
                ),
                float(kv_summary["trend_relative_cv_improvement"]),
            )
            kv_deltas.append(
                float(row["composition_applied_delta_training_window"])
            )
        self.assertTrue(
            math.isclose(
                sum(kv_deltas), 0.0, rel_tol=0.0, abs_tol=1e-12
            )
        )
        self.assertGreater(
            float(
                selected_rows["kv_layout.mla_share"][
                    "composition_applied_delta_training_window"
                ]
            ),
            0.0,
        )
        for group_name in ("mixer", "attention_access"):
            summary = composition_summaries[group_name]
            self.assertEqual(summary["selected_candidate_id"], "constant")
            for metric_id in ANALYZER.COMPOSITION_GROUPS[group_name]:
                row = selected_rows[metric_id]
                self.assertEqual(row["selected_candidate_id"], "constant")
                self.assertAlmostEqual(
                    float(
                        row[
                            "composition_applied_delta_training_window"
                        ]
                    ),
                    0.0,
                )

        candidates = {
            (row["metric_id"], row["candidate_id"]): row
            for row in _read_csv(
                self.output_dir / "trend_candidates.csv"
            )
        }
        mla_presence = selected_rows["technology.mla_presence"]
        mla_trend = candidates[
            ("technology.mla_presence", "trend")
        ]
        self.assertEqual(
            mla_presence["selected_candidate_id"], "constant"
        )
        self.assertEqual(mla_presence["evidence_grade"], "insufficient")
        self.assertIn(
            "fewer than two valid complete-year",
            mla_presence["selection_reason"],
        )
        self.assertEqual(
            int(mla_trend["rolling_complete_year_fold_count"]), 1
        )
        self.assertEqual(mla_trend["rolling_complete_year_score"], "")

        for spec in ANALYZER.METRIC_SPECS:
            self.assertEqual(
                selected_rows[spec.metric_id]["projection_role"],
                spec.projection_role,
            )

        moe_count = sum(
            int(row["has_moe"]) for row in observation_rows
        )
        self.assertEqual(
            int(
                selected_rows["moe.unconditional_layer_share"][
                    "observation_count"
                ]
            ),
            len(observation_rows),
        )
        self.assertEqual(
            int(
                selected_rows["moe.layer_share_given_moe"][
                    "observation_count"
                ]
            ),
            moe_count,
        )
        palm = observations["google:palm-540b:2022-04-05"]
        self.assertAlmostEqual(float(palm["moe_layer_share"]), 0.0)
        self.assertEqual(palm["moe_layer_share_given_moe"], "")
        mixtral = observations[
            "mistralai:mixtral-8x7b-instruct-v0.1:2023-12-11"
        ]
        self.assertAlmostEqual(
            float(mixtral["moe_layer_share_given_moe"]), 1.0
        )
        jamba = observations[
            "ai21labs:jamba-1.5-large:2024-08-22"
        ]
        self.assertAlmostEqual(
            float(jamba["moe_layer_share_given_moe"]), 0.5
        )

        glm52 = observations["zai-org:glm-5.2-fp8:2026-06-16"]
        self.assertAlmostEqual(
            float(glm52["kv_effective_bits_used"]), 8.0
        )
        self.assertAlmostEqual(
            float(glm52["index_effective_bits_used"]), 16.0
        )
        self.assertEqual(glm52["state_effective_bits_used"], "")
        kimi_linear = observations[
            "moonshotai:kimi-linear-48b-a3b-instruct:2025-10-30"
        ]
        expected_kimi_state_bits = (
            524_288 * 32 + 36_864 * 16
        ) / (524_288 + 36_864)
        self.assertAlmostEqual(
            float(kimi_linear["state_effective_bits_used"]),
            expected_kimi_state_bits,
        )
        self.assertEqual(kimi_linear["index_effective_bits_used"], "")
        self.assertEqual(
            int(
                selected_rows["precision.index_effective_bits"][
                    "observation_count"
                ]
            ),
            1,
        )
        self.assertEqual(
            int(
                selected_rows["precision.state_effective_bits"][
                    "observation_count"
                ]
            ),
            3,
        )

        projection_rows = _read_csv(
            self.output_dir / "trend_projection_grid.csv"
        )
        by_year: dict[int, dict[str, dict[str, str]]] = {}
        for row in projection_rows:
            by_year.setdefault(int(row["year"]), {})[
                row["metric_id"]
            ] = row
        for year, metrics in by_year.items():
            values = {
                metric_id: float(row["predicted_value"])
                for metric_id, row in metrics.items()
                if row["predicted_value"] != ""
            }
            with self.subTest(year=year, identity="parameters"):
                self.assertTrue(
                    math.isclose(
                        values["parameters.active_elements"],
                        values["parameters.resident_elements"]
                        * values["parameters.active_ratio"],
                        rel_tol=1e-12,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        values["parameters.sparsity_multiplier"],
                        1.0 / values["parameters.active_ratio"],
                        rel_tol=1e-12,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        values["parameters.sparsity_gap_bits"],
                        -math.log2(values["parameters.active_ratio"]),
                        rel_tol=1e-12,
                    )
                )
            for group_name, metric_ids in (
                ANALYZER.COMPOSITION_GROUPS.items()
            ):
                with self.subTest(
                    year=year, composition_group=group_name
                ):
                    self.assertTrue(
                        math.isclose(
                            sum(values[metric_id] for metric_id in metric_ids),
                            1.0,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    )

    def test_output_directory_inside_frozen_release_is_rejected(
        self,
    ) -> None:
        nested_output = self.release_dir / "nested-analysis-output"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(self.analyzer),
                "--release-dir",
                str(self.release_dir),
                "--output-dir",
                str(nested_output),
                "--no-plots",
            ],
            cwd=self.root,
            env=self.environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "analysis output directory must not be inside the frozen release",
            completed.stderr,
        )
        self.assertFalse(nested_output.exists())

    @unittest.skipUnless(
        MATPLOTLIB_AVAILABLE,
        "matplotlib is optional; install requirements-plot.txt to test plots",
    )
    def test_analysis_renders_ten_png_and_svg_figure_pairs(self) -> None:
        self._run_analysis(plots=True, timeout=240)
        figure_dir = self.output_dir / "figures"
        png_paths = sorted(figure_dir.glob("*.png"))
        svg_paths = sorted(figure_dir.glob("*.svg"))
        self.assertEqual(
            {path.stem for path in png_paths}, EXPECTED_FIGURE_STEMS
        )
        self.assertEqual(
            {path.stem for path in svg_paths}, EXPECTED_FIGURE_STEMS
        )
        self.assertEqual(len(png_paths), 10)
        self.assertEqual(len(svg_paths), 10)
        manifest = json.loads(
            (self.output_dir / "analysis_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["analysis_manifest_schema_version"], 2)
        expected_artifacts = {
            f"figures/{stem}.{suffix}"
            for stem in EXPECTED_FIGURE_STEMS
            for suffix in ("png", "svg")
        }
        self.assertEqual(
            set(manifest["artifacts"]["figures"]), expected_artifacts
        )
        expected_hashed_artifacts = (
            EXPECTED_TABLE_FILES - {"analysis_manifest.json"}
        ) | expected_artifacts
        self.assertEqual(
            set(manifest["artifacts"]["sha256"]),
            expected_hashed_artifacts,
        )
        for relative_path, expected_hash in manifest["artifacts"][
            "sha256"
        ].items():
            self.assertEqual(
                _sha256(self.output_dir / relative_path), expected_hash
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
