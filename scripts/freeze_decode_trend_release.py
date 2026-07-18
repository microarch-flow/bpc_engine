#!/usr/bin/env python3
"""Freeze a self-contained, checksummed Decode trend dataset release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "studies" / "decode_trend" / "models.json"
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "studies" / "decode_trend" / "releases"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an immutable, self-contained Decode trend release."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release version, for example v1.0.0.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
    )
    return parser


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: Sequence[str]) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _snapshot_sources(
    manifest_path: Path,
    release_dir: Path,
) -> Path:
    snapshot_root = release_dir / "source_snapshot"
    manifest = _load_object(manifest_path)
    audit_path = (
        manifest_path.parent / str(manifest["mechanism_audit_path"])
    ).resolve()

    snapshot_manifest = (
        snapshot_root / "studies" / "decode_trend" / "models.json"
    )
    _copy(manifest_path, snapshot_manifest)
    _copy(
        audit_path,
        snapshot_root
        / "studies"
        / "decode_trend"
        / "mechanism_audit.json",
    )
    for entry in manifest["models"]:
        config_path = (
            manifest_path.parent / str(entry["config_path"])
        ).resolve()
        relative = config_path.relative_to(PROJECT_ROOT)
        _copy(config_path, snapshot_root / relative)

    for source in sorted((PROJECT_ROOT / "decode_engine").glob("*.py")):
        _copy(source, snapshot_root / "decode_engine" / source.name)
    _copy(
        PROJECT_ROOT / "scripts" / "run_decode_trend.py",
        snapshot_root / "scripts" / "run_decode_trend.py",
    )
    for name in (
        "decode_trend_data_dictionary.md",
        "decode_trend_metrics.md",
        "decode_trend_p3_mechanism_audit.md",
    ):
        _copy(
            PROJECT_ROOT / "docs" / name,
            snapshot_root / "docs" / name,
        )
    return snapshot_manifest


def _write_release_readme(
    path: Path,
    *,
    version: str,
    model_count: int,
    row_count: int,
) -> None:
    path.write_text(
        f"""# Decode Trend Dataset {version}

这是2022–2026YTD代表模型的正式冻结数据版本。

- 模型数：{model_count}
- 数据行数：{row_count}
- 阶段：Decode
- 数据网格：公共 C 点、模型/机制锚点和主 B 点
- `data/`：正式结果
- `source_snapshot/`：生成该结果的配置、事实、P3审计、引擎和运行器快照
- `source_snapshot/docs/`：字段字典、指标合同和P3人工审计快照
- `release_manifest.json`：版本、来源与数据摘要
- `SHA256SUMS`：除自身之外所有文件的 SHA-256

复算：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B \\
  source_snapshot/scripts/run_decode_trend.py \\
  --manifest source_snapshot/studies/decode_trend/models.json \\
  --output-dir /tmp/decode-trend-{version}-reproduced
```

复算结果中的时间、run_id 和 Git 状态可能不同；数值字段应一致。
""",
        encoding="utf-8",
    )


def _write_checksums(release_dir: Path) -> None:
    paths = sorted(
        path
        for path in release_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(release_dir)}"
        for path in paths
    ]
    (release_dir / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.version.strip() or "/" in args.version:
        print("error: --version must be a non-empty path-safe label", file=sys.stderr)
        return 2

    manifest_path = args.manifest.resolve()
    release_dir = args.release_root.resolve() / args.version
    if release_dir.exists():
        print(
            f"error: frozen release already exists: {release_dir}",
            file=sys.stderr,
        )
        return 2

    try:
        release_dir.mkdir(parents=True)
        snapshot_manifest = _snapshot_sources(manifest_path, release_dir)
        data_dir = release_dir / "data"
        runner = (
            release_dir
            / "source_snapshot"
            / "scripts"
            / "run_decode_trend.py"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(runner),
                "--manifest",
                str(snapshot_manifest),
                "--output-dir",
                str(data_dir),
            ],
            cwd=release_dir / "source_snapshot",
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

        validation = _load_object(data_dir / "validation_report.json")
        if validation.get("status") != "pass":
            raise RuntimeError("generated validation report did not pass")
        with (data_dir / "decode_results.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
        model_count = int(validation["model_count"])
        if row_count != int(validation["row_count"]):
            raise RuntimeError("CSV and validation row counts disagree")

        created_at = datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z")
        source_manifest = _load_object(manifest_path)
        release_manifest = {
            "dataset_release_schema_version": 1,
            "dataset_version": args.version,
            "study_id": source_manifest["study_id"],
            "created_at_utc": created_at,
            "phase": "decode",
            "model_count": model_count,
            "row_count": row_count,
            "source_repository": {
                "git_commit": _git_value(("rev-parse", "HEAD")),
                "git_worktree_dirty": bool(_git_value(("status", "--short"))),
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
                "data_dictionary": (
                    "source_snapshot/docs/decode_trend_data_dictionary.md"
                ),
                "metric_contract": (
                    "source_snapshot/docs/decode_trend_metrics.md"
                ),
                "p3_human_audit": (
                    "source_snapshot/docs/"
                    "decode_trend_p3_mechanism_audit.md"
                ),
            },
            "canonical_data": {
                "csv": "data/decode_results.csv",
                "jsonl": "data/decode_results.jsonl",
                "model_profiles": "data/model_profiles.jsonl",
                "run_manifest": "data/run_manifest.json",
                "validation_report": "data/validation_report.json",
            },
            "validation": validation,
        }
        (release_dir / "release_manifest.json").write_text(
            json.dumps(
                release_manifest,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_release_readme(
            release_dir / "README.md",
            version=args.version,
            model_count=model_count,
            row_count=row_count,
        )
        _write_checksums(release_dir)
    except Exception as exc:
        shutil.rmtree(release_dir, ignore_errors=True)
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"froze {row_count} rows for {model_count} models at {release_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
