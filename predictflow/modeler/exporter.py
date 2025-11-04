"""
PredictFlow Workflow Exporter (Secure & Functional)
---------------------------------------------------

Purpose
-------
Safely export PredictFlow workflow definitions to disk or in-memory strings.

Security & Resilience Features
------------------------------
✅ Strict schema validation & size caps (prevents DoS via huge inputs)
✅ Sanitization of filenames (blocks traversal & weird chars)
✅ Secret redaction for known-sensitive keys
✅ Atomic writes to avoid partial/corrupt files
✅ Base directory jail: all writes confined to configured export root
✅ Overwrite protection disabled by default
✅ Optional ZIP bundle with JSON, CSV, and metadata
✅ Optional YAML support via PyYAML (safe_dump)

Usage
-----
    exporter = WorkflowExporter()  # defaults to ./exports
    path = exporter.export(workflow_dict, filename="invoice_flow.json", fmt="json")

    # In-memory strings:
    s = exporter.to_json_str(workflow_dict)
    y = exporter.to_yaml_str(workflow_dict)  # requires PyYAML

    # Bundle (zip) including JSON + steps.csv + metadata.json
    zpath = exporter.export_bundle(workflow_dict, filename="invoice_flow_bundle.zip")
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ----------------------------
# Tunables / Safety Limits
# ----------------------------
DEFAULT_EXPORT_DIR = Path.cwd() / "exports"
MAX_STEPS = 1000
MAX_NAME_LEN = 80
MAX_TOTAL_BYTES = 2_000_000  # ~2MB (post-serialization cap for safety)
ALLOWED_FMTS = {"json", "yaml", "yml", "zip"}

# Keys to redact during export (case-insensitive substring match)
SECRET_KEY_SUBSTRINGS = (
    "password", "secret", "token", "api_key", "apikey", "access_key", "private_key",
)


@dataclass(frozen=True)
class ExportStats:
    steps: int
    redacted_keys: int
    bytes_written: int


class WorkflowExporter:
    """
    Secure exporter for PredictFlow workflows.

    All file writes are sandboxed under `base_dir`. Paths are validated and
    writes are atomic. Overwrites are disabled by default.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None, *, allow_overwrite: bool = False):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_EXPORT_DIR
        self.base_dir = self.base_dir.resolve()
        self.allow_overwrite = bool(allow_overwrite)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("WorkflowExporter initialized under %s (overwrite=%s)", self.base_dir, self.allow_overwrite)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def export(self, workflow: Dict[str, Any], *, filename: Optional[str] = None, fmt: str = "json") -> Path:
        """
        Export a workflow to disk in the requested format (json|yaml|zip).

        Returns: Path to the written file.
        Raises: ValueError on invalid input/path or if size limits exceeded.
        """
        fmt = (fmt or "json").lower().strip()
        if fmt == "yml":
            fmt = "yaml"
        if fmt not in ALLOWED_FMTS:
            raise ValueError(f"Unsupported format '{fmt}'. Allowed: {sorted(ALLOWED_FMTS)}")

        wf = self._validate_and_sanitize_workflow(workflow)
        safe_filename = self._resolve_filename(filename, wf, fmt)
        out_path = self._safe_path(safe_filename)

        if out_path.exists() and not self.allow_overwrite:
            raise FileExistsError(f"File already exists: {out_path}")

        if fmt == "json":
            data = self._serialize_json(wf)
            self._ensure_size_within_limits(data)
            self._atomic_write(out_path, data)
            return out_path

        if fmt == "yaml":
            data = self._serialize_yaml(wf)
            self._ensure_size_within_limits(data)
            self._atomic_write(out_path, data)
            return out_path

        # ZIP bundle (json + csv + metadata)
        if fmt == "zip":
            return self._export_bundle_zip(wf, out_path)

        # not reachable
        raise AssertionError("Unhandled format")

    def export_bundle(self, workflow: Dict[str, Any], *, filename: Optional[str] = None) -> Path:
        """
        Convenience wrapper to export a standard ZIP bundle.
        """
        wf = self._validate_and_sanitize_workflow(workflow)
        safe_filename = self._resolve_filename(filename, wf, "zip")
        out_path = self._safe_path(safe_filename)
        if out_path.exists() and not self.allow_overwrite:
            raise FileExistsError(f"File already exists: {out_path}")
        return self._export_bundle_zip(wf, out_path)

    def to_json_str(self, workflow: Dict[str, Any]) -> str:
        """Serialize workflow to a JSON string with redaction and validation."""
        wf = self._validate_and_sanitize_workflow(workflow)
        data = self._serialize_json(wf)
        self._ensure_size_within_limits(data)
        return data.decode("utf-8")

    def to_yaml_str(self, workflow: Dict[str, Any]) -> str:
        """Serialize workflow to a YAML string with redaction and validation (requires PyYAML)."""
        wf = self._validate_and_sanitize_workflow(workflow)
        data = self._serialize_yaml(wf)
        self._ensure_size_within_limits(data)
        return data.decode("utf-8")

    # ---------------------------------------------------------------------
    # Validation & Sanitization
    # ---------------------------------------------------------------------
    def _validate_and_sanitize_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(workflow, dict):
            raise ValueError("Workflow must be a dictionary.")

        wf = dict(workflow)  # shallow copy
        wf_name = str(wf.get("name") or "unnamed").strip()
        if not wf_name:
            wf_name = "unnamed"
        wf["name"] = self._sanitize_name(wf_name)

        steps = wf.get("steps")
        if steps is None or not isinstance(steps, list):
            raise ValueError("Workflow must include a 'steps' list.")

        if len(steps) > MAX_STEPS:
            raise ValueError(f"Too many steps ({len(steps)}). Limit is {MAX_STEPS}.")

        sanitized_steps: List[Dict[str, Any]] = []
        for i, s in enumerate(steps):
            if not isinstance(s, dict):
                raise ValueError(f"Step at index {i} must be a dict.")
            step = dict(s)
            step_id = str(step.get("id") or f"step_{i}").strip()
            if not step_id:
                step_id = f"step_{i}"
            step["id"] = self._sanitize_name(step_id)

            # Optional description/action normalization
            for field in ("description", "action"):
                if field in step and step[field] is not None:
                    step[field] = self._clean_text(str(step[field]))

            # Redact secrets in step maps (shallow)
            step = self._redact_map(step)
            sanitized_steps.append(step)

        wf["steps"] = sanitized_steps

        # Redact secrets in top-level keys (shallow)
        wf = self._redact_map(wf)

        return wf

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Allow alphanumerics, underscore, hyphen; clamp length."""
        name = name[:MAX_NAME_LEN]
        if not re.match(r"^[A-Za-z0-9_\-]+$", name):
            # Replace invalid characters with underscore
            name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
        if not name:
            name = "unnamed"
        return name

    @staticmethod
    def _clean_text(text: str) -> str:
        """Collapse control characters and trim."""
        text = text.replace("\r", " ").replace("\n", " ").strip()
        return text[:5000]  # clamp long fields defensively

    def _redact_map(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Redact common secret keys (shallow only)."""
        redacted = {}
        for k, v in obj.items():
            if isinstance(k, str) and any(sub in k.lower() for sub in SECRET_KEY_SUBSTRINGS):
                redacted[k] = "***"
            else:
                redacted[k] = v
        return redacted

    # ---------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------
    @staticmethod
    def _serialize_json(wf: Dict[str, Any]) -> bytes:
        try:
            return json.dumps(wf, indent=2, ensure_ascii=False).encode("utf-8")
        except Exception as e:
            logger.exception("JSON serialization failed: %s", e)
            raise ValueError("Failed to serialize workflow as JSON") from e

    @staticmethod
    def _serialize_yaml(wf: Dict[str, Any]) -> bytes:
        try:
            import yaml  # optional dependency
        except Exception as e:
            raise RuntimeError("PyYAML is required for YAML export. Install 'pyyaml'.") from e

        try:
            data = yaml.safe_dump(wf, sort_keys=False, allow_unicode=True)
            return data.encode("utf-8")
        except Exception as e:
            logger.exception("YAML serialization failed: %s", e)
            raise ValueError("Failed to serialize workflow as YAML") from e

    # ---------------------------------------------------------------------
    # ZIP Bundle Export
    # ---------------------------------------------------------------------
    def _export_bundle_zip(self, wf: Dict[str, Any], out_path: Path) -> Path:
        import zipfile
        # Build in-memory files first
        json_bytes = self._serialize_json(wf)
        self._ensure_size_within_limits(json_bytes)

        steps_csv = self._steps_to_csv_bytes(wf["steps"])
        meta_json = self._bundle_metadata_bytes(wf, len(wf["steps"]))

        # Atomic write bundle
        tmp_fd, tmp_p = tempfile.mkstemp(dir=str(self.base_dir), prefix=".tmp_export_", suffix=".zip")
        os.close(tmp_fd)
        try:
            with zipfile.ZipFile(tmp_p, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("workflow.json", json_bytes)
                zf.writestr("steps.csv", steps_csv)
                zf.writestr("metadata.json", meta_json)
            os.replace(tmp_p, out_path)
        except Exception:
            # Clean temp on failure
            try:
                os.remove(tmp_p)
            except OSError:
                pass
            raise
        return out_path

    @staticmethod
    def _steps_to_csv_bytes(steps: List[Dict[str, Any]]) -> bytes:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "description", "action"])
        for s in steps:
            writer.writerow([
                s.get("id", ""),
                (s.get("description") or "")[:200],
                (s.get("action") or "")[:200],
            ])
        return buf.getvalue().encode("utf-8")

    @staticmethod
    def _bundle_metadata_bytes(wf: Dict[str, Any], step_count: int) -> bytes:
        now = datetime.now(timezone.utc).isoformat()
        meta = {
            "exported_at": now,
            "name": wf.get("name"),
            "steps": step_count,
            "schema_version": "1.0",
        }
        return json.dumps(meta, indent=2).encode("utf-8")

    # ---------------------------------------------------------------------
    # Path Safety & Atomic Write
    # ---------------------------------------------------------------------
    def _resolve_filename(self, filename: Optional[str], wf: Dict[str, Any], fmt: str) -> str:
        if filename:
            fn = str(filename).strip()
        else:
            base = self._sanitize_name(wf.get("name", "workflow") or "workflow")
            fn = f"{base}.{('yml' if fmt == 'yaml' else fmt)}"

        # Normalize improper extensions
        if fmt == "yaml" and not fn.lower().endswith((".yaml", ".yml")):
            fn += ".yaml"
        elif fmt != "yaml" and not fn.lower().endswith(f".{fmt}"):
            fn += f".{fmt}"

        # Kill directory components in provided filename
        fn = Path(fn).name
        if len(fn) > 120:
            fn = fn[:120]
        return fn

    def _safe_path(self, filename: str) -> Path:
        path = (self.base_dir / filename).resolve()
        base = self.base_dir
        if not str(path).startswith(str(base)):
            raise ValueError("Unsafe output path (path traversal blocked).")
        return path

    @staticmethod
    def _atomic_write(out_path: Path, data: bytes) -> ExportStats:
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use temp file in the same directory for atomic replace
        tmp_fd, tmp_p = tempfile.mkstemp(dir=str(out_dir), prefix=".tmp_export_", text=False)
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_p, str(out_path))
            return ExportStats(steps=0, redacted_keys=0, bytes_written=len(data))
        except Exception:
            try:
                os.remove(tmp_p)
            except OSError:
                pass
            raise

    @staticmethod
    def _ensure_size_within_limits(data: bytes) -> None:
        if len(data) > MAX_TOTAL_BYTES:
            raise ValueError(f"Serialized content too large ({len(data)} bytes). Limit is {MAX_TOTAL_BYTES}.")


# ---------------------------------------------------------------------
# Simple CLI (optional)
# ---------------------------------------------------------------------
def _main_cli() -> int:
    """
    Minimal CLI for manual testing:
      python -m predictflow.modeler.exporter input.json --fmt json --out exports/my.json
    """
    import argparse, sys

    parser = argparse.ArgumentParser(description="PredictFlow secure workflow exporter")
    parser.add_argument("input", help="Path to workflow JSON file")
    parser.add_argument("--fmt", choices=sorted(ALLOWED_FMTS), default="json")
    parser.add_argument("--out", help="Output filename (optional)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--base", help="Base export directory (default: ./exports)")
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        exporter = WorkflowExporter(base_dir=args.base, allow_overwrite=args.overwrite)
        path = exporter.export(workflow, filename=args.out, fmt=args.fmt)
        print(f"Exported to: {path}")
        return 0
    except Exception as e:
        print(f"Export failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main_cli())
