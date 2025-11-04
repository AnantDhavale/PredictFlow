"""
PredictFlow CLI Runner (Secure & Functional)
--------------------------------------------

Purpose:
  - Provide a safe and minimal command-line interface for PredictFlow
  - Allow users to run YAML or BPMN workflows with controlled parameters
  - Prevent path traversal, unsafe code execution, or arbitrary imports

Security Features:
  ✅ Validates input file paths (no traversal, must exist)
  ✅ Restricts workflow formats (YAML, YML, BPMN)
  ✅ Limits runtime environment access (no eval)
  ✅ Safe parsing via yaml.safe_load
  ✅ Structured exception handling and logging
  ✅ Optional dry-run mode for inspection without execution
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Internal PredictFlow modules
from predictflow.engine.parser import parse_workflow
from predictflow.engine.executor import Executor
from predictflow.fmea.report import FMEAReport
from predictflow.evaluator import Evaluator

# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------
logger = logging.getLogger("predictflow.cli")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ----------------------------------------------------
# Security Utilities
# ----------------------------------------------------
ALLOWED_EXTENSIONS = {".yaml", ".yml", ".bpmn"}
MAX_FILE_SIZE_MB = 10


def is_safe_file(file_path: str) -> bool:
    """Ensure the workflow path is safe and within limits."""
    try:
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            logger.error("File not found: %s", file_path)
            return False

        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            logger.error("Invalid file extension: %s", ext)
            return False

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            logger.error("File too large: %.2f MB (limit %.1f MB)", size_mb, MAX_FILE_SIZE_MB)
            return False

        # Ensure not outside project root
        project_root = Path(__file__).resolve().parents[2]
        if not str(path).startswith(str(project_root)):
            logger.error("Unsafe path detected: %s", path)
            return False

        return True
    except Exception as e:
        logger.error("File validation failed: %s", e)
        return False


# ----------------------------------------------------
# Core CLI Logic
# ----------------------------------------------------
def run_workflow(file_path: str, dry_run: bool = False, output: str | None = None) -> int:
    """Run a workflow securely and optionally export evaluation results."""
    if not is_safe_file(file_path):
        return 1

    try:
        logger.info("Loading workflow from: %s", file_path)
        workflow = parse_workflow(file_path)

        if dry_run:
            print(json.dumps(workflow, indent=2))
            return 0

        logger.info("Executing workflow...")
        executor = Executor(workflow)
        executor.run()

        # Generate reports
        logger.info("Analyzing results...")
        fmea_report = FMEAReport(executor.context)
        evaluator = Evaluator(executor.context)

        fmea_data = fmea_report.build_report()
        eval_results = evaluator.evaluate()

        # Display summaries
        print("\n=== FMEA Results ===")
        print(fmea_report.to_text())

        print("\n=== Workflow Evaluation Summary ===")
        for k, v in eval_results.items():
            print(f"{k}: {v}")

        # Optional export
        if output:
            safe_out = Path(output).resolve()
            if not str(safe_out).startswith(str(Path.cwd().resolve())):
                logger.error("Unsafe output path detected: %s", safe_out)
                return 1
            out_json = {
                "fmea": fmea_data,
                "evaluation": eval_results,
            }
            with open(safe_out, "w", encoding="utf-8") as f:
                json.dump(out_json, f, indent=2)
            logger.info("Results exported to %s", safe_out)

        logger.info("Workflow completed successfully.")
        return 0

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        return 130
    except Exception as e:
        logger.exception("Workflow execution failed: %s", e)
        return 1


# ----------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PredictFlow CLI Runner - Secure workflow execution tool"
    )
    parser.add_argument(
        "workflow",
        help="Path to workflow file (.yaml, .yml, .bpmn)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and display the workflow without executing it",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to export results as JSON",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for troubleshooting",
    )

    args = parser.parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    return run_workflow(args.workflow, dry_run=args.dry_run, output=args.output)


if __name__ == "__main__":
    sys.exit(main())
