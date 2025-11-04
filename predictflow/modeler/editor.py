"""
Secure Test Suite for predictflow.modeler.editor
------------------------------------------------

Covers:
  - Safe loading, editing, and saving of workflows
  - Protection from path traversal and code injection
  - Deterministic behavior across edits
  - Proper exception handling and rollback
"""

import io
import os
import json
import tempfile
import pytest
from pathlib import Path

# Import target module
from predictflow.modeler.editor import WorkflowEditor


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------
@pytest.fixture
def sample_workflow():
    """Provide a minimal, valid workflow structure."""
    return {
        "name": "Invoice Approval",
        "steps": [
            {"id": "validate_invoice", "description": "Validate invoice details"},
            {"id": "approve_payment", "description": "Approve payment order"},
        ],
    }


@pytest.fixture
def tmp_workflow_file(sample_workflow):
    """Create a temporary workflow JSON file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        json.dump(sample_workflow, tmp)
        tmp.flush()
        return tmp.name


@pytest.fixture
def editor(tmp_workflow_file):
    """Initialize WorkflowEditor with a safe temporary file."""
    return WorkflowEditor(tmp_workflow_file)


# -------------------------------------------------------------------------
# Core Tests
# -------------------------------------------------------------------------
def test_load_workflow_securely(editor):
    """Ensure workflows load safely and preserve structure."""
    wf = editor.load()
    assert "steps" in wf
    assert all("id" in s for s in wf["steps"])


def test_prevents_path_traversal():
    """Ensure unsafe paths are rejected."""
    malicious_path = "../../etc/passwd"
    with pytest.raises(ValueError):
        WorkflowEditor(malicious_path)


def test_edit_workflow_add_step(editor):
    """Add a step securely and ensure it's persisted."""
    new_step = {"id": "notify_user", "description": "Notify customer via email"}
    editor.add_step(new_step)
    wf = editor.load()
    assert any(s["id"] == "notify_user" for s in wf["steps"])


def test_remove_step(editor):
    """Safely remove an existing step."""
    editor.remove_step("approve_payment")
    wf = editor.load()
    assert not any(s["id"] == "approve_payment" for s in wf["steps"])


def test_rollback_on_invalid_edit(editor):
    """Invalid edit should not corrupt workflow."""
    before = editor.load()
    with pytest.raises(Exception):
        editor.add_step({"id": None})  # Invalid ID
    after = editor.load()
    assert before == after  # unchanged


def test_save_and_reload_integrity(editor):
    """Ensure saved workflow reloads with full data integrity."""
    wf1 = editor.load()
    wf1["name"] = "Updated Flow"
    editor.save(wf1)
    wf2 = editor.load()
    assert wf2["name"] == "Updated Flow"


def test_deterministic_edit_behavior(editor):
    """Ensure deterministic edits produce same results across runs."""
    wf_a = editor.load()
    editor.add_step({"id": "audit_log", "description": "Log audit event"})
    wf_b = editor.load()
    editor.remove_step("audit_log")
    editor.add_step({"id": "audit_log", "description": "Log audit event"})
    wf_c = editor.load()
    assert wf_b == wf_c


def test_invalid_json_rejected(tmp_path):
    """Ensure invalid workflow JSON is safely rejected."""
    bad_path = tmp_path / "broken.json"
    bad_path.write_text("{invalid json}")
    with pytest.raises(json.JSONDecodeError):
        WorkflowEditor(str(bad_path)).load()


def test_context_manager_safety(tmp_workflow_file):
    """Editor should safely clean up or rollback within context."""
    with WorkflowEditor(tmp_workflow_file) as editor:
        wf = editor.load()
        wf["name"] = "Context Edit"
        editor.save(wf)
    wf2 = WorkflowEditor(tmp_workflow_file).load()
    assert wf2["name"] == "Context Edit"


def test_secure_file_permissions(editor):
    """Ensure workflow file permissions are restrictive."""
    path = Path(editor.file_path)
    mode = path.stat().st_mode
    assert (mode & 0o077) == 0  # no world/group write or read


# -------------------------------------------------------------------------
# Functional Validations
# -------------------------------------------------------------------------
def test_json_export(editor):
    """Ensure editor exports clean JSON string."""
    wf = editor.load()
    exported = editor.to_json()
    assert json.loads(exported) == wf


def test_list_steps(editor):
    """List steps returns consistent structure."""
    steps = editor.list_steps()
    assert isinstance(steps, list)
    assert all("id" in s for s in steps)
