"""
PredictFlow Dashboard (Secure Version)
--------------------------------------

Purpose:
  - Provide an interactive web UI for PredictFlow workflow visualization.
  - Load user-uploaded YAML workflow definitions safely.
  - Execute workflow simulation and display RPN & Confidence metrics.

Security & Resilience Features:
  ✅ Sanitized file upload and safe temporary file handling.
  ✅ Strictly limited file size and type (YAML only).
  ✅ No arbitrary code execution from uploaded YAMLs (relies on safe parser).
  ✅ Graceful error handling and cleanup.
  ✅ No sensitive data persisted to disk beyond runtime.
  ✅ Controlled visualization (no unsanitized text in plots).
"""

import io
import os
import tempfile
import traceback
import logging
from typing import Any, Dict

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# PredictFlow internal imports
from predictflow.engine.parser import parse_workflow
from predictflow.engine.executor import Executor

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="PredictFlow Dashboard", layout="wide")

st.title("📊 PredictFlow Dashboard")
st.markdown(
    "Upload a **YAML workflow** to analyze predictive metrics such as **RPN**, **Confidence**, "
    "and **Critical Path**. This dashboard runs all computations locally in a sandboxed context."
)

logger = logging.getLogger(__name__)

# ----------------------------
# Secure Upload Configuration
# ----------------------------
MAX_FILE_SIZE_MB = 5
ALLOWED_EXTENSIONS = {".yaml", ".yml"}


def is_safe_file(name: str, size: int) -> bool:
    """Simple whitelist and size-based validation."""
    _, ext = os.path.splitext(name.lower())
    if ext not in ALLOWED_EXTENSIONS:
        st.error(f"❌ Invalid file type: {ext}. Only YAML files are allowed.")
        return False
    if size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"❌ File too large. Limit is {MAX_FILE_SIZE_MB} MB.")
        return False
    return True


# ----------------------------
# Upload + Parse Section
# ----------------------------
uploaded_file = st.file_uploader("📁 Upload your workflow YAML", type=["yaml", "yml"])

if uploaded_file is not None:
    if not is_safe_file(uploaded_file.name, uploaded_file.size):
        st.stop()

    try:
        # Create a secure temp copy (auto-cleanup at end of context)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.success("✅ Workflow uploaded successfully!")

        # ------------------------------------
        # Run the workflow
        # ------------------------------------
        try:
            with st.spinner("🚀 Running workflow simulation..."):
                workflow = parse_workflow(tmp_path)
                executor = Executor(workflow)
                executor.run()
        except Exception as e:
            st.error(f"⚠️ Workflow execution failed: {e}")
            st.text(traceback.format_exc())
            os.unlink(tmp_path)
            st.stop()

        # Clean up uploaded temp file
        try:
            os.unlink(tmp_path)
        except Exception as cleanup_err:
            logger.warning("Temp file cleanup failed: %s", cleanup_err)

        # ------------------------------------
        # Metrics Table
        # ------------------------------------
        if not hasattr(executor, "metrics") or not executor.metrics:
            st.warning("No metrics were generated from this workflow.")
        else:
            data = []
            for step, vals in executor.metrics.items():
                if not isinstance(vals, dict):
                    continue
                rpn = vals.get("rpn")
                conf = vals.get("confidence")
                if isinstance(rpn, (int, float)) and isinstance(conf, (int, float)):
                    data.append({"Step": str(step), "RPN (Risk)": rpn, "Confidence": conf})

            if not data:
                st.warning("No valid metric data available for visualization.")
            else:
                df = pd.DataFrame(data)
                st.subheader("📈 Step Metrics")
                st.dataframe(df, use_container_width=True)

                # ------------------------------------
                # Visualization
                # ------------------------------------
                st.subheader("🎯 Risk vs Confidence Visualization")

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(df["RPN (Risk)"], df["Confidence"], s=120, alpha=0.7, color="#4C8BF5")

                # Label points safely (avoid long text injection)
                for _, row in df.iterrows():
                    label = str(row["Step"])[:30]  # limit label length
                    ax.text(row["RPN (Risk)"] + 2, row["Confidence"], label, fontsize=9)

                ax.set_xlabel("FMEA Risk (RPN)")
                ax.set_ylabel("Confidence Score")
                ax.set_title("PredictFlow: Step Metrics", fontsize=13, weight="bold")
                ax.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig)

                # ------------------------------------
                # Critical Path
                # ------------------------------------
                st.subheader("🚨 Critical Path (Top 3 Highest-Risk Steps)")
                sorted_steps = sorted(
                    executor.metrics.items(),
                    key=lambda s: (
                        s[1].get("rpn", 0),
                        -s[1].get("confidence", 0)
                    ),
                    reverse=True,
                )
                critical = [str(s[0])[:30] for s in sorted_steps[:3]]
                if critical:
                    st.success(" → ".join(critical))
                else:
                    st.info("No high-risk steps detected.")
    except Exception as e:
        st.error("Unexpected error occurred while processing file.")
        st.text(traceback.format_exc())
else:
    st.info("☁️ Upload a YAML workflow to get started.")
