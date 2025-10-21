import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predictflow.engine.parser import parse_workflow
from predictflow.engine.executor import Executor
import tempfile
import os


st.set_page_config(page_title="PredictFlow Dashboard", layout="wide")

st.title("PredictFlow Dashboard")
st.markdown("Upload a YAML workflow to analyze its predictive metrics (RPN, Confidence, and Critical Path).")

# ------------------------------------
# Upload + Parse Section
# ------------------------------------
uploaded_file = st.file_uploader("Upload your workflow YAML", type=["yaml", "yml"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success("Workflow loaded successfully!")

    # Run the workflow
    workflow = parse_workflow(tmp_path)
    executor = Executor(workflow)
    with st.spinner("Running workflow simulation..."):
        executor.run()

    # Clean up temp file
    os.unlink(tmp_path)

    # ------------------------------------
    # Metrics Table
    # ------------------------------------
    data = [
        {
            "Step": step,
            "RPN (Risk)": vals["rpn"],
            "Confidence": vals["confidence"],
        }
        for step, vals in executor.metrics.items()
    ]
    df = pd.DataFrame(data)
    st.subheader("Step Metrics")
    st.dataframe(df, use_container_width=True)

    # ------------------------------------
    # Visualization
    # ------------------------------------
    st.subheader("Risk vs Confidence Visualization")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["RPN (Risk)"], df["Confidence"], s=120, alpha=0.7, color="#4C8BF5")

    # Label steps
    for i, row in df.iterrows():
        ax.text(row["RPN (Risk)"] + 2, row["Confidence"], row["Step"], fontsize=9)

    ax.set_xlabel("FMEA Risk (RPN)")
    ax.set_ylabel("Confidence Score")
    ax.set_title("PredictFlow: Step Metrics", fontsize=13, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

    # ------------------------------------
    # Critical Path
    # ------------------------------------
    st.subheader("Critical Path (Top 3 Highest Risk Steps)")
    sorted_steps = sorted(
        executor.metrics.items(),
        key=lambda s: (s[1]["rpn"], -s[1]["confidence"]),
        reverse=True
    )
    critical = [s[0] for s in sorted_steps[:3]]
    st.write(" → ".join(critical))
else:
    st.info("Upload a YAML workflow to get started.")
