import matplotlib.pyplot as plt
from predictflow.engine.parser import parse_workflow
from predictflow.engine.executor import Executor


def run_visualization(path: str):
    """Run workflow and visualize metrics."""
    workflow = parse_workflow(path)
    executor = Executor(workflow)
    executor.run()  # fills executor.metrics

    # Collect data
    steps = list(executor.metrics.keys())
    rpn = [executor.metrics[s]['rpn'] for s in steps]
    conf = [executor.metrics[s]['confidence'] for s in steps]

    # Identify critical path (top 3 by RPN)
    sorted_steps = sorted(
        executor.metrics.items(),
        key=lambda s: (s[1]['rpn'], -s[1]['confidence']),
        reverse=True
    )
    critical = [s[0] for s in sorted_steps[:3]]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(rpn, conf, s=100, alpha=0.7, label="Workflow Steps")

    # Highlight critical ones
    for step_id in critical:
        idx = steps.index(step_id)
        plt.scatter(rpn[idx], conf[idx], s=200, edgecolors='red',
                    facecolors='none', linewidths=2, label="Critical Step" if step_id == critical[0] else "")
        plt.text(rpn[idx] + 2, conf[idx], step_id, fontsize=9)

    plt.title("PredictFlow: Risk vs Confidence", fontsize=14, weight='bold')
    plt.xlabel("FMEA Risk (RPN)")
    plt.ylabel("Confidence Score")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 -m predictflow.visualize.metrics <workflow.yaml>")
    else:
        run_visualization(sys.argv[1])
