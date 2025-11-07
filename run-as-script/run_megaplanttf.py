import argparse, os, sys, uuid, json, base64, io
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt

current_directory = os.getcwd()
root_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
sys.path.append(root_directory)

from pretrained.predictor import SingleKModel

def main():
    parser = argparse.ArgumentParser(description="Run MegaPlantTF predictor in standalone mode.")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file.")
    parser.add_argument("--kmer", type=int, default=3, help="K-mer size (default: 3).")
    parser.add_argument("--voting", type=str, default="Two-Stage Voting", choices=["Max Voting", "Two-Stage Voting"],
                        help="Voting method to use (default: 'Two-Stage Voting').")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold (default: 0.5).")
    parser.add_argument("--output", type=str, default=None, help="Path to output HTML dashboard.")
    parser.add_argument("--jobid", type=str, default=None, help="Unique job ID.")
    args = parser.parse_args()

    if args.jobid is None:
        args.jobid = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]

    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    if args.output is None:
        args.output = os.path.join(output_dir, f"MegaPlantTF_Dashboard_{args.jobid}.html")

    csv_output = args.output.replace(".html", "_predictions.csv")

    print(f"\nRunning MegaPlantTF...")
    print("--------------------------------------------------")
    print(f"Input file     : {args.fasta}")
    print(f"K-mer size     : {args.kmer}")
    print(f"Voting method  : {args.voting}")
    print(f"Threshold      : {args.threshold}")
    print(f"Output folder  : {output_dir}")
    print(f"Dashboard file : {os.path.basename(args.output)}")
    print("--------------------------------------------------")

    model = SingleKModel(kmer_size=args.kmer)
    model.load(args.fasta, format="fasta")
    genboard = model.predict()

    preds = genboard.two_stage_prediction if args.voting == "Two-Stage Voting" else genboard.prediction
    df = preds.copy()
    df["Predicted_Class"] = df.idxmax(axis=1)
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df["Max_Prob"] = df[numeric_cols].max(axis=1)
    df.to_csv(csv_output, index_label="Sequence_ID")

    print(f"✅ Predictions saved: {csv_output}")

    html_dashboard = build_html_dashboard(df, args, jobid=args.jobid)
    with open(args.output, "w") as f:
        f.write(html_dashboard)
    print(f"✅ Interactive Dashboard saved: {args.output}")


def build_html_dashboard(df, args, jobid):
    """Generate HTML dashboard from an external template."""
    import plotly.express as px

    # 1. Create Plotly figure
    class_counts = df["Predicted_Class"].value_counts().reset_index()
    class_counts.columns = ["Predicted_Class", "Count"]
    fig = px.bar(
        class_counts,
        x="Predicted_Class",
        y="Count",
        color_discrete_sequence=["#1f77b4"],
        labels={"Predicted_Class": "Gene Family", "Count": "Predicted Count"},
        title=f"MegaPlantTF Predictions Overview — Job {jobid}"
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45, height=600)
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # 2. Serialize data to JSON
    data_json = df.to_json(orient="records")

    # 3. Load the HTML template
    template_path = os.path.join(os.path.dirname(__file__), "dashboard_template.html")
    with open(template_path, "r") as f:
        html_template = f.read()

    # 4. Replace placeholders
    html_filled = (
        html_template
        .replace("{{JOBID}}", jobid)
        .replace("{{KMER}}", str(args.kmer))
        .replace("{{VOTING}}", args.voting)
        .replace("{{THRESHOLD}}", str(args.threshold))
        .replace("{{PLOT_HTML}}", plot_html)
        .replace("{{DATA_JSON}}", data_json)
    )

    return html_filled




if __name__ == "__main__":
    main()
