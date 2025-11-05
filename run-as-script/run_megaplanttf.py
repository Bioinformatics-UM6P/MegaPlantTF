import argparse, os, sys, uuid
import pandas as pd
from datetime import datetime
from pretrained.predictor import SingleKModel

def main():
    # args
    parser = argparse.ArgumentParser(description="Run MegaPlantTF predictor in standalone mode.")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file.")
    parser.add_argument("--kmer", type=int, default=3, help="K-mer size (default: 3).")
    parser.add_argument("--voting", type=str, default="Two-Stage Voting", choices=["Max Voting", "Two-Stage Voting"],
                        help="Voting method to use (default: 'Two-Stage Voting').")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold (default: 0.5).")
    parser.add_argument("--output", type=str, default="MegaPlantTF_Report.html", help="Path to output HTML file.")
    parser.add_argument("--jobid", type=str, default=None, help="Unique job ID for naming outputs.")
    args = parser.parse_args()
    
    # setup
    if args.jobid is None:
        args.jobid = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
    if args.output is None:
        args.output = f"MegaPlantTF_Report_{args.jobid}.html"
    csv_output = args.output.replace(".html", "_predictions.csv")
    print(f"Running MegaPlantTF...")
    print(f"Input file     : {args.fasta}")
    print(f"K-mer size     : {args.kmer}")
    print(f"Voting method  : {args.voting}")
    print(f"Threshold      : {args.threshold}")
    print(f"Output report  : {args.output}")
    print("--------------------------------------------------")

    # run
    model = SingleKModel(kmer_size=args.kmer)
    model.load(args.fasta, format="fasta")
    # 3️⃣ Run inference
    genboard = model.predict()

    # 4️⃣ Build static HTML report
    print("🧠 Generating interactive HTML report...")
    html_report = build_html_report(genboard, args.voting, args.threshold)

    # 5️⃣ Save HTML to file
    with open(args.output, "w") as f:
        f.write(html_report)
    print(f"✅ Report saved: {args.output}")

    # 6️⃣ Export final predictions as CSV
    final_pred = export_final_predictions(genboard, args.voting, args.threshold)
    csv_output = args.output.replace(".html", "_predictions.csv")
    final_pred.to_csv(csv_output, index=False)
    print(f"✅ Predictions saved: {csv_output}")


def export_final_predictions(genboard, voting_method, threshold):
    """Return final predictions as DataFrame."""
    preds = genboard.prediction
    if voting_method == "Max Voting":
        binary_prediction = (preds > threshold).astype(int)
        final_prediction = binary_prediction.idxmax(axis=1)
        all_below = (preds.max(axis=1) <= threshold)
        final_prediction[all_below] = "Unknown"
    else:
        preds2 = genboard.two_stage_prediction
        binary_prediction = (preds2 > threshold).astype(int)
        final_prediction = binary_prediction.idxmax(axis=1)
        all_below = (preds2.max(axis=1) <= threshold)
        final_prediction[all_below] = "Unknown"

    df = pd.DataFrame({
        "Sequence_ID": genboard.init_df.index,
        "Predicted_Class": final_prediction.values
    })
    return df


def build_html_report(genboard, voting_method, threshold, jobid):
    import base64, io
    import matplotlib.pyplot as plt

    preds = genboard.prediction
    binary_prediction = (preds > threshold).astype(int)
    final_prediction = binary_prediction.idxmax(axis=1)
    all_below = (preds.max(axis=1) <= threshold)
    final_prediction[all_below] = "Unknown"
    gene_counts = final_prediction.value_counts()

    plt.figure(figsize=(10, 5))
    plt.bar(gene_counts.index, gene_counts.values, color="steelblue")
    plt.title(f"Predicted Gene Family Counts — Job {jobid}")
    plt.xticks(rotation=90)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()

    html = f"""
    <html>
    <head>
        <title>MegaPlantTF Report — {jobid}</title>
        <meta charset="utf-8"/>
        <style>
            body {{ font-family: Arial; margin: 20px; }}
            h1 {{ color: #1f77b4; }}
            button {{ background-color: #1f77b4; color: white; padding: 10px; border: none; cursor: pointer; }}
        </style>
    </head>
    <body>
        <h1>MegaPlantTF Report</h1>
        <p><b>Job ID:</b> {jobid}</p>
        <p><b>K-mer size:</b> {genboard.kmer_size}</p>
        <p><b>Voting:</b> {voting_method}</p>
        <p><b>Threshold:</b> {threshold}</p>

        <h2>Predicted Gene Family Counts</h2>
        <img src="data:image/png;base64,{img_base64}" />

        <div style="margin-top:20px;">
            <a href="MegaPlantTF_Report_{jobid}_predictions.csv" download>
                <button>Download Predictions CSV</button>
            </a>
        </div>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    main()
