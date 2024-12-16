import os, random, string, itertools, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from base64 import b64encode
from io import BytesIO
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    f1_score, 
    recall_score, 
    precision_score
)

class VISU:
    @staticmethod
    def save_meta(gene_familly, model_name, features, dataset_len, accuracy, f1, recall, precision):
        gene_familly_ = gene_familly.replace('/', '__')
        filepath = "Output/Model/" + gene_familly_ + "/meta.json"
        globpath = "Output/Metrics/meta.json"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        os.makedirs(os.path.dirname(globpath), exist_ok=True)

        # Initialize model data
        model_data = {
            "f1": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "df_length": dataset_len,
            "features_mask": {i + 1: features[i] for i in range(len(features))}
        }

        # [LOCAL & GLOBAL] Read existing data if file exists & Update the data
        for path in [filepath, globpath]:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as file:
                        data = json.load(file)
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}
            if gene_familly_ not in data:
                data[gene_familly_] = {}
            data[gene_familly_][model_name] = model_data
            with open(path, 'w') as file:
                json.dump(data, file, indent=4)
            
    @staticmethod
    def plot_curve_old(history, list_of_metrics):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        epochs = history.epoch
        hist = pd.DataFrame(history.history)
        for m in list_of_metrics:
            x = hist[m]
            plt.plot(epochs[1:], x[1:], '.-', label=m, lw=2, )
        plt.legend()


    @staticmethod
    def plot_curve(history, list_of_metrics):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        epochs = history.epoch
        hist = pd.DataFrame(history.history)
        
        for m in list_of_metrics:
            x = hist[m]
            # Scale loss and val_loss to [0, 1] if values are greater than 1
            if m in ['loss', 'val_loss']:
                if x.max() > 1:
                    x = x / x.max()  # Scale to [0, 1]
            plt.plot(epochs[1:], x[1:], label=m, lw=2)
            
        plt.legend()
            
    @staticmethod
    def test_report(X_test, y_test, model=None, args=["MODEL NAME", "SCORE", "bHLH", ['AAA', 'BBB', 'CCC'], "df_len"]):
        """
        Utils: For given model, and test data we run prediction and report peformance metrics
        """
        
        # Predict & Apply thresholding (default threshold is 0.5)
        predictions_proba = model.predict(X_test)
        y_pred = (predictions_proba >= 0.5).astype(int)
    
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        tn, fp, fn, tp = cm.ravel()
        report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'], zero_division=1,  digits=4)
        cf_id = ''.join(random.choices(string.ascii_uppercase+string.digits, k=8))
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f"Output/CFMatrix/confusion_matrix_{cf_id}.png")
        plt.close()
        
        report_html = f"""
        <div style="border: 2px solid #ddd;">
            <div style="padding: 0.6em; background-color: #ffdddd; font-weight: bold;">MODEL: {args[0]}</div>
            <div style="display: flex;">
                <div style="padding: 10px; width: 240px;">
                    <h2>Initial perfomance</h2>
                    <ul>
                        <li>Test accuracy: {args[1]}</li>
                    </ul>
                </div>
                <div style="flex: 1; padding: 10px;">
                    <h2>Classification Report</h2>
                    <pre>{report}</pre>
                    <h3>Metrics</h3>
                    <div style="display: flex;">
                        <ul>
                            <li>True Positives (TP): {tp}</li>
                            <li>True Negatives (TN): {tn}</li>
                        </ul>
                        <ul style="margin-left: 2em;">
                            <li>False Positives (FP): {fp}</li>
                            <li>False Negatives (FN): {fn}</li>
                        </ul>
                    </div>
                </div>
                <div style="flex: 1; padding: 10px;">
                    <h2 style="margin-left: 2em;">Confusion Matrix</h2>
                    <img src="Output/CFMatrix/confusion_matrix_{cf_id}.png" width="400">
                </div>
            </div>
        </div>
        """
        # Display report and confusion matrix side by side
        display(HTML(report_html))

        # Save meta
        # In your VISU.test_report function
        VISU.save_meta(gene_familly=args[2], model_name=args[0], features=args[3], dataset_len=args[4], accuracy=accuracy, f1=f1, recall=recall, precision=precision)

class VISUReport:
    def __init__(self, gene_name, dataset):
        self.gene_name = gene_name
        self.dataset = dataset
        self.reports = []
        self.header_html = f"""
        <html>
        <head>
            <title>GENE_FAMILY: {self.gene_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .container {{ border: 2px solid #ddd; margin: 20px; padding: 20px; }}
                .header {{ padding: 0.6em; background-color: #ffdddd; font-weight: bold; }}
                .title {{ padding: 0.6em; background-color: #80c4e6bd; font-weight: bold; }}
                .section {{ padding: 10px; }}
                .metrics {{ display: flex; }}
                .metrics ul {{ list-style: none; padding: 0; }}
                .metrics ul + ul {{ margin-left: 2em; }}
                .confusion-matrix img, .learning-curve img {{ width: 100%; max-width: 565px; }}
                .class_dist {{ width: 20em; }}
                .mod_sum {{ margin-bottom: 2em; }}
            </style>
        </head>
        <body>
        <div class="container title">
            <h3>GENE_FAMILY: <a href='https://planttfdb.gao-lab.org/family.php?fam={self.gene_name}' target='_blank'>{self.gene_name}</a></h3>
        </div>
        """
        self.footer_html = """
        </body>
        </html>
        """

    def plot_curve(self, history, list_of_metrics):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        epochs = history.epoch
        hist = pd.DataFrame(history.history)
        for m in list_of_metrics:
            x = hist[m]
            plt.plot(epochs[1:], x[1:], '.-', label=m, lw=2)
        plt.legend()
        curve_io = BytesIO()
        plt.savefig(curve_io, format='png')
        plt.close()
        return b64encode(curve_io.getvalue()).decode('utf-8')

    def add_report(self, X_test, y_test, model=None, history=None, args=["MODEL NAME", 0]):
        model_name = args[0]
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)

        learning_curve_base64 = self.plot_curve(history, ['loss', 'val_loss', 'accuracy', 'val_accuracy'])

        predictions_proba = model.predict(X_test)
        y_pred = (predictions_proba >= 0.5).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        tn, fp, fn, tp = cm.ravel()
        report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'], zero_division=1, digits=4)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        confusion_matrix_io = BytesIO()
        plt.savefig(confusion_matrix_io, format='png')
        plt.close()
        confusion_matrix_base64 = b64encode(confusion_matrix_io.getvalue()).decode('utf-8')

        class_counts = self.dataset['class'].value_counts()
        total_samples = len(self.dataset)
        class_counts_df = pd.DataFrame(class_counts)
        class_counts_df.columns = ['Count']
        class_counts_df['Percentage'] = (class_counts_df['Count'] / total_samples * 100).round(2)
        imbalance_ratio = class_counts_df['Count'].max() / class_counts_df['Count'].min()
        imbalance_threshold = 1.5

        report_html = f"""
        <div class="container">
            <div class="header">MODEL: {model_name}</div>
            <div style="display: flex;">
                <div class="section">
                    <h2 class='mod_sum'>Model Architecture</h2>
                    <pre>{model_summary}</pre>
                </div>
                <div class="section learning-curve">
                    <h2>Learning Curve</h2>
                    <img src="data:image/png;base64,{learning_curve_base64}" alt="Learning Curve">
                </div>
            </div>
            <div style="display: flex;">
                <div class="section class_dist">
                    <h2>Class Distribution</h2>
                    <pre>{class_counts_df}</pre>
                    <h3>Additional Metrics</h3>
                    <ul>
                        <li>Total Samples: {total_samples}</li>
                        <li>Imbalance Ratio: {imbalance_ratio:.2f}</li>
                    </ul>
                </div>
                <div class="section">
                    <h2>Classification Report</h2>
                    <pre>{report}</pre>
                    <h3>Metrics</h3>
                    <div class="metrics">
                        <ul>
                            <li>True Positives (TP): {tp}</li>
                            <li>True Negatives (TN): {tn}</li>
                        </ul>
                        <ul>
                            <li>False Positives (FP): {fp}</li>
                            <li>False Negatives (FN): {fn}</li>
                        </ul>
                    </div>
                </div>
                <div class="section confusion-matrix">
                    <h2>Confusion Matrix</h2>
                    <img src="data:image/png;base64,{confusion_matrix_base64}" alt="Confusion Matrix">
                </div>
            </div>
        </div>
        """
        self.reports.append(report_html)

    def save(self, tag=None):
        cf_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        if tag:
            cf_id = "full_" + cf_id
        complete_html = self.header_html + ''.join(self.reports) + self.footer_html

        os.makedirs(f"Output/Reports/{self.gene_name.replace('/', '__')}", exist_ok=True)
        with open(f"Output/Reports/{self.gene_name.replace('/', '__')}/report_{cf_id}.html", "w") as file:
            file.write(complete_html)

        print(f"Report saved as Output/Reports/{self.gene_name.replace('/', '__')}/report_{cf_id}.html")