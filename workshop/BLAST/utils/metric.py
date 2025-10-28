def show_metrics(true_labels, predicted_labels):
    import sklearn.metrics as metrics
    from IPython.core.display import display, HTML
    
    # accuracy
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    
    # sensitivity
    precision_micro = metrics.precision_score(true_labels, predicted_labels, average='micro', zero_division=1)
    precision_macro = metrics.precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
    precision_weighted = metrics.precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    # recall
    recall_micro = metrics.recall_score(true_labels, predicted_labels, average='micro', zero_division=1)
    recall_macro = metrics.recall_score(true_labels, predicted_labels, average='macro', zero_division=1)
    recall_weighted = metrics.recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    # f1-score
    f1_micro = metrics.f1_score(true_labels, predicted_labels, average='micro', zero_division=1)
    f1_macro = metrics.f1_score(true_labels, predicted_labels, average='macro', zero_division=1)
    f1_weighted = metrics.f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    overall_accuracy = f"""
    <h3 style='display: flex; justify-content: space-between;'>
        <span style='width: 100%;'>Overall Score</span> 
    </h3>
    <table style='width:100%; border-collapse: collapse; border: 1px solid black;'>
        <tr>
            <th style='text-align: left;'>Accuracy</th>
            <td>{accuracy:.4f}</td>
        </tr>
    </table>
    <table style="width: 100%; border-collapse: collapse; border: 1px solid black; table-layout: auto-;">
        <tr>
            <th>Metrics</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Metrics Description</th>
        </tr>
        <tr>
            <th>Micro</th>
            <td>{precision_micro:.4f}</td>
            <td>{recall_micro:.4f}</td>
            <td>{f1_micro:.4f}</td>
            <td>Calculates metrics globally by counting the total true positives, false negatives, and false positives.</td>
        </tr>
        <tr>
            <th>Macro</th>
            <td>{precision_macro:.4f}</td>
            <td>{recall_macro:.4f}</td>
            <td>{f1_macro:.4f}</td>
            <td>Calculates metrics for each class independently and then takes the average (treating all classes equally).</td>
        </tr>
        <tr>
            <th>Weighted</th>
            <td>{precision_weighted:.4f}</td>
            <td>{recall_weighted:.4f}</td>
            <td>{f1_weighted:.4f}</td>
            <td>Calculates metrics for each class independently and then takes the average, weighted by the number of instances of each class.</td>
        </tr>
    </table>
    """

    # Display metrics
    display(HTML(overall_accuracy))

def show_confusion(true_labels, predicted_labels, class_mapping_rules):
    import sklearn.metrics as metrics
    import pandas as pd
    import platform, base64, io
    from IPython.core.display import display, HTML
    import matplotlib.pyplot as plt
    import seaborn as sns

    def _fig_to_html(fig):
        """Converts matplotlib figure to base64 encoded HTML image string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = buf.getvalue()
        encoded_img = base64.b64encode(img_str).decode()
        html_img = f'<img src="data:image/png;base64,{encoded_img}" />'
        buf.close()
        return html_img
    
    cm = metrics.confusion_matrix(true_labels, predicted_labels)
    sorted_class_names = [k for k, v in sorted(class_mapping_rules.items(), key=lambda item: item[1])]
    cm_df = pd.DataFrame(cm, index=sorted_class_names, columns=sorted_class_names)

    fig, ax = plt.subplots(figsize=(20, 20))
    custom_palette = sns.color_palette("Set2", as_cmap=True) #OR - ("Set2", "rocket", "Blues")
    sns.heatmap(cm_df, annot=True, cmap=custom_palette, fmt='d', cbar=False, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    # Convert the plot to HTML string using _fig_to_html method
    cm_html = f"<h3>Confusion Matrix</h3><div>{_fig_to_html(fig)}</div>"
    confusion_matrix_html = cm_html
    plt.close(fig)

    # Display metrics
    display(HTML(cm_html))