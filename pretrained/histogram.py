import ipywidgets as widgets
from IPython.display import display, FileLink, HTML
from matplotlib.patches import Rectangle, FancyBboxPatch, Arrow
import numpy as np
import pandas as pd
import seaborn as sns
import platform, base64, io
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from PIL import Image
import shap
import lime
import lime.lime_tabular
import os, json
from sklearn.preprocessing import label_binarize

class GenBoard:
    def __init__(self, dataframe: pd.DataFrame):
        self.prediction = dataframe.sort_index(axis=1)
        self.init_df = None
        self.kmer_df = None
        self.model_dict = {}
        self.threshold_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            description='Threshold:',
            continuous_update=False
        )
        self.voting_method = widgets.RadioButtons(
            options=['Max Voting', 'Two-Stage Voting'], #'Weighted Max Voting',
            description='Voting Method:',
            disabled=False
        )
        self.model_weights = np.ones(self.prediction.shape[1])
        self.tab = widgets.Tab(layout=widgets.Layout(minwidth='1200px', height='800px'))
        self.two_stage_prediction = None
        self.show_counts = True
        self.kmer_size = None
        self.meta_model = None
        
    def get(self):
        return self.prediction

    def add_initial_set(self, df):
        self.init_df = df

    def add_kmer_set(self, df):
        self.kmer_df = df

    def add_model(self, model_dict):
        self.model_dict = model_dict

    def add_stage2_result(self, df):
        self.two_stage_prediction = df

    def set_stage2_model(self, model):
        self.meta_model = model

    def set_kmer_size(self, k):
        self.kmer_size = k

    def create_report_tab(self):
        report_output = widgets.Output()
        
        def toggle_show_counts(change):
            self.show_counts = change['new']
            update_report({'new': self.threshold_slider.value})
        
        def update_report(change):
            with report_output:
                report_output.clear_output(wait=True)
                threshold = self.threshold_slider.value
                voting_method = self.voting_method.value
    
                if voting_method == 'Max Voting':
                    binary_prediction = (self.prediction > threshold).astype(int)
                    final_prediction = binary_prediction.idxmax(axis=1)
                    all_below_threshold = (self.prediction.max(axis=1) <= threshold)
                    final_prediction[all_below_threshold] = 'Unknown'
                elif voting_method == 'Weighted Max Voting':
                    weighted_prediction = self.prediction * self.model_weights
                    binary_prediction = (weighted_prediction > threshold).astype(int)
                    final_prediction = binary_prediction.idxmax(axis=1)
                    all_below_threshold = (weighted_prediction.max(axis=1) <= threshold)
                    final_prediction[all_below_threshold] = 'Unknown'
                elif voting_method == 'Two-Stage Voting':
                    if self.two_stage_prediction is not None:
                        binary_prediction = (self.two_stage_prediction > threshold).astype(int)
                        final_prediction = binary_prediction.idxmax(axis=1)
                        all_below_threshold = (self.two_stage_prediction.max(axis=1) <= threshold)
                        final_prediction[all_below_threshold] = 'Unknown'
                    else:
                        raise ValueError("Two-stage prediction data is not available.")
                else:
                    raise ValueError("Unsupported voting method")
    
                gene_counts = final_prediction.value_counts()
                genes = gene_counts.index.tolist()
                
                # Ensure 'Unknown' is always included
                if 'Unknown' not in genes:
                    genes.append('Unknown')

                # Ensure gene_counts includes all original columns and 'Unknown'
                gene_counts = gene_counts.reindex(genes, fill_value=0)
    
                # Compute counts of predictions greater than the threshold
                above_threshold_counts = (self.prediction > threshold).sum(axis=0).reindex(genes, fill_value=0)
    
                plt.figure(figsize=(13, 7))
    
                above_threshold_bars = plt.bar(genes, above_threshold_counts, align='center', color='orange', alpha=0.6)
                bars = plt.bar(genes, gene_counts, align='center', color=['#1f77b4' if gene != 'Unknown' else '#eee' for gene in genes])
    
                plt.title('Number of Predicted Genes per Class')
                plt.xlabel('Gene Family')
                plt.ylabel('Number of Predicted Genes')
                plt.xticks(rotation=90)
    
                # Add counts on top of bars if show_counts is True
                if self.show_counts:
                    for bar, count in zip(bars, gene_counts):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2.0, height, str(count), ha='center', va='bottom')
                
                    for bar, count in zip(above_threshold_bars, above_threshold_counts):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2.0, height, str(count), ha='center', va='bottom')

                # Add legend with clickable behavior
                # legend = plt.legend(['Above Threshold Observations', 'Predicted Genes after voting'], loc='upper right')
                # for leg in legend.legendHandles:
                #     leg.set_picker(True)  # Enable picking on the legend handles
                #     leg.set_alpha(1)  # Set initial alpha to 1 (fully visible)

                legend = plt.legend(
                    ['Above Threshold Observations', 'Predicted Genes after voting'],
                    loc='upper right'
                )
                handles = getattr(legend, 'legendHandles', None)
                if handles is None:
                    handles = []
                    handles.extend(list(getattr(legend, 'get_lines', lambda: [])()))
                    handles.extend(list(getattr(legend, 'get_patches', lambda: [])()))
                for h in handles:
                    try:
                        h.set_picker(True)
                        h.set_alpha(1)
                    except Exception:
                        pass

                plt.tight_layout()
                plt.show()
    
        self.threshold_slider.observe(update_report, names='value')
        self.voting_method.observe(update_report, names='value')
        with report_output:
            update_report({'new': self.threshold_slider.value})
        
        show_counts_toggle = widgets.ToggleButton(value=True, description='Show Counts')
        show_counts_toggle.observe(toggle_show_counts, 'value')
        
        controls_layout = widgets.Layout(display='flex', justify_content='space-between', width='100%')
        controls = widgets.HBox([widgets.VBox([self.threshold_slider, show_counts_toggle]), self.voting_method], layout=controls_layout)
        combined_widget = widgets.VBox([controls, report_output])
        return combined_widget

    def create_meta_probabilities_tab(self):
        if 'meta' in self.init_df.columns:
            df  = self.init_df[['meta']].copy()
            df2 = self.init_df[['meta']].copy()
        else:
            df  = self.init_df[['id']].copy()
            df.rename(columns={'id': 'meta'}, inplace=True)
            df2  = self.init_df[['id']].copy()
            df2.rename(columns={'id': 'meta'}, inplace=True)
    
        # Add columns for gene family probabilities from prediction
        for gene_family in self.prediction.columns:
            df[gene_family] = self.prediction[gene_family]
        for gene_family in self.two_stage_prediction.columns:
            df2[gene_family] = self.two_stage_prediction[gene_family]
    
        # Add a prediction column with formatted string
        df['prediction'] = self.prediction.idxmax(axis=1) + ' (' + self.prediction.max(axis=1).astype(str) + ')'
        df2['prediction'] = self.two_stage_prediction.idxmax(axis=1) + ' (' + self.two_stage_prediction.max(axis=1).astype(str) + ')'
    
        # Add a column for unknown gene predictions
        df['Unknown Gene Family'] = '✓'
        df2['Unknown Gene Family'] = '✓'
    
        # Create widgets for threshold, voting method, search bar, and result count
        threshold_slider = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.01,
            description='Threshold:',
            continuous_update=False
        )
    
        voting_method = widgets.RadioButtons(
            options=['Max Voting', 'Two-Stage Voting'], #'Weighted Max Voting',
            description='Voting Method:',
            disabled=False
        )
        search_bar = widgets.Text(
            value='',
            placeholder='Search meta...',
            description='Filter:',
            continuous_update=True
        )
        result_count = widgets.Label(value="")
    
        df_output = widgets.Output()
    
        def update_df(change):
            with df_output:
                df_output.clear_output(wait=True)
                threshold = threshold_slider.value
                voting = voting_method.value
                search_value = search_bar.value.lower()
    
                # Update the 'Unknown Gene Family' column based on the threshold
                if voting == 'Two-Stage Voting' and self.two_stage_prediction is not None:
                    df2['Unknown Gene Family'] = np.where(self.two_stage_prediction.max(axis=1) < threshold, '✗', '✓')
                else:
                    df['Unknown Gene Family'] = np.where(self.prediction.max(axis=1) < threshold, '✗', '✓')
    
                # Filter DataFrame based on search bar input
                filtered_df = df[df['meta'].str.lower().str.contains(search_value)]
                filtered_df2 = df2[df2['meta'].str.lower().str.contains(search_value)]
    
                # Update result count
                if voting == 'Two-Stage Voting' and self.two_stage_prediction is not None:
                    result_count.value = f"({len(filtered_df2)} matches)"
                else:
                    result_count.value = f"({len(filtered_df)} matches)"
    
                # Highlight cells based on threshold
                def highlight_cells(val):
                    color = 'background-color: #ffcccc' if val < threshold else 'background-color: #a7c942'
                    return color
    
                # Highlight cells in the "Unknown Gene Family" column
                def highlight_unknown_cells(val):
                    return 'background-color: #add8e6' if val == '✗' else ''
    
                # if voting == 'Two-Stage Voting' and self.two_stage_prediction is not None:
                #     styled_df = filtered_df2.style.applymap(highlight_cells, subset=pd.IndexSlice[:, self.two_stage_prediction.columns])
                # else:
                #     styled_df = filtered_df.style.applymap(highlight_cells, subset=pd.IndexSlice[:, self.prediction.columns])
                # styled_df = styled_df.applymap(highlight_unknown_cells, subset=['Unknown Gene Family'])
                if voting == 'Two-Stage Voting' and self.two_stage_prediction is not None:
                    styled_df = filtered_df2.style.map(highlight_cells, subset=pd.IndexSlice[:, self.two_stage_prediction.columns])
                else:
                    styled_df = filtered_df.style.map(highlight_cells, subset=pd.IndexSlice[:, self.prediction.columns])
                styled_df = styled_df.map(highlight_unknown_cells, subset=pd.IndexSlice[:, ['Unknown Gene Family']])

                styled_df = styled_df.set_table_styles([{'selector': 'table', 'props': [('min-width', '1800px')]}])
                display(styled_df)
    
        # Observe changes in the threshold slider, voting method, and search bar
        threshold_slider.observe(update_df, names='value')
        voting_method.observe(update_df, names='value')
        search_bar.observe(update_df, names='value')
    
        # Initialize the display
        update_df(None)
    
        # Create the layout for the tab
        controls_layout = widgets.Layout(display='flex', justify_content='space-between', width='100%')
        controls = widgets.HBox([threshold_slider, voting_method], layout=controls_layout)
        search_layout = widgets.HBox([search_bar, result_count], layout=widgets.Layout(display='flex', align_items='center'))
        df_output_layout = widgets.Box([df_output], layout=widgets.Layout(overflow_x='scroll', width='100%'))
        tab_content = widgets.VBox([controls, search_layout, df_output_layout])
        return tab_content


    def create_data_transformation_tab(self):
        init_df_output = widgets.Output()
        kmer_df_output = widgets.Output()

        def save_df(df, filename):
            df.to_csv(filename, index=False)
            return FileLink(filename)
        with init_df_output:
            display(self.init_df)
        with kmer_df_output:
            display(self.kmer_df)

        init_df_button = widgets.Button(description="Download Initial DataFrame")
        kmer_df_button = widgets.Button(description="Download Transformed DataFrame")
        init_df_file_link = widgets.HTML()
        kmer_df_file_link = widgets.HTML()

        def on_init_df_button_clicked(b):
            filename = 'initial_dataframe.csv'
            save_df(self.init_df, filename)
            init_df_file_link.value = f'<a href="{filename}" download>Download Initial DataFrame</a>'
        def on_kmer_df_button_clicked(b):
            filename = 'transformed_dataframe.csv'
            save_df(self.kmer_df, filename)
            kmer_df_file_link.value = f'<a href="{filename}" download>Download Transformed DataFrame</a>'

        init_df_button.on_click(on_init_df_button_clicked)
        kmer_df_button.on_click(on_kmer_df_button_clicked)
        init_df_vbox = widgets.VBox([widgets.HTML('<h3>Initial DataFrame</h3>'), init_df_output, init_df_button, init_df_file_link])
        kmer_df_vbox = widgets.VBox([widgets.HTML('<h3>Transformed DataFrame</h3>'), kmer_df_output, kmer_df_button, kmer_df_file_link])
        return widgets.VBox([init_df_vbox, kmer_df_vbox])

    def create_pipeline_diagram(self):
        # Create a widget to hold the output
        out = widgets.Output()

        with out:
            fig, ax = plt.subplots(figsize=(14, 10))

            # Define component positions
            positions = {
                "load_data": (0.1, 0.8),
                "build_kmer_set": (0.1, 0.6),
                "load_models": (0.4, 0.8),
                "make_predictions": (0.4, 0.6),
                "aggregate_predictions": (0.7, 0.6),
                "generate_report": (0.7, 0.4),
                "display_results": (0.4, 0.4),
            }
            
            # Function to add rectangles for each component
            def add_component(ax, text, position):
                text_width = len(text) * 0.015
                text_height = 0.08
                rect = Rectangle(position, text_width, text_height, linewidth=1, edgecolor='#eee', facecolor='#1f77b4', alpha=0.7)
                ax.add_patch(rect)
                ax.text(position[0] + text_width / 2, position[1] + text_height / 2, text, ha="center", va="center",
                        fontsize=10, color="white", weight="bold")

            components = {
                "Load Data": positions["load_data"],
                "Build K-mer Set": positions["build_kmer_set"],
                "Load Models": positions["load_models"],
                "Make Predictions": positions["make_predictions"],
                "Aggregate Predictions": positions["aggregate_predictions"],
                "Generate Report": positions["generate_report"],
                "Display Results": positions["display_results"],
            }

            for text, position in components.items():
                add_component(ax, text, position)

            # Function to add arrows between components
            def add_arrow(ax, start, end):
                arrow = Arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                              width=0.01, edgecolor="black", facecolor="black", alpha=0.7)
                ax.add_patch(arrow)

            # Define the connections between components using positions
            connections = [
                ("load_data", "build_kmer_set"),
                ("build_kmer_set", "make_predictions"),
                ("load_models", "make_predictions"),
                ("make_predictions", "aggregate_predictions"),
                ("aggregate_predictions", "generate_report"),
                ("generate_report", "display_results"),
            ]

            for start, end in connections:
                add_arrow(ax, positions[start], positions[end])

            # Add titles and descriptions
            ax.text(0.5, 0.95, "Tool Pipeline Diagram", ha="center", va="center", fontsize=16, weight="bold")
            ax.text(0.5, 0.9, "This diagram illustrates the steps, model architecture, and pipeline.", ha="center", va="center", fontsize=12)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            # Instead of plt.show(), display the figure directly
            plt.close(fig)
            display(fig)

            # Display model summaries for each model used in predictions
            for gene_family in self.prediction.columns:
                if gene_family in self.model_dict:
                    model, features = self.model_dict[gene_family]
                    
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    model_summary = "\n".join(model_summary)
                    
                    display(widgets.HTML(f"<h3>{gene_family} Model Summary</h3>"))
                    display(widgets.HTML(f"<pre>{model_summary}</pre>"))

        return out

    def create_about_tab(self):
        version_info = {
            "Application Name": "MegaPlantTF",
            "Version": "1.0.0",
            "Python Version": platform.python_version(),
            "Operating System": platform.system(),
        }
        version_html = "<h3>About</h3>"
        version_html += "<ul>"
        for key, value in version_info.items():
            version_html += f"<li><b>{key}:</b> {value}</li>"
        version_html += "</ul>"
        return widgets.VBox([widgets.HTML(version_html)])

    def create_explainable_tab(self):
        output = widgets.Output()
        # Explain predictions using SHAP
        X_test = self.prediction
        explainer = shap.Explainer(self.meta_model, X_test)
        shap_values = explainer(X_test)
        plt.figure(figsize=(12, 6))

        # Check shap_values
        print(shap_values.shape)
        print(X_test.shape)
        print(X_test.columns.values)
        
        shap.summary_plot(shap_values, X_test, feature_names=["f"+str(i) for i in range(58)])
        plt.tight_layout()
        plt.show()
        with output:
            display(plt.gcf())
        return output

    """def create_explainable_tab(self):
        output = widgets.Output()
        X_test = self.prediction
        
        # Print type, shape, and content of X_test for debugging
        print(f"Type of X_test: {type(X_test)}")
        print(f"Shape of X_test: {X_test.shape}")
        print(f"Content of X_test:\n{X_test}")
        
        X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(X_test_np, mode="regression")
        
        # Assuming you want to explain the first instance in X_test
        exp = explainer.explain_instance(X_test_np[0], self.meta_model.predict)
        
        # Print the explanation
        exp.show_in_notebook(show_table=True)
        
        # Display the explanation within the output widget
        with output:
            exp.show_in_notebook(show_table=True)
        
        return output"""
    
    def display(self):
        self.tab.children = [
            self.create_report_tab(), 
            self.create_meta_probabilities_tab(), 
            self.create_data_transformation_tab(), 
            self.create_pipeline_diagram(), 
            self.create_about_tab()
            #,self.create_explainable_tab()
        ]
        self.tab.set_title(0, 'Stats Report')
        self.tab.set_title(1, 'Meta Report')
        self.tab.set_title(2, 'Data Transformation')
        self.tab.set_title(3, 'Model Pipeline')
        self.tab.set_title(4, 'About')
        #self.tab.set_title(5, 'Explainable AI')
        display(self.tab)

    def _fig_to_html(self, fig):
        """Converts matplotlib figure to base64 encoded HTML image string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = buf.getvalue()
        encoded_img = base64.b64encode(img_str).decode()
        html_img = f'<img src="data:image/png;base64,{encoded_img}" />'
        buf.close()
        return html_img

    def show_eval_metric(self, true_label, class_mapping_rules, voting_method="Max Voting", voting_threshold=0.5, binary_class_threshold=0.5, 
                         components=['confusion_matrix', 'general_accuracy', 'accuracy_per_family'],
                         metrics_storage_path=None):
        # Perform voting
        if voting_method == 'Max Voting':
            binary_prediction = (self.prediction > voting_threshold).astype(int)
            final_prediction = binary_prediction.idxmax(axis=1)
            all_below_threshold = (self.prediction.max(axis=1) <= voting_threshold)
            final_prediction[all_below_threshold] = 'Unknown'
        elif voting_method == 'Weighted Max Voting':
            weighted_prediction = self.prediction * self.model_weights
            binary_prediction = (weighted_prediction > voting_threshold).astype(int)
            final_prediction = binary_prediction.idxmax(axis=1)
            all_below_threshold = (weighted_prediction.max(axis=1) <= voting_threshold)
            final_prediction[all_below_threshold] = 'Unknown'
        elif voting_method == 'Two-Stage Voting':
            if self.two_stage_prediction is not None:
                binary_prediction = (self.two_stage_prediction > voting_threshold).astype(int)
                final_prediction = binary_prediction.idxmax(axis=1)
                all_below_threshold = (self.two_stage_prediction.max(axis=1) <= voting_threshold)
                final_prediction[all_below_threshold] = 'Unknown'
            else:
                raise ValueError("Two-stage prediction data is not available.")
        else:
            raise ValueError("Unsupported voting method")
        
        # Map final predicted classes
        encoded_predictions = final_prediction.map(class_mapping_rules)

        # Evaluation metrics
        overall_accuracy = None
        accuracy_per_family = None
        confusion_matrix_html = None

        if 'general_accuracy' in components:
            accuracy = metrics.accuracy_score(true_label, encoded_predictions)
        
            # Precision
            precision_micro = metrics.precision_score(true_label, encoded_predictions, average='micro', zero_division=1)
            precision_macro = metrics.precision_score(true_label, encoded_predictions, average='macro', zero_division=1)
            precision_weighted = metrics.precision_score(true_label, encoded_predictions, average='weighted', zero_division=1)
            
            # Recall
            recall_micro = metrics.recall_score(true_label, encoded_predictions, average='micro', zero_division=1)
            recall_macro = metrics.recall_score(true_label, encoded_predictions, average='macro', zero_division=1)
            recall_weighted = metrics.recall_score(true_label, encoded_predictions, average='weighted', zero_division=1)
            
            # F1 Score
            f1_micro = metrics.f1_score(true_label, encoded_predictions, average='micro', zero_division=1)
            f1_macro = metrics.f1_score(true_label, encoded_predictions, average='macro', zero_division=1)
            f1_weighted = metrics.f1_score(true_label, encoded_predictions, average='weighted', zero_division=1)

            if metrics_storage_path != None:
                if not os.path.exists(metrics_storage_path):
                    os.makedirs(metrics_storage_path)
                
                # Create a dictionary to store the metrics
                report = {
                    'accuracy': accuracy,
                    'precision_micro': precision_micro,
                    'precision_macro': precision_macro,
                    'precision_weighted': precision_weighted,
                    'recall_micro': recall_micro,
                    'recall_macro': recall_macro,
                    'recall_weighted': recall_weighted,
                    'f1_micro': f1_micro,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted
                }
            
                # Save the metrics to a JSON file
                report_file_path = os.path.join(metrics_storage_path, f'overall_classifier_{self.kmer_size}.json')
                with open(report_file_path, 'w') as report_file:
                    json.dump(report, report_file, indent=4)
                
            overall_accuracy = f"""
            <h3 style='display: flex; justify-content: space-between;'>
                <span style='width: 100%;'>Overall Score</span> 
                <span style='padding: 0.3em;background-color: #1f77b4;color: #fff;font-size: 0.85em;'>kmer_size={self.kmer_size}</span>
            </h3>
            <table style='width:100%; border-collapse: collapse; border: 1px solid black;'>
                <tr>
                    <th style='text-align: left;'>Accuracy</th>
                    <td>{accuracy:.2f}</td>
                </tr>
            </table>
            <table style="width: 100%; border-collapse: collapse; border: 1px solid black; table-layout: auto;">
                <tr>
                    <th>Metrics</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <!--<th>Metrics Description</th>-->
                </tr>
                <tr>
                    <th>Micro</th>
                    <td>{precision_micro:.2f}</td>
                    <td>{recall_micro:.2f}</td>
                    <td>{f1_micro:.2f}</td>
                    <!--<td>Calculates metrics globally by counting the total true positives, false negatives, and false positives.</td>-->
                </tr>
                <tr>
                    <th>Macro</th>
                    <td>{precision_macro:.2f}</td>
                    <td>{recall_macro:.2f}</td>
                    <td>{f1_macro:.2f}</td>
                    <!--<td>Calculates metrics for each class independently and then takes the average (treating all classes equally).</td>-->
                </tr>
                <tr>
                    <th>Weighted</th>
                    <td>{precision_weighted:.2f}</td>
                    <td>{recall_weighted:.2f}</td>
                    <td>{f1_weighted:.2f}</td>
                    <!--<td>Calculates metrics for each class independently and then takes the average, weighted by the number of instances of each class</td>-->
                </tr>
            </table>
            """
        
        if 'accuracy_per_family' in components:
            accuracy_per_family_left = "<h3>Accuracy per Gene Family</h3><div style=''>\
                <table style='width:calc(50% - 1em); float:left; border-collapse: collapse; border: 1px solid black; margin-left: 0.5em;'>\
                <tr><th>Gene Family</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr>"
            accuracy_per_family_right = "<table style='width:50%; float:right; border-collapse: collapse; border: 1px solid black;'>\
                <tr><th>Gene Family</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr>"
            
            column_count = 0
            metrics_per_family = {}
            for gene_family in self.prediction.columns:
                if gene_family in class_mapping_rules:
                    true_labels_for_family = [1 if true_label[i] == class_mapping_rules[gene_family] else 0 for i in range(len(true_label))]
                    pred_labels_for_family = [1 if self.prediction[gene_family][i] >= binary_class_threshold else 0 for i in range(len(self.prediction))]
                    # Todo: CORRECT -> max among all class for specific observation
                    
                    accuracy = metrics.accuracy_score(true_labels_for_family, pred_labels_for_family)
                    precision = metrics.precision_score(true_labels_for_family, pred_labels_for_family, zero_division=1)
                    recall = metrics.recall_score(true_labels_for_family, pred_labels_for_family, zero_division=1)
                    f1 = metrics.f1_score(true_labels_for_family, pred_labels_for_family, zero_division=1)

                    metrics_per_family[gene_family] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                            
                    if column_count % 2 == 0:
                        accuracy_per_family_left += f"<tr><td>{gene_family}</td><td><b>{accuracy:.2f}</b></td><td>{precision:.2f}</td><td>{recall:.2f}</td><td>{f1:.2f}</td></tr>"
                    else:
                        accuracy_per_family_right += f"<tr><td>{gene_family}</td><td><b>{accuracy:.2f}</b></td><td>{precision:.2f}</td><td>{recall:.2f}</td><td>{f1:.2f}</td></tr>"
                    column_count += 1
            
            accuracy_per_family_left += "</table>"
            accuracy_per_family_right += "</table></div>"
            accuracy_per_family = accuracy_per_family_left + accuracy_per_family_right
            
            # Save metrics to a JSON file
            if metrics_storage_path is not None:
                if not os.path.exists(metrics_storage_path):
                    os.makedirs(metrics_storage_path)
                
                report_file_path = os.path.join(metrics_storage_path, f'binary_classifier_{self.kmer_size}.json')
                with open(report_file_path, 'w') as report_file:
                    json.dump(metrics_per_family, report_file, indent=4)

        if 'confusion_matrix' in components:
            cm = metrics.confusion_matrix(true_label, encoded_predictions)
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
            cm_html = f"<h3>Confusion Matrix</h3><div>{self._fig_to_html(fig)}</div>"
            confusion_matrix_html = cm_html
            plt.close(fig)
        
        # Display the HTML tables
        display_html = ""
        if overall_accuracy:
            display_html += overall_accuracy
        if accuracy_per_family:
            display_html += accuracy_per_family
        if confusion_matrix_html:
            display_html += confusion_matrix_html

        # Add save button HTML and JavaScript
        save_button_html = """
            <div style='margin-top: 20px; text-align: center;'>
                <button onclick="saveReport()" style='background-color: blue; color: white; padding: 10px 20px; border: none; cursor: pointer;'>Save Report</button>
            </div>
            <script>
                function saveReport() {
                    var content = `""" + display_html.replace('\n', '') + """`;
                    var blob = new Blob([content], { type: 'text/html' });
                    var url = URL.createObjectURL(blob);
                    var a = document.createElement('a');
                    a.href = url;
                    a.download = 'report.html';
                    a.click();
                    URL.revokeObjectURL(url);
                }
            </script>
        """
        display_html += save_button_html
        
        # Output the HTML tables
        display(HTML(display_html))

    def show_roc_curve(
        self,
        true_label,
        class_mapping_rules,
        voting_method="Max Voting",
        voting_threshold=0.5,
        save_path=None,
        figsize=(10, 8)
    ):
        """
        Compute and display ROC curves for each class and macro-average ROC.
        Optionally saves the figure as a PNG.
        """
        # --- Step 1: Compute final predictions (same voting logic as in show_eval_metric)
        if voting_method == 'Max Voting':
            prob_matrix = self.prediction.copy()
        elif voting_method == 'Two-Stage Voting' and self.two_stage_prediction is not None:
            prob_matrix = self.two_stage_prediction.copy()
        else:
            raise ValueError("Unsupported or missing voting method/prediction data")

        # --- Step 2: Prepare labels
        y_true = np.array(true_label)
        y_score = prob_matrix.values
        class_names = list(prob_matrix.columns)
        y_bin = label_binarize(y_true, classes=list(class_mapping_rules.values()))

        n_classes = y_bin.shape[1]

        # --- Step 3: Compute ROC and AUC for each class
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # --- Step 4: Compute macro-average ROC curve and AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        # --- Step 5: Plot ROC curves
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            ax.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})",
            )

        # Plot macro-average ROC
        ax.plot(
            fpr["macro"],
            tpr["macro"],
            color="black",
            linestyle="--",
            lw=3,
            label=f"Macro-average (AUC = {roc_auc['macro']:.2f})",
        )

        # Random baseline
        ax.plot([0, 1], [0, 1], "r--", lw=1, label="Chance")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves per Class and Macro Average")
        ax.legend(loc="lower right")
        plt.tight_layout()

        # --- Step 6: Display and optionally save
        display(fig)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"ROC curve saved to {save_path}")

        plt.close(fig)

    def compute_and_save_roc_per_family(
        self,
        true_label,
        class_mapping_rules,
        voting_method="Max Voting",
        voting_threshold=0.5,
        save_folder="results/roc_data",
    ):
        """
        Compute ROC curves per gene family and save data + plots.
        Generates one plot per family and one JSON file per model run.
        """
        # --- Step 1: Choose prediction source
        if voting_method == "Max Voting":
            prob_matrix = self.prediction.copy()
        elif voting_method == "Two-Stage Voting" and self.two_stage_prediction is not None:
            prob_matrix = self.two_stage_prediction.copy()
        else:
            raise ValueError("Unsupported or missing voting method/prediction data")

        # --- Step 2: Align classes (skip Unknown etc.)
        common_classes = [c for c in prob_matrix.columns if c in class_mapping_rules]
        filtered_mapping = {c: class_mapping_rules[c] for c in common_classes}

        y_true = np.array(true_label)
        y_score = prob_matrix[common_classes].values
        y_bin = label_binarize(y_true, classes=list(filtered_mapping.values()))

        os.makedirs(save_folder, exist_ok=True)

        roc_dict = {}

        # --- Step 3: Compute and save per-family ROC
        for i, fam in enumerate(common_classes):
            if i >= y_bin.shape[1]:
                # Skip mismatch if class not in label set
                continue
            fpr, tpr, _ = metrics.roc_curve(y_bin[:, i], y_score[:, i])
            auc_value = metrics.auc(fpr, tpr)

            roc_dict[fam] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc_value),
            }

            # Plot and save individual ROC
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"{fam} (AUC={auc_value:.3f})")
            ax.plot([0, 1], [0, 1], "r--", lw=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {fam} (k={self.kmer_size})")
            ax.legend(loc="lower right")
            plt.tight_layout()

            fam_dir = os.path.join(save_folder, "plots")
            os.makedirs(fam_dir, exist_ok=True)
            plot_path = os.path.join(fam_dir, f"{fam}_k{self.kmer_size}.png")
            fig.savefig(plot_path, dpi=300)
            plt.close(fig)

        # --- Step 4: Save all ROC data for later comparison
        json_path = os.path.join(save_folder, f"roc_data_k{self.kmer_size}.json")
        with open(json_path, "w") as f:
            json.dump(roc_dict, f, indent=4)

        print(f"✅ Saved ROC data and plots for k={self.kmer_size} → {save_folder}")

    def compare_roc_across_k(family_name, roc_folders, save_path=None):
        """
        Load saved ROC JSONs from multiple k-folders and overlay curves for one family.
        Example roc_folders: ["results/k3", "results/k4", "results/k5"]
        """
        plt.figure(figsize=(7, 6))

        for folder in roc_folders:
            json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
            if not json_files:
                continue
            json_path = os.path.join(folder, json_files[0])

            with open(json_path, "r") as f:
                roc_dict = json.load(f)

            if family_name not in roc_dict:
                continue

            roc_data = roc_dict[family_name]
            fpr = np.array(roc_data["fpr"])
            tpr = np.array(roc_data["tpr"])
            auc_val = roc_data["auc"]

            # Extract k value from folder name
            k_label = os.path.basename(folder).replace("k", "")
            plt.plot(fpr, tpr, lw=2, label=f"k={k_label} (AUC={auc_val:.3f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Comparison — {family_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()