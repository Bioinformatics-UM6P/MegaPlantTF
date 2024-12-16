import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_testset(true_label_df_path="", class_mapping_df_path=""):
    true_label = pd.read_csv(true_label_df_path)['true_label'].values
    with open(class_mapping_df_path, 'r') as json_file:
        class_mapping = json.load(json_file)
    reverse_class_mapping = {v: k for k, v in class_mapping.items()}
    label_counts = pd.Series(true_label).value_counts().sort_index()
    label_counts.index = label_counts.index.map(reverse_class_mapping)
    label_counts = label_counts.sort_values(ascending=False)
    
    # Plot the bar plot
    # Plot the bar plot
    plt.figure(figsize=(16, 6))
    ax = label_counts.plot(kind='bar')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Distribution of True Labels')
    plt.xticks(rotation=90)
    
    # Annotate each bar with the value counts
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.tight_layout()
    plt.show()