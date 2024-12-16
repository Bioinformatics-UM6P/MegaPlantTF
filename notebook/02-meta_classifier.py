#!/usr/bin/env python
# coding: utf-8

# <div style="hwidth: 100%; background-color: #ddd; overflow:hidden; ">
#     <div style="display: flex; justify-content: center; align-items: center; border-bottom: 10px solid #80c4e7; padding: 3px;">
#         <h2 style="position: relative; top: 3px; left: 8px;">S2 Project: DNA Classification - (part2: Approach 2)</h2>
#         <!--<img style="position: absolute; height: 68px; top: -2px;; right: 18px" src="./Content/Notebook-images/dna1.png"/>-->
#     </div>
#     <div style="padding: 3px 8px;">
#         
# 1. <strong>Description</strong>:
#    - In this approach, we represent DNA sequences using k-mer frequencies. Each sequence is encoded as a vector where each element represents the frequency of a specific k-mer in the sequence. This vector representation is then used as input to a neural network architecture for classification.
# 
# 2. <strong>Pros</strong>:
#    - Utilizes frequency analysis: By representing sequences based on the frequency of k-mers, the model can capture important patterns and motifs in the DNA sequences.
#    - Flexible architecture: Neural networks provide a flexible framework for learning complex relationships between features, allowing the model to adapt to different types of data.
# 
# 3. <strong>Cons</strong>:
#    - Curse of dimensionality: Depending on the value of k and the size of the alphabet (e.g., DNA bases A, C, G, T), the feature space can become very large, leading to increased computational complexity and potential overfitting.
#    - Loss of sequence information: By focusing solely on k-mer frequencies, the model may overlook important sequential dependencies and structural information present in the DNA sequences.
#     </div>    
# </div>

# ### 1 - Importing utils
# The following code cells will import necessary libraries.

# In[1]:


import os, random, string, itertools, warnings, sys, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display, HTML
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    f1_score, 
    recall_score, 
    precision_score
)

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction import DictVectorizer
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import LSTM, SimpleRNN
from keras.layers import Flatten, Embedding, BatchNormalization, Dropout, MaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import sequence
from keras.utils import Sequence
from keras.layers import Conv1D

# OS
current_directory = os.getcwd()
root_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
utils_directory = os.path.join(root_directory, 'processing')
sys.path.append(utils_directory)

# Import Utils
import fasta
from representation import DNA
from visualization import VISU, VISUReport


# In[2]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# ### 2 - Pretrained model Utils

# In[3]:


# Paths
gene_info_path = "../data/gene_info.json"
dataset_path = "../data/one_vs_other/"
pretrained_model_path = "../notebook/Output/Model"
gene_bank_folder = "../data/raw_data"

# Load gene info
with open(gene_info_path, 'r') as json_file:
    gene_info = json.load(json_file)

# Utils
def load_models(k):
    """Load multiple models from given paths."""
    models_dict = []
    for gene, info in gene_info.items():
        model_path = f"{pretrained_model_path}/{info['file_code']}/FEEDFORWARD_k{k}.keras"
        meta_path  = f"{pretrained_model_path}/{info['file_code']}/meta.json"
        model = load_model(model_path)
        with open(meta_path, 'r') as json_file:
            meta = json.load(json_file)
        feature = meta[gene.replace('/', '__')][f"FEEDFORWARD_k{k}"]["features_mask"].values()
        models_dict.append((model, feature))
    return models_dict


# ### 3 - Load data

# In[4]:


GENE_FAMILY = gene_info.keys()
gene_families_index = {gene_family: index for index, gene_family in enumerate(GENE_FAMILY)}


# In[5]:


def build_combined_df():
    combined_train_df = pd.DataFrame()
    combined_test_df  = pd.DataFrame()
    for gene_family, info in gene_info.items():
        file_path = "../data/raw_data/"+info["filename"]
        df = fasta.read(file_path, gene_families_index[gene_family])
        
        # Split the data to take 80%
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Combine the DataFrame
        combined_train_df = pd.concat([combined_train_df, train_df], ignore_index=True)
        combined_test_df = pd.concat([combined_test_df, test_df], ignore_index=True)
    
    return combined_train_df, combined_test_df


# Build the combined DataFrame
train_df, test_df = build_combined_df()


# # 3 - Pipeline

# * **Data Mask fit**

# In[6]:


class DataFrameProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.sort_index(axis=1)
    
    def fit_mask(self, feature_array):
        df_copy = self.dataframe.copy()
        existing_columns = df_copy.columns.intersection(feature_array)
        df_copy = df_copy[existing_columns]
        new_df = pd.DataFrame(0, index=df_copy.index, columns=feature_array)
        for col in existing_columns:
            new_df[col] = df_copy[col]
        return new_df

    def get(self):
        return self.dataframe


# * **Data Generator**

# In[7]:


class DataGenerator(Sequence):
    def __init__(self, df, models_dict, gene_info, batch_size=32, k=2):
        self.df = df
        self.models_dict = models_dict
        self.gene_info = gene_info
        self.batch_size = batch_size
        self.k = k
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[indexes]
        X, y = self.__data_generation(batch_df)
        return X, y

    def __data_generation(self, batch_df):
        kmer_features, y_kmer, _ = DNA.build_kmer_representation_v2(train_data, k=self.k)
        X_kmer = DataFrameProcessor(kmer_features)
        
        predictions = []
        for model, feature in self.models_dict:
            X_test = X_kmer.fit_mask(feature)
            pred = model.predict(X_test, verbose=0)
            predictions.append(pred)
        predictions = np.array(predictions)
        shape = predictions.shape
        X_batch = predictions.reshape(shape[0], shape[1]).T
        y_batch = np.array(y_kmer)
        
        return X_batch, y_batch


# * **Model Utils**

# In[8]:


domaine = "ACDEFGHIKLMNPQRSTVWYX"
def model_checkpoint(model_name):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath="Output/MetaClassifier/"+model_name+".keras", 
        monitor='val_loss', 
        verbose=0, 
        save_best_only=True, 
        save_weights_only=False
    )
def early_stopping(patience=10):
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        verbose=0,
    )


# ### 4 - Training and Testing

# <h4 style="background-color: #80c4e6; display: flex;">
#     <ul><li>k=2</li></ul>
# </h4>

# In[9]:


k = 2
models_dict = load_models(k)


# In[10]:


train_data, validation_data = train_test_split(train_df, train_size=0.8, stratify=train_df['class'])

batch_size=4096
training_generator   = DataGenerator(train_data, models_dict, gene_info, batch_size=batch_size, k=k)
validation_generator = DataGenerator(validation_data, models_dict, gene_info, batch_size=batch_size, k=k)
test_generator       = DataGenerator(test_df, models_dict, gene_info, batch_size=batch_size, k=k)


# * <span style="color: blue; font-weight: bold;">FEED-FORWARD META CLASSIFIER</span>

# In[ ]:


name="META_STAGE2_k2"
def build_stage2_classifier():
    meta_model = Sequential(name=name)
    meta_model.add(Dense(128, input_dim=len(models_dict), activation='relu'))
    meta_model.add(Dropout(0.1))
    #meta_model.add(Dense(64, activation='relu'))
    meta_model.add(Dense(len(gene_info), activation='softmax'))
    meta_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    meta_model.summary()
    return meta_model

# Build
meta_model = build_stage2_classifier()
stop_callback = early_stopping(patience=5)
save_callback = model_checkpoint(name)

# Train
history = meta_model.fit(training_generator, validation_data=validation_generator, epochs=10, callbacks=[stop_callback, save_callback])

# Evaluate
# Evaluate and score
test_scores = model.evaluate(test_generator, verbose=0)
train_score = history.history.get('accuracy')[-1]
print("\n[Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}%]".format(train_score*100, test_scores[1]*100))


# In[ ]:


VISU.plot_curve(history, ['loss', 'val_loss', 'accuracy', 'val_accuracy'])


# In[ ]:

######################################### DEBUG TO OPTIMIZE K-MER LOEADER FUNCTION ###########################################




