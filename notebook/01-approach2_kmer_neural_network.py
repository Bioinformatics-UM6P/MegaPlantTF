#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys

def main(gene_familly):
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
    
    # In[2]:
    
    
    import os, random, string, itertools, warnings, sys, json
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from IPython.display import display, HTML
    from sklearn.metrics import (
        confusion_matrix, 
        classification_report, 
        accuracy_score, 
        f1_score, 
        recall_score, 
        precision_score
    )
    
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.sequence import pad_sequences
    from sklearn.feature_extraction import DictVectorizer
    from keras.models import Sequential
    from keras.layers import Dense, Input
    from keras.layers import LSTM, SimpleRNN
    from keras.layers import Flatten, Embedding, BatchNormalization, Dropout, MaxPooling1D, GlobalAveragePooling1D
    from keras.preprocessing import sequence
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
    
    
    # In[3]:
    
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    
    # ### 2 - Importing Dataset
    # The following function will read our preprocessed **.csv file** and return a pandas dataframe
    
    # In[4]:
    
    
    # READ GENE_INFO JSON FILE
    
    gene_info_path = "../data/gene_info.json"
    dataset_path   = "../data/one_vs_other/"
    with open(gene_info_path, 'r') as json_file:
        gene_info = json.load(json_file)
    
    
    # In[5]:
    
    
    # FOCUS ON GENE FAMALLY
    
    gene_dict = gene_info[gene_familly]
    df_path = dataset_path+gene_dict['file_code']+".csv"
    
    
    # In[6]:
    
    
    USE_FULL_DF = True
    
    if USE_FULL_DF:
        dataset = pd.read_csv(df_path)
    else:
        dataset_ = pd.read_csv(df_path)
        r = min(5000/len(dataset_), 1)
        _, dataset = train_test_split(dataset_, test_size=r, stratify=dataset_['class'], random_state=42)
    dataset.head()
    
    
    # In[7]:
    
    
    dataset.info()
    
    
    # In[8]:
    
    
    sns.set(style="whitegrid")
    sns.violinplot(x=dataset.length)
    plt.title("sequence length distribution in test sequences")
    plt.show()
    
    
    # In[154]:
    
    
    report = VISUReport(gene_familly, dataset)
    
    
    # * **Model Utils**
    
    # In[155]:
    
    
    domaine = "ACDEFGHIKLMNPQRSTVWYX"
    def model_checkpoint(model_name):
        gene_familly_ = gene_familly.replace('/', '__')
        return tf.keras.callbacks.ModelCheckpoint(
            filepath="Output/Model/"+gene_familly_+"/"+model_name+".keras", 
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
    
    # In[156]:
    
    """
    k = 2
    X, y, features_k2 = DNA.build_kmer_representation_v2(dataset, domaine=domaine, k=k, dtypes=['float16', 'int8'], asCudaDF=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) #, random_state=42
    X_test.head()
    
    
    # In[157]:
    
    
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    
    NUM_CLASS  = 1
    SEQ_LENGTH = X_train.shape[1]
    
    
    # * <span style="color: blue; font-weight: bold;">MODEL 1 : FEED-FORWARD NETWORKS</span>
    
    # In[158]:
    
    
    name="FEEDFORWARD_k2"
    def feedforward_net1(name=name, num_output=NUM_CLASS, seq_length=SEQ_LENGTH):
        model = Sequential(name=name)
        model.add(Input(shape=(SEQ_LENGTH,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(rate=0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(rate=0.1))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(rate=0.1))
        model.add(Dense(num_output, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
    
    # Build & train the model
    model = feedforward_net1()
    stop_callback = early_stopping(patience=10)
    save_callback = model_checkpoint(name)
    
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, train_size=0.8, stratify=y_train)
    history = model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=100, batch_size=64, callbacks=[stop_callback])
    
    # Evaluate and score
    test_scores = model.evaluate(X_test, y_test, verbose=0)
    train_score = history.history.get('accuracy')[-1]
    print("\n[Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}%]".format(train_score*100, test_scores[1]*100))
    
    
    # In[160]:
    
    
    VISU.plot_curve(history, ['loss', 'val_loss', 'accuracy', 'val_accuracy'])
    
    
    # In[172]:
    
    
    #VISU.test_report(X_test, y_test, model=model,  args=[model.name, test_scores[1]*100, gene_familly, features_k2, len(dataset)]) 
    report.add_report(X_test, y_test, model=model, history=history, args=[model.name, "---"])
    
    
    # <h4 style="background-color: #80c4e6; display: flex;">
    #     <ul><li>k=3</li></ul>
    # </h4>
    
    # In[14]:
    
    
    k = 3
    X, y, features_k3 = DNA.build_kmer_representation_v2(dataset, domaine=domaine, k=k, dtypes=['float16', 'int8'], asCudaDF=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test.head()
    
    
    # In[15]:
    
    
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    
    NUM_CLASS  = 1
    SEQ_LENGTH = X_train.shape[1]
    
    
    # * <span style="color: blue; font-weight: bold;">MODEL 1 : FEED-FORWARD NETWORKS</span>
    
    # In[16]:
    
    
    name="FEEDFORWARD_k3"
    def feedforward_net1(name=name, num_output=NUM_CLASS, seq_length=SEQ_LENGTH):
        model = Sequential(name=name)
        model.add(Input(shape=(SEQ_LENGTH,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(rate=0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(rate=0.1))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(rate=0.1))
        model.add(Dense(num_output, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
    
    # Build & train the model
    model = feedforward_net1()
    stop_callback = early_stopping(patience=10)
    save_callback = model_checkpoint(name)
    
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, train_size=0.8, stratify=y_train)
    history = model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=100, batch_size=64, callbacks=[stop_callback])
    
    # Evaluate and score
    test_scores = model.evaluate(X_test, y_test, verbose=0)
    train_score = history.history.get('accuracy')[-1]
    print("\n[Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}%]".format(train_score*100, test_scores[1]*100))
    
    
    # In[17]:
    
    
    VISU.plot_curve(history, ['loss', 'val_loss', 'accuracy', 'val_accuracy'])
    
    
    # In[18]:
    
    
    #VISU.test_report(X_test, y_test, model=model,  args=[model.name, test_scores[1]*100, gene_familly, features_k3, len(dataset)]) 
    report.add_report(X_test, y_test, model=model, history=history, args=[model.name, "---"])
    
    
    # <h4 style="background-color: #80c4e6; display: flex;">
    #     <ul><li>k=4</li></ul>
    # </h4>
    
    # In[19]:
    
    
    k = 4
    X, y, features_k4 = DNA.build_kmer_representation_v2(dataset, domaine=domaine, k=k, dtypes=['float16', 'int8'], asCudaDF=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test.head()
    
    
    # In[20]:
    
    
    NUM_CLASS  = 1
    SEQ_LENGTH = X_train.shape[1]
    
    
    # * <span style="color: blue; font-weight: bold;">MODEL 2 : FEED-FORWARD NETWORKS</span>
    
    # In[21]:
    
    
    name="FEEDFORWARD_k4"
    def feedforward_net1(name=name, num_output=NUM_CLASS, seq_length=SEQ_LENGTH):
        model = Sequential(name=name)
        model.add(Input(shape=(SEQ_LENGTH,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(num_output, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
    
    # Build & train the model
    model = feedforward_net1()
    stop_callback = early_stopping(patience=10)
    save_callback = model_checkpoint(name)
    
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, train_size=0.8, stratify=y_train)
    history = model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=100, batch_size=64, callbacks=[stop_callback])
    
    
    # Evaluate and score
    test_scores = model.evaluate(X_test, y_test, verbose=0)
    train_score = history.history.get('accuracy')[-1]
    print("\n[Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}%]".format(train_score*100, test_scores[1]*100))
    
    
    # In[22]:
    
    
    VISU.plot_curve(history, ['loss', 'val_loss', 'accuracy', 'val_accuracy'])
    
    
    # In[23]:
    
    
    #VISU.test_report(X_test, y_test, model=model,  args=[model.name, test_scores[1]*100, gene_familly, features_k4, len(dataset)]) 
    report.add_report(X_test, y_test, model=model, history=history, args=[model.name, "---"])
    
    
    # In[24]:
    
    
    #report.save()
    
    
    # <h4 style="background-color: #80c4e6; display: flex;">
    #     <ul><li>k=5</li></ul>
    # </h4>
    
    # In[25]:
    """
    
    k = 5
    print('k=5')
    X, y, features_k5 = DNA.build_kmer_representation_v2(dataset, domaine=domaine, k=k, dtypes=['float16', 'int8'], asCudaDF=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test.head()
    
    
    # In[26]:
    
    
    NUM_CLASS  = 1
    SEQ_LENGTH = X_train.shape[1]
    
    
    # * <span style="color: blue; font-weight: bold;">MODEL 3 : FEED-FORWARD NETWORKS</span>
    
    # In[27]:
    
    
    name="FEEDFORWARD_k5"
    def feedforward_net1(name=name, num_output=NUM_CLASS, seq_length=SEQ_LENGTH):
        model = Sequential(name=name)
        model.add(Input(shape=(SEQ_LENGTH,)))
        model.add(Dense(2*256, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(num_output, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
    
    # Build & train the model
    model = feedforward_net1()
    stop_callback = early_stopping(patience=10)
    save_callback = model_checkpoint(name)
    
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, train_size=0.8, stratify=y_train)
    history = model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=100, batch_size=64, callbacks=[stop_callback])
    
    # Evaluate and score
    test_scores = model.evaluate(X_test, y_test, verbose=0)
    train_score = history.history.get('accuracy')[-1]
    print("\n[Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}%]".format(train_score*100, test_scores[1]*100))
    
    
    # In[28]:
    
    
    VISU.plot_curve(history, ['loss', 'val_loss', 'accuracy', 'val_accuracy'])
    
    
    # In[29]:
    
    
    #VISU.test_report(X_test, y_test, model=model,  args=[model.name, test_scores[1]*100, gene_familly, features_k5, len(dataset)]) 
    report.add_report(X_test, y_test, model=model, history=history, args=[model.name, "---"])
    report.save('end')
    
    
    # In[30]:
    
    
    # END
    
    
    # In[31]:
    
    
    ######################################### DEBUG TO OPTIMIZE K-MER LOEADER FUNCTION ###########################################
    
    
    # In[ ]:

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 01-approach2_kmer_neural_network.py <gene_family>")
        sys.exit(1)
    gene_family = sys.argv[1]
    main(gene_family)


