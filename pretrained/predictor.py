# utils
import json, sys, os
current_directory = os.getcwd()
current_abs_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
utils_directory = os.path.join(root_directory, 'processing')
sys.path.append(utils_directory)
sys.path.append(current_abs_directory)

# import
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import load_model
from representation import DNA
import fasta
from generator import DataFrameProcessor
from histogram import GenBoard
import joblib

class SingleKModel:
    def __init__(self, kmer_size):
        self.k = kmer_size
        self.domaine="ACDEFGHIKLMNPQRSTVWYX"
        self.gene_info_path = "../data/gene_info.json"
        #self.pretained_model_path = "../models/v1-beta/models"
        self.pretained_model_path = "../notebook/Output/Model-ANOVA"
        self.stage2_classifier_path = f"../notebook/Output/MetaClassifier-ANOVA/META_k{self.k}.keras"
        self.use_weight = False
        with open(self.gene_info_path, 'r') as json_file:
            self.gene_info = json.load(json_file)
        self.TestSet=None
        self.kmerSet=None
        if "joblib" in self.stage2_classifier_path:
            self.metaModel = joblib.load(self.stage2_classifier_path)
        else:
            self.metaModel = load_model(self.stage2_classifier_path)

    def load(self, fasta_path, format, type='gene_file'):
        """Read a FASTA file and return a list of sequences."""
        if not os.path.isfile(fasta_path):
            raise FileNotFoundError(f"FASTA file '{fasta_path}' not found.")
            
        if format == "fasta" and type == "gene_file":
            self.TestSet = fasta.read_fas(fasta_path)
        elif format == "csv" and type == "gene_file":
            self.TestSet = fasta.read_csv(fasta_path)
        elif format == "csv" and type == "kmer_file":
            self.TestSet = pd.read_csv(fasta_path)
        else:
            raise ValueError(f"The '{format}' format is not supported !")

        if type == "gene_file":
            temp_df = DNA.build_kmer_prediction_set(self.TestSet, domaine=self.domaine, k=self.k, dtypes=['float64', 'int64'], batch_size=1000, feature_mask=None)
        else:
            self.TestSet.drop('class', axis=1, inplace=True)
            temp_df = self.TestSet.drop('id', axis=1)
        self.kmerSet = DataFrameProcessor(temp_df)

    def load_models(self):
        """Load multiple models from given paths."""
        models_dict = []
        for gene, info in self.gene_info.items():
            model_path = f"{self.pretained_model_path}/{info['file_code']}/FEEDFORWARD_k{self.k}.keras"
            meta_path  = f"{self.pretained_model_path}/{info['file_code']}/meta.json"
            model = load_model(model_path)
            with open(meta_path, 'r') as json_file:
                meta = json.load(json_file)
            feature = meta[gene.replace('/', '__')][f"FEEDFORWARD_k{self.k}"]["features_mask"].values()
            models_dict.append((model, feature))
        return models_dict
    
    def weighted_average_predictions(self, predictions, weights):
        """Compute the weighted average of predictions."""
        weighted_preds = np.zeros(predictions[0].shape)
        for pred, weight in zip(predictions, weights):
            weighted_preds += weight * pred
        return weighted_preds / sum(weights)
    
    def predict(self):
        """Load models and make predictions."""
        model_dict = self.load_models()
        data_sizes = [self.gene_info[gene]['count']*2 for gene in self.gene_info.keys()]
        weights = [size / sum(data_sizes) for size in data_sizes]

        # predict
        predictions = []
        for model, feature_mask in tqdm(model_dict, desc="Predicting"):
            X_test = self.kmerSet.fit_mask(feature_mask)
            pred = model.predict(X_test, verbose=0)
            predictions.append(pred)
        # Compute final weighted prediction
        final_prediction = np.array(self.weighted_average_predictions(predictions, weights) if self.use_weight else predictions)
        shape = final_prediction.shape
        final_prediction = final_prediction.reshape(shape[0], shape[1]).T

        # Convert to DataFrame --> (final_prediction > 0.5).astype(int)
        final_prediction_df = pd.DataFrame(final_prediction, columns=self.gene_info.keys())
        genboard = GenBoard(final_prediction_df)
        genboard.add_initial_set(self.TestSet)
        genboard.add_kmer_set(self.kmerSet.get())
        genboard.add_model(model_dict)

        # Stage 2 prediction
        predictions2 = self.metaModel.predict_proba(final_prediction) if "joblib" in self.stage2_classifier_path else self.metaModel.predict(final_prediction, verbose=0)
        predictions_stage2_df = pd.DataFrame(predictions2, columns=self.gene_info.keys())
        genboard.add_stage2_result(predictions_stage2_df)
        genboard.set_kmer_size(self.k)
        genboard.set_stage2_model(self.metaModel)

        return genboard

class BatchSingleKModel:
    def __init__(self, kmer_size, batch_size):
        self.k = kmer_size
        self.batch_size = batch_size
        self.domaine = "ACDEFGHIKLMNPQRSTVWYX"
        self.gene_info_path = "../data/gene_info.json"
        self.pretained_model_path = "../notebook/Output/Model-ANOVA"
        self.stage2_classifier_path = f"../notebook/Output/MetaClassifier-ANOVA/META_k{self.k}.keras"
        self.use_weight = False
        with open(self.gene_info_path, 'r') as json_file:
            self.gene_info = json.load(json_file)
        self.TestSet = None
        self.kmerSet = None
        if "joblib" in self.stage2_classifier_path:
            self.metaModel = joblib.load(self.stage2_classifier_path)
        else:
            self.metaModel = load_model(self.stage2_classifier_path)
        self.input_file_type = None
        self.input_format = None
        self.input_path = None

    def set_load_config(self, fasta_path, format, type='gene_file'):
        """Read a FASTA file and return a list of sequences."""
        if not os.path.isfile(fasta_path):
            raise FileNotFoundError(f"FASTA file '{fasta_path}' not found.")
            
        elif format == "csv" and type == "kmer_file":
            self.input_file_type = type
            self.input_format = format
            self.input_path = fasta_path
            #self.TestSet = pd.read_csv(fasta_path)
            #self.TestSet.drop('class', axis=1, inplace=True)
        else:
            raise ValueError(f"The '{format}' format is not supported !")

    def load_models(self):
        """Load multiple models from given paths."""
        models_dict = []
        for gene, info in self.gene_info.items():
            model_path = f"{self.pretained_model_path}/{info['file_code']}/FEEDFORWARD_k{self.k}.keras"
            meta_path = f"{self.pretained_model_path}/{info['file_code']}/meta.json"
            model = load_model(model_path)
            with open(meta_path, 'r') as json_file:
                meta = json.load(json_file)
            feature = meta[gene.replace('/', '__')][f"FEEDFORWARD_k{self.k}"]["features_mask"].values()
            models_dict.append((model, feature))
        return models_dict

    def weighted_average_predictions(self, predictions, weights):
        """Compute the weighted average of predictions."""
        weighted_preds = np.zeros(predictions[0].shape)
        for pred, weight in zip(predictions, weights):
            weighted_preds += weight * pred
        return weighted_preds / sum(weights)

    def predict(self):
        """Load models and make predictions in batches."""
        model_dict = self.load_models()
        data_sizes = [self.gene_info[gene]['count'] * 2 for gene in self.gene_info.keys()]
        weights = [size / sum(data_sizes) for size in data_sizes]

        all_predictions = []

        # Process data in batches
        for chunk in tqdm(pd.read_csv(self.input_path, chunksize=self.batch_size), desc="Batch Predictions"):
            batch_test_set = chunk
            temp_df = batch_test_set.drop(columns=['id', 'class'], axis=1)

            self.TestSet = batch_test_set
            self.kmerSet = kmerSet_batch = DataFrameProcessor(temp_df)

            batch_predictions = []
            for model, feature_mask in model_dict:
                X_test = kmerSet_batch.fit_mask(feature_mask)
                pred = model.predict(X_test, verbose=0)
                batch_predictions.append(pred)

            # Compute final weighted prediction
            final_batch_prediction = np.array(self.weighted_average_predictions(batch_predictions, weights) if self.use_weight else batch_predictions)
            shape = final_batch_prediction.shape
            final_batch_prediction = final_batch_prediction.reshape(shape[0], shape[1]).T
            all_predictions.append(final_batch_prediction)

        # Concatenate all batch predictions
        final_prediction = np.vstack(all_predictions)

        # Convert to DataFrame
        final_prediction_df = pd.DataFrame(final_prediction, columns=self.gene_info.keys())
        genboard = GenBoard(final_prediction_df)
        genboard.add_initial_set(self.TestSet)
        genboard.add_kmer_set(pd.DataFrame(columns=list(self.gene_info.keys())))
        genboard.add_model(model_dict)

        # Stage 2 prediction
        predictions2 = self.metaModel.predict_proba(final_prediction) if "joblib" in self.stage2_classifier_path else self.metaModel.predict(final_prediction, verbose=0)
        predictions_stage2_df = pd.DataFrame(predictions2, columns=self.gene_info.keys())
        genboard.add_stage2_result(predictions_stage2_df)
        genboard.set_kmer_size(self.k)

        return genboard
        
class MultiKModel:
    def __init__(self, kmer_size=[2, 3, 4, 5]):
        """Will be developed"""
        pass

class OneTestKModel:
    def __init__(self, kmer_size):
        self.k = kmer_size
        self.domaine = "ACDEFGHIKLMNPQRSTVWYX"
        self.gene_info_path = "../data/gene_info.json"
        self.pretained_model_path = "../notebook/Output/Model"
        self.use_weight = False
        with open(self.gene_info_path, 'r') as json_file:
            self.gene_info = json.load(json_file)
        self.TestSet = None
        self.kmerSet = None

    def process_sequence(self, sequence):
        """Convert a sequence string into a k-mer set DataFrame."""
        temp_df = DNA.kmer_count_v2(sequence, self.domaine, self.k, step=1) 
        kmer_data = {'kmer': [temp_df]}
        df = pd.DataFrame(kmer_data)
        self.kmerSet = DataFrameProcessor(df)

    def load_models(self):
        """Load multiple models from given paths."""
        models_dict = []
        for gene, info in self.gene_info.items():
            model_path = f"{self.pretained_model_path}/{info['file_code']}/FEEDFORWARD_k{self.k}.keras"
            meta_path = f"{self.pretained_model_path}/{info['file_code']}/meta.json"
            model = load_model(model_path)
            with open(meta_path, 'r') as json_file:
                meta = json.load(json_file)
            feature = list(meta[gene.replace('/', '__')][f"FEEDFORWARD_k{self.k}"]["features_mask"].values())
            models_dict.append((model, feature))
        return models_dict

    def weighted_average_predictions(self, predictions, weights):
        """Compute the weighted average of predictions."""
        weighted_preds = np.zeros(predictions[0].shape)
        for pred, weight in zip(predictions, weights):
            weighted_preds += weight * pred
        return weighted_preds / sum(weights)

    def analyze(self):
        """Process a sequence string, make predictions, and return a DataFrame of predictions."""
        model_dict = self.load_models()
        data_sizes = [self.gene_info[gene]['count'] * 2 for gene in self.gene_info.keys()]
        weights = [size / sum(data_sizes) for size in data_sizes]

        predictions = []
        for model, feature_mask in tqdm(model_dict, desc="Predicting"):
            X_test = self.kmerSet.fit_mask(feature_mask)
            pred = model.predict(X_test, verbose=0)
            predictions.append(pred)
        
        final_prediction = np.array(self.weighted_average_predictions(predictions, weights) if self.use_weight else predictions)
        shape = final_prediction.shape
        final_prediction = final_prediction.reshape(shape[0], shape[1]).T

        final_prediction_df = pd.DataFrame(final_prediction, columns=self.gene_info.keys())
        return final_prediction_df
        