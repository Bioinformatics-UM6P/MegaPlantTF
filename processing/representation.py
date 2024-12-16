import numpy as np
import itertools
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from multiprocessing import Pool
import time
from numba import jit
import numpy as np
import pandas as pd
import os
import csv
import sklearn
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from collections import defaultdict
from keras.utils import Sequence
from scipy.sparse import save_npz
import uuid

class DNA:
    @staticmethod
    def kmer_count_v2(sequence, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, step=1):
        """
        Utils: to count kmer occurrence in DNA sequence and compute frequency
        """
        kmers_count = defaultdict(int)
        total_kmers = 0
        for i in range(0, len(sequence) - k + 1, step):
            kmer = sequence[i:i + k]
            kmers_count[kmer] += 1
            total_kmers += 1
        for key in kmers_count:
            kmers_count[key] /= total_kmers
        return kmers_count

    @staticmethod
    def build_kmer_representation_v3(df, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, dtypes=['float64', 'int64'], asCudaDF=False, batch_size=1000, feature_mask=None):
        sequences = df['sequence']
        y = df['class']
        
        # Initialize DictVectorizer
        v = DictVectorizer(sparse=True)
        
        # Create a temporary directory for storing batch results
        temp_dir = "./Temp/kmer_batch_results"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process in batches and save to files
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_kmers_count = [DNA.kmer_count_v2(sequence, domaine, k=k, step=1) for sequence in batch_sequences]
            batch_file = os.path.join(temp_dir, f"batch_{i // batch_size}.npz")
            feature_values = v.fit_transform(batch_kmers_count)
            save_npz(batch_file, feature_values)
            with open(os.path.join(temp_dir, f"batch_{i // batch_size}_labels.pkl"), 'wb') as f:
                pickle.dump(y[i:i + batch_size], f)
        
        return v, temp_dir, len(sequences), batch_size

    @staticmethod
    def build_kmer_representation_v2(df, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, dtypes=['float64', 'int64'], asCudaDF=False, batch_size=1000, feature_mask=None):
        """
        Utils: For given k-mer generate dataset and return vectorized version
        """
        sequences = df['sequence']
        y = df['class']#.astype(dtypes[1])
        
        # Initialize DictVectorizer
        v = DictVectorizer(sparse=True)
        kmers_count_list = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_kmers_count = [DNA.kmer_count_v2(sequence, domaine, k=k, step=1) for sequence in batch_sequences]
            kmers_count_list.extend(batch_kmers_count)
        
        # Vectorize the kmer counts
        feature_values = v.fit_transform(kmers_count_list)
        feature_names = v.get_feature_names_out()
        
        # Convert to DataFrame
        X = pd.DataFrame.sparse.from_spmatrix(feature_values, columns=feature_names)#.astype(dtypes[0])
        X = X.sparse.to_dense()

        # Apply feature mask if provided
        if feature_mask is not None:
            # Ensure feature_mask is a set for quick lookup
            feature_mask_set = set(feature_mask)
            current_features = set(X.columns)
            for feature in feature_mask_set - current_features:
                X[feature] = 0
            X = X[feature_mask]
        
        if asCudaDF:
            import cudf
            X_cuda = cudf.DataFrame.from_pandas(X)
            y_cuda = cudf.Series(y)
            return X_cuda, y_cuda, feature_names
        
        return X, y, feature_names
        
class DNA_FULL:
    @staticmethod
    def one_hot_encoding(sequences, max_length=100):
        """
        One-hot encode a list of DNA sequences.

        Parameters:
        sequences (list of str): List of DNA sequences.
        max_length (int): Maximum length of the sequences. Sequences longer than this will be truncated,
                          and sequences shorter than this will be padded with 'N'.

        Returns:
        np.ndarray: A 3D numpy array of shape (num_sequences, max_length, 4) representing the one-hot encoded sequences.
        """
        # Define a dictionary to map nucleotides to integers
        nucleotide_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

        # Initialize an empty array for the one-hot encoded sequences
        one_hot_encoded = np.zeros((len(sequences), max_length, 4), dtype=int)

        for i, sequence in enumerate(sequences):
            # Truncate or pad the sequence to the maximum length
            sequence = sequence[:max_length].ljust(max_length, 'N')
            for j, nucleotide in enumerate(sequence):
                if nucleotide in nucleotide_to_int and nucleotide != 'N':
                    one_hot_encoded[i, j, nucleotide_to_int[nucleotide]] = 1
        return one_hot_encoded
        
    @staticmethod
    def kmer_count(sequence, domaine="ATCG", k=3, step=1):
        """
        Utils: to count kmer occurrence in DNA sequence and compute frequency.
        """
        start_time = time.time()  # Start the timer
        kmers = [''.join(p) for p in itertools.product(domaine, repeat=k)]
        kmers_count = {kmer: 0 for kmer in kmers}
        s = 0
        
        for i in range(0, len(sequence) - k + 1, step):
            kmer = sequence[i:i + k]
            s += 1
            if kmer in kmers:
                kmers_count[kmer] += 1
        for key in kmers_count:
            kmers_count[key] = kmers_count[key] / s
            
        end_time = time.time()
        elapsed_time = end_time - start_time        
        return kmers_count
    
    @staticmethod
    def build_kmer_representation(df, domaine="ATCG", k=3, dtypes=['float64', 'int64'], asCudaDF=False):
        """
        Utils: For given k-mer generate dataset and return vectorised version
        """
        # Count
        sequences   = df['sequence']
        kmers_count = np.array([DNA_FULL.kmer_count(sequence, domaine, k=k, step=1) for sequence in sequences])
        
        # Vectorize
        v = DictVectorizer(sparse=False)
        feature_values = v.fit_transform(kmers_count)
        feature_names = v.get_feature_names_out()

        # dtypes and save df
        dtype = {col: dtypes[0] for col in feature_names}
        X = pd.DataFrame(feature_values, columns=feature_names).astype(dtype)
        y = df['class'].astype(dtypes[1])

        
        if asCudaDF:
            import cudf 
            X_cuda = cudf.DataFrame.from_pandas(X)
            y_cuda = cudf.Series(y)
            return X_cuda, y_cuda, feature_names
            
        return X, y, feature_names

class DNA_MPI:
    def kmer_count_v2(sequence, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, step=1):
        """
        Utils: to count kmer occurrence in DNA sequence and compute frequency
        """
        kmers_count = defaultdict(int)
        total_kmers = 0
        for i in range(0, len(sequence) - k + 1, step):
            kmer = sequence[i:i + k]
            kmers_count[kmer] += 1
            total_kmers += 1
        for key in kmers_count:
            kmers_count[key] /= total_kmers
        return kmers_count

    @staticmethod
    def build_kmer_representation_v2(input_file, domaine, k=3, workers=4, output_file='./Content/Data/kmer_sample.csv', script_path='./path/to/mpi_dna.py'):
        from mpi4py import MPI
        # Execute the MPI command with the full path to the script
        os.system(f"mpirun -n {workers} python3 {script_path} {input_file} {domaine} {k} {output_file}")
        
        # Read the output file to get the data
        kmer_df = pd.read_csv(output_file)
        X = kmer_df.drop(columns=['class'])  # All columns except 'class'
        y = kmer_df['class']  # The 'class' column
        return X, y, X.columns.values
        
    """@staticmethod
    def build_kmer_representation_v2(input_file, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, workers=4, output_file='./Content/Data/kmer_sample.csv'):
        os.system(f"mpirun -n {workers} python3 mpi_dna.py {input_file} {domaine} {k} {output_file}")
        
        # Read the output file to get the data
        kmer_df = pd.read_csv(output_file)
        X = kmer_df.drop(columns=['class'])
        y = kmer_df['class']
        return X, y, X.columns.values"""
    
    @staticmethod
    def _build_kmer_representation_v2(input_file, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, output_file='./Content/Data/kmer_sample.csv'):
        """
        Utils: For given k-mer generate dataset and return vectorized version or
               MPI version to count kmer occurrence in DNA sequences and save a vectorized
               representation to a CSV file.
        """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            df = pd.read_csv(input_file)
            chunks = np.array_split(df, size)
        else:
            chunks = None

        chunk = comm.scatter(chunks, root=0)
        kmer_counts = [DNA_MPI.kmer_count_v2(seq, domaine, k) for seq in chunk['sequence']]
        all_kmer_counts = comm.gather(kmer_counts, root=0)
        all_y = comm.gather(chunk['class'].tolist(), root=0)  # gather class column

        if rank == 0:
            all_kmer_counts = [item for sublist in all_kmer_counts for item in sublist]
            all_y = [item for sublist in all_y for item in sublist]

            v = DictVectorizer(sparse=False)
            feature_values = v.fit_transform(all_kmer_counts)
            feature_names = v.get_feature_names_out()

            # Create DataFrame for features and class column
            X = pd.DataFrame(feature_values, columns=feature_names)
            y = pd.Series(all_y, name='class')

            # Combine features and class column
            kmer_df = pd.concat([X, y], axis=1)
            kmer_df.to_csv(output_file, index=False)

    """
    # Vectorize
        v = DictVectorizer(sparse=False)
        feature_values = v.fit_transform(kmers_count)
        feature_names = v.get_feature_names_out()

        # dtypes and save df
        dtype = {col: dtypes[0] for col in feature_names}
        X = pd.DataFrame(feature_values, columns=feature_names).astype(dtype)
        y = df['class'].astype(dtypes[1])
    """
    """@staticmethod
    def _build_kmer_representation_v2(input_file="path/to/df.csv", domaine="", k=3, output_file='path/to/output.csv'):
        ""
        Utils: For given k-mer generate dataset and return vectorized version using multi processing and efficient memory sage
        ""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if rank == 0:
            # Master process
            df = pd.read_csv(input_file)
            chunk_size = len(df) // workers
            chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(workers)]

            if len(df) % workers != 0:
                chunks[-1] = pd.concat([chunks[-1], df.iloc[workers*chunk_size:]])

            # Save the chunks as temporary CSV files
            os.makedirs('temp_chunks', exist_ok=True)
            for i, chunk in enumerate(chunks):
                chunk.to_csv(f'temp_chunks/chunk_{k}_{i}.csv', index=False)
            
            # Create empty output files for each worker
            for i in range(workers):
                with open(f'temp_chunks/output_{i}.csv', 'w') as f:
                    pass

        # Synchronize all processes
        comm.Barrier()
        if rank < workers:
            chunk_df = pd.read_csv(f'temp_chunks/chunk_{k}_{rank}.csv')
            sequences = chunk_df['sequence']
            targets = chunk_df['class']

            # Compute k-mer counts
            kmer_dicts = []
            for i, sequence in enumerate(sequences):
                kmer_dict = DNA.kmer_count(sequence, domaine, k=k, step=1)
                kmer_dict['class'] = targets.iloc[i]
                kmer_dicts.append(kmer_dict)

            # Save k-mer counts to CSV
            temp_output_file = f'temp_chunks/output_{rank}.csv'
            with open(temp_output_file, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(kmer_dicts[0].keys()))
                writer.writeheader()
                writer.writerows(kmer_dicts)

        # Synchronize all processes
        comm.Barrier()
        if rank == 0:
            with open(output_file, mode='w', newline='') as final_output:
                writer = None
                for i in range(workers):
                    temp_output_file = f'temp_chunks/output_{i}.csv'
                    with open(temp_output_file, mode='r') as file:
                        reader = csv.DictReader(file)
                        if writer is None:
                            writer = csv.DictWriter(final_output, fieldnames=reader.fieldnames)
                            writer.writeheader()
                        for row in reader:
                            writer.writerow(row)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[workers={workers}]\t - Execution time: {elapsed_time:.6f} seconds")
            shutil.rmtree('temp_chunks')
            return True
            return pd.read_csv(output_file)"""
                
class DNA:  
    @staticmethod
    def kmer_count_v2(sequence, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, step=1):
        """
        Utils: to count kmer occurrence in DNA sequence and compute frequency
        """
        kmers_count = defaultdict(int)
        total_kmers = 0
        for i in range(0, len(sequence) - k + 1, step):
            kmer = sequence[i:i + k]
            kmers_count[kmer] += 1
            total_kmers += 1
        for key in kmers_count:
            kmers_count[key] /= total_kmers
        return kmers_count

    def kmer_count_v3(sequence, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, step=1, feature_mask=None):
        """
        Utils: to count kmer occurrence in DNA sequence and compute frequency
        """
        kmers_count = defaultdict(int)
        total_kmers = 0
        
        if feature_mask is not None:
            feature_mask_set = set(feature_mask)
        else:
            feature_mask_set = None
        
        for i in range(0, len(sequence) - k + 1, step):
            kmer = sequence[i:i + k]
            if feature_mask_set is None or kmer in feature_mask_set:
                kmers_count[kmer] += 1
            total_kmers += 1
        
        for key in kmers_count:
            kmers_count[key] /= total_kmers
        
        return kmers_count

    @staticmethod
    def build_kmer_representation_v2(df, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, dtypes=['float64', 'int64'], asCudaDF=False, batch_size=1000, feature_mask=None):
        """
        Utils: For given k-mer generate dataset and return vectorized version
        """
        import os
        import pickle
    
        sequences = df['sequence']
        y = df['class']
        
        # Initialize DictVectorizer
        v = DictVectorizer(sparse=True)
        kmers_count_list = []
        
        # Create a temporary file for storing batch results
        temp_file = f"kmer_batch_results_{str(uuid.uuid4())}.pkl"
        
        # Process in batches and save to file
        with open(temp_file, 'wb') as f:
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i + batch_size]
                batch_kmers_count = [DNA.kmer_count_v3(sequence, domaine, k=k, step=1, feature_mask=feature_mask) for sequence in batch_sequences]
                kmers_count_list.extend(batch_kmers_count)
                # Save the current batch to file
                pickle.dump(batch_kmers_count, f)
        
        # Load all batches from file
        kmers_count_list = []
        with open(temp_file, 'rb') as f:
            while True:
                try:
                    kmers_count_list.extend(pickle.load(f))
                except EOFError:
                    break
        
        # Remove the temporary file
        os.remove(temp_file)
        
        # Vectorize the kmer counts
        feature_values = v.fit_transform(kmers_count_list)
        feature_names = v.get_feature_names_out()
        
        # Convert to DataFrame
        X = pd.DataFrame.sparse.from_spmatrix(feature_values, columns=feature_names)
        X = X.sparse.to_dense()
    
        # Apply feature mask if provided
        if feature_mask is not None:
            # Ensure feature_mask is a set for quick lookup
            feature_mask_set = set(feature_mask)
            current_features = set(X.columns)
            for feature in feature_mask_set - current_features:
                X[feature] = 0
            X = X[feature_mask]
        
        if asCudaDF:
            import cudf
            X_cuda = cudf.DataFrame.from_pandas(X)
            y_cuda = cudf.Series(y)
            return X_cuda, y_cuda, feature_names
        
        return X, y, feature_names
        
    """@staticmethod
    def build_kmer_representation_v2(df, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, dtypes=['float64', 'int64'], asCudaDF=False, batch_size=1000, feature_mask=None):
        ""
        Utils: For given k-mer generate dataset and return vectorized version
        ""
        sequences = df['sequence']
        y = df['class']#.astype(dtypes[1])
        
        # Initialize DictVectorizer
        v = DictVectorizer(sparse=True)
        kmers_count_list = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_kmers_count = [DNA.kmer_count_v2(sequence, domaine, k=k, step=1) for sequence in batch_sequences]
            kmers_count_list.extend(batch_kmers_count)
        
        # Vectorize the kmer counts
        feature_values = v.fit_transform(kmers_count_list)
        feature_names = v.get_feature_names_out()
        
        # Convert to DataFrame
        X = pd.DataFrame.sparse.from_spmatrix(feature_values, columns=feature_names)#.astype(dtypes[0])
        X = X.sparse.to_dense()

        # Apply feature mask if provided
        if feature_mask is not None:
            # Ensure feature_mask is a set for quick lookup
            feature_mask_set = set(feature_mask)
            current_features = set(X.columns)
            for feature in feature_mask_set - current_features:
                X[feature] = 0
            X = X[feature_mask]
        
        if asCudaDF:
            import cudf
            X_cuda = cudf.DataFrame.from_pandas(X)
            y_cuda = cudf.Series(y)
            return X_cuda, y_cuda, feature_names
        
        return X, y, feature_names"""

    @staticmethod
    def build_kmer_prediction_set(df, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, dtypes=['float64', 'int64'], batch_size=1000, feature_mask=None):
        """
        Utils: For given k-mer generate dataset and return vectorized version
        """
        sequences = df['sequence']
        
        # Initialize DictVectorizer
        v = DictVectorizer(sparse=True)
        kmers_count_list = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_kmers_count = [DNA.kmer_count_v2(sequence, domaine, k=k, step=1) for sequence in batch_sequences]
            kmers_count_list.extend(batch_kmers_count)
        
        # Vectorize the kmer counts
        feature_values = v.fit_transform(kmers_count_list)
        feature_names = v.get_feature_names_out()
        
        # Convert to DataFrame
        X = pd.DataFrame.sparse.from_spmatrix(feature_values, columns=feature_names)#.astype(dtypes[0])
        X = X.sparse.to_dense()

        # Apply feature mask if provided
        if feature_mask is not None:
            # Ensure feature_mask is a set for quick lookup
            feature_mask_set = set(feature_mask)
            current_features = set(X.columns)
            for feature in feature_mask_set - current_features:
                X[feature] = 0
            X = X[feature_mask]
            
        return X

class DataGenerator(Sequence):
    def __init__(self, dataset, feature_mask, batch_size=64, shuffle=True, domaine=None, k=None):
        self.dataset = dataset
        self.feature_mask = feature_mask
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.domaine = domaine
        self.k = k
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.dataset) / self.batch_size))
    
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch, y_batch, _ = DNA.build_kmer_representation_v2(
            self.dataset[indices], feature_mask=self.feature_mask, domaine=self.domaine, k=self.k, 
            dtypes=['float16', 'int8'], asCudaDF=False
        )
        return X_batch, y_batch
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)