import numpy as np
import pandas as pd

def read(file_path, family, domaine="ACDEFGHIKLMNPQRSTVWY"):
    """
    Utils: to read fasta file and extract content as dataframe, replacing any character not in the specified domaine with 'X'.
    """
    sequences = []
    with open(file_path, 'r') as file:
        current_id = None
        current_sequence = ''
        for line in file:
            if line.startswith('>'):
                if current_id:
                    sequences.append({
                        'id': current_id,
                        'sequence': ''.join([char if char in domaine else 'X' for char in current_sequence]),
                        'length': len(current_sequence),
                        'class': family
                    })
                current_id = line.strip().split('|')[0][1:].strip()
                current_sequence = ''
            else:
                current_sequence += line.strip()
        if current_id:
            sequences.append({
                'id': current_id,
                'sequence': ''.join([char if char in domaine else 'X' for char in current_sequence]),
                'length': len(current_sequence),
                'class': family
            })
    
    df = pd.DataFrame(sequences)
    return df
    
def read_fas(file_path, domaine="ACDEFGHIKLMNPQRSTVWY"):
    """
    Utils: to read fasta file and extract content as dataframe, replacing any character not in the specified domaine with 'X'.
    """
        
    sequences = []
    with open(file_path, 'r') as file:
        current_id = None
        current_sequence = ''
        is_fasta_format = False
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                is_fasta_format = True
                if current_id:
                    if current_sequence == '':
                        raise ValueError("FASTA format error: sequence missing for identifier '{}'.".format(current_id))
                    sequences.append({
                        'id': current_id,
                        'meta': meta_data,
                        'sequence': ''.join([char if char in domaine else 'X' for char in current_sequence]),
                    })
                current_id = line.split('|')[0][1:].strip()
                meta_data = line
                current_sequence = ''
            elif is_fasta_format:
                if line == '' or line.startswith('>'):
                    raise ValueError("FASTA format error: sequence missing or misplaced '>' character.")
                current_sequence += line
        
        if current_id:
            if current_sequence == '':
                raise ValueError("FASTA format error: sequence missing for identifier '{}'.".format(current_id))
            sequences.append({
                'id': current_id,
                'meta': meta_data,
                'sequence': ''.join([char if char in domaine else 'X' for char in current_sequence]),
            })
    
    if not sequences:
        raise ValueError("FASTA format error: no valid sequences found.")

    df = pd.DataFrame(sequences)
    return df


def read_csv(file_path, domaine="ACDEFGHIKLMNPQRSTVWY"):
    """
    Utils: to read CSV file and extract content as dataframe, replacing any character not in the specified domaine with 'X'.
    """
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file '{file_path}' is empty.")
    
    if 'sequence' not in df.columns or 'id' not in df.columns:
        raise ValueError("CSV format error: Missing 'sequence' or 'id' column in CSV file.")
    
    sequences = []
    for index, row in df.iterrows():
        sequence = row['sequence']
        seq_id = row['id']
        if not isinstance(sequence, str):
            raise ValueError(f"CSV format error: 'sequence' column value at index {index} is not a string.")
        processed_sequence = ''.join([char if char in domaine else 'X' for char in sequence])
        sequences.append({
            'id': seq_id,
            'sequence': processed_sequence
        })
    
    df_processed = pd.DataFrame(sequences)
    return df_processed