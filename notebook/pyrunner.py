### RUN FROM PYTHON
"""import os
import json
import multiprocessing
import subprocess

def run_script(gene_family):
    # Command to source the pygenomics environment and run the script
    command = f"python 01-approach2_kmer_neural_network.py {gene_family}"
    #command = f"source activate pygenomics && python 01-approach2_kmer_neural_network.py {gene_family}"
    subprocess.run(command, shell=True, executable="/bin/bash")

if __name__ == "__main__":
    # Load the gene info
    gene_info_path = "../data/gene_info_small.json"
    with open(gene_info_path, 'r') as json_file:
        gene_info = json.load(json_file)

    # Create output directory if it doesn't exist
    os.makedirs("AutoSave", exist_ok=True)

    # Choose whether to run sequentially or concurrently
    multiprocess = False

    if multiprocess:
        # Run scripts concurrently using multiprocessing
        num_processes = multiprocessing.cpu_count()
        print('NUMBER OF PROCESSES: ', num_processes)
        with multiprocessing.Pool(num_processes) as pool:
            pool.map(run_script, gene_info.keys())
    else:
        # Run scripts sequentially
        for gene_family in gene_info.keys():
            run_script(gene_family)

"""

import os
import json
import multiprocessing
import papermill as pm

# Utils
def run_notebook(gene):
    input_notebook = "01-approach2_kmer_neural_network.ipynb"
    notebook_name = os.path.splitext(input_notebook)[0]
    gene_ = gene.replace('/', '__')
    output_notebook = f"AutoSave/{notebook_name}-{gene_}.ipynb"

    # Run the notebook with the specified gene
    pm.execute_notebook(
        input_notebook,
        output_notebook,
        parameters=dict(gene_familly=gene),
        timeout=-1,
        kernel_name='python3'
    )

if __name__ == "__main__":
    # List of genes 
    gene_info_path = "../data/gene_info_small.json"
    with open(gene_info_path, 'r') as json_file:
        gene_info = json.load(json_file)

    # Output directory
    os.makedirs("AutoSave", exist_ok=True)

    # EXEC NATURE
    multiprocess = True

    if multiprocess:
        # Run notebooks concurrently using multiprocessing
        num_processes = min(5, multiprocessing.cpu_count())
        print('NUMBER OF PROCESSES: ', num_processes)
        with multiprocessing.Pool(num_processes) as pool:
            pool.map(run_notebook, gene_info.keys())
    else:
        # Run notebooks sequentially
        for gene in gene_info.keys():
            run_notebook(gene)
