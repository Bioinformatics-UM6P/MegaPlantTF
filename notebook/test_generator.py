import os
import json
import multiprocessing
import papermill as pm

# Utils
def run_notebook(gene):
    input_notebook = "05-extract_testset-0.ipynb"
    notebook_name = os.path.splitext(input_notebook)[0]
    gene_ = gene.replace('/', '__')
    output_notebook = f"AutoSave/TestGen/{notebook_name}-{gene_}.ipynb"

    # Run the notebook with the specified gene
    pm.execute_notebook(
        input_notebook,
        output_notebook,
        parameters=dict(gene_familly=gene),
        timeout=-1,
        kernel_name='pygenomics'
    )

if __name__ == "__main__":
    # List of genes 
    gene_info_path = "../data/gene_info.json"
    with open(gene_info_path, 'r') as json_file:
        gene_info = json.load(json_file)

    # Output directory
    os.makedirs("AutoSave/TestGen", exist_ok=True)

    # EXEC NATURE
    multiprocess = False

    if multiprocess:
        # Run notebooks concurrently using multiprocessing
        num_processes = multiprocessing.cpu_count()
        print('NUMBER OF PROCESSES: ', num_processes)
        with multiprocessing.Pool(num_processes) as pool:
            pool.map(run_notebook, gene_info.keys())
    else:
        # Run notebooks sequentially
        for gene in gene_info.keys():
            run_notebook(gene)
