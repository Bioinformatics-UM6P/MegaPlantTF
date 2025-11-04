<div align="center" style="">
  <br>
  <img src="./logo.png"/ style="height: 5em;">
  <br>
  <!-- <h1>GeneLM</h1> -->
  MegaPlantTF: a comprehensive machine learning framework for the identification and classification of plant transcription factors.
  <br>
  <br>

  [![DOI:10.1101/2025.03.20.644312](https://zenodo.org/badge/DOI/10.1093/bib/bbaf311.svg)]()
  [![Hugging Face](https://img.shields.io/badge/Hugging_Face-MegaPlantTF-orange?style=flat&logo=HuggingFace&logoColor=)](https://huggingface.co/Genereux-akotenou/genomics-tf-prediction)
  [![Conda](https://img.shields.io/badge/Conda-Supported-brightgreen?style=flat&logo=anaconda&logoColor=white)](https://anaconda.org/bioinformatics-um6p/megaplanttf)
</div>

## MegaPlantTF

`MegaPlantTF` is the first machine learning–based framework designed to identify and classify plant transcription factors (TFs) across multiple species. The project leverages curated data from [PlantTFDB](https://planttfdb.gao-lab.org/) and advanced k-mer–based feature representations to train robust, family-specific binary classifiers. With `MegaPlantTF`, you can:

- Predict Transcription Factors: Identify and classify TF families from plant proteomes using pretrained binary and stacking models.
- Comprehensive Evaluation: Generate detailed classification reports with accuracy, precision, recall, F1-score, and confidence thresholds.
- Flexible Inference Options: Apply max-voting or two-stage stacking classifiers for improved family-level predictions.

<br>

![Step 1 - Install MegaPlantTF](https://img.shields.io/badge/Step%201-Install%20MegaPlantTF-0b75b6?style=for-the-badge&logo=python&logoColor=white)


<img src="https://img.shields.io/badge/Step%201-Install%20MegaPlantTF-0b75b6?style=for-the-badge&logo=python&logoColor=white" width="100%; height: 1em;"/>


#### Step 1: Create & Activate Conda Environment

Open your terminal in the current folder then create the `MegaPlantTF` environment from the provided YAML file.

```bash
cd MegaPlantTF
conda env create -f MegaPlantTF.yml
```

Activate the MegaPlantTF environment.

```bash
conda activate MegaPlantTF
```

#### Step 2: Register Environment in Jupyter
```bash
python -m ipykernel install --user --name MegaPlantTF --display-name "MegaPlantTF"
```

#### Step 3: Start Jupyter Notebook

Start Jupyter Notebook to begin working with MegaPlantTF.

```bash
jupyter notebook
```

<br>
<div style="padding: 0.5em; background: #0b75b6; color: #fff; font-size: 1.1em;">
2- Build pretrained model
</div>
<!-- ## 2- Build pretrained model -->

We have to move into notebook folder and execute the python file named `pyrunner`

```bash
cd notebook
```

The python file should look like this. Depending on if we wanna run the program using multiprocess we have to set either `multiprocess=True` or `multiprocess=False`.

```python
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
        kernel_name='pygenomics'
    )

if __name__ == "__main__":
    # List of genes 
    gene_info_path = "../data/gene_info.json"
    with open(gene_info_path, 'r') as json_file:
        gene_info = json.load(json_file)

    # Output directory
    os.makedirs("AutoSave", exist_ok=True)

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
```

The next step is to run this file then till the program finish

```bash
python pyrunner
```

## 3- Pretrained Model and documentation

After running the notebook, you can find the results in the `Output` directory. Here's what you will find:

1. **Model Files**:
    - Located in `Output/Model`.
    - Inside this directory, you will find folders named after gene families.
    - Each gene family folder contains:
        - Model `.h5` files for various k-mer sizes.
        - `feature_mask.json` files.

2. **Reports**:
    - Located in `Output/Reports`.
    - Each report is specific to a gene family.
    - Reports include:
        - Model architecture and parameters.
        - Learning curve.
        - Train set class distribution.
        - Classification metrics: F1 score, recall, accuracy, precision.
        - Confusion matrix for each k-mer size.

## 4- How to Make Predictions Using the Model

To make predictions using the trained model, follow these steps:

1. **Set the k-mer Size**:
    - Choose the k-mer size you want to use for predictions.
    - You can use a single k-mer model or a multi k-mer model.

2. **Import the Prediction Classes**:
    - Import `SingleKModel` or `MultiKModel` from the `pretrained.predictor` module located in the `notebook` directory.

3. **Create a Notebook or Python File**:
    - Create a new notebook or Python file and include the following code:

```python
from pretrained.predictor import SingleKModel, MultiKModel

# Example for SingleKModel
kmodel = SingleKModel(kmer_size=3)
kmodel.load("Ach_pep_kiwi.fas", format="fasta")
genboard = kmodel.predict()
genboard.display()
```
<img src="genboard.png" alt="genbaord beta image" style="width: 97%;"/>