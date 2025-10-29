## MegaPlantTF — Hands-On Workshop

This hands-on workshop walks you through:
1. Predicting transcription factors (TFs) with **MegaPlantTF**
2. Running large-scale benchmarks
3. Comparing performance with a **traditional BLAST** baseline

---

### Step 1: Create & Activate Conda Environment

Open your terminal in the current folder then create the `MegaPlantTF` environment from the provided YAML file.

```bash
conda env create -f ../env.yml
```

Activate the MegaPlantTF environment.

```bash
conda activate MegaPlantTF
```

### Step 2: Register Environment in Jupyter
```bash
python -m ipykernel install --user --name MegaPlantTF --display-name "MegaPlantTF"
```

### Step 3: Download Pretrained Model Weights & testset for lab
```bash
# d into workshop folder
cd MegaPlantTF/workshop

# clone to temp
tmpdir="$(mktemp -d)"
git clone https://huggingface.co/Genereux-akotenou/genomics-tf-prediction "$tmpdir/repo"

# force overwrite into your local root
rsync -av --delete "$tmpdir/repo/Binary-Classifier/" "../models/Binary-Classifier/"
rsync -av --delete "$tmpdir/repo/MetaClassifier/"   "../models/MetaClassifier/"
rsync -av "$tmpdir/repo/testset/testset.csv" "../data/testset-full/k3/"
rm -rf "$tmpdir"
```

### Step 4: Start Jupyter-lab

Start JupyterLab to begin working with MegaPlantTF.

```bash
jupyter-lab
```

Then open and run the following notebooks **in order**:
- **[1-hands-on-MegaPlantTF.ipynb](./1-hands-on-MegaPlantTF.ipynb)** --> run TF prediction and benchmark  
- **[2-hands-on-BLAST.ipynb](./2-hands-on-BLAST.ipynb)** --> run BLAST baseline and compare results
