## MegaPlantTF 
---

gUIDE FOR HANDONS


### Step 1: Create Conda Environment

First, create the `MegaPlantTF` environment from the provided YAML file.

```bash
conda env create -f ../pygenomics.yml
```

### Step 2: Activate Conda Environment

Activate the MegaPlantTF environment.

```bash
conda activate MegaPlantTF
```

### Step 3: Download model weights from hugginface
```bash
# d into workshop folder
cd MegaPlantTF/workshop

# clone to temp
tmpdir="$(mktemp -d)"
git clone https://huggingface.co/Genereux-akotenou/genomics-tf-prediction "$tmpdir/repo"

# force overwrite into your local root
rsync -av --delete "$tmpdir/repo/" "../models/"
rm -rf "$tmpdir"
```


### Step 3: Start Jupyter-lab

Start JupyterLab to begin working with MegaPlantTF.

```bash
jupyter-lab
```

# Traditinonal blast

# mega...

# analysis f performance

# running mega for any fasta...