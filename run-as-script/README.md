<div align="center" style="">
    <br>
    <img src="../logo.png"/ style="height: 5em;">
    <br>
    <br>
    MegaPlantTF: a comprehensive machine learning framework for the identification and classification of plant transcription factors.
    <br>
    <br>
    <p>****</p>
    <p>MegaPlantTF-Script Runner</p>
</div>

---

#### Quick start

First of all you need to setup MegaPlantTF conda environment tobe able to predict TF.

- Download:
Download the whole project but make sre to be into this folder to run the annotation scripts (after you install the env):
`MegaPlantTF/run-as-script`
<br>
- Setup:
Please refer to main README(clone - env - pull models)
<br>



#### Usage
**1. Start fast**

```bash
python run_megaplanttf.py --fasta=__temp__/Ach_pep_kiwi-small.fas --kmer=3
```
<br>

**2. Detailed**
  
```bash
python run_megaplanttf.py \
  --fasta __temp__/Ach_pep_kiwi-small.fas \
  --kmer 3 \
  --voting 'Two-Stage Voting' #or  'Max Voting' (leave to default 'Two-Stage Voting' )\
  --output "path to ouput folder. default is ouput folder in current script active directory" \
  --jobid "Unique job ID for naming outputs. but can be generated automatically" \
```

**Ouput**

Be like:

```
output/
 ├── MegaPlantTF_Dashboard_<jobid>.html
 └── MegaPlantTF_Dashboard_<jobid>_predictions.csv
```