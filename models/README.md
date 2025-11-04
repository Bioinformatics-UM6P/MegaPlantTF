### Download model weights from hugginface

```bash
# cd into MegaPlantTF folder
cd MegaPlantTF

# clone to temp
tmpdir="$(mktemp -d)"
git clone https://huggingface.co/Genereux-akotenou/genomics-tf-prediction "$tmpdir/repo"

# force overwrite into your local root
rsync -av --delete "$tmpdir/repo/Binary-Classifier/" "./models/Binary-Classifier/"
rsync -av --delete "$tmpdir/repo/MetaClassifier/"   "./models/MetaClassifier/"

# rsync -av "$tmpdir/repo/testset/testset.csv" "./data/testset-full/k3/"
rm -rf "$tmpdir"
```
