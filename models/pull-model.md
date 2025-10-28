
### Download MegaPlantTF models

Open a terminal in this folder and run this code to download the weights from hugginface.

```bash
# clone to temp
tmpdir="$(mktemp -d)"
git clone https://huggingface.co/Genereux-akotenou/genomics-tf-prediction "$tmpdir/repo"

# force overwrite into your local root
rsync -av --delete "$tmpdir/repo/" "../models/"
rm -rf "$tmpdir"
```