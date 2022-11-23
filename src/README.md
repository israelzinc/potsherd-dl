# How to run this program

## Creating conda environment

```console
$ conda create --name archeological python=3.10
$ conda activate archeological
$ conda install -c conda-forge mamba=0.24
$ mamba install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
$ mamba install pandas scikit-learn scikit-learn-intelex tqdm ipykernel albumentations -c conda-forge
```

alternatively, run

```console
$ conda env create --file environment.yml
$ conda activate archeological
```

## Editing Configuration

Edit `config.json` if necessary.

## Creating CSV file of the datasetst

```console
$ python create_csv.py
```
