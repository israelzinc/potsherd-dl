# How to run this program

## Installing the dependencies

```console
$ pip3 install -r requirements.txt
```

## Preprocessing the data
```console
$./preprocess.py conf/[the selected config file]
```

## Training a model

```console
$ ./train.py conf/[the selected config file]
```

## Testing a model
```console
$ ./test.py conf/[the selected config file]
```

## Config Files
A config file is a json with instructions of how you want to train a model. It contains information regarding the following:
- datasets
- model
- Dataloader
- optimizer
- scheduler
- training

For information about each, see the example `test-run.json` located inside the `conf` folder.