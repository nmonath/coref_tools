# coref-tools #
Tools for coreference resolution.


## Setup ##

Install xcluster:

```bash
git clone https://github.com/iesl/xcluster.git
cd xcluster
# if you need to install maven
sh bin/util/install_mvn.sh
# build
sh bin/build.sh
# Set environment variables
source bin/setup.sh
```

Install dependencies:

```bash
# CPU pytorch suffices (otherwise install the GPU version from here: https://pytorch.org/)
conda install pytorch-cpu torchvision-cpu -c pytorch
pip install unidecode
```

Install fasttext
```bash
pip install fasttext
```

If fasttext doesn't install properly, you can try running
```bash
conda install libgcc
```
and then try to reinstall.

Install `jq`: [https://stedolan.github.io/jq/](https://stedolan.github.io/jq/)

## Run ##


Before running any code, run (in every shell session):

```
source bin/setup.sh
```

### Training a Model ###

Use the following commands to train a model on Rexa:

```bash
# Pairwise Model, 1 is the number of different models to train
sh bin/train/pw/train_n.sh conf/rexa/Rexa-PW-Lin-No-Entity.json  1 rexa
# Entity Model, 1 is the number of different models to train
sh bin/train/pw/train_n.sh conf/rexa/Rexa-PW-Lin-BaseSubEnt-lin.json  1 rexa
```

### Evaluating a Model ###

Make sure you have done the following to set XCLUSTER environment variables:

```bash
cd xcluster
source bin/setup.sh
```

Use the following commands to evaluate a trained model

```bash
# Usage sh bin/inf/hac/approx_inf_hac_n_test.sh <config file of run to use> <list of canopies to run on> <path to canopy directories> <number of shufflings of data to use>
sh bin/inf/hac/approx_inf_hac_n_test.sh  exp_out/rexa/2018-04-30-12-20-26/run_1/config.json    data/rexa/eval/dev/dev_canopies.txt  data/rexa/eval/dev/canopy/ 1
```

You can create config files to use in the above command for PERCH and variants using:

```bash
# Usage python -m coref.util.CreateExperimentConfigsFromEHAC exp_out/rexa/2018-04-30-12-20-26/run_1/config.json num_nearest_neighbors nsw_r_parameter 
python -m coref.util.CreateExperimentConfigsFromEHAC exp_out/rexa/2018-04-30-12-20-26/run_1/config.json 5 3
```
To get Micro F1 Numbers for your run :

```bash
sh bin/eval/micro_f1_all.sh exp_out/rexa/2018-05-05-05-34-21
```

You can collect the output using:

```bash
# Usage: python -m coref.util.CollectResultsInc <result output dir>
python -m coref.util.CollectResultsInc exp_out/rexa/2018-05-05-05-34-21/run_9/results/ehac/2018-05-06-15-01-58/run_9
```

