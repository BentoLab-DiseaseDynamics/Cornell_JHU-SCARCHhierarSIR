# Cornell_JHU-SCARCHhierarSIR

An SIR model wrapped in a Bayesian hierarchical inference framework for short-term infectious disease forecasting. Implemented using `pyMC v6` and `arviz v1`. Successor to Cornell_JHU-hierarchSIR.

## Installation (local)

Available platforms: macOS and Linux.

### Setup and activate a conda environment

Update conda to make sure your version is up-to-date,

```
conda update conda
```

Setup/update the `environment`: All dependencies needed to run the scripts are collected in the conda `SCARCHhierarSIR_env.yml` file. To set up the environment,

```
conda env create -f SCARCHhierarSIR_env.yml
conda activate BENTOLAB-SCARCH_HIERARSIR
```

or alternatively, to update the environment (needed after adding a dependency),

```
conda activate BENTOLAB-SCARCH_HIERARSIR
conda env update -f SCARCHhierarSIR_env.yml --prune
```

### Install the `SCARCHhierarSIR` package

Install the `SCARCHhierarSIR` Python package inside the conda environment using,

```
conda activate BENTOLAB-SCARCH_HIERARSIR
pip install -e . --force-reinstall
```

### Model training and forecasting

#### Clustering 

Instead of modeling all 52 U.S. states and territories at once, the code can automatically loop over clusters of states and territories. The clustering pipeline interfaces with the `SCARCHhierarSIR` model through `~/data/interim/geography/cluster.csv`.

#### Training (execute once before season start)

```
cd ~/scripts/operational/
python train.py
```

#### Forecast (performed weekly through GH actions)

```
cd ~/scripts/operational/
python forecast.py
```

## Training on a cluster

Information on how to train the model on the Cornell Seneca cluster is provided in `~/CU-SENECA_README.md`.

## Workflows

Automated workflows were ported from `hierarchSIR`.
