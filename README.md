# Cornell_JHU-SCARCHhierarSIR

An SIR model wrapped in a Bayesian hierarchical inference framework for short-term infectious disease forecasting. Forward simulation model integrated using `diffrax`, probablistic model implemented using `NumPyro`. Successor to Cornell_JHU-hierarchSIR.

## Installation (local)

Available platforms: macOS and Linux.

### Setup and activate a conda environment

Update conda to make sure your version is up-to-date,

```
conda update conda
```

Setup/update the `environment`: All dependencies needed to run the scripts are collected in the conda `SCARCHhierarSIR_env.yml` file. To set up the environment,

```bash
conda env create -f SCARCHhierarSIR_env.yml
conda activate SCARCH_HIERARSIR
```

or alternatively, to update the environment (needed after adding a dependency),

```bash
conda activate SCARCH_HIERARSIR
conda env update -f SCARCHhierarSIR_env.yml --prune
```

### Install the `SCARCHhierarSIR` package

Install the `SCARCHhierarSIR` Python package inside the conda environment using,

```bash
conda activate SCARCH_HIERARSIR
pip install -e .
```

### Model training and forecasting

See `~/model_description.pdf`.

#### Training (execute once before season start)

```bash
cd ~/scripts/operational/
python hierarchical_training.py
```

#### Forecast (performed weekly through GH actions)

```bash
cd ~/scripts/operational/
python forecast.py
```

## Training on a cluster

Information on how to train the model on the Cornell Seneca cluster is provided in `~/CU-SENECA_README.md`.

## Workflows

Automated workflows were ported from `hierarchSIR`.
