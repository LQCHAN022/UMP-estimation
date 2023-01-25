# Remote Sensing Landuse

## Setup

Create a conda enviroment:

```
conda create --name remote-sensing-landuse python=3.8
```

Install the python deps

```
 conda env update --file enviroment.yml
```

Install terracatalogueclient

```
pip install --extra-index-url https://artifactory.vgt.vito.be/api/pypi/python-packages/simple terracatalogueclient 
```
When running the authentication with `Catalogue().authenticate()`, run it in a Jupyter Notebook using the appropriate kernel.

You need to install gcloud in order to run the program in the CLI, see: https://cloud.google.com/cli

