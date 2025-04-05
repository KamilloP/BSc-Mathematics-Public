# BSc-Maths-KP

# Description
This is the repository for my own BSc thesis in Mathematics.
Thesis focuses on numerical approximation of solutions of ordinary differiential equations defined on graphs and it's applications in graph classification/description.
It is under construction project, especially notes.


Author: Kamil Pilkiewicz

Supervisor: prof. dr hab. P. R.

## Usage
`python3 analyze_graph.py`

We will be asked to give three arguments:
- `nameTUD` - Name of dataset in (TUD),
- `T` - time duration we want to compute,
- `minV` - Minimal number of vertices

From dataset `nameTUD` every graph with number of vertices not smaller than `minV` will be analyzed in period of time: [0, T].
As a result of analysis in the browser we will see generated graphs (2 versions - computated by closed and open scheme).
Vertex color shows it's heat, the scale is on the right side. 
We can choose time we want to observe, unit of time is 0.1s.
Options:
- play - shows change heat values of vertices in time
- ctrl + mouse: graph translation
- alt + mouse: changing graph size
- mouse: graph rotation

In the terminal computated values of heat will be shown (2 versions: computated by open and closed Euler scheme).

## Name of data sets in TUD
IMDB-BINARY

## Setup instructions

### python3
`python3 --version` - checks Python version

### venv
`python3 -m venv < path >` - creates virtual enviroment (venv) in the new folder *path* 

`python3 -m venv venv` - creates new folder in current folder containing venv

`source < path >/bin/activate` - activates virtual enviroment in the path

`source venv/bin/activate`

### pip
`python -m pip install --upgrade pip` - upgrades pip

`pip install < package_name >` - installs Python package

`pip install ...`

### git
`git clone < repository_name >` - downloads repository

`git clone https://github.com/KamilloP/BSc-Maths-KP/`

## Running Python script in Linux terminal
`python3 < name >` - runs script named *name*, should be done after venv activation


## Useful links
https://docs.python.org/3/library/venv.html - venv

https://pip.pypa.io/en/stable/installation/

https://packaging.python.org/en/latest/tutorials/installing-packages/

https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.TUDataset.html#torch_geometric.datasets.TUDataset

https://chrsmrrs.github.io/datasets/docs/datasets/

https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets

https://pytorch-geometric.readthedocs.io/en/2.4.0/modules/utils.html

https://networkx.org/

https://pypi.org/project/oct2py/

https://numpy.org/

https://plotly.com/python/graph-objects/

https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls

https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens

https://pypi.org/project/oct2py/

For author:

https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax
