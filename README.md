# A-MCTS

United We Stand: Decentralized Multi-Agent Planning With Attrition

This project contains the source code for the environment constructor, planning algorithms, and simulations used for the paper listed above, implemented in the Python programming language. The repository contains several folders:

- CODE: the environment constructor, planning algorithms, and simulations considered in the study,
- Data: the environment configurations and simulation output files.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements libraries.
```bash
pip install -r requirements.txt
```

Or use the [conda](https://docs.conda.io/projects/conda/en/stable/) to create a testing environment.
```bash
conda create --name <env> --file requirements.txt
```

## Usage
To run a simulation of A-MCTS on a multi-agent path planning task.
```bash
python3 simulation.py [-h] [-s] -a {Dec,Global,Reset,RM,Greedy} [-f FOLDER] [-v] [-p PARAMS [PARAMS ...]]
optional arguments:
  -h, --help            show this help message and exit
  -s, --save            Save performance
  -a {Dec,Global,Reset,RM,Greedy}, --adapt {Dec,Global,Reset,RM,Greedy}
                        Type of adaptation to environmental changes
  -f FOLDER, --folder FOLDER
                        Folder name to store simulation data
  -v, --verbose         Print details
  -p PARAMS [PARAMS ...], --params PARAMS [PARAMS ...]
                        Parameter testing

```

To construct new environment configurations.
```bash
python3 graph_helper.py [-h] [-a] [-n N_CONFIGS] [-d DRAW]

Graph constructor

optional arguments:
  -h, --help            show this help message and exit
  -a, --animation       Show constructed graph
  -n N_CONFIGS, --n_configs N_CONFIGS
                        No of configurations
  -d DRAW, --draw DRAW  Draw existing graph
```

To use Dec-MCTS for your problems.
```python
from function import *
from mcts import growTree
```
