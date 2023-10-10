# Battlesnake Reinforcement Learning

## Installation

#### Cloning this repository with submodule
Use the following command to clone this Repo with the included Submodule. You need access rights from TNT-LUH.
```
git clone <url>
git submodule init
git submodule update
```

You need to either create an environment or update an existing environment.
**After** creating an environment you have to activate it:
#### Create environment
This may take a long while
```
mamba env create -f environment.yml
```
#### Update environment (if env exists)
```
mamba env update -f environment.yml
```

#### Install Cuda if necessary/possible
```
mamba remove pytorch
mamba install pytorch-cuda==11.7 cudatoolkit pytorch==2.0.1 torchvision -c pytorch -c nvidia
```

#### Installation of EsCNN is a complete Mess, comprehensive tutorial below
Note that it was not possible for me to install the following two libraries via pip :/

``
git clone https://github.com/AMLab-Amsterdam/lie_learn.git
``

Then go to setup.py in the repository and change in line 28:
``
extensions = cythonize(extensions)
``
to
``
extensions = cythonize(files)
``

```
cd lie_learn
python setup.py install
git clone https://github.com/fujiisoup/py3nj.git
cd py3nj
python setup.py install
```


#### Activate environment
```
mamba activate battlesnake-rl
```

#### Compile C++ Module
```
# Windows
cd src/cpp
cmd < compile.bat

# Linux
cd src/cpp
sh compile.sh
```

## Folder Structure
* scripts: Python Scripts for generating training configs, tournament evaluations and plots of the results
* test: Python Unittests. Mirrors the folder structure of src
* src: 
  * agent: Interface and implementation of agents playing a game
  * analysis: GUI for displaying neural network predictions in the game of Battlesnake
  * cpp: C++ Library for Battlesnake game and various game theoretic algorithms
  * depth: Code for parallel evaluation of different tree search depths
  * equilibria: Python interface for game theoretic algorithms. Internally call the C++ code.
  * evaluation: Parallel tournament evaluation of different Agents
  * frontend: Frontend webserver for Battlesnake online leaderboard
  * game: Game interface
    * battlesnake: BattleSnake game
    * bootcamp: various game modes and test environments of Battlesnake
    * exploitability: Converts any game into a single player game against fixed agents
    * extensive_form: Extensive form games and random initialization
    * normal_form: Normal form games and random initialization
    * oshi_zumo: Game of OshiZumo
    * overcooked: Game of Overcooked
  * misc: Various code snippets for multiprocessing, plotting or training
  * modelling: Maximum likelihood estimation interface
  * network: Neural Network Architectures, notably ResNet, MobileNetV3 and MobileOne
  * optimization: Bayesian Optimization wrapper for SMAC3
  * search: Interface for different Search algorithms. Notable implementations are MCTS, Fixed Depth Search, Iterative Deepening and SM-OOS. All variants have standard interface for selection, expansion, backup and extraction functions.
  * supervised: Code for supervised training. Includes optimization, loss computation and annealing.
  * trainer: Parallelized AlphaZero framework







