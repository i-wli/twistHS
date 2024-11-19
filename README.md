# Twisted Heterostructure Generation Toolkit
A Python-based solution for generating twisted heterostructures through both a Command-Line Interface (CLI) and a Graphical User Interface (GUI). Whether you are a researcher, student, or hobbyist interested in the fascinating world of moiré patterns, this toolkit is designed to provide a flexible and user-friendly approach to exploring these patterns.

Only first-order moirés are considered for direct comparison of different twist angles. Please check 2012 J. Phys.: Condens. Matter 24 314210 for theoretical background.

## Features

- CLI Tool (`twist_cli.py`): generate structures through your terminal or command prompt.
- GUI Tool (`twist_gui.py`): an user-friendly interface, allowing for interactive adjustments and real-time visualization.


## Installation

1. **Set Up Environment**:
It is recommended to create a new Conda environment for running this project to avoid any conflicts with existing packages. You can create a new environment with Python 3.12 as follows:
```
conda create --name moire_env python=3.12 pip
conda activate moire_env
```
2. **Install the Package Directly from GitHub**:
```
pip install git+https://github.com/i-wli/twistHS.git
```

2.1 **Development Installation with source code**:
```
git clone https://github.com/i-wli/twistHS.git
cd twistHS
pip install .
```

## Usage

**Option 1. Using the CLI Tool (`twist_cli.py`)**

Open your command-line interface and run:
```
twist_cli [options]
```
This should open a matplotlib interface:
![image](https://github.com/i-wli/twistHS/blob/main/docs/cli.gif)


**Option 2. Using the GUI Tool (`twist_gui.py`)**

To launch the GUI, simply run:
```
twist_gui
```

Once the GUI is open, you should see:
![image](https://github.com/i-wli/twistHS/blob/main/docs/gui.gif)
