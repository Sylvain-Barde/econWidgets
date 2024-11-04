# econWidgets
General collection of Jupyter Widgets for teaching maths/stats for economics

## Requirements and Installation

A `requirements.txt` file is provided. The toolbox is built around the [jupyter widget]([https://gpytorch.ai/](https://ipywidgets.readthedocs.io/en/8.1.2/)) extension to notebooks. Additional (standard) packages required are: `matplotlib, numpy, pandas, scipy`. Note, `nbconvert` is used to convert the `econWidgets_demo.ipynb` demonstration notebook to HTML for display purposes, but is not required for running the widgets themselves.

## Overview

My [econWidgets](https://sylvain-barde.github.io/projects/econwidgets/) project page provides an overview of the three types of widgets provided (mathematics, statistics and economic applications), as well as illustrations of a selection of each type of widget. 

## Demo notebook

Two files are provides to demonstrate the functionality of each widget
- `econWidgets_demo.ipynb`: A full notebook environment within which the interactive functionality of each widget can be explored. Requires a functional jupyter notebook environment as specified in the `requirements.txt` file.
- `econWidgets_demo.htmi`: A static rendering of the full demmo notebook. This file can be downnloaded and inspected on any browser without requireing a full innstallation, but the displayed widgets are not fully interactive.
