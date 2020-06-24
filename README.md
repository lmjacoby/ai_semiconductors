<img height="140" align="left" src="https://github.com/lmjacoby/ai_semiconductors/blob/master/ai_semiconductors/static/AI_semicond_logo3.jpg">


[![Build Status](https://travis-ci.com/lmjacoby/ai_semiconductors.svg?branch=master)](https://travis-ci.com/lmjacoby/ai_semiconductors)
![License](https://img.shields.io/github/license/Chenyi-Mao/formulation)

We are working with researchers at Argonne National Lab to build a tool that uses machine learning models to effectively and accurately predict the formation energy and transition energy levels of novel impurity doped II-VI, III-V, and IV-IV semiconductors and their alloys. This project aims to assist scientists and researchers in the field of semiconductor research as they explore novel materials with targeted optoelectronic properties. We are currently building three different machine learning models based on Kernel Ridge Regression, Random Forest Regression and Neural Networks.


### Prediction Tool
The chemical space of impurity doped semiconductors is vast, and we wanted to create a tool to allow researchers to quickly scan through compounds for targeted properties. So, we created this prediction tool that allows a user to explore the formation energies and transition energy levels of ~12,500 doped semiconductor compounds.

The tool gives the predicted formation energies and transition levels for an impurity doped semiconductor based on three separate types of machine-learning models: kernel ridge regression, random forest regression, and a neural network.

There's an (almost) endless wealth of impurity doped semiconductors to explore, so read the instructions below and then get to exploring!

**How to use the tool**
1. Select a semiconductor from a dropdown menu. We have all traditional semiconductors (II-VI, III-V, IV-IV) available for prediction.
2. Select an impurity dopant from a dropdown menu. We have most s, p and d-block elements to select from.
3. Select a site for the impurity dopant from a dropdown menu. The choices are substitutional doping for either A or B of the parent semiconductor (M_A, M_B) or interstitial doping in an A-rich site, B-rich site or neutral site (M_i_A, M_i_B, M_i_neut)
4. The tool will output a table and plot of predicted formation energies, and transition levels for the compound of choice (8 targets in total, 2 formation energies and 6 transition energy levels). There will be three different predictions per target from a kernel ridge regression model, random forest regression model, and neural network.

To get started, navigate [here](https://notebooks.gesis.org/binder/jupyter/user/lmjacoby-ai_semiconductors-u7lnukpm/apps/ai_semiconductors/prediction_tool/Energy_plot.ipynb?appmode_scroll=0).

### Repository Structure
This repository serves as a home for our prediction tool, and example notebooks to display the code we wrote to build our machine learning models and make predictions. The data we used to train our models is **private data**, so we have put example notebooks in the notebooks folder if you are interested in checking out our code in action.

- ai_semiconductors
  - doc
  - notebooks
    - Feature Selection
    - ModelExploration
    - OutlierDetection
  - prediction_tool
  - static
  - tests

#### Authors
- Robert Biegaj
- Xiaofeng Xiang
- Laura Jacoby

#### License
This project is licensed under the MIT License - see LICENSE.md for details.

#### Acknowledgments
The data to train our machine learning models came from our project supervisors at ANL. They also contribute guidance and support in building our machine learning tools.

This project is part of the UW DIRECT Capstone 2020, and is supported by Argonne National Lab.
<img height="100" align="right" src="https://github.com/lmjacoby/ai_semiconductors/blob/master/ai_semiconductors/static/Argonnelogo.png"> <img height="100" align="left" src="ai_semiconductors/static/DIRECTlogo.png">
