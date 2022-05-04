<img height="140" align="left" src="https://github.com/lmjacoby/ai_semiconductors/blob/master/ai_semiconductors/static/AI_semicond_logo3.jpg">


[![Build Status](https://travis-ci.com/lmjacoby/ai_semiconductors.svg?branch=master)](https://travis-ci.com/lmjacoby/ai_semiconductors)
![License](https://img.shields.io/github/license/Chenyi-Mao/formulation)

We are working with researchers at Argonne National Lab to build a tool that uses machine learning models to effectively and accurately predict the formation energy and transition energy levels of novel impurity doped II-VI, III-V, and IV-IV semiconductors and their alloys. This project aims to assist scientists and researchers in the field of semiconductor research as they explore novel materials with targeted optoelectronic properties. We are currently building three different machine learning models based on Kernel Ridge Regression, Random Forest Regression and Neural Networks.

If you are interested in learning more, check out our [paper](<https://www.cell.com/patterns/fulltext/S2666-3899(22)00023-X>).


### Prediction Tool
The chemical space of impurity doped semiconductors is vast, and we wanted to create a tool to allow researchers to quickly scan through compounds for targeted properties. So, we created this prediction tool that allows a user to explore the formation energies and transition energy levels of ~12,500 doped semiconductor compounds.

The tool gives the predicted formation energies and transition levels for an impurity doped semiconductor based on three separate types of machine-learning models: kernel ridge regression, random forest regression, and a neural network.

There's an (almost) endless wealth of impurity doped semiconductors to explore, so read the instructions below and then get to exploring!

**How to use the tool**
1. Select a semiconductor from a dropdown menu. We have all traditional semiconductors (II-VI, III-V, IV-IV) available for prediction.
2. Select an impurity dopant from a dropdown menu. We have most s, p and d-block elements to select from.
3. Select a site for the impurity dopant from a dropdown menu. The choices are substitutional doping for either A or B of the parent semiconductor (M_A, M_B) or interstitial doping in an A-rich site, B-rich site or neutral site (M_i_A, M_i_B, M_i_neut)
4. The tool will output a table and plot of predicted formation energies, and transition levels for the compound of choice (8 targets in total, 2 formation energies and 6 transition energy levels). There will be three different predictions per target from a kernel ridge regression model, random forest regression model, and neural network.

To get started, navigate [here](https://mybinder.org/v2/gh/lmjacoby/ai_semiconductors/master?urlpath=%2Fapps%2Fai_semiconductors%2Fprediction_tool%2FEnergy_plot.ipynb)

![prediction tool gif](https://github.com/lmjacoby/ai_semiconductors/blob/master/ai_semiconductors/static/predict_tool_app.gif)
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

### GitHub Authors
- Xiaofeng Xiang
- Laura Jacoby
- Robert Biegaj

### Manuscript
A manuscript describing the data, ML models, and prediction results can be found [here](<https://www.cell.com/patterns/fulltext/S2666-3899(22)00023-X)>). The following authors contributed to this work:

University of Washington
- Xiaofeng Xiang, xiaofx2@uw.edu
- Laura Jacoby, ljacoby@uw.edu
- Robert Biegaj, rbiegaj@uw.edu
- Scott T. Dunham, dunham@ece.uw.edu
- Daniel R. Gamelin, gamelin@chem.washington.edu

Purdue University
- Arun Mannodi-Kanakkithodi, amannodi@purdue.edu

Argonne National Lab
- Maria K.Y. Chan, mchan@anl.gov

### Citation
If you find this prediction tool useful, please cite the following in your research:
```
Mannodi-Kanakkithodi, A., Xiang, X., Jacoby, L., Biegaj, R., Dunham, S. T., Gamelin, D. R., & Chan, M. K. (2022). Universal Machine Learning Framework for Defect Predictions in Zinc Blende Semiconductors. Patterns, 3(3), 100450.
```

**Bibtex**
```
@article{mannodi2022universal,
  title={Universal machine learning framework for defect predictions in zinc blende semiconductors},
  author={Mannodi-Kanakkithodi, Arun and Xiang, Xiaofeng and Jacoby, Laura and Biegaj, Robert and Dunham, Scott T and Gamelin, Daniel R and Chan, Maria KY},
  journal={Patterns},
  volume={3},
  number={3},
  pages={100450},
  year={2022},
  publisher={Elsevier}
}
```
### License
This project is licensed under the MIT License - see LICENSE.md for details.

### Acknowledgments
The data to train our machine learning models came from our project supervisors at Argonne National Lab. They also contributed guidance and support in building our machine learning tools.

This project was part of the University of Washington DIRECT 2020 Capstone. It was funded under the Data Intensive Research Enabling Clean Technology (DIRECT) NSF National Research Traineeship (DGE-1633216) and was also supported by Argonne National Lab.
<img height="100" align="right" src="https://github.com/lmjacoby/ai_semiconductors/blob/master/ai_semiconductors/static/Argonnelogo.png"> <img height="100" align="left" src="ai_semiconductors/static/DIRECTlogo.png">
