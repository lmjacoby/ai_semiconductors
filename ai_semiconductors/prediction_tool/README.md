### AI Semiconductors Prediction Tool
Welcome to our impurity doped semiconductor prediction tool. Feel free to check out the example notebook we have, and when you're ready click on the button at the bottom to use the app live, without cloning or installing anything! The tool has the ability to predict formation energies and transition levels for ~12,500 impurity doped semiconductor compounds, so explore around for however long you would like!

**How to use the tool**
1. Select a semiconductor from a dropdown menu. We have all traditional semiconductors (II-VI, III-V, IV-IV) available for prediction.
2. Select an impurity dopant from a dropdown menu. We have most s, p and d-block elements to select from.
3. Select a site for the impurity dopant from a dropdown menu. The choices are substitutional doping for either A or B of the parent semiconductor (M_A, M_B) or interstitial doping in an A-rich site, B-rich site or neutral site (M_i_A, M_i_B, M_i_neut)
4. The tool will output a table and plots of predicted formation energies, and transition levels for the compound of choice (8 targets in total, 2 formation energies and 6 transition energy levels). There will be three different predictions per target from a kernel ridge regression model, random forest regression model, and neural network.

**CLICK HERE**
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lmjacoby/ai_semiconductors/master?filepath=%2Fai_semiconductors%2Fprediction_tool%2Fapps%2FEnergy_plot.ipynb)

Disclaimer: we cannot make the "true" DFT data available at this time, but please stay tuned as we will be able to release it in the fall of 2020.
