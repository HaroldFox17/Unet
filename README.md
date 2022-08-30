##  UNet

This project contains my implementation of the UNet algorithm (Python, Tensorflow).

The data I used to train and test my model can be found at this address: https://www.kaggle.com/c/airbus-ship-detection/overview.

Scripts:

-  data_analysis.ipynb contains data analysis and masks visualisation;

-  model.py contains a class for creating the model;

-  losses.py contains loss function used to compile the model: 

*  train.py shows the code used to train the model;

*  datasets.py contains code to create datasets;

*  utils.py contains some functions used in other scripts;

*  inference.py creates .csv file with model predictions.

To install the project load all the project files and dataset into 2 different folders (note: you might need to change some absolute path names to make project run succesfully).

To run project run train.py file to train the model and save it in another file, then run inference.py file to make predictions on dataset test folder and save results in csv file.
