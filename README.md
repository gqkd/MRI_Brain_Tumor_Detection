# MRI_Brain_Tumor_Detection

This code is a Python script that builds and trains a CNN model to classify brain tumors in MRI scans.

## Dependencies

* keras: This library is used to build and train the CNN model.
* tensorflow: This library is used to provide the computational backend for keras.
* matplotlib: This library is used to visualize the results of the model.
* numpy: This library is used to handle numerical data.

## Instructions

1. Install the dependencies by running the following command:

```
pip install keras tensorflow matplotlib numpy
```

2. Run the script by providing the path to the MRI scan data as an argument:

```
python brain_tumor_classification.py <path_to_data>
```

For example, to train the model on the data set provided in the README, you would run the following command:

```
python brain_tumor_classification.py data/
```

The script will first load the MRI scan data and split it into a training set and a test set. The training set will be used to train the CNN model, and the test set will be used to evaluate the model's performance.

The script will then train the CNN model using the training set. The model will be trained for a specified number of epochs, and the loss and accuracy of the model will be monitored during training.

Once the model is trained, the script will evaluate the model's performance on the test set. The accuracy of the model will be reported.

## Results



## Conclusion


## Future Work


