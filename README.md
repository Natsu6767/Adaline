# Implementation of Adaline

This is an implentation of an ADAptive LInear NEuron (Adaline) in Python3. 
The code in this repository is based on the Adaline example given in the book: Python Machine Learning by Sebastian Raschka.

## AdalineGD

Batch gradient descent is used to optimise the model. In this method,
the weights are updated only after passing the whole training data
through the model.

## AdalineSGD

Stochastic gradient descent is used to optimize the model. In this
method, the weights are updated after each training sample.

## Result

Both the Adaline models were trained on the Iris dataset.

<p float="left">
  <img src="images/AdalineGD Decision Region.png" width="400" />
  <img src="images/AdalineGD Training Error.png" width="400" /> 
</p>

<p float="left">
  <img src="images/AdalineSGD Decision Region.png" width="400" />
  <img src="images/AdalineSGD Training Error.png" width="400" /> 
</p>
