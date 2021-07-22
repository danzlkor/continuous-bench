# BENCH
Bayesian EstimatioN of CHange

## What is it?
BENCH is a toolbox for identifying and estimating changes in parameters of a biophysical model between two groups of data (e.g. patients and controls). It is an alternative for model inversion, that is estimating the parameters for each group separately and compare the estimations. The advantage is it allows for using over parameterised models where the model inversion approaches fail because of parameter degeneraceis.  

## when it is useful?
Bench is usful when the aim is comparing two groups of data in terms of parameters of a biophysical model; but not estimating the parameters pre.se. It is particularly advantagous when the model has more free parameters than what can be estimated given the measurements.  

## How it works?
Using simulated data, we generate models of change that can predict how a baseline measurement changes if any of the parameters of the model changes. Then, in a bayesian framework, we estimate which of the `` change models'' can better explain the observed change between two groups. FOr more details, please refer to the paper(link). 


## How to install?
Here are the steps to install BENCH: 

1. clone the project in github. 
2. open a command line in the downloded directory (where there is a setup.py file)
3. run ``python setup.py install`` 

## What are the required inputs?


## How to use it?


## What are the outputs?

