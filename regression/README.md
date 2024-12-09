# Overview
Tensorflow implementation of the experiment "Supervised Learning on Synthetic Graph Data" from the paper "Fully Distributed Online Training of Graph Neural Networks in Networked Systems". TODO: add link to the paper

## Dependencies

* **python>=3.8**
* **tensorflow>=2.13.0**: https://tensorflow.org
* **numpy**
* **matplotlib**

## Structure
* [regression](https://github.com/RostyslavUA/grad_cons/blob/master/regression.py): The main code to run node regression on a synthetic graph data.
* [data_generation](https://github.com/RostyslavUA/grad_cons/blob/master/data_generation.py): The data generating code 
* [gcns](https://github.com/RostyslavUA/grad_cons/blob/master/gcns.py): Implementation of GCNs. 
* [optimizers](https://github.com/RostyslavUA/grad_cons/blob/master/optimizers.py): The file containing optimizers, such as D-SGD and D-Adam.
* [utils](https://github.com/RostyslavUA/grad_cons/blob/master/utils.py): General utility functions.
* [training_utils](https://github.com/RostyslavUA/grad_cons/blob/master/training_utils.py): Training-specific utility functions.
* [results_plot](https://github.com/RostyslavUA/grad_cons/blob/master/results_plot.ipynb): Notebook for plotting the results.
RostyslavUA
## Usage
To run the experiment, execute `regression.py` with the desired flags. For example, to run distributed training with 
sigmoid activation functions, batch size of 100 and D-Adam optimizer, execute `python regression.py 3 distributed mix 1.0 sigmoid 100 dadam`.

Notice: to reproduce the results from the paper, please run the experiment with the parameter settings specified in the provided [bash script](https://github.com/RostyslavUA/fdTrainGNN/blob/main/regression/regression.sh).

## Feedback
For questions and comments, feel free to contact [Rostyslav Olshevskyi](mailto:ro22@rice.edu).

## Citation
Please cite (TODO: add citation) in your work when using this library in your experiments.
