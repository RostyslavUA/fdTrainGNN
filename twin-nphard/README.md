# Overview
Tensorflow implementation of the experiment "Graph-based Policy Gradient Descent for Link Scheduling" from the paper "Fully Distributed Online Training of Graph Neural Networks in Networked Systems". TODO: add link to the paper

## Dependencies
The following instructions assume that Python3.9 is your default Python3.
Other versions of Python3 may also work. 

`pip3 install -r requirements.txt`

Install any missing packages while running the code or notebook.

## Directory
```bash
├── bash # bash command
├── data # training and testing datasets
├── gcn # GCN modules
├── model # Trained models
├── output # Raw experiment outputs
├── plot_test_results.ipynb # Scripts of figure plotting
├── plot_training.ipynb # Plotting training curves
├── LICENSE
├── README.md
└── requirements.txt
```

## Usage
To run MWIS experiment execute `mwis_gcn_train_twin.py` with the desired flags (see `runtime_config.py`). For example, distributed training with
Adam optimizer and the learning rate of 0.00005, run
`python mwis_gcn_train_twin.py --training_set=dist_GCN --architecture=decentralized --optimizer=Adam --learning_rate=0.001`.

## Feedback
For questions and comments, feel free to contact [Rostyslav Olshevskyi](mailto:ro22@rice.edu).

## Citation
Please cite (TODO: add citation) in your work when using this library in your experiments.
