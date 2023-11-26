# Disruption Prediction Codes

## Prerequisitions

h5py

numpy

pytorch

scipy \(optional\)

matplotlib \(optional\)

## Brief Introduction

The main function of this set of code is to sort out the disruption signals from the normal signals of the Alcator C-Mod tokamak. This is achieved with a set of sub convolutional neural networks \(CNN\). Each of the subnetworks process a signal channel \(i.e. the ip, the vloop, or the horizontal/vertical displacement\). Then a voting network is used to evaluate the weight for the signal channels.

## How to run the training codes?

1. Download the C-Mod database to the base directory.
2. Copy the Solver_\*\*\*.py and the LuNet7_\*\*\*.py in the directory ./\*\*\* to the base directory.
3. Modify the DisruptionLu.py accordingly \(i.e. _from Solver\_\*\*\* import \*_\). If you want to use cuda, just uncomment the lines transferring the tensors to the cuda. If you want to continue training the existing model, just also copy the LuNet7_\*\*\*.pkl to the base directory and uncomment the line _net.load\_state\_dict\(\*\*\*\)_.
4. Run DisruptionLu.py.
5. Enter 1 to save the trained model.
6. Copy the \*\*\*.pkl file to the directory ./\*\*\*.

## How to run the test codes?

1. Keep the C-Mod database in the base directory.
2. run test.py.
