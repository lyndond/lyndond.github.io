# Neural Net C++

A simple Multi-Layer Perceptron (MLP) built from scratch in C++ that successfully learns to approximate the sin(x)^2 function.

## How to run (on Mac)

CFirst, compile and run the C++ program to train the model and generate the output data:

```bash
g++ -std=c++20 -Wall main.cpp -o my_program  
./my_program  
```

Then, plot the results using the Python script (assuming you're in a Python environment with matplotlib, numpy, and seaborn):

```bash
python3 plot_mlp_training.py
```

## Change Log

### 2025-09-08

- Fixed a series of critical bugs that prevented the neural network from training successfully.
- The initial "noise" problem was a bug in the random number generator. It was re-seeding on every call, causing the model's weights to be initialized with the same values every time. This was fixed by making the random number engine static so it is only seeded once.
- After fixing the initialization, the training loop was debugged by:
  - Reducing an overly high learning rate that caused chaotic, non-convergent behavior.
  - Fixing a bug where the learning rate hyperparameter was being ignored, which allowed for proper tuning.
  - Increasing the data logging frequency to enable correct visualization of the loss curve.
  - The build process was also updated to use the C++20 standard to support modern language features.
