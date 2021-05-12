# Eurographics 2021: Learning and Exploring Motor Skills with Spacetime Bounds

## 1. Introduction
This repository contains code reproduces results in "Learning and Exploring Motor Skills with Spacetime Bounds"(https://milkpku.github.io/project/spacetime.html). 
![spacetimeBounds](https://milkpku.github.io/project/spacetime/teaser.svg)

It also contains code implements feature extraction algorithms proposed by "Towards Robust Direction Invariance in Character Animation"(https://milkpku.github.io/project/hairyball.html)
![motionDirection](https://milkpku.github.io/project/hairyball/teaser.svg)
## 2. Requirements

### C++
name | version
----|----
Clang | 10.0.1 (required)
PyBullet | 2.89 (required)
Eigen | 3.3.7 (or later)
swig | 4.0.2 (or later)
CMake | 3.11.0 (or later)
gtest (optional) |

In our experiments, we use Clang to compile C++ code and use swig to build modules for Python.

Note:
1. We suspect there are bugs in linux gcc > 9.2 or kernel > 5.3 or our code somehow is not compatible with that. Our code has large numerical errors from unknown source given the new C++ compiler. Please use Clang 10.0.1 or test the project on Windows.
2. Later versions of pybullet remove SPD for joint control, which is critical for our system, thus version 2.89 is required.

### Python Pakages:
The version shouldn't matter. You should be safe to install the latest versions of these packages.

name | version
--|--
PyTorch | 1.8.0 (or later)
numpy | 1.19.1 (or later)
numba | 0.50.0 (or later)
tensorboardX | 2.2 (or later)

## 3. Setup
Before running pretrained models or training new models, code in folder `Kinematic` should be compiled. Following are compile commands:
```bash
cd Kinematic && mkdir build && cd build
cmake .. -DEIGEN_INCLUDE_PATH=<path_to_eigen_src_dir> -DPYTHON_INCLUDE_PATH=<path_to_python_include_dir>
make -j8
```

You should replace `<path_to_eigen_src_dir>` and `<path_to_python_include_dir>` with your own path to Eigen source and Python include files. 

## 4. Usage
Since we use `multiprocessing` in Python, there may be conflicts with OpenMP used by NumPy, so run
```bash
export OMP_NUM_THREADS=1
```
before training models.

To run pretrained models, firstly clone repo from https://github.com/milkpku/spacetimeBounds_policies and copy the contents to path `spacetimeBounds/data/policies`, then run:
```bash
python run_model.py args/demo_spacetime_walk.json
```

To train models:
```bash
python train.py args/train_spacetime_run.json --id spacetime_run
```