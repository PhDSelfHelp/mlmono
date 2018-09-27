# mlmono
![](https://travis-ci.com/PhDSelfHelp/mlmono.svg?branch=master)
[![](https://codecov.io/gh/PhDSelfHelp/mlmono/branch/master/graph/badge.svg)](https://codecov.io/gh/PhDSelfHelp/mlmono)

Mono repo for all machine learning work.
For now this package rely on the tensorflow estimator feature.

## Adding a model

Models should all go under the directory `models/`.

1. Write a class that inherits from `MLGraph`
```python
class CNN(MLGraph):
    def __init__(self):
        # define the variables that you want to interface with the outside
        # mostly for debugging
        # input and output are the default names for the forward model as defined in `MLGraph`
        # self.input = None
        # self.output = None
```
2. Defined a forward-pass function for the model
```python
class CNN(MLGraph):
    ...
    def add_forward_pass(self, features):
        # define the forward model
        # returns the prediction output
        # self.input and self.output must now refer to a variable node
```

3. Write a config file

Config files are YAML files that describes a test case.
They technically can be anywhere but we suggest that you put them under `configs/` and keep track of them along with the version controll system.
A config file should have at least `io > dataset`, `model` and `trainer` fields.
The corresponding modules of the model will be initialized according to the variables under those fields.
Refer to the example config for more details.

## How to run

Before you can run the script, you usually need to run
```
python setup.py develop
```
in order to set up the entry point.
This is necessary to make the import paths correct.

The entry script is `cli.py` so basically you can run the `mlmono.cli` module in most of the cases.

### To train a model

```
mlmono train --config configs/mnist.yml
```

### To do prediction

```
mlmono predict --config configs/mnist.yml
```

## How it works
