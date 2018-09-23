# mlmono
Mono repo for all machine learning work.
For now this package rely on the tensorflow estimator feature.

## Directory structure

## Adding a model

Models should all go under the directory `models/`.

1. Write a class that inherits from `Graph`
```python
class Cnn(Graph):
    def __init__(self):
        # define the variables that you want to interface with the outside
        # mostly for debugging
```
2. Defined a forward-pass function for the model
```python
class Cnn(Graph):
    ...
    def add_forward_pass(self, features):
        # define the forward model
        # returns the prediction output
```

3. Write a config file

Config files are YAML files that describes a test case.
They technically can be anywhere but we suggest that you also put them under `configs/`.
A config file should have at least `io > dataset`, `model` and `trainer` fields.
The corresponding modules of the model will be initialized according to the variables under those fields.

## How to run

Before you can run the script, you usually need to run
```
python setup.py develop
```
in order to set up the entry point. The entry script is `mlmono.py`

### To train a model

```
python mlmono.py -m mlmono.cli --config configs/mnist.yml --train
```

### To do prediction

```
python mlmono.py -m mlmono.cli --config configs/mnist.yml --predict
```

## How it works
