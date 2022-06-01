# Tensorflow-DARTS

A collection of MixedOperations for automatic architecture search.  
The goal is to make architectural search more accessible.
The collection therefore mainly contains operations with good scalability.
These operations should be able to run on the personal computer even for non-trivial problems.

For a quick start, have a look at the [Cifar10 example](./src/Examples/cifar10_keras.py) and read the short introduction below.

# MixedOp
```python
MixedOp(operations: List[tf.keras.layers.Layer], num_on_samples: int, name=None)
```
A MixedOp expects a list of layers from which num_on_samples many will be chosen for the final architecture.

# ContinuousMixedOp
```python
ContinuousMixedOp(operations: List[tf.keras.layers.Layer], num_on_samples: int, name=None)
```

The ContinuousMixedOp updates all operations and architectur weights simultaneously. It is described in [DARTS](https://arxiv.org/abs/1806.09055).

# BinaryMixedOp
```python
BinaryMixedOp(operations: List[tf.keras.layers.Layer], num_on_samples: int, name=None)
```
Combination of [BinaryConnect](https://arxiv.org/abs/1511.00363) and [DARTS](https://arxiv.org/abs/1806.09055). Updates all architecture weights but only num_on_sampels many operations at once. Uses the trick from the appendix of [ProxylessNAS](https://arxiv.org/abs/1812.00332) for a reduced memory footprint. Often produces better results than ContinuousMixedOps because the performance gap due to the discretisation between architecture search and fine tuning does not exist.

# BinaryMaskedMixedOp
```python
BinaryMaskedMixedOp(operations: List[Callable], num_eval_samples: int, num_on_samples: int, name=None)
```
Like BinaryMixedOp, but only num_eval_samples operations are evaluated to determine the update of the architecture. The number of updated operations is still num_on_samples. Both methods have the same memory footprint. The computational effort and thus the required evaluation time scales much better with BinaryMaskedMixedOps.

# BinaryProxylessNASStyleMixedOp
```python
BinaryProxylessNASStyleMixedOp(operations: List[Callable], num_eval_samples: int, num_on_samples: int, name=None)
```
BinaryMaskedMixedOp but with the post processing step described in [ProxylessNAS](https://arxiv.org/abs/1812.00332).

# BinaryMovingAverageMixedOp
```python
BinaryMovingAverageMixedOp(operations: List[Callable], num_eval_samples: int, num_on_samples: int, learning_rate: float)
```
BinaryMaskedMixedOp but the expected output of all operations is estimated using historic data. The learning_rate controls how far the horizon is for this. For example, a learning_rate of 1/100 would always take the average of the last 100 calls as the expected value. This type of MixedOps updates all architectural weights but only num_on_samples many operations per pass. Advantageous if the list of operations is very long.

# ArchitectureSearchModel

A tf.keras.model with 3 additional options.  
These are useful if, as is common in the literature, the architecture and the operations themselves are to be updated separately.
```
model.trainable_architecture_variables
```
Returns all architecture variables of the model. Requires that all layers of the model are either ArchitectureSearchModels themselves or normal layers and thus leaves of the structure.
```
model.trainable_operational_variables
```
Returns all non-architecture variables of the model. The same prerequisite as for trainable_architecture_variables applies.
```
model.architecture_summary()
```
Returns a string specifying the current distribution in the MixedOps.

# GraphMode

Unfortunately, even after a long search, I could not find a way to wrap the call of the MixedOps as a tf.function.
Therefore, the attempt will now throw an exception until further notice.
If you have an idea how this can be achieved, please take a look at my related [stackoverflow thread](https://stackoverflow.com/questions/72360420/how-to-evaluate-only-a-random-subset-of-all-possible-operations-per-pass-inside/).
