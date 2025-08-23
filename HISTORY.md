# Release 0.5

## Interface Changes

An additional layer of indirection, `AbstractAlgorithms` has been added.
Previously, all variational inference algorithms were assumed to run SGD in parameter space.
This desing however, is proving to be too rigid.
Instead, each algorithm is now assumed to implement three simple interfaces: `init`, `step`, and `output`.
Algorithms that run SGD in parameter space now need to implement the `AbstractVarationalObjective` interface of `ParamSpaceSGD <: AbstractAlgorithms`, which is a general implementation of the new interface.
Therefore, the old behavior of `AdvancedVI` is fully inhereted by `ParamSpaceSGD`.

## Internal Changes

The state of the objectives now use a concrete type.
Related to this, the objective `state` argument in `estimate_gradient!` has been moved to the front to avoid type ambiguities.
