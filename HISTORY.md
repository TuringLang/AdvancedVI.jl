# Release 0.5

## Default Configuration Changes

The default parameters for the parameter-free optimizers `DoG` and `DoWG` has been changed.
Now, the choice of parameter should be more invariant to dimension such that convergence will become faster than before on high dimensional problems.

The default value of the `operator` keyword argument of `KLMinRepGradDescent` has been changed to `IdentityOperator` from `ClipScale`. This means that for variational families `<:MvLocationScale`, optimization may fail since there is nothing enforcing the scale matrix to be positive definite.
Therefore, in case a variational family of `<:MvLocationScale` is used in combination with `IdentityOperator`, a warning message instruting to use `ClipScale` will be displayed.

## Interface Changes

An additional layer of indirection, `AbstractVariationalAlgorithms` has been added.
Previously, all variational inference algorithms were assumed to run SGD in parameter space.
This desing however, is proving to be too rigid.
Instead, each algorithm is now assumed to implement three simple interfaces: `init`, `step`, and `output`.
Algorithms that run SGD in parameter space now need to implement the `AbstractVarationalObjective` interface of `ParamSpaceSGD <: AbstractVariationalAlgorithms`, which is a general implementation of the new interface.
Therefore, the old behavior of `AdvancedVI` is fully inhereted by `ParamSpaceSGD`.

## Internal Changes

The state of the objectives now use a concrete type.
Related to this, the objective `state` argument in `estimate_gradient!` has been moved to the front to avoid type ambiguities.
