
# Release 0.6.2

Some subtle bugs in the implementations of `KLMinWassFwdBwd`, `KLMinNaturalGradDescent`, `KLMinSqrtNaturalGradDescent` have been fixed.
Mainly, when calling to `gauss_expected_grad_hess`, these wrapped the Hessian estimate into a `Symmetric` adaptor. For the reparametrization gradient, however (which is used when dealing with log-density problems with only first-order capabilities), the estimate is not almost surely symmetric.
So wrapping with `Symmetric` induces unexpected behavior.

# Release 0.6.1

`ReshufflingBatchSubsampling(dataset, batchsize)` now ensures that all batches have the size exactly `batchsize`. This means that, if "last batch" has an uneven batch size, it will be ignored. This is necessary to ensure that target log-density problems with `DifferentiationInterface.prepare_gradient` return the correct outcome.

In addition, a subtle related bug with algorithms sharing the `step` function in `common.jl` has been fixed. When calling `DifferentiationInterface.prepare_gradient` on the objective forward paths, now the subsampled objective is used instead of the non-subsampled objective. This means that `DifferentiationInterface.prepare_gradient` will see the correct type of objective used during optimization.

# Release 0.6

## New Algorithms

This update adds new variational inference algorithms in light of the flexibility added in the v0.5 update.
Specifically, the following measure-space optimization algorithms have been added:

  - `KLMinWassFwdBwd`
  - `KLMinNaturalGradDescent`
  - `KLMinSqrtNaturalGradDescent`
  - `FisherMinBatchMatch`

## Interface Change

The objective value returned by `estimate_objective` is now the value to be *minimized* by the algorithm.
For instance, for ELBO maximization algorithms, `estimate_objective` will return the negative ELBO.

## Behavior Change

In addition, `KLMinRepGradDescent`, `KLMinRepGradProxDescent`, `KLMinScoreGradDescent` will now throw a `RuntimException` if the objective value estimated at each step turns out to be degenerate (`Inf` or `NaN`). Previously, the algorithms ran until `max_iter` even if the optimization run has failed.

# Release 0.5

## Default Configuration Changes

The default parameters for the parameter-free optimizers `DoG` and `DoWG` has been changed.
Now, the choice of parameter should be more invariant to dimension such that convergence will become faster than before on high dimensional problems.

The default value of the `operator` keyword argument of `KLMinRepGradDescent` has been changed to `IdentityOperator` from `ClipScale`. This means that for variational families `<:MvLocationScale`, optimization may fail since there is nothing enforcing the scale matrix to be positive definite.
Therefore, in case a variational family of `<:MvLocationScale` is used in combination with `IdentityOperator`, a warning message instruting to use `ClipScale` will be displayed.

## Interface Changes

An additional layer of indirection, `AbstractVariationalAlgorithms` has been added.
Previously, all variational inference algorithms were assumed to run SGD in parameter space.
This design however, has proved to be too rigid.
Instead, each algorithm is now assumed to implement three simple interfaces: `init`, `step`, and `output`.

A new specialization of `estimate_objective` have been added that takes the variational algorithm `alg` as an argument.
Therefore, each algorithm should now implement `estimate_objective`.
This will automatically choose the right strategy for estimating the associated objective without having to worry about internal implementation details.

## Internal Changes

The state of the objectives `state` may now use a concrete type.
Therefore, to be able to dispatch based on the type of `state` while avoiding type ambiguities, the `state` argument in `estimate_gradient!` has been moved to the front.

Under the new interface `AbstractVariationalAlgorithms`, the algorithms running SGD in parameter space, currently `KLMinRepGradDescent`, `KLMinRepGradProxDescent`, `KLMinScoreGradDescent`, are treated as distinct algorithms.
However, they all implicitly share the same `step` function in `src/algorithms/common.jl` and the same fields for the `state ` object.
This may change in the future.
