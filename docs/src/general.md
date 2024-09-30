# [General Usage](@id general)

Each VI algorithm provides the followings:

 1. Variational families supported by each VI algorithm.
 2. A variational objective corresponding to the VI algorithm.
    Note that each variational family is subject to its own constraints.
    Thus, please refer to the documentation of the variational inference algorithm of interest.

## Optimizing a Variational Objective

After constructing a *variational objective* `objective` and initializing a *variational approximation*, one can optimize `objective` by calling `optimize`:

```@docs
optimize
```

## Estimating the Objective

In some cases, it is useful to directly estimate the objective value.
This can be done by the following funciton:

```@docs
estimate_objective
```

!!! info
    
    Note that `estimate_objective` is not expected to be differentiated through, and may not result in optimal statistical performance.

## Advanced Usage

Each variational objective is a subtype of the following abstract type:

```@docs
AdvancedVI.AbstractVariationalObjective
```

Furthermore, `AdvancedVI` only interacts with each variational objective by querying gradient estimates.
Therefore, to create a new custom objective to be optimized through `AdvancedVI`, it suffices to implement the following function:

```@docs
AdvancedVI.estimate_gradient!
```

If an objective needs to be stateful, one can implement the following function to inialize the state.

```@docs
AdvancedVI.init
```
