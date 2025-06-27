
# [General Usage](@id general)

AdvancedVI provides multiple variational inference (VI) algorithms.
Given a `<: AbstractAlgorithm` object associated with each algorithm, it suffices to call the function `optimize`:

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
