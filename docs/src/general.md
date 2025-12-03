# [General Usage](@id general)

AdvancedVI provides multiple variational inference (VI) algorithms.
Each algorithm defines its subtype of [`AdvancedVI.AbstractVariationalAlgorithm`](@ref) with some corresponding methods (see [this section](@ref algorithm)).
Then the algorithm can be executed by invoking `optimize`. (See [this section](@ref optimize)).

## [Running Variational Inference](@id optimize)

Given a subtype of `AbstractVariationalAlgorithm` associated with each algorithm, it suffices to call the function `optimize`:

```@docs
optimize
```

Each algorithm may interact differently with the arguments of `optimize`.
Therefore, please refer to the documentation of each different algorithm for a detailed description on their behavior and their requirements.

The `prob` argument to `optimize` must satisfy the LogDensityProblems.jl interface.
Some algorithms in AdvancedVI will call `logdensity_and_gradient` or `logdensity_gradient_and_hessian` methods using not a vector, but a view of an array.
If this is not supported by the `prob` argument, you should define this method to return `false`:

```@docs
use_view_in_gradient
```

## [Monitoring the Objective Value](@id estimate_objective)

Furthermore, each algorithm has an associated variational objective subject to *minimization*. (By convention, we assume all objectives are minimized rather than maximized.)
The progress made by each optimization algorithm can be diagnosed by monitoring the variational objective value.
This can be done by calling the following method.

```@docs
estimate_objective(::Random.AbstractRNG, ::AdvancedVI.AbstractVariationalAlgorithm, ::Any, ::Any; kwargs...)
```

## [Algorithm Interface](@id algorithm)

A variational inference algorithm supported by `AdvancedVI` should define its own subtype of `AbstractVariationalAlgorithm`:

```@docs
AdvancedVI.AbstractVariationalAlgorithm
```

The functionality of each algorithm is then implemented through the following methods:

```@docs
AdvancedVI.init(::Random.AbstractRNG, ::AdvancedVI.AbstractVariationalAlgorithm, ::Any, ::Any)
AdvancedVI.step
AdvancedVI.output
```

The role of each method should be self-explanatory and should be clear once we take a look at how `optimize` interacts with each algorithm.
The operation of `optimize` can be simplified as follows:

```julia
function optimize(rng, algorithm, max_iter, q_init, objargs; kwargs...)
    info_total = NamedTuple[]
    state = init(rng, algorithm, q_init, prob)
    for t in 1:max_iter
        info = (iteration=t,)
        state, terminate, info′ = step(
            rng, algorithm, state, callback, objargs...; kwargs...
        )
        info = merge(info′, info)

        if terminate
            break
        end

        push!(info_total, info)
    end
    out = output(algorithm, state)
    return out, info_total, state
end
```
