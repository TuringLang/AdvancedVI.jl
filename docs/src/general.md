
# [General Usage](@id general)

AdvancedVI provides multiple variational inference (VI) algorithms.
Each algorithm defines its subtype of [`AdvancedVI.AbstractAlgorithm`](@ref) with some corresponding methods (see [this section](@ref algorithm)).
Then the algorithm can be executed by invoking `optimize`. (See [this section](@ref optimize)).

## [Optimize](@id optimize)

Given a subtype of `AbstractAlgorithm` associated with each algorithm, it suffices to call the function `optimize`:

```@docs
optimize
```

Each algorithm may interact differently with the arguments of `optimize`.
Therefore, please refer to the documentation of each different algorithm for a detailed description on their behavior and their requirements.

## [Algorithm Interface](@id algorithm)

A variational inference algorithm supported by `AdvancedVI` should define its own subtype of `AbstractAlgorithm`:

```@docs
AdvancedVI.AbstractAlgorithm
```

The functionality of each algorithm is then implemented through the following methods:

```@docs
AdvancedVI.init(::Random.AbstractRNG, ::AdvancedVI.AbstractAlgorithm, ::Any, ::Any)
AdvancedVI.step
AdvancedVI.output
```

The role of each method should be self-explanatory and should be clear once we take a look at how `optimize` interacts with each algorithm.
The operation of `optimize` can be simplified as follows:

```julia
function optimize([rng,] algorithm, max_iter, q_init, objargs; kwargs...)
    info_total = NamedTuple[]
    state = init(rng, algorithm, q_init)
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
