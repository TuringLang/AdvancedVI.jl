
"""
    optimize(
        [rng::Random.AbstractRNG = Random.default_rng(),]
        algorithm::AbstractAlgorithm,
        max_iter::Int,
        prob,
        q_init,
        args...;
        kwargs...
    )

Run variational inference `algorithm` on the `problem` implementing the `LogDensityProblems` interface.
For more details on the usage, refer to the documentation corresponding to `algorithm`.

# Arguments
- `rng`: Random number generator.
- `algorithm`: Variational inference algorithm.
- `max_iter::Int`: Maximum number of iterations.
- `prob`: Target `LogDensityProblem` 
- `q_init`: Initial variational distribution.
- `args...`: Arguments to be passed to `algorithm`.

# Keyword Arguments
- `show_progress::Bool`: Whether to show the progress bar. (Default: `true`.)
- `state::Union{<:Any,Nothing}`: Initial value for the internal state of optimization. Used to warm-start from the state of a previous run. (See the returned values below.)
- `callback`: Callback function called after every iteration. See further information below. (Default: `nothing`.)
- `progress::ProgressMeter.AbstractProgress`: Progress bar configuration. (Default: `ProgressMeter.Progress(n_max_iter; desc="Optimizing", barlen=31, showspeed=true, enabled=prog)`.)
- `kwargs...`: Keyword arguments to be passed to `algorithm`.

# Returns
- `output`: The output of the variational inference algorithm.
- `info`: Array of `NamedTuple`s, where each `NamedTuple` contains information generated at each iteration.
- `state`: Collection of the final internal states of optimization. This can used later to warm-start from the last iteration of the corresponding run.

# Callback
The signature of the callback function depends on the `algorithm` in use.
Thus, see the documentation for each `algorithm`.
However, a callback should return either a `nothing` or a `NamedTuple` containing information generated during the current iteration.
The content of the `NamedTuple` will be concatenated into the corresponding entry in the `info` array returns in the end of the call to `optimize` and will be displayed on the progress meter.
"""
function optimize(
    rng::Random.AbstractRNG,
    algorithm::AbstractAlgorithm,
    max_iter::Int,
    prob,
    q_init,
    args...;
    show_progress::Bool=true,
    state::Union{<:Any,Nothing}=nothing,
    callback=nothing,
    progress::ProgressMeter.AbstractProgress=ProgressMeter.Progress(
        max_iter; desc="Optimizing", barlen=31, showspeed=true, enabled=show_progress
    ),
    kwargs...,
)
    info_total = NamedTuple[]
    state = if isnothing(state)
        init(rng, algorithm, prob, q_init)
    else
        state
    end

    for t in 1:max_iter
        info = (iteration=t,)

        state, terminate, info′ = step(rng, algorithm, state, callback, args...; kwargs...)
        info = merge(info′, info)

        if terminate
            break
        end

        pm_next!(progress, info)
        push!(info_total, info)
    end
    out = output(algorithm, state)
    return out, map(identity, info_total), state
end

function optimize(
    algorithm::AbstractAlgorithm, max_iter::Int, prob, q_init, objargs...; kwargs...
)
    return optimize(
        Random.default_rng(), algorithm, max_iter, prob, q_init, objargs...; kwargs...
    )
end
