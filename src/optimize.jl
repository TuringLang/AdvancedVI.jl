
"""
    optimize(
        problem,
        objective   ::AbstractVariationalObjective,
        restructure,
        param_init  ::AbstractVector{<:Real},
        max_iter    ::Int,
        objargs...;
        kwargs...
    )              

Optimize the variational objective `objective` targeting the problem `problem` by estimating (stochastic) gradients, where the variational approximation can be constructed by passing the variational parameters `param_init` to the function `restructure`.

    optimize(
        problem,
        objective             ::AbstractVariationalObjective,
        variational_dist_init,
        max_iter              ::Int,
        objargs...;
        kwargs...
    )              

Optimize the variational objective `objective` targeting the problem `problem` by estimating (stochastic) gradients, where the initial variational approximation `variational_dist_init` supports the `Optimisers.destructure` interface.

# Arguments
- `objective`: Variational Objective.
- `param_init`: Initial value of the variational parameters.
- `restruct`: Function that reconstructs the variational approximation from the flattened parameters.
- `variational_dist_init`: Initial variational distribution. The variational parameters must be extractable through `Optimisers.destructure`.
- `max_iter`: Maximum number of iterations.
- `objargs...`: Arguments to be passed to `objective`.
- `kwargs...`: Additional keywoard arguments. (See below.)

# Keyword Arguments
- `adbackend::ADtypes.AbstractADType`: Automatic differentiation backend. 
- `optimizer::Optimisers.AbstractRule`: Optimizer used for inference. (Default: `Adam`.)
- `rng::AbstractRNG`: Random number generator. (Default: `Random.default_rng()`.)
- `show_progress::Bool`: Whether to show the progress bar. (Default: `true`.)
- `callback!`: Callback function called after every iteration. See further information below. (Default: `nothing`.)
- `prog`: Progress bar configuration. (Default: `ProgressMeter.Progress(n_max_iter; desc="Optimizing", barlen=31, showspeed=true, enabled=prog)`.)
- `state::NamedTuple`: Initial value for the internal state of optimization. Used to warm-start from the state of a previous run. (See the returned values below.)

# Returns
- `params`: Variational parameters optimizing the variational objective.
- `stats`: Statistics gathered during optimization.
- `state`: Collection of the final internal states of optimization. This can used later to warm-start from the last iteration of the corresponding run.

# Callback
The callback function `callback!` has a signature of

    cb(; stat, state, param, restructure, gradient)

The arguments are as follows:
- `stat`: Statistics gathered during the current iteration. The content will vary depending on `objective`.
- `state`: Collection of the internal states used for optimization.
- `param`: Variational parameters.
- `restructure`: Function that restructures the variational approximation from the variational parameters. Calling `restructure(param)` reconstructs the variational approximation. 
- `gradient`: The estimated (possibly stochastic) gradient.

`cb` can return a `NamedTuple` containing some additional information computed within `cb`.
This will be appended to the statistic of the current corresponding iteration.
Otherwise, just return `nothing`.

"""
function optimize(
    problem,
    objective    ::AbstractVariationalObjective,
    restructure,
    params_init  ::AbstractVector{<:Real},
    max_iter     ::Int,
    objargs...;
    adbackend    ::ADTypes.AbstractADType, 
    optimizer    ::Optimisers.AbstractRule = Optimisers.Adam(),
    rng          ::Random.AbstractRNG      = Random.default_rng(),
    show_progress::Bool                    = true,
    state_init   ::NamedTuple              = NamedTuple(),
    callback!                              = nothing,
    prog                                   = ProgressMeter.Progress(
        max_iter;
        desc      = "Optimizing",
        barlen    = 31,
        showspeed = true,
        enabled   = show_progress
    )
)
    λ        = copy(params_init)
    opt_st   = maybe_init_optimizer(state_init, optimizer, λ)
    obj_st   = maybe_init_objective(state_init, rng, objective, λ, restructure)
    grad_buf = DiffResults.DiffResult(zero(eltype(λ)), similar(λ))
    stats    = NamedTuple[]

    for t = 1:max_iter
        stat = (iteration=t,)

        grad_buf, obj_st, stat′ = estimate_gradient!(
            rng, problem, adbackend, objective, obj_st,
            λ, restructure, grad_buf, objargs...
        )
        stat = merge(stat, stat′)

        g         = DiffResults.gradient(grad_buf)
        opt_st, λ = Optimisers.update!(opt_st, λ, g)

        if !isnothing(callback!)
            stat′ = callback!(
                ; stat, restructure, params=λ, gradient=g,
                state=(optimizer=opt_st, objective=obj_st)
            )
            stat = !isnothing(stat′) ? merge(stat′, stat) : stat
        end
        
        @debug "Iteration $t" stat...

        pm_next!(prog, stat)
        push!(stats, stat)
    end
    state  = (optimizer=opt_st, objective=obj_st)
    stats  = map(identity, stats)
    params = λ
    params, stats, state
end

function optimize(problem,
                  objective ::AbstractVariationalObjective,
                  variational_dist_init,
                  n_max_iter::Int,
                  objargs...;
                  kwargs...)
    λ, restructure = Optimisers.destructure(variational_dist_init)
    λ, logstats, state = optimize(
        problem, objective, restructure, λ, n_max_iter, objargs...; kwargs...
    )
    restructure(λ), logstats, state
end
