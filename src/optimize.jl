
"""
    optimize(problem, objective, variational_dist_init, max_iter; kwargs...)              

Optimize the variational objective `objective` targeting the problem `problem` by estimating (stochastic) gradients.

# Arguments
- `objective::AbstractVariationalObjective`: Variational Objective.
- `variational_dist_init`: Initial variational distribution. The variational parameters must be extractable through `Optimisers.destructure`.
- `max_iter::Int`: Maximum number of iterations.

# Keyword Arguments
- `adtype::ADtypes.AbstractADType`: Automatic differentiation backend. 
- `optimizer::Optimisers.AbstractRule`: Optimizer used for inference. (Default: `Adam`.)
- `rng::AbstractRNG`: Random number generator. (Default: `Random.default_rng()`.)
- `show_progress::Bool`: Whether to show the progress bar. (Default: `true`.)
- `callback`: Callback function called after every iteration. See further information below. (Default: `nothing`.)
- `prog`: Progress bar configuration. (Default: `ProgressMeter.Progress(n_max_iter; desc="Optimizing", barlen=31, showspeed=true, enabled=prog)`.)
- `state::NamedTuple`: Initial value for the internal state of optimization. Used to warm-start from the state of a previous run. (See the returned values below.)

Additional keyword arguments may apply depending on `objective`.

# Returns
- `variational_dist`: Variational distribution optimizing the variational objective.
- `stats`: Statistics gathered during optimization.
- `state`: Collection of the final internal states of optimization. This can used later to warm-start from the last iteration of the corresponding run.

# Callback
The callback function `callback` has a signature of

    callback(; stat, state, params, restructure, gradient)

The arguments are as follows:
- `stat`: Statistics gathered during the current iteration. The content will vary depending on `objective`.
- `state`: Collection of the internal states used for optimization.
- `params`: Variational parameters.
- `restructure`: Function that restructures the variational approximation from the variational parameters. Calling `restructure(param)` reconstructs the variational approximation. 
- `gradient`: The estimated (possibly stochastic) gradient.

`cb` can return a `NamedTuple` containing some additional information computed within `cb`.
This will be appended to the statistic of the current corresponding iteration.
Otherwise, just return `nothing`.

!!! info
    Some AD backends may only operator on "flattened" vectors.
    In this case, `AdvancedVI` will leverage `Optimisers.destructure` to flatten the variational distribution.
    (This is determined according to the value of `adtype`.)
    For this to automatically work however, `variational_dist_init` must be marked as a functor through `Functors.@functor`.
    Variational families provided by `AdvancedVI` will all be marked as functors already.
    Otherwise, one can simply use an AD backend that supported structured gradients such as `Zygote`.
"""
function optimize(
    rng          ::Random.AbstractRNG,
    problem,
    objective    ::AbstractVariationalObjective,
    q_init,
    max_iter     ::Int;
    adtype       ::ADTypes.AbstractADType, 
    optimizer    ::Optimisers.AbstractRule = Optimisers.Adam(),
    show_progress::Bool                    = true,
    state_init   ::NamedTuple              = NamedTuple(),
    callback                               = nothing,
    prog                                   = ProgressMeter.Progress(
        max_iter;
        desc      = "Optimizing",
        barlen    = 31,
        showspeed = true,
        enabled   = show_progress
    ),
    kwargs...
)
    q          = deepcopy(q_init)
    params, re = maybe_destructure(adtype, q)
    opt_st     = maybe_init_optimizer(state_init, optimizer, params)
    obj_st     = maybe_init_objective(state_init, rng, objective, params, q; kwargs...)
    stats      = NamedTuple[]

    for t = 1:max_iter
        stat = (iteration=t,)

        grad, obj_st, stat′ = estimate_gradient(
            rng,
            objective,
            adtype,
            problem,
            params,
            re,
            obj_st;
            kwargs...
        )
        stat = merge(stat, stat′)

        opt_st, params = update_variational_params!(
            typeof(q), opt_st, params, re, grad
        )

        if !isnothing(callback)
            stat′ = callback(
                ; stat, restructure=re, params=params, gradient=grad,
                state=(optimizer=opt_st, objective=obj_st)
            )
            stat = !isnothing(stat′) ? merge(stat′, stat) : stat
        end
        
        @debug "Iteration $t" stat...

        pm_next!(prog, stat)
        push!(stats, stat)
    end
    state = (optimizer=opt_st, objective=obj_st)
    stats = map(identity, stats)
    re(params), stats, state
end

function optimize(
    problem,
    objective              ::AbstractVariationalObjective,
    variational_dist_init,
    max_iter               ::Int,
    objargs...;
    kwargs...
)
    optimize(
        Random.default_rng(),
        problem,
        objective,
        variational_dist_init,
        max_iter,
        objargs...;
        kwargs...
    )
end
