
"""
    optimize(problem, objective, q_init, max_iter, objargs...; kwargs...)              

Optimize the variational objective `objective` targeting the problem `problem` by estimating (stochastic) gradients.

The trainable parameters in the variational approximation are expected to be extractable through `Optimisers.destructure`.
This requires the variational approximation to be marked as a functor through `Functors.@functor`.

# Arguments
- `objective::AbstractVariationalObjective`: Variational Objective.
- `q_init`: Initial variational distribution. The variational parameters must be extractable through `Optimisers.destructure`.
- `max_iter::Int`: Maximum number of iterations.
- `objargs...`: Arguments to be passed to `objective`.

# Keyword Arguments
- `adtype::ADtypes.AbstractADType`: Automatic differentiation backend. 
- `optimizer::Optimisers.AbstractRule`: Optimizer used for inference. (Default: `Adam`.)
- `averager::AbstractAverager` : Parameter averaging strategy. (Default: `NoAveraging()`)
- `operator::AbstractOperator` : Operator applied to the parameters after each optimization step. (Default: `IdentityOperator()`)
- `rng::AbstractRNG`: Random number generator. (Default: `Random.default_rng()`.)
- `show_progress::Bool`: Whether to show the progress bar. (Default: `true`.)
- `callback`: Callback function called after every iteration. See further information below. (Default: `nothing`.)
- `prog`: Progress bar configuration. (Default: `ProgressMeter.Progress(n_max_iter; desc="Optimizing", barlen=31, showspeed=true, enabled=prog)`.)
- `state::NamedTuple`: Initial value for the internal state of optimization. Used to warm-start from the state of a previous run. (See the returned values below.)

# Returns
- `averaged_params`: Variational parameters generated by the algorithm averaged according to `averager`.
- `params`: Last variational parameters generated by the algorithm.
- `stats`: Statistics gathered during optimization.
- `state`: Collection of the final internal states of optimization. This can used later to warm-start from the last iteration of the corresponding run.

# Callback
The callback function `callback` has a signature of

    callback(; stat, state, params, averaged_params, restructure, gradient)

The arguments are as follows:
- `stat`: Statistics gathered during the current iteration. The content will vary depending on `objective`.
- `state`: Collection of the internal states used for optimization.
- `params`: Variational parameters.
- `averaged_params`: Variational parameters averaged according to the averaging strategy.
- `restructure`: Function that restructures the variational approximation from the variational parameters. Calling `restructure(param)` reconstructs the variational approximation. 
- `gradient`: The estimated (possibly stochastic) gradient.

`callback` can return a `NamedTuple` containing some additional information computed within `cb`.
This will be appended to the statistic of the current corresponding iteration.
Otherwise, just return `nothing`.

"""
function optimize(
    rng::Random.AbstractRNG,
    problem,
    objective::AbstractVariationalObjective,
    q_init,
    max_iter::Int,
    objargs...;
    adtype::ADTypes.AbstractADType,
    optimizer::Optimisers.AbstractRule=Optimisers.Adam(),
    averager::AbstractAverager=NoAveraging(),
    operator::AbstractOperator=IdentityOperator(),
    show_progress::Bool=true,
    state_init::NamedTuple=NamedTuple(),
    callback=nothing,
    prog=ProgressMeter.Progress(
        max_iter; desc="Optimizing", barlen=31, showspeed=true, enabled=show_progress
    ),
)
    params, restructure = Optimisers.destructure(deepcopy(q_init))
    opt_st = maybe_init_optimizer(state_init, optimizer, params)
    obj_st = maybe_init_objective(state_init, rng, objective, problem, params, restructure)
    avg_st = maybe_init_averager(state_init, averager, params)
    grad_buf = DiffResults.DiffResult(zero(eltype(params)), similar(params))
    stats = NamedTuple[]

    for t in 1:max_iter
        stat = (iteration=t,)
        grad_buf, obj_st, stat′ = estimate_gradient!(
            rng,
            objective,
            adtype,
            grad_buf,
            problem,
            params,
            restructure,
            obj_st,
            objargs...,
        )
        stat = merge(stat, stat′)

        grad = DiffResults.gradient(grad_buf)
        opt_st, params = Optimisers.update!(opt_st, params, grad)
        params = operate(operator, typeof(q_init), params, restructure)
        avg_st = average(averager, avg_st, params)

        if !isnothing(callback)
            averaged_params = value(averager, avg_st)
            stat′ = callback(;
                stat,
                restructure,
                params=params,
                averaged_params=averaged_params,
                gradient=grad,
                state=(optimizer=opt_st, averager=avg_st, objective=obj_st),
            )
            stat = !isnothing(stat′) ? merge(stat′, stat) : stat
        end

        @debug "Iteration $t" stat...

        pm_next!(prog, stat)
        push!(stats, stat)
    end
    state = (optimizer=opt_st, averager=avg_st, objective=obj_st)
    stats = map(identity, stats)
    averaged_params = value(averager, avg_st)
    return restructure(averaged_params), restructure(params), stats, state
end

function optimize(
    problem,
    objective::AbstractVariationalObjective,
    q_init,
    max_iter::Int,
    objargs...;
    kwargs...,
)
    return optimize(
        Random.default_rng(), problem, objective, q_init, max_iter, objargs...; kwargs...
    )
end
