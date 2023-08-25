
function pm_next!(pm, stats::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

"""
    optimize(
        objective    ::AbstractVariationalObjective,
        restructure,
        λ₀           ::AbstractVector{<:Real},
        n_max_iter   ::Int;
        kwargs...
    )              

Optimize the variational objective `objective` by estimating (stochastic) gradients, where the variational approximation can be constructed by passing the variational parameters `λ₀` to the function `restructure`.

    optimize(
        objective ::AbstractVariationalObjective,
        q,
        n_max_iter::Int;
        kwargs...
    )              

Optimize the variational objective `objective` by estimating (stochastic) gradients, where the initial variational approximation `q₀` supports the `Optimisers.destructure` interface.

# Arguments
- `objective`: Variational Objective.
- `λ₀`: Initial value of the variational parameters.
- `restruct`: Function that reconstructs the variational approximation from the flattened parameters.
- `q`: Initial variational approximation. The variational parameters must be extractable through `Optimisers.destructure`.
- `n_max_iter`: Maximum number of iterations.

# Keyword Arguments
- `adbackend`: Automatic differentiation backend. (Type: `<: ADtypes.AbstractADType`.)
- `optimizer`: Optimizer used for inference. (Type: `<: Optimisers.AbstractRule`; Default: `Adam`.)
- `rng`: Random number generator. (Type: `<: AbstractRNG`; Default: `Random.default_rng()`.)
- `show_progress`: Whether to show the progress bar. (Type: `<: Bool`; Default: `true`.)
- `callback!`: Callback function called after every iteration. The signature is `cb(; obj_state, stats, restructure, λ, g)`, which returns a dictionary-like object containing statistics to be displayed on the progress bar. The variational approximation can be reconstructed as `restructure(λ)`. If the estimator associated with `objective` is stateful, `obj_state` contains its state. (Default: `nothing`.) `g` is the stochastic gradient.
- `prog`: Progress bar configuration. (Default: `ProgressMeter.Progress(n_max_iter; desc="Optimizing", barlen=31, showspeed=true, enabled=prog)`.)

When resuming from the state of a previous run, use the following keyword arguments:
- `opt_state`: Initial state of the optimizer.
- `obj_state`: Initial state of the objective.

# Returns
- `λ`: Variational parameters optimizing the variational objective.
- `stats`: Statistics gathered during inference.
- `opt_state`: Final state of the optimiser.
- `obj_state`: Final state of the objective.
"""
function optimize(
    objective    ::AbstractVariationalObjective,
    restructure,
    λ₀           ::AbstractVector{<:Real},
    n_max_iter   ::Int;
    adbackend::AbstractADType, 
    optimizer    ::Optimisers.AbstractRule = Optimisers.Adam(),
    rng          ::AbstractRNG             = default_rng(),
    show_progress::Bool                    = true,
    opt_state                              = nothing,
    obj_state                              = nothing,
    callback!                              = nothing,
    prog                                   = ProgressMeter.Progress(
        n_max_iter;
        desc      = "Optimizing",
        barlen    = 31,
        showspeed = true,
        enabled   = show_progress
    )              
)
    λ         = copy(λ₀)
    opt_state = isnothing(opt_state) ? Optimisers.setup(optimizer, λ)       : opt_state
    obj_state = isnothing(obj_state) ? init(rng, objective, λ, restructure) : obj_state
    grad_buf  = DiffResults.GradientResult(λ)
    stats     = NamedTuple[]

    for t = 1:n_max_iter
        stat = (iteration=t,)

        grad_buf, obj_state, stat′ = estimate_gradient!(
            rng, adbackend, objective, obj_state, λ, restructure, grad_buf)
        stat = merge(stat, stat′)

        g            = DiffResults.gradient(grad_buf)
        opt_state, λ = Optimisers.update!(opt_state, λ, g)
        stat′ = (iteration = t,)
        stat = merge(stat, stat′)

        if !isnothing(callback!)
            stat′ = callback!(; obj_state, stat, restructure, λ, g)
            stat = !isnothing(stat′) ? merge(stat′, stat) : stat
        end
        
        @debug "Iteration $t" stat...

        pm_next!(prog, stat)
        push!(stats, stat)
    end
    λ, map(identity, stats), opt_state, obj_state
end

function optimize(objective ::AbstractVariationalObjective,
                  q₀,
                  n_max_iter::Int;
                  kwargs...)
    λ, restructure = Optimisers.destructure(q₀)
    λ, stats, opt_state, obj_state = optimize(
        objective, restructure, λ, n_max_iter; kwargs...
    )
    restructure(λ), stats, opt_state, obj_state
end
