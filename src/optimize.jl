
function pm_next!(pm, stats::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

"""
    optimize(
        objective    ::AbstractVariationalObjective,
        restructure,
        λ₀           ::AbstractVector{<:Real},
        n_max_iter   ::Int,
        objargs...;
        kwargs...
    )              

Optimize the variational objective `objective` by estimating (stochastic) gradients, where the variational approximation can be constructed by passing the variational parameters `λ₀` to the function `restructure`.

    optimize(
        objective ::AbstractVariationalObjective,
        q,
        n_max_iter::Int,
        objargs...;
        kwargs...
    )              

Optimize the variational objective `objective` by estimating (stochastic) gradients, where the initial variational approximation `q₀` supports the `Optimisers.destructure` interface.

# Arguments
- `objective`: Variational Objective.
- `λ₀`: Initial value of the variational parameters.
- `restruct`: Function that reconstructs the variational approximation from the flattened parameters.
- `q`: Initial variational approximation. The variational parameters must be extractable through `Optimisers.destructure`.
- `n_max_iter`: Maximum number of iterations.
- `objargs...`: Arguments to be passed to `objective`.
- `kwargs...`: Additional keywoard arguments. (See below.)

# Keyword Arguments
- `adbackend`: Automatic differentiation backend. (Type: `<: ADtypes.AbstractADType`.)
- `optimizer`: Optimizer used for inference. (Type: `<: Optimisers.AbstractRule`; Default: `Adam`.)
- `rng`: Random number generator. (Type: `<: AbstractRNG`; Default: `Random.default_rng()`.)
- `show_progress`: Whether to show the progress bar. (Type: `<: Bool`; Default: `true`.)
- `callback!`: Callback function called after every iteration. The signature is `cb(; stats, restructure, λ, g)`, which returns a dictionary-like object containing statistics to be displayed on the progress bar. The variational approximation can be reconstructed as `restructure(λ)`, `g` is the stochastic estimate of the gradient. (Default: `nothing`.)
- `prog`: Progress bar configuration. (Default: `ProgressMeter.Progress(n_max_iter; desc="Optimizing", barlen=31, showspeed=true, enabled=prog)`.)
- `state`: Initial value for the internal state of optimization. Used to warm-start from the state of a previous run. (See the returned values below.) (Type: `<: NamedTuple`.)

# Returns
- `λ`: Variational parameters optimizing the variational objective.
- `logstats`: Statistics and logs gathered during optimization.
- `states`: Collection of the final internal states of optimization. This can used later to warm-start from the last iteration of the corresponding run.
"""
function optimize(
    objective    ::AbstractVariationalObjective,
    restructure,
    λ₀           ::AbstractVector{<:Real},
    n_max_iter   ::Int,
    objargs...;
    adbackend    ::AbstractADType, 
    optimizer    ::Optimisers.AbstractRule = Optimisers.Adam(),
    rng          ::AbstractRNG             = default_rng(),
    show_progress::Bool                    = true,
    state        ::NamedTuple              = NamedTuple(),
    callback!                              = nothing,
    prog                                   = ProgressMeter.Progress(
        n_max_iter;
        desc      = "Optimizing",
        barlen    = 31,
        showspeed = true,
        enabled   = show_progress
    )
)
    λ        = copy(λ₀)
    opt_st   = haskey(state, :opt) ? state.opt : Optimisers.setup(optimizer, λ)
    obj_st   = haskey(state, :obj) ? state.obj : init(rng, objective, λ, restructure)
    grad_buf = DiffResults.GradientResult(λ)
    logstats = NamedTuple[]

    for t = 1:n_max_iter
        stat = (iteration=t,)

        grad_buf, obj_st, stat′ = estimate_gradient(
            rng, adbackend, objective, obj_st,
            λ, restructure, grad_buf; objargs...
        )
        stat = merge(stat, stat′)

        g         = DiffResults.gradient(grad_buf)
        opt_st, λ = Optimisers.update!(opt_st, λ, g)

        if !isnothing(callback!)
            stat′ = callback!(; stat, restructure, λ, g)
            stat = !isnothing(stat′) ? merge(stat′, stat) : stat
        end
        
        @debug "Iteration $t" stat...

        pm_next!(prog, stat)
        push!(logstats, stat)
    end
    state    = (opt=opt_st, obj=obj_st)
    logstats = map(identity, logstats)
    λ, logstats, state
end

function optimize(objective ::AbstractVariationalObjective,
                  q₀,
                  n_max_iter::Int;
                  kwargs...)
    λ, restructure = Optimisers.destructure(q₀)
    λ, logstats, state = optimize(
        objective, restructure, λ, n_max_iter; kwargs...
    )
    restructure(λ), logstats, state
end
