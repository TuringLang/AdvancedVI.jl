
function pm_next!(pm, stats::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

"""
    optimize!(vo, alg::VariationalInference{AD}, q::VariationalPosterior, model::Model, θ; optimizer = TruncatedADAGrad())

Iteratively updates parameters by calling `grad!` and using the given `optimizer` to compute
the steps.
"""
function optimize(
    objective ::AbstractVariationalObjective,
    restructure,
    λ         ::AbstractVector{<:Real},
    n_max_iter::Int;
    optimizer ::Optimisers.AbstractRule = Optimisers.Adam(),
    rng       ::AbstractRNG             = default_rng(),
    progress  ::Bool                    = true,
    callback!                           = nothing,
    terminate                           = (args...) -> false,
    adback::AbstractADType              = AutoForwardDiff(), 
)
    opt_state = Optimisers.init(optimizer, λ)
    est_state = init(objective)
    grad_buf  = DiffResults.GradientResult(λ)

    prog = ProgressMeter.Progress(n_max_iter;
                                  barlen    = 0,
                                  enabled   = progress,
                                  showspeed = true)
    stats = Vector{NamedTuple}(undef, n_max_iter)

    for t = 1:n_max_iter
        stat = (iteration=t,)

        grad_buf, est_state, stat′ = estimate_gradient(
            rng, adback, objective, est_state, λ, restructure, grad_buf)
        g    = DiffResults.gradient(grad_buf)
        stat = merge(stat, stat′)

        opt_state, Δλ = Optimisers.apply!(optimizer, opt_state, λ, g)
        Optimisers.subtract!(λ, Δλ)

        stat′ = (iteration=t, Δλ=norm(Δλ), gradient_norm=norm(g))
        stat = merge(stat, stat′)

        q = restructure(λ)

        if !isnothing(callback!)
            stat′ = callback!(q, stat)
            stat = !isnothing(stat′) ? merge(stat′, stat) : stat
        end
        
        AdvancedVI.DEBUG && @debug "Step $t" stat...

        pm_next!(prog, stat)
        stats[t] = stat

        # Termination decision is work in progress
        if terminate(rng, λ, q, objective, stat)
            stats = stats[1:t]
            break
        end
    end
    λ, stats
end

function optimize(objective::AbstractVariationalObjective,
                  q,
                  n_max_iter::Int;
                  kwargs...)
    λ, restructure = Optimisers.destructure(q)
    λ, stats = optimize(objective, restructure, λ, n_max_iter; kwargs...)
    restructure(λ), stats
end
