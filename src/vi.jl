
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
    rebuild,
    n_max_iter::Int,
    λ         ::AbstractVector{<:Real};
    optimizer ::Optimisers.AbstractRule = TruncatedADAGrad(),
    rng       ::Random.AbstractRNG      = Random.GLOBAL_RNG,
    progress  ::Bool                    = true,
    callback!                           = nothing,
    terminate                           = (args...) -> false,
)
    # TODO: really need a better way to warn the user about potentially
    # not using the correct accumulator
    if (optimizer isa TruncatedADAGrad) && (λ ∉ keys(optimizer.acc))
        # this message should only occurr once in the optimization process
        @info "[$(string(objective))] Should only be seen once: optimizer created for θ" objectid(λ)
    end

    optstate = Optimisers.init(optimizer, λ)
    grad_buf = DiffResults.GradientResult(λ)

    q = rebuild(λ)
    i = 0
    prog = ProgressMeter.Progress(
        n_max_iter;
        barlen    = 0,
        enabled   = progress,
        showspeed = true)

    for i = 1:n_max_iter
        grad_buf, stats = estimate_gradient!(rng, objective, λ, rebuild, grad_buf)
        g = DiffResults.gradient(grad_buf)

        optstate, Δλ = Optimisers.apply!(optimizer, optstate, λ, g)
        Optimisers.subtract!(λ, Δλ)

        stat′ = (Δλ=norm(Δλ), gradient_norm=norm(g))
        stats = merge(stats, stat′)
        q     = rebuild(λ)

        if !isnothing(callback!)
            stat′  = callback!(q, stats)
            stats = !isnothing(stat′) ? merge(stat′, stats) : stats
        end
        
        AdvancedVI.DEBUG && @debug "Step $i" stats...
            pm_next!(prog, stats)

        # Termination decision is work in progress
        if terminate(rng, q, objective, stats)
            break
        end
    end
    λ
end

# function vi(grad_estimator, q, θ_init; optimizer = TruncatedADAGrad(), rng = Random.GLOBAL_RNG)
#     θ = copy(θ_init)
#     optimize!(grad_estimator, rebuild, n_max_iter, λ, optimizer = optimizer, rng = rng)

#     # If `q` is a mean-field approx we use the specialized `update` function
#     if q isa Distribution
#         return update(q, θ)
#     else
#         # Otherwise we assume it's a mapping θ → q
#         return q(θ)
#     end
# end
