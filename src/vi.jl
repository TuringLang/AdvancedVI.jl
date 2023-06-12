
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

    prog = ProgressMeter.Progress(n_max_iter;
                                  barlen    = 0,
                                  enabled   = progress,
                                  showspeed = true)
    stats = Vector{NamedTuple}(undef, n_max_iter)

    for t = 1:n_max_iter
        grad_buf, stat = estimate_gradient!(rng, objective, λ, restructure, grad_buf)
        g = DiffResults.gradient(grad_buf)

        optstate, Δλ = Optimisers.apply!(optimizer, optstate, λ, g)
        Optimisers.subtract!(λ, Δλ)

        stat′ = (Δλ=norm(Δλ), gradient_norm=norm(g))
        stat  = merge(stat, stat′)
        q     = restructure(λ)

        if !isnothing(callback!)
            stat′ = callback!(q, stat)
            stat = !isnothing(stat′) ? merge(stat′, stat) : stat
        end
        
        AdvancedVI.DEBUG && @debug "Step $i" stat...

        pm_next!(prog, stat)
        stats[t] = stat

        # Termination decision is work in progress
        if terminate(rng, q, objective, stat)
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
