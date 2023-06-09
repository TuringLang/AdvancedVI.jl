
function pm_next!(pm, stats::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

"""
    optimize!(vo, alg::VariationalInference{AD}, q::VariationalPosterior, model::Model, θ; optimizer = TruncatedADAGrad())

Iteratively updates parameters by calling `grad!` and using the given `optimizer` to compute
the steps.
"""
function optimize(
    objective::AbstractVariationalObjective,
    rebuild::Function,
    n_max_iter::Int,
    λ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad(),
    rng       = Random.default_rng(),
    adbackend = AD.ForwardDiffBackend()
)
    # TODO: really need a better way to warn the user about potentially
    # not using the correct accumulator
    if (optimizer isa TruncatedADAGrad) && (λ ∉ keys(optimizer.acc))
        # this message should only occurr once in the optimization process
        @info "[$(string(objective))] Should only be seen once: optimizer created for θ" objectid(λ)
    end

    i = 0
    prog = ProgressMeter.Progress(
        n_max_iter; desc="[$(string(objective))] Optimizing...", barlen=0, enabled=PROGRESS[])

    # add criterion? A running mean maybe?
    time_elapsed = @elapsed begin
        for i = 1:n_max_iter
            Δλ, stats = estimate_gradient!(adbackend, rng, objective, λ, rebuild)
            
            # apply update rule
            Δλ = apply!(optimizer, λ, Δλ)
            @. λ = λ - Δλ

            stat′ = (Δλ=norm(Δλ),)
            stats = merge(stats, stat′)
        
            AdvancedVI.DEBUG && @debug "Step $i" stats...
            pm_next!(prog, stats)
        end
    end
    return λ
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
