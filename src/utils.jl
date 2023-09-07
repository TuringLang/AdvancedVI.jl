
function pm_next!(pm, stats::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

function maybe_init_optimizer(
    state_init::Union{Nothing, NamedTuple},
    optimizer ::Optimisers.AbstractRule,
    λ         ::AbstractVector
)
    haskey(state_init, :optimizer) ? state_init.optimizer : Optimisers.setup(optimizer, λ)
end

function maybe_init_objective(
    state_init::Union{Nothing, NamedTuple},
    rng       ::Random.AbstractRNG,
    objective ::AbstractVariationalObjective,
    λ         ::AbstractVector,
    restructure
)
    haskey(state_init, :objective) ? state_init.objective : init(rng, objective, λ, restructure)
end

