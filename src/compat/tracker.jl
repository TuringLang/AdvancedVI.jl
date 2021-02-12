using .Tracker

ADBackend(::Val{:Tracker}) = ReverseDiffAD{getcache()}
function setadbackend(::Val{:Tracker})
    ADBACKEND[] = :Tracker
end

struct TrackerAD <: ADBackend end

ADBackend(::Val{:Tracker}) = TrackerAD

function grad!(
    vo,
    alg::VariationalInference{<:TrackerAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    θ_tracked = Tracker.param(θ)
    y = if (q isa Distribution)
        - vo(alg, update(q, θ_tracked), model, args...)
    else
        - vo(alg, q(θ_tracked), model, args...)
    end
    Tracker.back!(y, 1.0)

    DiffResults.value!(out, Tracker.data(y))
    DiffResults.gradient!(out, Tracker.grad(θ_tracked))
end