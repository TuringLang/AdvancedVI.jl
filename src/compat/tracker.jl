using .Tracker

ADBackend(::Val{:tracker}) = ReverseDiffAD{getcache()}
function setadbackend(::Val{:tracker})
    ADBACKEND[] = :tracker
end

struct TrackerAD <: ADBackend end

ADBackend(::Val{:tracker}) = TrackerAD

function grad!(
    out::DiffResults.MutableDiffResult,
    f,
    x,
    ::VariationalInference{<:TrackerAD},
)
    x_tracked = Tracker.param(x)
    y = f(x_tracked)
    Tracker.back!(y, one(eltype(y)))

    DiffResults.value!(out, Tracker.data(y))
    DiffResults.gradient!(out, Tracker.grad(x_tracked))
end