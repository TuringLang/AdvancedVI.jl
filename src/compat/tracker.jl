using .Tracker

ADBackend(::Val{:Tracker}) = ReverseDiffAD{getcache()}
function setadbackend(::Val{:Tracker})
    ADBACKEND[] = :Tracker
end

struct TrackerAD <: ADBackend end

ADBackend(::Val{:Tracker}) = TrackerAD

function grad!(
    out::DiffResults.MutableDiffResult,
    f,
    x,
    ::VariationalInference{<:TrackerAD},
)
    x_tracked = Tracker.param(x)
    y = f(x_tracked)
    Tracker.back!(y, 1.0)

    DiffResults.value!(out, Tracker.data(y))
    DiffResults.gradient!(out, Tracker.grad(x_tracked))
end