using .ReverseDiff: compile, GradientTape
using .ReverseDiff.DiffResults: GradientResult

tape(f, x) = GradientTape(f, x)
function taperesult(f, x)
    return tape(f, x), GradientResult(x)
end

# Precompiled tapes are not properly supported yet.
function AdvancedVI.grad!(
    f::Function,
    ::AutoReverseDiff,
    λ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    )
    tp = tape(f, λ)
    ReverseDiff.gradient!(out, tp, λ)
    return out
end
