using .Zygote

struct ZygoteAD <: ADBackend end
ADBackend(::Val{:zygote}) = ZygoteAD
function setadbackend(::Val{:zygote})
    ADBACKEND[] = :zygote
end
export ZygoteAD

function AdvancedVI.grad!(
    alg::VariationalInference{<:AdvancedVI.ZygoteAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    f(θ) = if (q isa Distribution)
        - vo(alg, update(q, θ), model, args...)
    else
        - vo(alg, q(θ), model, args...)
    end
    y, back = Zygote.pullback(f, θ)
    dy = first(back(1.0))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, dy)
    return out
end