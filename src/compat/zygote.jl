using .Zygote

struct ZygoteAD <: ADBackend end
ADBackend(::Val{:zygote}) = ZygoteAD
function setadbackend(::Val{:zygote})
    return ADBACKEND[] = :zygote
end
export ZygoteAD

function AdvancedVI.grad!(
    out::DiffResults.MutableDiffResult, f, x, ::VariationalInference{<:AdvancedVI.ZygoteAD}
)
    y, back = Zygote.pullback(f, x)
    dy = first(back(one(eltype(y))))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, dy)
    return out
end
