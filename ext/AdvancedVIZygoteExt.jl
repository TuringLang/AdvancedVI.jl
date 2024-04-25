module AdvancedVIZygoteExt

if isdefined(Base, :get_extension)
    using AdvancedVI: AdvancedVI, ADTypes, DiffResults, Distributions
    using Zygote: Zygote
else
    using ..AdvancedVI: AdvancedVI, ADTypes, DiffResults, Distributions
    using ..Zygote: Zygote
end

AdvancedVI.ADBackend(::Val{:zygote}) = ADTypes.AutoZygote()
function AdvancedVI.setadbackend(::Val{:zygote})
    Base.depwarn("`setadbackend` is deprecated. Please pass a `ADTypes.AbstractADType` as a keyword argument to the VI algorithm.", :setadbackend)
    AdvancedVI.ADBACKEND[] = :zygote
end

function AdvancedVI.grad!(
    vo,
    alg::AdvancedVI.VariationalInference{<:ADTypes.AutoZygote},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    f(θ) =
        if (q isa Distributions.Distribution)
            -vo(alg, AdvancedVI.update(q, θ), model, args...)
        else
            -vo(alg, q(θ), model, args...)
        end
    y, back = Zygote.pullback(f, θ)
    dy = first(back(1.0))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, dy)
    return out
end

end
