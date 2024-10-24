module AdvancedVIEnzymeExt

if isdefined(Base, :get_extension)
    using AdvancedVI: AdvancedVI, ADTypes, DiffResults, Distributions
    using Enzyme: Enzyme
else
    using ..AdvancedVI: AdvancedVI, ADTypes, DiffResults, Distributions
    using ..Enzyme: Enzyme
end

AdvancedVI.ADBackend(::Val{:enzyme}) = ADTypes.AutoEnzyme()
function AdvancedVI.setadbackend(::Val{:enzyme})
    Base.depwarn("`setadbackend` is deprecated. Please pass a `ADTypes.AbstractADType` as a keyword argument to the VI algorithm.", :setadbackend)
    AdvancedVI.ADBACKEND[] = :enzyme
end

function AdvancedVI.grad!(
    vo,
    alg::AdvancedVI.VariationalInference{<:ADTypes.AutoEnzyme},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    f(θ) = if (q isa Distributions.Distribution)
        -vo(alg, AdvancedVI.update(q, θ), model, args...)
    else
        -vo(alg, q(θ), model, args...)
    end

    y = f(θ)
    DiffResults.value!(out, y)
    dy = DiffResults.gradient(out)
    fill!(dy, 0)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal, true),
        Enzyme.Const(f),
        Enzyme.Active,
        Enzyme.Duplicated(θ, dy)
    )
    return out
end

end
