module AdvancedVIEnzymeExt

if isdefined(Base, :get_extension)
    using AdvancedVI: AdvancedVI, ADTypes, DiffResults, Distributions
    using Enzyme: Enzyme
else
    using ..AdvancedVI: ADTypes, AdvancedVI
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
    f(θ) =
        if (q isa Distributions.Distribution)
            -vo(alg, AdvancedVI.update(q, θ), model, args...)
        else
            -vo(alg, q(θ), model, args...)
        end
    # Use `Enzyme.ReverseWithPrimal` once it is released:
    # https://github.com/EnzymeAD/Enzyme.jl/pull/598
    y = f(θ)
    DiffResults.value!(out, y)
    dy = DiffResults.gradient(out)
    fill!(dy, 0)
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(θ, dy))
    return out
end

end
