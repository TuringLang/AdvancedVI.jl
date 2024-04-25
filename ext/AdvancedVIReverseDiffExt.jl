module AdvancedVIReverseDiffExt

if isdefined(Base, :get_extension)
    using AdvancedVI: AdvancedVI, ADTypes, DiffResults, Distributions
    using ReverseDiff: ReverseDiff
else
    using ..AdvancedVI: ADTypes, AdvancedVI
    using ..ReverseDiff: ReverseDiff
end

AdvancedVI.ADBackend(::Val{:reversediff}) = ADTypes.AutoReverseDiff()

function AdvancedVI.setadbackend(::Val{:reversediff})
    Base.depwarn("`setadbackend` is deprecated. Please pass a `ADTypes.AbstractADType` as a keyword argument to the VI algorithm.", :setadbackend)
    AdvancedVI.ADBACKEND[] = :reversediff
end

tape(f, x) = ReverseDiff.GradientTape(f, x)

function AdvancedVI.grad!(
    vo,
    alg::AdvancedVI.VariationalInference{<:ADTypes.AutoReverseDiff},
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
    tp = tape(f, θ)
    ReverseDiff.gradient!(out, tp, θ)
    return out
end

end
