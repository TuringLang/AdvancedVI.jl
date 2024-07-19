
module AdvancedVIZygoteExt

if isdefined(Base, :get_extension)
    using AdvancedVI
    using AdvancedVI: ADTypes, DiffResults
    using ChainRulesCore
    using Zygote
else
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes, DiffResults
    using ..ChainRulesCore
    using ..Zygote
end

function AdvancedVI.stop_gradient(::ADTypes.AutoZygote, x)
    return ChainRulesCore.ignore_derivatives(x)
end

function AdvancedVI.value_and_gradient!(
    ::ADTypes.AutoZygote,
    f,
    x::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult
)
    y, back = Zygote.pullback(f, x)
    ∇x = back(one(y))
    if only(∇x) === nothing
        # this is necessary in case of non-diff function
        # since nothing can't be stored in DiffResults
        grad = zeros(length(x))
    else
        grad = only(∇x)
    end
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, grad)
    return out
end

function AdvancedVI.value_and_gradient!(
    ad::ADTypes.AutoZygote,
    f,
    x::AbstractVector{<:Real},
    aux,
    out::DiffResults.MutableDiffResult
)
    AdvancedVI.value_and_gradient!(ad, x′ -> f(x′, aux), x, out)
end

end
