module AdvancedVIMooncakeExt

using ADTypes: ADTypes
using Accessors
using AdvancedVI
using LogDensityProblems
using Mooncake

Mooncake.@is_primitive(
    Mooncake.MinimalCtx,
    Tuple{
        typeof(LogDensityProblems.logdensity),
        AdvancedVI.MixedADLogDensityProblem,
        AbstractArray{<:Base.IEEEFloat},
    }
)

Mooncake.tangent_type(::Type{<:AdvancedVI.MixedADLogDensityProblem}) = Mooncake.NoTangent

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(LogDensityProblems.logdensity)},
    mixedad_prob::Mooncake.CoDual{<:AdvancedVI.MixedADLogDensityProblem},
    x::Mooncake.CoDual{<:AbstractArray{<:Base.IEEEFloat}},
)
    x, dx = Mooncake.arrayify(x)
    ℓπ, ∇ℓπ = LogDensityProblems.logdensity_and_gradient(
        Mooncake.primal(mixedad_prob).problem, x
    )
    function logdensity_pb(∂y)
        view(dx, 1:length(x)) .+= ∂y' * ∇ℓπ
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return Mooncake.zero_fcodual(ℓπ), logdensity_pb
end

function AdvancedVI._prepare_gradient(
    f, adtype::ADTypes.AutoMooncake, x::AbstractVector{<:Real}, aux
)
    config = if isnothing(adtype.config)
        Mooncake.Config(; friendly_tangents=false)
    else
        cfg = adtype.config
        Accessors.@set cfg.friendly_tangents = false
    end
    return Mooncake.prepare_gradient_cache(f, x, aux; config)
end

function AdvancedVI._value_and_gradient(
    f, prep, ::ADTypes.AutoMooncake, x::AbstractVector{<:Real}, aux
)
    y, (_, ∇y, _) = Mooncake.value_and_gradient!!(prep, f, x, aux)
    return y, copy(∇y)
end

end
