module AdvancedVIMooncakeExt

using AdvancedVI
using LogDensityProblems
using Mooncake

Mooncake.@is_primitive(
    Mooncake.MinimalCtx,
    Tuple{
        typeof(LogDensityProblems.logdensity),
        AdvancedVI.MixedADLogDensityProblem,
        <:AbstractArray{<:Base.IEEEFloat},
    }
)

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

end
