module AdvancedVIMooncakeExt

using AdvancedVI
using Base: IEEEFloat
using LogDensityProblems
using Mooncake

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{
    typeof(LogDensityProblems.logdensity),
    AdvancedVI.MixedADLogDensityProblem,
    Array{<:IEEEFloat, 1},
}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{
    typeof(LogDensityProblems.logdensity),
    AdvancedVI.MixedADLogDensityProblem,
    SubArray{<:IEEEFloat,1},
}

end
