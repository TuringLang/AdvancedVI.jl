module AdvancedVIMooncakeExt

using AdvancedVI
using Base: IEEEFloat
using LogDensityProblems
using Mooncake

# Array types explicitly tested by Mooncake
const SupportedArray{P,N} = Union{Array{P,N},AbstractGPUArray{P,N}}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{
    typeof(LogDensityProblems.logdensity),
    AdvancedVI.MixedADLogDensityProblem,
    SupportedArray{<:IEEEFloat,1},
}

end
