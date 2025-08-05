module AdvancedVIReverseDiffExt

using AdvancedVI
using LogDensityProblems
using ReverseDiff

ReverseDiff.@grad_from_chainrules LogDensityProblems.logdensity(
    prob::AdvancedVI.MixedADLogDensityProblem, x::ReverseDiff.TrackedArray
)

ReverseDiff.@grad_from_chainrules LogDensityProblems.logdensity(
    prob::AdvancedVI.MixedADLogDensityProblem, x::AbstractArray{<:ReverseDiff.TrackedReal}
)

end
