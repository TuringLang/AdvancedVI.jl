module AdvancedVIReverseDiffExt

using AdvancedVI
using LogDensityProblems
using ReverseDiff

ReverseDiff.@grad_from_chainrules LogDensityProblems.logdensity(
    prob::AdvancedVI.MixedADLogDensityProblem, x::ReverseDiff.TrackedArray
)

end
