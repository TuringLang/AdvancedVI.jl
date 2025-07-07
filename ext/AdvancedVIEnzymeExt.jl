module AdvancedVIEnzymeExt

using AdvancedVI
using LogDensityProblems
using Enzyme

Enzyme.@import_rrule(typeof(LogDensityProblems.logdensity), AdvancedVI.MixedADLogDensityProblem, AbstractVector)

end 
