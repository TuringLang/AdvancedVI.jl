module AdvancedVIFluxExt

if isdefined(Base, :get_extension)
    using AdvancedVI: AdvancedVI
    using Flux: Flux
else
    using ..AdvancedVI: AdvancedVI
    using ..Flux: Flux
end

AdvancedVI.apply!(o::Flux.Optimise.AbstractOptimizer, x, Δ) = Flux.Optimise.apply!(o, x, Δ)

end
