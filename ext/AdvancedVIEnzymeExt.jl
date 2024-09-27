module AdvancedVIEnzymeExt

if isdefined(Base, :get_extension)
    using Enzyme
    using AdvancedVI
    using AdvancedVI: ADTypes
else
    using ..Enzyme
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes
end

function AdvancedVI.restructure_ad_forward(::ADTypes.AutoEnzyme, restructure, params)
    return restructure(params)::typeof(restructure.model)
end

end
