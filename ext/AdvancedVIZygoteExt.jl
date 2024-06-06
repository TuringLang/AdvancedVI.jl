
module AdvancedVIZygoteExt

if isdefined(Base, :get_extension)
    using AdvancedVI
    using AdvancedVI: ADTypes
    using Zygote
else
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes
    using ..Zygote
end

function AdvancedVI.value_and_gradient(ad::ADTypes.AutoZygote, f, θ)
    y, back = Zygote.pullback(f, θ)
    ∇θ = back(one(y))
    only(∇θ), y
end

end
