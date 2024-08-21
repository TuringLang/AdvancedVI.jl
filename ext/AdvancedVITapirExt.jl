
module AdvancedVITapirExt

if isdefined(Base, :get_extension)
    using AdvancedVI
    using AdvancedVI: ADTypes, DiffResults
    using Tapir
else
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes, DiffResults
    using ..Tapir
end

AdvancedVI.init_adbackend(::ADTypes.AutoTapir, f, x) = Tapir.build_rrule(f, x)

function AdvancedVI.value_and_gradient!(
    ::ADTypes.AutoTapir,
    st_ad,
    f,
    x::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
)
    rule = st_ad
    y, g = Tapir.value_and_gradient!!(rule, f, x)
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, last(g))
    return out
end

AdvancedVI.init_adbackend(::ADTypes.AutoTapir, f, x, aux) = Tapir.build_rrule(f, x, aux)

function AdvancedVI.value_and_gradient!(
    ::ADTypes.AutoTapir,
    st_ad,
    f,
    x::AbstractVector{<:Real},
    aux,
    out::DiffResults.MutableDiffResult,
)
    rule = st_ad
    y, g = Tapir.value_and_gradient!!(rule, f, x, aux)
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, g[2])
    return out
end

end
