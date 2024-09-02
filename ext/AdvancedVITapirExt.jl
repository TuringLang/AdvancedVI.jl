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

function AdvancedVI.value_and_gradient!(
    ::ADTypes.AutoTapir,
    f,
    x::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
)
    rule = Tapir.build_rrule(f, x)
    y, g = Tapir.value_and_gradient!!(rule, f, x)
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, last(g))
    return out
end

function AdvancedVI.value_and_gradient!(
    ::ADTypes.AutoTapir,
    f,
    x::AbstractVector{<:Real},
    aux,
    out::DiffResults.MutableDiffResult,
)
    rule = Tapir.build_rrule(f, x, aux)
    y, g = Tapir.value_and_gradient!!(rule, f, x, aux)
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, g[2])
    return out
end

end
