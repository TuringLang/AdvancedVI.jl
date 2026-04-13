module AdvancedVIEnzymeExt

using ADTypes: ADTypes
using AdvancedVI
using Enzyme: Enzyme, EnzymeRules
using LogDensityProblems
using Statistics: Statistics

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(LogDensityProblems.logdensity)},
    ::Type{<:Union{Enzyme.Active,Enzyme.Duplicated,Enzyme.DuplicatedNoNeed}},
    prob::Enzyme.Const{<:AdvancedVI.MixedADLogDensityProblem},
    x::Enzyme.Duplicated{<:Vector},
)
    ℓπ, ∇ℓπ = LogDensityProblems.logdensity_and_gradient(prob.val.problem, x.val)
    primal = EnzymeRules.needs_primal(config) ? ℓπ : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(ℓπ) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, copy(∇ℓπ))
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(LogDensityProblems.logdensity)},
    dret::Enzyme.Active,
    ∇ℓπ,
    ::Enzyme.Const{<:AdvancedVI.MixedADLogDensityProblem},
    x::Enzyme.Duplicated{<:Vector},
)
    @inbounds for i in eachindex(x.dval, ∇ℓπ)
        x.dval[i] += dret.val * ∇ℓπ[i]
    end
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(Statistics.mean)},
    ::Type{<:Union{Enzyme.Active,Enzyme.Duplicated,Enzyme.DuplicatedNoNeed}},
    x::Enzyme.Duplicated{<:AbstractVector{<:Real}},
)
    total = zero(eltype(x.val))
    @inbounds for xi in x.val
        total += xi
    end
    n = length(x.val)
    mean_x = total / n
    primal = EnzymeRules.needs_primal(config) ? mean_x : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(mean_x) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, n)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(Statistics.mean)},
    dret::Enzyme.Active,
    n::Int,
    x::Enzyme.Duplicated{<:AbstractVector{<:Real}},
)
    scale = dret.val / n
    @inbounds for i in eachindex(x.dval)
        x.dval[i] += scale
    end
    return (nothing,)
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(Statistics.mean)},
    ::Type{<:Union{Enzyme.Active,Enzyme.Duplicated,Enzyme.DuplicatedNoNeed}},
    ::Enzyme.Const{typeof(abs2)},
    x::Enzyme.Duplicated{<:AbstractVector{<:Real}},
)
    # Keep Enzyme out of Statistics.mapreduce_dim for the vector score-gradient path.
    total = zero(eltype(x.val))
    @inbounds for xi in x.val
        total += abs2(xi)
    end
    n = length(x.val)
    mean_abs2_x = total / n
    primal = EnzymeRules.needs_primal(config) ? mean_abs2_x : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(mean_abs2_x) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, n)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(Statistics.mean)},
    dret::Enzyme.Active,
    n::Int,
    ::Enzyme.Const{typeof(abs2)},
    x::Enzyme.Duplicated{<:AbstractVector{<:Real}},
)
    scale = dret.val / n
    @inbounds for i in eachindex(x.dval, x.val)
        x.dval[i] += scale * (x.val[i] + x.val[i])
    end
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(Statistics.mean)},
    ::Type{<:Union{Enzyme.Active,Enzyme.Duplicated,Enzyme.DuplicatedNoNeed}},
    f::Union{
        Enzyme.Const{
            <:Base.Fix1{
                typeof(LogDensityProblems.logdensity),<:AdvancedVI.MixedADLogDensityProblem
            },
        },
        Enzyme.MixedDuplicated{
            <:Base.Fix1{
                typeof(LogDensityProblems.logdensity),<:AdvancedVI.MixedADLogDensityProblem
            },
        },
        Enzyme.Active{
            <:Base.Fix1{
                typeof(LogDensityProblems.logdensity),<:AdvancedVI.MixedADLogDensityProblem
            },
        },
    },
    xs::Enzyme.Duplicated{<:Base.ColumnSlices},
)
    # Bypass Statistics._foldl_impl for this mixed-AD pattern; Enzyme miscompiles the
    # generic eachcol path when custom rules are involved.
    total = zero(eltype(parent(xs.val)))
    n = 0
    @inbounds for x in xs.val
        total += LogDensityProblems.logdensity(f.val.x, x)
        n += 1
    end
    mean_logdensity = total / n
    primal = EnzymeRules.needs_primal(config) ? mean_logdensity : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(mean_logdensity) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, n)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(Statistics.mean)},
    dret::Enzyme.Active,
    n::Int,
    f::Union{
        Enzyme.Const{
            <:Base.Fix1{
                typeof(LogDensityProblems.logdensity),<:AdvancedVI.MixedADLogDensityProblem
            },
        },
        Enzyme.MixedDuplicated{
            <:Base.Fix1{
                typeof(LogDensityProblems.logdensity),<:AdvancedVI.MixedADLogDensityProblem
            },
        },
    },
    xs::Enzyme.Duplicated{<:Base.ColumnSlices},
)
    prob = f.val.x
    scale = dret.val / n
    @inbounds for (dx, x) in zip(xs.dval, xs.val)
        _, ∇ℓπ = LogDensityProblems.logdensity_and_gradient(prob.problem, x)
        for i in eachindex(dx, ∇ℓπ)
            dx[i] += scale * ∇ℓπ[i]
        end
    end
    return (nothing, nothing)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfigWidth{1},
    ::Enzyme.Const{typeof(Statistics.mean)},
    dret::Enzyme.Active,
    n::Int,
    f::Enzyme.Active{
        <:Base.Fix1{
            typeof(LogDensityProblems.logdensity),<:AdvancedVI.MixedADLogDensityProblem
        },
    },
    xs::Enzyme.Duplicated{<:Base.ColumnSlices},
)
    prob = f.val.x
    scale = dret.val / n
    @inbounds for (dx, x) in zip(xs.dval, xs.val)
        _, ∇ℓπ = LogDensityProblems.logdensity_and_gradient(prob.problem, x)
        for i in eachindex(dx, ∇ℓπ)
            dx[i] += scale * ∇ℓπ[i]
        end
    end
    return (Enzyme.make_zero(f.val), nothing)
end

function AdvancedVI._prepare_gradient(
    f,
    adtype::ADTypes.AutoEnzyme{<:Union{Nothing,Enzyme.EnzymeCore.ReverseMode}},
    x::AbstractVector{<:Real},
    aux,
)
    return (; dx=similar(x))
end

function AdvancedVI._prepare_gradient(
    f, adtype::ADTypes.AutoEnzyme, x::AbstractVector{<:Real}, aux
)
    return nothing
end

function AdvancedVI._value_and_gradient(
    f,
    prep::NamedTuple{(:dx,)},
    adtype::ADTypes.AutoEnzyme{M,A},
    x::AbstractVector{<:Real},
    aux,
) where {M,A}
    isempty(x) && return f(x, aux), copy(x)

    dx = prep.dx
    fill!(dx, zero(eltype(dx)))
    f = A === Nothing ? f : A(f)
    mode = if isnothing(adtype.mode)
        Enzyme.ReverseWithPrimal
    else
        Enzyme.EnzymeCore.WithPrimal(adtype.mode)
    end
    _, val = Enzyme.autodiff(
        mode, f, Enzyme.Active, Enzyme.Duplicated(x, dx), Enzyme.Const(aux)
    )
    return val, copy(dx)
end

function AdvancedVI._value_and_gradient(
    f, ::Nothing, adtype::ADTypes.AutoEnzyme{M,A}, x::AbstractVector{<:Real}, aux
) where {M,A}
    isempty(x) && return f(x, aux), copy(x)

    f = A === Nothing ? f : A(f)
    mode = if isnothing(adtype.mode)
        Enzyme.ReverseWithPrimal
    else
        Enzyme.EnzymeCore.WithPrimal(adtype.mode)
    end
    result = Enzyme.gradient(mode, f, x, Enzyme.Const(aux))
    return result.val, copy(first(result.derivs))
end

end
