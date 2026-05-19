module AdvancedVIMooncakeExt

using ADTypes: AutoMooncake, AutoMooncakeForward
using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Prepared, VectorEvaluator
using AdvancedVI
using LogDensityProblems
using Mooncake

Mooncake.@is_primitive(
    Mooncake.MinimalCtx,
    Tuple{
        typeof(LogDensityProblems.logdensity),
        AdvancedVI.MixedADLogDensityProblem,
        AbstractArray{<:Base.IEEEFloat},
    }
)

Mooncake.tangent_type(::Type{<:AdvancedVI.MixedADLogDensityProblem}) = Mooncake.NoTangent

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(LogDensityProblems.logdensity)},
    mixedad_prob::Mooncake.CoDual{<:AdvancedVI.MixedADLogDensityProblem},
    x::Mooncake.CoDual{<:AbstractArray{<:Base.IEEEFloat}},
)
    x, dx = Mooncake.arrayify(x)
    ℓπ, ∇ℓπ = LogDensityProblems.logdensity_and_gradient(
        Mooncake.primal(mixedad_prob).problem, x
    )
    function logdensity_pb(∂y)
        view(dx, 1:length(x)) .+= ∂y' * ∇ℓπ
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return Mooncake.zero_fcodual(ℓπ), logdensity_pb
end

const _MooncakePrepared = Prepared{<:AutoMooncake,<:VectorEvaluator}

# Order-1 LDP methods are inherited from the AbstractADType fallback in
# AdvancedVI core.
function LogDensityProblems.capabilities(::Type{<:_MooncakePrepared})
    LogDensityProblems.LogDensityOrder{2}()
end

# Mooncake forward-over-reverse Hessian: a fresh forward-mode Jacobian cache
# is built per call, so this is fine for occasional use but costly inside a
# tight per-sample loop.
function LogDensityProblems.logdensity_gradient_and_hessian(p::_MooncakePrepared, x)
    val, grad = LogDensityProblems.logdensity_and_gradient(p, x)
    grad_fn = y -> LogDensityProblems.logdensity_and_gradient(p, y)[2]
    fwd_jac = AbstractPPL.prepare(AutoMooncakeForward(), grad_fn, x)
    _, H = AbstractPPL.value_and_jacobian!!(fwd_jac, x)
    return val, grad, copy(H)
end

end
