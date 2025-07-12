
"""
    MixedADLogDensityProblem(problem)

A `LogDensityProblem` wrapper for mixing AD frameworks.
 Whenever the outer AD framework attempts to differentiate through `logdensity(problem)`
the pullback calls `logdensity_and_gradient`, which invokes the inner AD framework.
"""
struct MixedADLogDensityProblem{Problem}
    problem::Problem
end

function LogDensityProblems.dimension(mixedad_prob::MixedADLogDensityProblem)
    return LogDensityProblems.dimension(mixedad_prob.problem)
end

function LogDensityProblems.logdensity(
    mixedad_prob::MixedADLogDensityProblem, x::AbstractArray
)
    return LogDensityProblems.logdensity(mixedad_prob.problem, x)
end

function LogDensityProblems.logdensity_and_gradient(
    mixedad_prob::MixedADLogDensityProblem, x::AbstractArray
)
    return LogDensityProblems.logdensity_and_gradient(mixedad_prob.problem, x)
end

function LogDensityProblems.logdensity_gradient_and_hessian(
    mixedad_prob::MixedADLogDensityProblem, x::AbstractArray
)
    return LogDensityProblems.logdensity_and_gradient(mixedad_prob.problem, x)
end

function LogDensityProblems.capabilities(mixedad_prob::MixedADLogDensityProblem)
    return LogDensityProblems.capabilities(mixedad_prob.problem)
end

function ChainRulesCore.rrule(
    ::typeof(LogDensityProblems.logdensity),
    mixedad_prob::MixedADLogDensityProblem,
    x::AbstractArray,
)
    ℓπ, ∇ℓπ = LogDensityProblems.logdensity_and_gradient(mixedad_prob.problem, x)
    function logdensity_pullback(∂y)
        ∂x = @thunk(∂y' * ∇ℓπ)
        return NoTangent(), NoTangent(), ∂x
    end
    return ℓπ, logdensity_pullback
end
