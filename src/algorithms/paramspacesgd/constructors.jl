
"""
    KLMinRepGradDescent(adtype; entropy, optimizer, n_samples, averager, operator)

KL divergence minimization by running stochastic gradient descent with the reparameterization gradient in the Euclidean space of variational parameters.

# Arguments
- `adtype::ADTypes.AbstractADType`: Automatic differentiation backend. 

# Keyword Arguments
- `entropy`: Entropy gradient estimator to be used. Must be one of `ClosedFormEntropy`, `StickingTheLandingEntropy`, `MonteCarloEntropy`. (default: `ClosedFormEntropy()`)
- `optimizer::Optimisers.AbstractRule`: Optimization algorithm to be used. (default: `DoWG()`)
- `n_samples::Int`: Number of Monte Carlo samples to be used for estimating each gradient. (default: `1`)
- `averager::AbstractAverager`: Parameter averaging strategy. 
- `operator::Union{<:IdentityOperator, <:ClipScale}`: Operator to be applied after each gradient descent step. (default: `ClipScale()`)

# Requirements
- The trainable parameters in the variational approximation are expected to be extractable through `Optimisers.destructure`. This requires the variational approximation to be marked as a functor through `Functors.@functor`.
- The variational approximation ``q_{\\lambda}`` implements `rand`.
- The target distribution and the variational approximation have the same support.
- The target `LogDensityProblems.logdensity(prob, x)` must be differentiable with respect to `x` by the selected AD backend.
- Additonal requirements on `q` may apply depending on the choice of `entropy`.
"""
function KLMinRepGradDescent(
    adtype::ADTypes.AbstractADType;
    entropy::Union{<:ClosedFormEntropy,<:StickingTheLandingEntropy,<:MonteCarloEntropy}=ClosedFormEntropy(),
    optimizer::Optimisers.AbstractRule=DoWG(),
    n_samples::Int=1,
    averager::AbstractAverager=PolynomialAveraging(),
    operator::Union{<:IdentityOperator,<:ClipScale}=ClipScale(),
)
    objective = RepGradELBO(n_samples; entropy=entropy)
    return ParamSpaceSGD(objective, adtype, optimizer, averager, operator)
end

const ADVI = KLMinRepGradDescent

"""
    KLMinRepGradProxDescent(adtype; entropy_zerograd, optimizer, n_samples, averager)

KL divergence minimization by running stochastic proximal gradient descent with the reparameterization gradient in the Euclidean space of variational parameters of a location-scale family.

This algorithm only supports subtypes of `MvLocationScale`.
Also, since the stochastic proximal gradient descent does not use the entropy of the gradient, the entropy estimator to be used must have a zero-mean gradient.
Thus, only the entropy estimators with a "ZeroGradient" suffix are allowed.

# Arguments
- `adtype`: Automatic differentiation backend. 

# Keyword Arguments
- `entropy_zerograd`: Estimator of the entropy with a zero-mean gradient to be used. Must be one of `ClosedFormEntropyZeroGrad`, `StickingTheLandingEntropyZeroGrad`. (default: `ClosedFormEntropyZeroGrad()`)
- `optimizer::Optimisers.AbstractRule`: Optimization algorithm to be used. Only `DoG`, `DoWG` and `Optimisers.Descent` are supported. (default: `DoWG()`)
- `n_samples::Int`: Number of Monte Carlo samples to be used for estimating each gradient.
- `averager::AbstractAverager`: Parameter averaging strategy. (default: `PolynomialAveraging()`)

# Requirements
- The variational family is `MvLocationScale`.
- The target distribution and the variational approximation have the same support.
- The target `LogDensityProblems.logdensity(prob, x)` must be differentiable with respect to `x` by the selected AD backend.
- Additonal requirements on `q` may apply depending on the choice of `entropy_zerograd`.
"""
function KLMinRepGradProxDescent(
    adtype::ADTypes.AbstractADType;
    entropy_zerograd::Union{
        <:ClosedFormEntropyZeroGradient,<:StickingTheLandingEntropyZeroGradient
    }=ClosedFormEntropyZeroGradient(),
    optimizer::Union{<:Descent,<:DoG,<:DoWG}=DoWG(),
    n_samples::Int=1,
    averager::AbstractAverager=PolynomialAveraging(),
)
    objective = RepGradELBO(n_samples; entropy=entropy_zerograd)
    operator = ProximalLocationScaleEntropy()
    return ParamSpaceSGD(objective, adtype, optimizer, averager, operator)
end

"""
    KLMinScoreGradDescent(adtype; optimizer, n_samples, averager, operator)

KL divergence minimization by running stochastic gradient descent with the score gradient in the Euclidean space of variational parameters.

# Arguments
- `adtype`: Automatic differentiation backend. 

# Keyword Arguments
- `optimizer::Optimisers.AbstractRule`: Optimization algorithm to be used. Only `DoG`, `DoWG` and `Optimisers.Descent` are supported. (default: `DoWG()`)
- `n_samples::Int`: Number of Monte Carlo samples to be used for estimating each gradient.
- `averager::AbstractAverager`: Parameter averaging strategy. (default: `PolynomialAveraging()`)
- `operator::Union{<:IdentityOperator, <:ClipScale}`: Operator to be applied after each gradient descent step. (default: `IdentityOperator()`)

# Requirements
- The trainable parameters in the variational approximation are expected to be extractable through `Optimisers.destructure`. This requires the variational approximation to be marked as a functor through `Functors.@functor`.
- The variational approximation ``q_{\\lambda}`` implements `rand`.
- The variational approximation ``q_{\\lambda}`` implements `logpdf(q, x)`, which should also be differentiable with respect to `x`.
- The target distribution and the variational approximation have the same support.
"""
function KLMinScoreGradDescent(
    adtype::ADTypes.AbstractADType;
    optimizer::Union{<:Descent,<:DoG,<:DoWG}=DoWG(),
    n_samples::Int=1,
    averager::AbstractAverager=PolynomialAveraging(),
    operator::Union{<:IdentityOperator,<:ClipScale}=IdentityOperator(),
)
    objective = ScoreGradELBO(n_samples)
    return ParamSpaceSGD(objective, adtype, optimizer, averager, operator)
end

const BBVI = KLMinScoreGradDescent
