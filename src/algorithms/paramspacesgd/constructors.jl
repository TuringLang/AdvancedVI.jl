
"""
    BBVIRepGrad(problem, adtype; entropy, optimizer, n_samples, operator)

Black-box variational inference with the reparameterization gradient with stochastic gradient descent.

# Arguments
- `problem`: Target problem.
- `adtype::ADTypes.AbstractADType`: Automatic differentiation backend. 

# Keyword Arguments
- `entropy`: Entropy gradient estimator to be used. Must be one of `ClosedFormEntropy`, `StickingTheLandingEntropy`, `MonteCarloEntropy`. (default: `ClosedFormEntropy()`)
- `optimizer::Optimisers.AbstractRule`: Optimization algorithm to be used. (default: `DoWG()`)
- `n_samples::Int`: Number of Monte Carlo samples to be used for estimating each gradient. (default: `1`)
- `operator::Union{<:IdentityOperator, <:ClipScale}`: Operator to be applied after each gradient descent step. (default: `ClipScale()`)
- `averager::AbstractAverager`: Parameter averaging strategy. 

"""
function BBVIRepGrad(
    problem,
    adtype::ADTypes.AbstractADType;
    entropy::Union{<:ClosedFormEntropy,<:StickingTheLandingEntropy,<:MonteCarloEntropy}=ClosedFormEntropy(),
    optimizer::Optimisers.AbstractRule=DoWG(),
    n_samples::Int=1,
    averager::AbstractAverager=PolynomialAveraging(),
    operator::Union{<:IdentityOperator,<:ClipScale}=ClipScale(),
)
    objective = RepGradELBO(n_samples; entropy=entropy)
    return ParamSpaceSGD(problem, objective, adtype, optimizer, averager, operator)
end

"""
    BBVIRepGradProxLocScale(problem, adtype; entropy, optimizer, n_samples, operator)

Black-box variational inference with the reparameterization gradient and stochastic proximal gradient descent on location scale families.

This algorithm only supports subtypes of `MvLocationScale`.
Also, since the stochastic proximal gradient descent does not use the entropy of the gradient, the entropy estimator to be used must have a zero-mean gradient.
Thus, only the entropy estimators with a "ZeroGradient" suffix are allowed.

# Arguments
- `problem`: Target problem.
- `adtype`: Automatic differentiation backend. 

# Keyword Arguments
- `entropy_zerograd`: Estimator of the entropy with a zero-mean gradient to be used. Must be one of `ClosedFormEntropyZeroGrad`, `StickingTheLandingEntropyZeroGrad`. (default: `ClosedFormEntropyZeroGrad()`)
- `optimizer::Optimisers.AbstractRule`: Optimization algorithm to be used. Only `DoG`, `DoWG` and `Optimisers.Descent` are supported. (default: `DoWG()`)
- `n_samples::Int`: Number of Monte Carlo samples to be used for estimating each gradient.
- `averager::AbstractAverager`: Parameter averaging strategy. (default: `PolynomialAveraging()`)

"""
function BBVIRepGradProxLocScale(
    problem,
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
    return ParamSpaceSGD(problem, objective, adtype, optimizer, averager, operator)
end
