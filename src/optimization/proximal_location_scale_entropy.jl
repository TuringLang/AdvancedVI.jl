
"""
    ProximalLocationScaleEntropy()

Proximal operator for the entropy of a location-scale distribution, which is defined as
```math
    \\mathrm{prox}(\\lambda) = \\argmin_{\\lambda^{\\prime}} - \\mathbb{H}(q_{\\lambda^{\\prime}}) + \\frac{1}{2 \\gamma_t} \\left\\lVert \\lambda - \\lambda^{\\prime} \\right\\rVert ,
```
where \$\\gamma_t\$ is the stepsize the optimizer used with the proximal operator.
This assumes the variational family is `<:VILocationScale` and the optimizer is one of the following:
- `DoG`
- `DoWG`
- `Descent`

For ELBO maximization, since this proximal operator handles the entropy, the gradient estimator for the ELBO must ingore the entropy term.
That is, the `entropy` keyword argument of `RepGradELBO` muse be one of the following:
- `ClosedFormEntropyZeroGradient`
- `StickingTheLandingEntropyZeroGradient`
"""
struct ProximalLocationScaleEntropy <: AbstractOperator end

function apply(::ProximalLocationScaleEntropy, family, state, params, restructure)
    return error("`ProximalLocationScaleEntropy` only supports `<:MvLocationScale`.")
end

function stepsize_from_optimizer_state(rule::Optimisers.AbstractRule, state)
    return error(
        "`ProximalLocationScaleEntropy` does not support optimization rule $(typeof(rule))."
    )
end

stepsize_from_optimizer_state(rule::Descent, ::Any) = rule.eta

function stepsize_from_optimizer_state(::DoG, state)
    _, v, r = state
    return r / sqrt(v)
end

function stepsize_from_optimizer_state(::DoWG, state)
    _, v, r = state
    return r * r / sqrt(v)
end

function apply(
    ::ProximalLocationScaleEntropy,
    ::Type{<:MvLocationScale},
    leaf::Optimisers.Leaf{<:Union{<:DoG,<:DoWG,<:Descent},S},
    params,
    restructure,
) where {S}
    q = restructure(params)

    stepsize = stepsize_from_optimizer_state(leaf.rule, leaf.state)
    diag_idx = diagind(q.scale)
    scale_diag = q.scale[diag_idx]
    @. q.scale[diag_idx] =
        scale_diag + 1 / 2 * (sqrt(scale_diag^2 + 4 * stepsize) - scale_diag)

    params, _ = Optimisers.destructure(q)

    return params
end
