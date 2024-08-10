
"""
    NoAveraging()

No averaging. This returns the last-iterate of the optimization rule.
"""
struct NoAveraging <: AbstractAverager end

init(::NoAveraging, x) = x

apply(::NoAveraging, state, x) = x

value(::NoAveraging, state) = state

"""
    PolynomialAveraging(eta)

Polynomial averaging rule proposed [Shamir and Zhang](https://proceedings.mlr.press/v28/shamir13.html)
At iteration `t`, the polynomial averaging rule is given as
```julia
    params_avg = (1 - w) * params_avg + w * x
```
where the averaging weight is 
```julia
    w = (eta + 1) / (t + eta)
```
Higher `eta` down-weights earlier iterations.
The [DoG paper](https://arxiv.org/abs/2302.12022) suggests `eta = 8`.

# Parameters
- eta: Regularization term. (default: `8`)
"""
struct PolynomialAveraging{F} <: AbstractAverager
    eta::F
end

PolynomialAveraging() = PolynomialAveraging(8)

init(::PolynomialAveraging, x) = (x, 1)

function apply(avg::PolynomialAveraging, state, x::AbstractVector{T}) where {T}
    eta = T(avg.eta)
    x_bar, t = state

    w = (eta + 1) / (t + eta)
    x_bar = (1 - w) * x_bar + w * x
    return (x_bar, t + 1)
end

value(::PolynomialAveraging, state) = first(state)
