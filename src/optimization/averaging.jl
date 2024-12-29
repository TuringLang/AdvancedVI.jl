
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

Polynomial averaging rule proposed Shamir and Zhang[^SZ2013].
At iteration `t`, the parameter average \$ \\bar{\\lambda}_t \$ according to the polynomial averaging rule is given as
```math
    \\bar{\\lambda}_t = (1 - w_t) \\bar{\\lambda}_{t-1} + w_t \\lambda_t \\, ,
```
where the averaging weight is 
```math
    w_t = \\frac{\\eta + 1}{t + \\eta} \\, .
```
Higher `eta` (\$\\eta\$) down-weights earlier iterations.
When \$\\eta=0\$, this is equivalent to uniformly averaging the iterates in an online fashion.
The DoG paper[^IHC2023] suggests \$\\eta=8\$.

# Parameters
- `eta`: Regularization term. (default: `8`)

[^SZ2013]: Shamir, O., & Zhang, T. (2013). Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. In International conference on machine learning (pp. 71-79). PMLR.
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
