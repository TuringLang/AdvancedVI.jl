
"""
    DoWG(repsilon)

Distance over weighted gradient (DoWG[^KMJ2024]) optimizer.
It's only parameter is the initial guess of the Euclidean distance to the optimum repsilon.

# Parameters
- `repsilon`: Initial guess of the Euclidean distance between the initial point and
            the optimum. (default value: `1e-6`)

[^KMJ2024]: Khaled, A., Mishchenko, K., & Jin, C. (2023). Dowg unleashed: An efficient universal parameter-free gradient descent method. Advances in Neural Information Processing Systems, 36, 6748-6769.
"""
Optimisers.@def struct DoWG <: Optimisers.AbstractRule
    repsilon = 1e-6
end

Optimisers.init(o::DoWG, x::AbstractArray{T}) where {T} = (copy(x), zero(T), T(o.repsilon))

function Optimisers.apply!(::DoWG, state, x::AbstractArray{T}, dx) where {T}
    x0, v, r = state

    r = max(sqrt(sum(abs2, x - x0)), r)
    r2 = r * r
    v = v + r2 * sum(abs2, dx)
    η = r2 / sqrt(v)
    dx′ = Optimisers.@lazy dx * η
    return (x0, v, r), dx′
end

"""
    DoG(repsilon)

Distance over gradient (DoG[^IHC2023]) optimizer.
It's only parameter is the initial guess of the Euclidean distance to the optimum repsilon.
The original paper recommends \$ 10^{-4} ( 1 + \\lVert \\lambda_0 \\rVert ) \$, but the default value is \$ 10^{-6} \$.

# Parameters
- `repsilon`: Initial guess of the Euclidean distance between the initial point and the optimum. (default value: `1e-6`)

[^IHC2023]: Ivgi, M., Hinder, O., & Carmon, Y. (2023). Dog is sgd's best friend: A parameter-free dynamic step size schedule. In International Conference on Machine Learning (pp. 14465-14499). PMLR.
"""
Optimisers.@def struct DoG <: Optimisers.AbstractRule
    repsilon = 1e-6
end

Optimisers.init(o::DoG, x::AbstractArray{T}) where {T} = (copy(x), zero(T), T(o.repsilon))

function Optimisers.apply!(::DoG, state, x::AbstractArray{T}, dx) where {T}
    x0, v, r = state

    r = max(sqrt(sum(abs2, x - x0)), r)
    v = v + sum(abs2, dx)
    η = r / sqrt(v)
    dx′ = Optimisers.@lazy dx * η
    return (x0, v, r), dx′
end

"""
    COCOB(alpha)

Continuous Coin Betting (COCOB[^OT2017]) optimizer.
We use the "COCOB-Backprop" variant, which is closer to the Adam optimizer.
It's only parameter is the maximum change per parameter α, which shouldn't need much tuning.

# Parameters
- `alpha`: Scaling parameter. (default value: `100`)

[^OT2017]: Orabona, F., & Tommasi, T. (2017). Training deep networks without learning rates through coin betting. Advances in Neural Information Processing Systems, 30.
"""
Optimisers.@def struct COCOB <: Optimisers.AbstractRule
    alpha = 100
end

function Optimisers.init(::COCOB, x::AbstractArray{T}) where {T}
    return (zero(x), zero(x), zero(x), zero(x), copy(x))
end

function Optimisers.apply!(o::COCOB, state, x::AbstractArray{T}, dx) where {T}
    α = T(o.alpha)
    L, G, R, θ, x1 = state

    Optimisers.@.. L = max(L, abs(dx))
    Optimisers.@.. G = G + abs(dx)
    Optimisers.@.. R = max(R + (x - x1) * -dx, 0)
    Optimisers.@.. θ = θ + -dx
    dx′ = Optimisers.@lazy -(x1 - x) - (θ / (L * max(G + L, α * L)) * (L + R))
    return (L, G, R, θ, x1), dx′
end
