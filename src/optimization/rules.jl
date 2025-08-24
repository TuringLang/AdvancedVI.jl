
"""
    DoWG(alpha)

Distance over weighted gradient (DoWG[^KMJ2024]) optimizer.
Its only parameter is the guess for the distance between the optimum and the initialization `alpha`, which shouldn't need much tuning.

DoWG is a minor modification to DoG so that the step sizes are always provably larger than DoG.
Similarly to DoG, it works by starting from a AdaGrad-like update rule with a small step size, but then automatically increases the step size ("warming up") to be as large as possible.
If `alpha` is too large, the optimzier can initially diverge, while if it is too small, the warm up period can be too long.
Depending on the problem, DoWG can be too aggressive and result in unstable behavior.
If this is suspected, try using DoG instead.

# Parameters
- `alpha`: Scaling factor for initial guess (`repsilon` in the original paper) of the Euclidean distance between the initial point and the optimum. For the initial parameter `x0`, `repsilon` is calculated as `repsilon = alpha*(1 + norm(x0)`. (default value: `1e-6`)
"""
Optimisers.@def struct DoWG <: Optimisers.AbstractRule
    alpha = 1e-6
end

Optimisers.init(o::DoWG, x::AbstractArray{T}) where {T} = (copy(x), zero(T), T(o.alpha)*(1 + norm(x)))

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
    DoG(alpha)

Distance over gradient (DoG[^IHC2023]) optimizer.
Its only parameter is the guess for the distance between the optimum and the initialization `alpha`, which shouldn't need much tuning.

DoG works by starting from a AdaGrad-like update rule with a small step size, but then automatically increases the step size ("warming up") to be as large as possible.
If `alpha` is too large, the optimzier can initially diverge, while if it is too small, the warm up period can be too long.

# Parameters
- `alpha`: Scaling factor for initial guess (`repsilon` in the original paper) of the Euclidean distance between the initial point and the optimum. For the initial parameter `x0`, `repsilon` is calculated as `repsilon = alpha*(1 + norm(x0)`. (default value: `1e-6`)
"""
Optimisers.@def struct DoG <: Optimisers.AbstractRule
    alpha = 1e-6
end

Optimisers.init(o::DoG, x::AbstractArray{T}) where {T} = (copy(x), zero(T), T(o.alpha)*(1 + norm(x)))

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
Its only parameter is the maximum change per parameter `alpha`, which shouldn't need much tuning.

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
