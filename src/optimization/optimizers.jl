
"""
    DoWG(repsilon = 1e-8)

[DoWG](https://arxiv.org/abs/2305.16284) optimizer. It's only parameter is the 
initial guess of the Euclidean distance to the optimum repsilon.
The [DoG](https://arxiv.org/abs/2302.12022) paper recommends 1e-4*(1 + norm(x0)).

# Parameters
- repsilon: Initial guess of the Euclidean distance between the initial point and
            the optimum.
"""
Optimisers.@def struct DoWG <: Optimisers.AbstractRule
    repsilon = 1e-8
end

Optimisers.init(o::DoWG, x::AbstractArray{T}) where T = (copy(x), zero(T), T(o.repsilon))

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
    DoG(repsilon = 1e-8)

[DoG](https://arxiv.org/abs/2305.16284) optimizer. It's only parameter is the 
initial guess of the Euclidean distance to the optimum repsilon.
The [DoG](https://arxiv.org/abs/2302.12022) paper recommends 1e-4*(1 + norm(x0)).

# Parameters
- repsilon: Initial guess of the Euclidean distance between the initial point and
            the optimum.
"""

Optimisers.@def struct DoG <: Optimisers.AbstractRule
    repsilon = 1e-8
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
    COCOB(α = 100)

[Continuous Coin Betting](https://arxiv.org/abs/1705.07795) optimizer.
It's only parameter is the maximum change per parameter α, which shouldn't need much tuning.
The paper suggests α = 100 as a generally default value.

# Parameters
- alpha (α): Scaling parameter.
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
