
struct DiagMvNormal{T,Tμ<:AbstractVector{T},TΓ<:AbstractVector{T}} <:
       AbstractPosteriorMvNormal{T}
    dim::Int
    μ::Tμ
    Γ::TΓ
    function DiagMvNormal(μ::AbstractVector{T}, Γ::AbstractVector{T}) where {T}
        return new{T,typeof(μ),typeof(Γ)}(length(μ), μ, Γ)
    end
    function DiagMvNormal(
        dim::Int, μ::Tμ, Γ::TΓ
    ) where {T,Tμ<:AbstractVector{T},TΓ<:AbstractVector{T}}
        return new{T,Tμ,TΓ}(dim, μ, Γ)
    end
end

function Distributions._rand!(
    rng::AbstractRNG, d::DiagMvNormal{T}, x::AbstractVector
) where {T}
    nDim = length(x)
    nDim == d.dim || error("Wrong dimensions")
    return x .= d.μ + d.Γ .* randn(rng, T, nDim)
end

function Distributions._rand!(
    rng::AbstractRNG, d::DiagMvNormal{T}, x::AbstractMatrix
) where {T}
    nDim, nPoints = size(x)
    nDim == d.dim || error("Wrong dimensions")
    return x .= d.μ .+ d.Γ .* randn(rng, T, nDim, nPoints)
end

Distributions.cov(d::DiagMvNormal) = Diagonal(abs2.(d.Γ))

@functor DiagMvNormal

function reparametrize!(x, q::DiagMvNormal, z)
    return x .= q.μ .+ q.Γ .* z
end

function to_vec(q::DiagMvNormal)
    return vcat(q.μ, q.Γ)
end

function to_dist(q::DiagMvNormal, θ::AbstractVector)
    return DiagMvNormal(θ[1:length(q)], θ[(length(q) + 1):end])
end
