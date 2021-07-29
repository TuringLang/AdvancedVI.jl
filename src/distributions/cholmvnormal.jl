## Traditional Cholesky representation where Γ is Lower Triangular
struct CholMvNormal{T,Tμ<:AbstractVector{T},TΓ<:LowerTriangular{T}} <:
       AbstractPosteriorMvNormal{T}
    dim::Int
    μ::Tμ
    Γ::TΓ
    function CholMvNormal(μ::AbstractVector{T}, Γ::LowerTriangular{T}) where {T}
        length(μ) == size(Γ, 1) ||
            throw(DimensionMismatch("μ and Γ have incompatible sizes"))
        return new{T,typeof(μ),typeof(Γ)}(length(μ), μ, Γ)
    end
    function CholMvNormal(
        dim::Int, μ::Tμ, Γ::TΓ
    ) where {T,Tμ<:AbstractVector{T},TΓ<:LowerTriangular{T}}
        length(μ) == size(Γ, 1) ||
            throw(DimensionMismatch("μ and Γ have incompatible sizes"))
        return new{T,Tμ,TΓ}(dim, μ, Γ)
    end
end

Distributions.cov(d::CholMvNormal) = d.Γ * d.Γ'
Distributions.logdetcov(d::CholMvNormal) = 2 * logdet(d.Γ)

@functor CholMvNormal

function reparametrize!(x, q::CholMvNormal, z)
    return x .= q.μ .+ q.Γ * z
end

function to_vec(q::CholMvNormal)
    return vcat(q.μ, vec(q.Γ))
end

function to_dist(q::CholMvNormal, θ::AbstractVector)
    return CholMvNormal(
        θ[1:length(q)],
        LowerTriangular(reshape(θ[(length(q) + 1):end], length(q), length(q))),
    )
end
