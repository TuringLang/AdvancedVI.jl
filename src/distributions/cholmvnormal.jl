## Traditional Cholesky representation where Γ is Lower Triangular
struct CholMvNormal{T, Tμ<:AbstractVector{T}, TΓ<:LowerTriangular{T}} <: AbstractLowRankMvNormal{T}
    dim::Int
    μ::Tμ
    Γ::TΓ
    function CholMvNormal(μ::AbstractVector{T}, Γ::LowerTriangular{T}) where {T}
        length(μ) == size(Γ, 1) || throw(DimensionMismatch("μ and Γ have incompatible sizes")) 
        new{T,typeof(μ),typeof(Γ)}(length(μ), μ, Γ)
    end
    function CholMvNormal(
        dim::Int,
        μ::Tμ,
        Γ::TΓ
    ) where {
        T,
        Tμ<:AbstractVector{T},
        TΓ<:LowerTriangular{T},
    }
        length(μ) == size(Γ, 1) || throw(DimensionMismatch("μ and Γ have incompatible sizes")) 
        new{T,Tμ,TΓ}(dim, μ, Γ)
    end
end

Distributions.cov(d::CholMvNormal) = XXt(d.Γ)
Distributions.entropy(d::CholMvNormal) = logdet(d.Γ) + 0.5 * length(d) * log2π 


function reparametrize!(x, q::CholMvNormal, z)
    x .= q.μ .+ q.Γ * z
end


function to_vec(q::CholMvNormal)
    vcat(q.μ, vec(q.Γ))
end

function to_dist(q::CholMvNormal, θ::AbstractVector)
    CholMvNormal(θ[1:length(q)], LowerTriangular(reshape(θ[(length(q)+1):end], length(q), length(q))))
end