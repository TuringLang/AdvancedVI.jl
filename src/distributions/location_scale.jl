
function LocationScale(μ::AbstractVector,
                       L::Union{<: AbstractTriangular,
                                <: Diagonal},
                       q₀::ContinuousMultivariateDistribution)
    @assert (length(μ) == size(L,1)) && (length(μ) == size(L,2))
    transformed(q₀, Bijectors.Shift(μ) ∘ Bijectors.Scale(L))
end

function StatsBase.entropy(
    q_trans::MultivariateTransformed{<: ContinuousMultivariateDistribution,
                                     <: Bijectors.ComposedFunction{
                                         <: Bijectors.Shift,
                                         <: Bijectors.Scale}})
    q_base = q_trans.dist
    scale  = q_trans.transform.inner.a
    entropy(q_base) + first(logabsdet(scale))
end

function Distributions.logpdf(
    q_trans::MultivariateTransformed{<: ContinuousMultivariateDistribution,
                                     <: Bijectors.ComposedFunction{
                                         <: Bijectors.Shift,
                                         <: Bijectors.Scale}},
    z::AbstractVector)
    q_base  = q_trans.dist
    reparam = q_trans.transform
    scale   = q_trans.transform.inner.a
    η       = inverse(reparam)(z)
    logpdf(q_base, η) - first(logabsdet(scale))
end

function FullRankGaussian(μ::AbstractVector{T},
                          L::AbstractTriangular{T,S}) where {T <: Real, S}
    @assert (length(μ) == size(L,1)) && (length(μ) == size(L,2))
    n_dims = length(μ)
    q_base = MvNormal(FillArrays.Zeros{T}(n_dims),
                      PDMats.ScalMat{T}(n_dims, one(T)))
    LocationScale(μ, L, q_base)
end

function MeanFieldGaussian(μ::AbstractVector{T},
                           L::Diagonal{T,V}) where {T <: Real, V}
    @assert (length(μ) == size(L,1))
    n_dims = length(μ)
    q_base = MvNormal(FillArrays.Zeros{T}(n_dims),
                      PDMats.ScalMat{T}(n_dims, one(T)))
    LocationScale(μ, L, q_base)
end
