
LocationScale(μ::LinearAlgebra.AbstractVector,
              L::Union{<: LinearAlgebra.AbstractTriangular,
                       <: LinearAlgebra.Diagonal},
              q₀::Distributions.ContinuousMultivariateDistribution) =
                  transformed(q₀, Bijectors.Shift(μ) ∘ Bijectors.Scale(L))

function location_scale_entropy(
    q₀::Distributions.ContinuousMultivariateDistribution,
    locscale_bijector::Bijectors.ComposedFunction)
end

function entropy(q_trans::MultivariateTransformed{
    <: Distributions.ContinuousMultivariateDistribution,
    <: Bijectors.ComposedFunction{
        <: Bijectors.Shift,
        <: Bijectors.Scale}})
    q_base = q_trans.dist
    scale  = q_trans.transform.inner.a
    entropy(q_base) + first(logabsdet(scale))
end

function FullRankGaussian(μ::AbstractVector,
                          L::LinearAlgebra.AbstractTriangular)
    q₀ = MvNormal(zeros(eltype(μ), length(μ)), one(eltype(μ)))
    LocationScale(μ, L, q₀)
end

function MeanFieldGaussian(μ::AbstractVector,
                           L::LinearAlgebra.Diagonal)
    q₀ = MvNormal(zeros(eltype(μ), length(μ)), one(eltype(μ)))
    LocationScale(μ, L, q₀)
end
