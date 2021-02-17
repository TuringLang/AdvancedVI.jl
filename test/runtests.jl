using Test
using Distributions, DistributionsAD
using AdvancedVI
using LinearAlgebra
using Flux
include("optimisers.jl")

target = MvNormal(ones(2))
logπ(z) = logpdf(target, z)
advi = ADVI(10, 1000)

# Using a function z ↦ q(⋅∣z)
getq(θ) = TuringDiagMvNormal(θ[1:2], exp.(θ[3:4]))
q = AdvancedVI.CholMvNormal(zeros(2), LowerTriangular(randn(2,2)))
q = vi(logπ, advi, q, opt=ADAM(0.01))
q = vi(logπ, advi, getq, randn(4))

xs = rand(target, 10)
@test mean(abs2, logpdf(q, xs) - logpdf(target, xs)) ≤ 0.05

# OR: implement `update` and pass a `Distribution`
function AdvancedVI.update(d::TuringDiagMvNormal, θ::AbstractArray{<:Real})
    return TuringDiagMvNormal(θ[1:length(q)], exp.(θ[length(q) + 1:end]))
end

q0 = TuringDiagMvNormal(zeros(2), ones(2))
q = vi(logπ, advi, q0, randn(4))

xs = rand(target, 10)
@test mean(abs2, logpdf(q, xs) - logpdf(target, xs)) ≤ 0.05

