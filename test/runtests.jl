using AdvancedVI
using Test
using Distributions, DistributionsAD
using LinearAlgebra
using Flux
# include("optimisers.jl")

target = MvNormal(ones(2))
logπ(z) = logpdf(target, z)
advi = ADVI(10, 1000)
bbvi = AdvancedVI.BBVI(10, 1000)
# Using a function z ↦ q(⋅∣z)
getq(θ) = TuringDiagMvNormal(θ[1:2], exp.(θ[3:4]))
q1 = AdvancedVI.CholMvNormal(zeros(2), LowerTriangular(randn(2,2)))
q2 = AdvancedVI.CholMvNormal(zeros(2), LowerTriangular(randn(2,2)))
q1 = vi(logπ, advi, q1, opt=ADAM(0.01))
q2 = vi(logπ, bbvi, q2, opt=ADAM(0.01))

xs = rand(target, 10)
@test mean(abs2, logpdf(q1, xs) - logpdf(target, xs)) ≤ 0.05
@test mean(abs2, logpdf(q2, xs) - logpdf(target, xs)) ≤ 0.05