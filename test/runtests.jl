
using ReTest: @testset, @test
using Random
using Random: default_rng
using Statistics
using Distributions, DistributionsAD
using LinearAlgebra
using AdvancedVI

include("ad.jl")
include("distributions.jl")
include("exact.jl")

