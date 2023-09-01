
using ReTest
using ReTest: @testset, @test

using Comonicon
using Random, StableRNGs
using Statistics
using Distributions
using LinearAlgebra
using SimpleUnPack: @unpack
using FillArrays
using PDMats

using Functors
using DistributionsAD
@functor TuringDiagMvNormal

using Bijectors
using LogDensityProblems
using Optimisers
using ADTypes
using ForwardDiff, ReverseDiff, Zygote

using AdvancedVI

# Models for Inference Tests
struct TestModel{M,L,S}
    model::M
    μ_true::L
    L_true::S
    n_dims::Int
    is_meanfield::Bool
end

include("models/normallognormal.jl")

# Tests
include("ad.jl")
include("advi_locscale.jl")
include("optimize.jl")

@main function runtests(patterns...; dry::Bool = false)
    retest(patterns...; dry = dry, verbose = Inf)
end

