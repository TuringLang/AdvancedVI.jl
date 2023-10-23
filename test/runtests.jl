
using Test
using Test: @testset, @test

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
include("models/normal.jl")

# Tests
include("interface/ad.jl")
include("interface/optimize.jl")
include("interface/advi.jl")

include("inference/advi_distributionsad.jl")
