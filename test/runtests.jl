
using Test
using Test: @testset, @test

using Bijectors
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
include("models/normal.jl")
include("models/normallognormal.jl")

# Tests
include("interface/ad.jl")
include("interface/optimize.jl")
include("interface/repgradelbo.jl")

include("inference/repgradelbo_distributionsad.jl")
include("inference/repgradelbo_distributionsad_bijectors.jl")
