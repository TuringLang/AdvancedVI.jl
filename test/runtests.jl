
using Test
using Test: @testset, @test

using Base.Iterators
using Bijectors
using DiffResults
using Distributions
using FillArrays
using LinearAlgebra
using LogDensityProblems
using Optimisers
using PDMats
using Pkg
using Random, StableRNGs
using Statistics
using StatsBase

using Functors
using DistributionsAD
@functor TuringDiagMvNormal

using ADTypes
using ForwardDiff, ReverseDiff, Zygote, Mooncake

using AdvancedVI

const PROGRESS = haskey(ENV, "PROGRESS")
const TEST_GROUP = get(ENV, "TEST_GROUP", "All")

if TEST_GROUP == "Enzyme"
    using Enzyme
end

# Models for Inference Tests
struct TestModel{M,L,S,SC}
    model::M
    μ_true::L
    L_true::S
    n_dims::Int
    strong_convexity::SC
    is_meanfield::Bool
end
include("models/normal.jl")
include("models/normallognormal.jl")

if TEST_GROUP == "All" || TEST_GROUP == "General"
    # Interface tests that do not involve testing on Enzyme
    include("general/optimize.jl")
    include("general/rules.jl")
    include("general/averaging.jl")
    include("general/clip_scale.jl")
    include("general/proximal_location_scale_entropy.jl")
end

if TEST_GROUP == "All" || TEST_GROUP == "General" || TEST_GROUP == "Enzyme"
    # Interface tests that involve testing on Enzyme
    include("interface/ad.jl")
end

if TEST_GROUP == "All" || TEST_GROUP == "Families"
    include("families/location_scale.jl")
    include("families/location_scale_low_rank.jl")
end

if TEST_GROUP == "All" || TEST_GROUP == "ParamSpaceSGD" || TEST_GROUP == "Enzyme"
    include("algorithms/paramspacesgd/repgradelbo.jl")
    include("algorithms/paramspacesgd/scoregradelbo.jl")
    include("algorithms/paramspacesgd/repgradelbo_distributionsad.jl")
    include("algorithms/paramspacesgd/repgradelbo_locationscale.jl")
    include("algorithms/paramspacesgd/repgradelbo_locationscale_bijectors.jl")
    include("algorithms/paramspacesgd/repgradelbo_proximal_locationscale.jl")
    include("algorithms/paramspacesgd/repgradelbo_proximal_locationscale_bijectors.jl")
    include("algorithms/paramspacesgd/scoregradelbo_distributionsad.jl")
    include("algorithms/paramspacesgd/scoregradelbo_locationscale.jl")
    include("algorithms/paramspacesgd/scoregradelbo_locationscale_bijectors.jl")
end
