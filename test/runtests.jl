
using Test
using Test: @testset, @test

using ADTypes
using Base.Iterators
using Bijectors
using DiffResults
using Distributions
using FillArrays
using ForwardDiff, ReverseDiff
using LinearAlgebra
using LogDensityProblems
using Optimisers
using PDMats
using Pkg
using Random, StableRNGs
using Statistics
using StatsBase

using AdvancedVI

const PROGRESS = haskey(ENV, "PROGRESS")
const GROUP = get(ENV, "GROUP", "All")
const AD_str = get(ENV, "AD", "ReverseDiff")

const AD = if AD_str == "ReverseDiff"
    AutoReverseDiff()
elseif AD_str == "ForwardDiff"
    AutoForwardDiff()
elseif AD_str == "Zygote"
    using Zygote
    AutoZygote()
elseif AD_str == "Mooncake"
    using Mooncake
    AutoMooncake(; config=Mooncake.Config())
elseif AD_str == "Enzyme"
    using Enzyme
    AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Const,
    )
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
include("models/subsamplednormals.jl")

if GROUP == "All" || GROUP == "GENERAL"
    include("general/optimize.jl")
    include("general/proximal_location_scale_entropy.jl")
    include("general/rules.jl")
    include("general/averaging.jl")
    include("general/clip_scale.jl")

    include("families/location_scale.jl")
    include("families/location_scale_low_rank.jl")

    include("general/gauss_expected_grad_hess.jl")
    include("algorithms/klminwassfwdbwd.jl")
    include("algorithms/klminsqrtnaturalgraddescent.jl")
    include("algorithms/klminnaturalgraddescent.jl")
    include("algorithms/fisherminbatchmatch.jl")
end

if GROUP == "All" || GROUP == "AD"
    # Tests that need to check correctness of the integration with AD backends
    include("general/ad.jl")
    include("general/mixedad_logdensity.jl")
    include("general/subsampledobj.jl")

    include("algorithms/klminrepgraddescent.jl")
    include("algorithms/klminscoregraddescent.jl")
    include("algorithms/klminrepgradproxdescent.jl")
    include("algorithms/klminrepgraddescent_bijectors.jl")
    include("algorithms/klminrepgradproxdescent_bijectors.jl")
    include("algorithms/klminscoregraddescent_bijectors.jl")
end
