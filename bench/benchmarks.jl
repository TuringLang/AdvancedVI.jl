
using ADTypes, ForwardDiff, ReverseDiff, Zygote
using AdvancedVI
using BenchmarkTools
using Bijectors
using Distributions
using DistributionsAD
using FillArrays
using InteractiveUtils
using LinearAlgebra
using LogDensityProblems
using Optimisers
using Random

BLAS.set_num_threads(min(4, Threads.nthreads()))

@info sprint(versioninfo)
@info "BLAS threads: $(BLAS.get_num_threads())"

include("utils.jl")
include("normallognormal.jl")

const SUITES = BenchmarkGroup()

# Comment until https://github.com/TuringLang/Bijectors.jl/pull/315 is merged
# SUITES["normal + bijector"]["meanfield"]["Zygote"] =
#     @benchmarkable normallognormal(
#         ;
#         fptype       = Float64,
#         adtype       = AutoZygote(),
#         family       = :meanfield,
#         objective    = :RepGradELBO,
#         n_montecarlo = 4,
#     )

SUITES["normal + bijector"]["meanfield"]["ReverseDiff"] =
    @benchmarkable normallognormal(
        ;
        fptype       = Float64,
        adtype       = AutoReverseDiff(),
        family       = :meanfield,
        objective    = :RepGradELBO,
        n_montecarlo = 4,
    )

SUITES["normal + bijector"]["meanfield"]["ForwardDiff"] =
    @benchmarkable normallognormal(
        ;
        fptype       = Float64,
        adtype       = AutoForwardDiff(),
        family       = :meanfield,
        objective    = :RepGradELBO,
        n_montecarlo = 4,
    )

BenchmarkTools.tune!(SUITES; verbose=true)
results = BenchmarkTools.run(SUITES; verbose=true)
display(median(results))

BenchmarkTools.save(joinpath(@__DIR__, "benchmark_results.json"), median(results))
