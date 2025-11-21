using ADTypes
using AdvancedVI
using BenchmarkTools
using Distributions
using DistributionsAD
using Enzyme, ForwardDiff, ReverseDiff, Zygote, Mooncake
using FillArrays
using InteractiveUtils
using LinearAlgebra
using LogDensityProblems
using Optimisers
using Random

BLAS.set_num_threads(min(4, Threads.nthreads()))

@info sprint(versioninfo)
@info "BLAS threads: $(BLAS.get_num_threads())"

include("normallognormal.jl")
include("unconstrdist.jl")

const SUITES = BenchmarkGroup()

function variational_standard_mvnormal(type::Type, n_dims::Int, family::Symbol)
    if family == :meanfield
        MeanFieldGaussian(zeros(type, n_dims), Diagonal(ones(type, n_dims)))
    else
        FullRankGaussian(zeros(type, n_dims), Matrix(type, I, n_dims, n_dims))
    end
end

begin
    T = Float64

    for (probname, prob) in [("normal", normal(; n_dims=10, realtype=T))]
        max_iter = 10^4
        d = LogDensityProblems.dimension(prob)
        opt = Optimisers.Adam(T(1e-3))

        for (objname, entropy) in [
                ("RepGradELBO", ClosedFormEntropy()),
                ("RepGradELBO + STL", StickingTheLandingEntropy()),
            ],
            (adname, adtype) in [
                ("Zygote", AutoZygote()),
                ("ReverseDiff", AutoReverseDiff()),
                ("Mooncake", AutoMooncake(; config=Mooncake.Config())),
                # ("Enzyme", AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)),
            ],
            (familyname, family) in [
                ("meanfield", MeanFieldGaussian(zeros(T, d), Diagonal(ones(T, d)))),
                (
                    "fullrank",
                    FullRankGaussian(zeros(T, d), LowerTriangular(Matrix{T}(I, d, d))),
                ),
            ]

            q = family
            alg = KLMinRepGradDescent(adtype; optimizer=opt, entropy, operator=ClipScale())

            SUITES[probname][objname][familyname][adname] = begin
                @benchmarkable AdvancedVI.optimize(
                    $alg, $max_iter, $prob, $q; show_progress=false
                )
            end
        end
    end
end

BenchmarkTools.tune!(SUITES; verbose=true)
results = BenchmarkTools.run(SUITES; verbose=true)
display(median(results))

BenchmarkTools.save(joinpath(@__DIR__, "benchmark_results.json"), median(results))
