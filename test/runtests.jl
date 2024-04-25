using Test
using Distributions, DistributionsAD
using ADTypes
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Tracker: Tracker
using Zygote: Zygote
using Enzyme: Enzyme
Enzyme.API.runtimeActivity!(true);
Enzyme.API.typeWarning!(false);

using AdvancedVI

function AdvancedVI.update(q::TuringDiagMvNormal, θ::AbstractArray{<:Real})
    return TuringDiagMvNormal(θ[1:length(q)], exp.(θ[length(q)+1:end]))
end

include("optimisers.jl")

@testset "$adtype" for adtype in [
    AutoForwardDiff(),
    AutoReverseDiff(),
    AutoTracker(),
    AutoZygote(),
    # AutoEnzyme()  # results in incorrect result
]
    target = MvNormal(ones(2))
    logπ(z) = logpdf(target, z)
    advi = ADVI(10, 1000; adtype)

    # Using a function z ↦ q(⋅∣z)
    getq(θ) = TuringDiagMvNormal(θ[1:2], exp.(θ[3:4]))
    q = vi(logπ, advi, getq, randn(4))

    xs = rand(target, 10)
    @test mean(abs2, logpdf(q, xs) - logpdf(target, xs)) ≤ 0.05

    # OR: implement `update` and pass a `Distribution`
    q0 = TuringDiagMvNormal(zeros(2), ones(2))

    q = vi(logπ, advi, q0, randn(4))

    xs = rand(target, 10)
    @test mean(abs2, logpdf(q, xs) - logpdf(target, xs)) ≤ 0.05

end
