
using Test

struct SubsampledNormals{D <: Normal, F <: Real}
    dists::Vector{D}
    likeadj::F
end

function SubsampledNormals(rng::Random.AbstractRNG, n_normals::Int)
    μs = randn(rng, n_normals)
    σs = ones(n_normals)
    dists = Normal.(μs, σs)
    SubsampledNormals{eltype(dists), Float64}(dists, 1.0)
end

function LogDensityProblems.logdensity(m::SubsampledNormals, x)
    @unpack likeadj, dists = m
    likeadj*mapreduce(Base.Fix2(logpdf, only(x)), +, dists)
end

function AdvancedVI.subsample(m::SubsampledNormals, idx)
    n_data = length(m.dists)
    SubsampledNormals(m.dists[idx], n_data/length(idx))
end

@testset "interface Subsampled" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    n_data = 16
    prob = SubsampledNormals(rng, n_data)

    q0 = MeanFieldGaussian(zeros(Float64, 1), Diagonal(ones(Float64, 1)))
    full_obj = RepGradELBO(10)
    sub_obj = Subsampled(full_obj, 1, 1:n_data)

    adtype = AutoForwardDiff()
    optimizer = Optimisers.Adam(1e-2)
    averager = PolynomialAveraging()

    T = 128
    @testset "determinism" begin
        rng = StableRNG(seed)
        q_avg, q, _, _ = optimize(
            rng, prob, sub_obj, q0, T; optimizer, averager, show_progress=false, adtype
        )

        rng = StableRNG(seed)
        q_avg_ref, q_ref, _, _ = optimize(
            rng, prob, sub_obj, q0, T; optimizer, averager, show_progress=false, adtype
        )

        @test q_avg == q_avg_ref
        @test q == q_ref

        rng = StableRNG(seed)
        sub_objval_ref = estimate_objective(rng, sub_obj, q0, prob)

        rng = StableRNG(seed)
        sub_objval = estimate_objective(rng, sub_obj, q0, prob)
        @test sub_objval == sub_objval_ref
    end

    @testset "exactness estimate_objective batchsize=$(batchsize)" for batchsize in [1, 3, 4]
        sub_obj′ = @set sub_obj.batchsize = batchsize
        full_objval = estimate_objective(full_obj, q0, prob; n_samples=10^6)
        sub_objval = estimate_objective(sub_obj′, q0, prob; n_samples=10^6)
        @test full_objval ≈ sub_objval rtol=0.1
    end

    @testset "exactness estimate_gradient batchsize=$(batchsize)" for batchsize in [1, 3, 4]
        params, restructure = Optimisers.destructure(q0)
        out = DiffResults.DiffResult(zero(eltype(params)), similar(params))
        n_batches_per_epoch = ceil(Int, n_data/batchsize)
        sub_obj = Subsampled(full_obj, 1, 1:n_data)

        full_state = AdvancedVI.init(rng, full_obj, prob, params, restructure)
        AdvancedVI.estimate_gradient!(
            rng, full_obj, adtype, out, prob, params, restructure, full_state
        )
        grad_ref = DiffResults.gradient(out)

        sub_state = AdvancedVI.init(rng, sub_obj, prob, params, restructure)
        grad = mean(1:n_batches_per_epoch) do _
            # Using a fixed RNG so that the same Monte Carlo samples are used across the batches
            rng = StableRNG(seed)
            AdvancedVI.estimate_gradient!(
                rng, sub_obj, adtype, out, prob, params, restructure, sub_state
            )
            DiffResults.gradient(out)
        end
        @test grad ≈ grad_ref
    end
end
