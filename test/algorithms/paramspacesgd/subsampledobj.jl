
struct SubsampledNormals{D<:Normal,F<:Real}
    dists::Vector{D}
    likeadj::F
end

function SubsampledNormals(rng::Random.AbstractRNG, n_normals::Int)
    μs = randn(rng, n_normals)
    σs = ones(n_normals)
    dists = Normal.(μs, σs)
    SubsampledNormals{eltype(dists),Float64}(dists, 1.0)
end

function LogDensityProblems.logdensity(m::SubsampledNormals, x)
    (; likeadj, dists) = m
    likeadj*mapreduce(Base.Fix2(logpdf, only(x)), +, dists)
end

function LogDensityProblems.logdensity_and_gradient(m::SubsampledNormals, x)
    return (
        LogDensityProblems.logdensity(m, x),
        ForwardDiff.gradient(Base.Fix1(LogDensityProblems.logdensity, m), x),
    )
end

function LogDensityProblems.capabilities(::Type{<:SubsampledNormals})
    return LogDensityProblems.LogDensityOrder{1}()
end

function AdvancedVI.subsample(m::SubsampledNormals, idx)
    n_data = length(m.dists)
    SubsampledNormals(m.dists[idx], n_data/length(idx))
end

@testset "SubsampledObjective" begin
    seed = (0x38bef07cf9cc549d)
    n_data = 8
    prob = SubsampledNormals(Random.default_rng(), n_data)

    μ0 = [mean([mean(dist) for dist in prob.dists])]
    q0 = MeanFieldGaussian(μ0, Diagonal(ones(1)))
    full_obj = RepGradELBO(10)

    @testset "determinism" begin
        T = 128
        sub_obj = SubsampledObjective(full_obj, 1, 1:n_data)
        alg = ParamSpaceSGD(sub_obj, AD, DoWG(), PolynomialAveraging(), ClipScale())

        rng = StableRNG(seed)
        q_avg, _, _ = optimize(rng, alg, T, prob, q0; show_progress=false)

        rng = StableRNG(seed)
        q_avg_ref, _, _ = optimize(rng, alg, T, prob, q0; show_progress=false)
        @test q_avg == q_avg_ref

        rng = StableRNG(seed)
        sub_objval_ref = estimate_objective(rng, sub_obj, q0, prob)

        rng = StableRNG(seed)
        sub_objval = estimate_objective(rng, sub_obj, q0, prob)
        @test sub_objval == sub_objval_ref
    end

    @testset "estimate_objective batchsize=$(batchsize)" for batchsize in [1, 3, 4]
        sub_obj′ = SubsampledObjective(full_obj, batchsize, 1:n_data)
        full_objval = estimate_objective(full_obj, q0, prob; n_samples=10^8)
        sub_objval = estimate_objective(sub_obj′, q0, prob; n_samples=10^8)
        @test full_objval ≈ sub_objval rtol=0.1
    end

    @testset "estimate_gradient! batchsize=$(batchsize)" for batchsize in [1, 3, 4]
        params, restructure = Optimisers.destructure(q0)

        out = DiffResults.DiffResult(zero(eltype(params)), similar(params))
        sub_obj = SubsampledObjective(full_obj, batchsize, 1:n_data)

        # Estimate using full batch
        rng = StableRNG(seed)
        full_state = AdvancedVI.init(rng, full_obj, AD, prob, params, restructure)
        AdvancedVI.estimate_gradient!(
            rng, full_obj, AD, out, full_state, params, restructure
        )
        grad_ref = DiffResults.gradient(out)

        # Estimate the full batch gradient by averaging the minibatch gradients
        rng = StableRNG(seed)
        sub_state = AdvancedVI.init(rng, sub_obj, AD, prob, params, restructure)
        grad = mean(1:length(sub_obj.subsampling)) do _
            # Fixing the RNG so that the same Monte Carlo samples are used across the batches
            rng = StableRNG(seed)
            AdvancedVI.estimate_gradient!(
                rng, sub_obj, AD, out, sub_state, params, restructure
            )
            DiffResults.gradient(out)
        end
        @test grad ≈ grad_ref
    end
end
