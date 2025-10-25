
@testset "SubsampledObjective" begin
    seed = (0x38bef07cf9cc549d)
    n_data = 8

    modelstats = subsamplednormal(Random.default_rng(), n_data)
    (; model, n_dims, μ_true, L_true) = modelstats

    q0 = MeanFieldGaussian(μ_true, Diagonal(diag(L_true)))
    full_obj = RepGradELBO(10)

    @testset "algorithm constructors" begin
        @testset for batchsize in [1, 3, 4]
            sub = ReshufflingBatchSubsampling(1:n_data, batchsize)
            alg = KLMinRepGradDescent(
                AD; n_samples=10, subsampling=sub, operator=ClipScale()
            )
            _, info, _ = optimize(alg, 10, model, q0; show_progress=false)
            @test isfinite(last(info).elbo)

            alg = KLMinRepGradProxDescent(AD; n_samples=10, subsampling=sub)
            _, info, _ = optimize(alg, 10, model, q0; show_progress=false)
            @test isfinite(last(info).elbo)

            alg = KLMinScoreGradDescent(
                AD; n_samples=100, subsampling=sub, operator=ClipScale()
            )
            _, info, _ = optimize(alg, 10, model, q0; show_progress=false)
            @test isfinite(last(info).elbo)
        end
    end

    @testset "determinism" begin
        T = 128
        sub = ReshufflingBatchSubsampling(1:n_data, 1)
        alg = KLMinRepGradDescent(AD; subsampling=sub, operator=ClipScale())
        sub_obj = alg.objective

        rng = StableRNG(seed)
        q_avg, _, _ = optimize(rng, alg, T, model, q0; show_progress=false)

        rng = StableRNG(seed)
        q_avg_ref, _, _ = optimize(rng, alg, T, model, q0; show_progress=false)
        @test q_avg == q_avg_ref

        rng = StableRNG(seed)
        sub_objval_ref = estimate_objective(rng, sub_obj, q0, model)

        rng = StableRNG(seed)
        sub_objval = estimate_objective(rng, sub_obj, q0, model)
        @test sub_objval == sub_objval_ref
    end

    @testset "estimate_objective batchsize=$(batchsize)" for batchsize in [1, 3, 4]
        sub = ReshufflingBatchSubsampling(1:n_data, batchsize)
        sub_obj′ = SubsampledObjective(full_obj, sub)
        full_objval = estimate_objective(full_obj, q0, model; n_samples=10^8)
        sub_objval = estimate_objective(sub_obj′, q0, model; n_samples=10^8)
        @test full_objval ≈ sub_objval rtol=0.1
    end

    @testset "estimate_gradient! batchsize=$(batchsize)" for batchsize in [1, 3, 4]
        params, restructure = Optimisers.destructure(q0)

        out = DiffResults.DiffResult(zero(eltype(params)), similar(params))
        sub = ReshufflingBatchSubsampling(1:n_data, batchsize)
        sub_obj = SubsampledObjective(full_obj, sub)

        # Estimate using full batch
        rng = StableRNG(seed)
        full_state = AdvancedVI.init(rng, full_obj, AD, q0, model, params, restructure)
        AdvancedVI.estimate_gradient!(
            rng, full_obj, AD, out, full_state, params, restructure
        )
        grad_ref = DiffResults.gradient(out)

        # Estimate the full batch gradient by averaging the minibatch gradients
        rng = StableRNG(seed)
        sub_state = AdvancedVI.init(rng, sub_obj, AD, q0, model, params, restructure)
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
