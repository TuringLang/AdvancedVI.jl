@testset "use_view_in_gradient" begin
    # Set up a LogDensityProblem that does not accept views
    struct LogDensityNoView end
    dims = 2
    LogDensityProblems.dimension(::LogDensityNoView) = dims
    LogDensityProblems.capabilities(::Type{<:LogDensityNoView}) =
        LogDensityProblems.LogDensityOrder{1}()
    function LogDensityProblems.logdensity(::LogDensityNoView, x::AbstractArray)
        return sum(x .^ 2)
    end
    function LogDensityProblems.logdensity(::LogDensityNoView, ::SubArray)
        error("Cannot use view")
    end
    function LogDensityProblems.logdensity_and_gradient(::LogDensityNoView, x::AbstractArray)
        ld = sum(x .^ 2)
        grad = 2 .* x
        return ld, grad
    end
    function LogDensityProblems.logdensity_and_gradient(::LogDensityNoView, ::SubArray)
        error("Cannot use view")
    end

    names_and_algs = [
        ("KLMinNaturalGradDescent", KLMinNaturalGradDescent(; stepsize=1e-2, n_samples=10)),
        (
            "KLMinSqrtNaturalGradDescent",
            KLMinSqrtNaturalGradDescent(; stepsize=1e-2, n_samples=10),
        ),
        ("KLMinWassFwdBwd", KLMinWassFwdBwd(; stepsize=1e-2, n_samples=10)),
        ("FisherMinBatchMatch", FisherMinBatchMatch()),
    ]

    # Attempt to run VI without setting `use_view_in_gradient` to false
    AdvancedVI.use_view_in_gradient(::LogDensityNoView) = true
    @testset "$name" for (name, algorithm) in names_and_algs
        @test_throws "Cannot use view" optimize(
            algorithm,
            10,
            LogDensityNoView(),
            FullRankGaussian(zeros(dims), LowerTriangular(Matrix{Float64}(0.6 * I, dims, dims)));
            show_progress=false,
        )
    end

    # Then run VI with `use_view_in_gradient` set to false
    AdvancedVI.use_view_in_gradient(::LogDensityNoView) = false
    @testset "$name" for (name, algorithm) in names_and_algs
        @test optimize(
            algorithm,
            10,
            LogDensityNoView(),
            FullRankGaussian(zeros(dims), LowerTriangular(Matrix{Float64}(0.6 * I, dims, dims)));
            show_progress=false,
        ) isa Any
    end

end
