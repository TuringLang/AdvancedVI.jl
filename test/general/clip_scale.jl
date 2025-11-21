
@testset "interface ClipScale" begin
    @testset "MvLocationScale" begin
        @testset "$(string(covtype)) $(realtype)" for covtype in [:meanfield, :fullrank],
            realtype in [Float32, Float64]

            d = 5
            μ = zeros(realtype, d)
            ϵ = sqrt(realtype(0.5))
            q = if covtype == :fullrank
                L = LowerTriangular(Matrix{realtype}(I, d, d))
                FullRankGaussian(μ, L)
            elseif covtype == :meanfield
                L = Diagonal(ones(realtype, d))
                MeanFieldGaussian(μ, L)
            end

            params, re = Optimisers.destructure(q)
            opt_st = Optimisers.setup(Descent(1e-2), params)
            params′ = AdvancedVI.apply(ClipScale(ϵ), typeof(q), opt_st, params, re)
            q′ = re(params′)

            @test all(var(q′) .≥ ϵ^2)
        end
    end

    @testset "MvLocationScaleLowRank" begin
        @testset "$(realtype)" for realtype in [Float32, Float64]
            n_rank = 2
            d = 5
            μ = zeros(realtype, d)
            ϵ = sqrt(realtype(0.5))
            D = ones(realtype, d)
            U = randn(realtype, d, n_rank)
            q = MvLocationScaleLowRank(
                μ, D, U, Normal{realtype}(zero(realtype), one(realtype))
            )

            params, re = Optimisers.destructure(q)
            opt_st = Optimisers.setup(Descent(1e-2), params)
            params′ = AdvancedVI.apply(ClipScale(ϵ), typeof(q), opt_st, params, re)
            q′ = re(params′)

            @test all(var(q′.dist) .≥ ϵ^2)
        end
    end
end
