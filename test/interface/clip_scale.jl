
@testset "interface ClipScale" begin
    @testset "MvLocationScale" begin
        @testset "$(string(covtype)) $(realtype) $(bijector)" for covtype in
                                                                  [:meanfield, :fullrank],
            realtype in [Float32, Float64],
            bijector in [nothing, :identity]

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
            q = if isnothing(bijector)
                q
            else
                Bijectors.TransformedDistribution(q, identity)
            end

            params, re = Optimisers.destructure(q)
            params′ = AdvancedVI.apply(ClipScale(ϵ), typeof(q), params, re)
            q′ = re(params′)

            if isnothing(bijector)
                @test all(var(q′) .≥ ϵ^2)
            else
                @test all(var(q′.dist) .≥ ϵ^2)
            end
        end
    end

    @testset "MvLocationScaleLowRank" begin
        @testset "$(realtype) $(bijector)" for realtype in [Float32, Float64],
            bijector in [nothing, :identity]

            n_rank = 2
            d = 5
            μ = zeros(realtype, d)
            ϵ = sqrt(realtype(0.5))
            D = ones(realtype, d)
            U = randn(realtype, d, n_rank)
            q = MvLocationScaleLowRank(
                μ, D, U, Normal{realtype}(zero(realtype), one(realtype))
            )
            q = if isnothing(bijector)
                q
            else
                Bijectors.TransformedDistribution(q, bijector)
            end

            params, re = Optimisers.destructure(q)
            params′ = AdvancedVI.apply(ClipScale(ϵ), typeof(q), params, re)
            q′ = re(params′)

            if isnothing(bijector)
                @test all(var(q′) .≥ ϵ^2)
            else
                @test all(var(q′.dist) .≥ ϵ^2)
            end
        end
    end
end
