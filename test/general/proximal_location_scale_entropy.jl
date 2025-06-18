
@testset "interface ProximalLocationScaleEntropy" begin
    @testset "MvLocationScale" begin
        @testset "$(string(covtype)) $(realtype) $(bijector)" for covtype in
                                                                  [:meanfield, :fullrank],
            realtype in [Float32, Float64],
            bijector in [nothing, :identity]

            stepsize = 1e-2
            optimizer = Descent(stepsize)

            d = 5
            μ = zeros(realtype, d)
            ϵ = sqrt(realtype(0.5))
            L = if covtype == :fullrank
                LowerTriangular(Matrix{realtype}(I, d, d))
            elseif covtype == :meanfield
                Diagonal(ones(realtype, d))
            end
            q = if covtype == :fullrank
                FullRankGaussian(μ, L)
            elseif covtype == :meanfield
                MeanFieldGaussian(μ, L)
            end
            q = if isnothing(bijector)
                q
            else
                Bijectors.TransformedDistribution(q, identity)
            end

            # The proximal operator for the entropy of a location scale distribution 
            # solves the subproblem:
            #
            # argmin_{L} - logabsdet(L) + 1/(2η) norm(ab2, L - L')
            #
            # for some fixed L' with respect to L over the set of triangular matrices 
            # that have strictly positive eigenvalues.
            # 
            # The solution L to this convex program is the solution to
            #
            #     ∇logabsdet(L) = ∇ 1/(2η) norm(abs2, L - L') .
            #
            # This unit test will check that this equation is satisfied.

            params, re = Optimisers.destructure(q)
            opt_st = Optimisers.setup(optimizer, params)
            params′ = AdvancedVI.apply(
                ProximalLocationScaleEntropy(), typeof(q), opt_st, params, re
            )

            q′ = re(params′)
            scale′ = isnothing(bijector) ? q′.scale : q′.dist.scale

            grad_left = only(Zygote.gradient(L_ -> first(logabsdet(L_)), scale′))
            grad_right = only(
                Zygote.gradient(L_ -> sum(abs2, L_ - L) / (2 * stepsize), scale′)
            )

            @test grad_left ≈ grad_right
        end
    end
end
