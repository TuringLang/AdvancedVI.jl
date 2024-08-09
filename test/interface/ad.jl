
using Test

@testset "ad" begin
    @testset "$(adname)" for (adname, adtype) ∈ Dict(
          :ForwardDiff => AutoForwardDiff(),
          :ReverseDiff => AutoReverseDiff(),
          :Zygote      => AutoZygote(),
          :Tapir       => AutoTapir(),
          :Enzyme      => AutoEnzyme() 
        )
        D = 10
        A = randn(D, D)
        λ = randn(D)
        f(λ′) = λ′'*A*λ′ / 2

        ad_st = AdvancedVI.init_adbackend(adtype, f, λ)
        grad_buf = DiffResults.GradientResult(λ)
        AdvancedVI.value_and_gradient!(adtype, ad_st, f, λ, grad_buf)
        ∇ = DiffResults.gradient(grad_buf)
        f = DiffResults.value(grad_buf)
        @test ∇ ≈ (A + A')*λ/2
        @test f ≈ λ'*A*λ / 2
    end
end
