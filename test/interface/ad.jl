
using Test

@testset "ad" begin
    @testset "$(adname)" for (adname, adtype) in Dict(
        :ForwardDiff => AutoForwardDiff(),
        :ReverseDiff => AutoReverseDiff(),
        :Zygote => AutoZygote(),
        :Tapir => AutoTapir(; safe_mode=false),
        #:Enzyme      => AutoEnzyme() 
    )
        D = 10
        A = randn(D, D)
        λ = randn(D)
        f(λ′) = λ′' * A * λ′ / 2

        ad_st = AdvancedVI.init_adbackend(adtype, f, λ)
        grad_buf = DiffResults.GradientResult(λ)
        AdvancedVI.value_and_gradient!(adtype, ad_st, f, λ, grad_buf)
        ∇ = DiffResults.gradient(grad_buf)
        f = DiffResults.value(grad_buf)
        @test ∇ ≈ (A + A') * λ / 2
        @test f ≈ λ' * A * λ / 2
    end

    @testset "$(adname) with auxiliary input" for (adname, adsymbol) in Dict(
        :ForwardDiff => AutoForwardDiff(),
        :ReverseDiff => AutoReverseDiff(),
        :Zygote => AutoZygote(),
        :Enzyme => AutoEnzyme(),
    )
        D = 10
        A = randn(D, D)
        λ = randn(D)
        b = randn(D)
        grad_buf = DiffResults.GradientResult(λ)
        f(λ′, aux) = λ′' * A * λ′ / 2 + dot(aux.b, λ′)
        AdvancedVI.value_and_gradient!(adsymbol, f, λ, (b=b,), grad_buf)
        ∇ = DiffResults.gradient(grad_buf)
        f = DiffResults.value(grad_buf)
        @test ∇ ≈ (A + A') * λ / 2 + b
        @test f ≈ λ' * A * λ / 2 + dot(b, λ)
    end
end
