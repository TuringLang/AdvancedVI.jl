
using Test

@testset "ad" begin
    @testset "$(adname)" for (adname, adsymbol) ∈ Dict(
          :ForwardDiff => AutoForwardDiff(),
          :ReverseDiff => AutoReverseDiff(),
          :Zygote      => AutoZygote(),
          # :Enzyme      => AutoEnzyme() # Currently not tested against
        )
        D = 10
        A = randn(D, D)
        λ = randn(D)
        f(λ′) = λ′'*A*λ′ / 2
        ∇, f = AdvancedVI.value_and_gradient(adsymbol, f, λ)
        @test ∇ ≈ (A + A')*λ/2
        @test f ≈ λ'*A*λ / 2
    end
end
