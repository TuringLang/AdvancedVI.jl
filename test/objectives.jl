@testset "objectives" begin
    using AdvancedVI: ELBO, FreeEnergy, VariationalObjective, evaluate
    L = AdvancedELBO()
    @test L isa VariationalObjective
    @test L isa FreeEnergy
    alg = ADVI(1000, 1)
    q = CholMvNormal(zeros(2), LowerTriangular(randn(2, 2)))
    logπ(x) = logpdf(MvNormal(ones(2)), x)
    @test evaluate(L, alg, q, logπ) ≈ (expec_logπ(alg, q, logπ) - entropy(alg, q)) atol=1e-1
    @test elbo(alg, q, logπ) ≈ evaluate(L, alg, q, logπ) atol=1e-1
end