@testset "objectives" begin
    using AdvancedVI: ELBO, FreeEnergy, VariationalObjective, evaluate
    using AdvancedVI: elbo, entropy, expec_logπ
    L = ELBO()
    @test L isa VariationalObjective
    @test L isa FreeEnergy
    alg = ADVI(1000, 1)
    q = AdvancedVI.CholMvNormal(zeros(2), LowerTriangular(diagm(ones(2))))
    logπ(x) = logpdf(MvNormal(ones(2)), x)
    @test evaluate(L, alg, q, logπ) ≈ (expec_logπ(alg, q, logπ) - entropy(alg, q)) atol=1e0
    @test elbo(alg, q, logπ) ≈ evaluate(L, alg, q, logπ) atol=1e0
end