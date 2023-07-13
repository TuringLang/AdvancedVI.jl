
function normallognormal_fullrank(realtype; rng = default_rng())
    n_dims = 5

    μ_x  = randn(rng, realtype)
    σ_x  = π
    μ_y  = randn(rng, realtype, n_dims)
    L₀_y = randn(rng, realtype, n_dims, n_dims) |> LowerTriangular
    ϵ    = realtype(1.0)
    Σ_y  = (L₀_y*L₀_y' + ϵ*I) |> Hermitian

    Turing.@model function normallognormal()
        x ~ LogNormal(μ_x, σ_x)
        y ~ MvNormal(μ_y, Σ_y)
    end
    model = normallognormal()

    Σ = Matrix{realtype}(undef, n_dims+1, n_dims+1)
    Σ[1,1]         = σ_x^2
    Σ[2:end,2:end] = Σ_y
    Σ = Σ |> Hermitian

    μ = vcat(μ_x, μ_y)
    L = cholesky(Σ).L |> LowerTriangular

    TestModel(model, μ, L, n_dims+1, false)
end

function normallognormal_meanfield(realtype)
    n_dims = 5

    μ_x  = randn(realtype)
    σ_x  = π
    μ_y  = randn(realtype, n_dims)
    ϵ    = realtype(1.0)
    Σ_y  = Diagonal(exp.(randn(realtype, n_dims)))

    Turing.@model function normallognormal()
        x ~ LogNormal(μ_x, σ_x)
        y ~ MvNormal(μ_y, Σ_y)
    end
    model = normallognormal()

    σ²        = Vector{realtype}(undef, n_dims+1)
    σ²[1]     = σ_x^2
    σ²[2:end] = diag(Σ_y)

    μ = vcat(μ_x, μ_y)
    L = sqrt.(σ²) |> Diagonal

    TestModel(model, μ, L, n_dims+1, true)
end
