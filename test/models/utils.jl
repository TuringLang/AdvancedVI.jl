
function sample_cholesky(rng::AbstractRNG, type::Type, n_dims::Int)
    A   = randn(rng, type, n_dims, n_dims) 
    L   = tril(A)
    idx = diagind(L)
    @. L[idx] = log(exp(L[idx]) + 1)
    L |> LowerTriangular
end