struct ELBO <: VariationalObjective end

function (elbo::ELBO)(alg, q, logπ, num_samples; kwargs...)
    return elbo(Random.default_rng(), alg, q, logπ, num_samples; kwargs...)
end

const elbo = ELBO()
