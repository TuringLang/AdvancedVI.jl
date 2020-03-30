using Random: GLOBAL_RNG

struct ELBO <: VariationalObjective end

function (elbo::ELBO)(alg, q, logπ, num_samples; kwargs...)
    return elbo(GLOBAL_RNG, alg, q, logπ, num_samples; kwargs...)
end

const elbo = ELBO()
