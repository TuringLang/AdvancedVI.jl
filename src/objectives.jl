abstract type VariationalObjective end

struct ELBO <: VariationalObjective end

const FreeEnergy = ELBO


## Generic evaluation of the free energy
function evaluate(::ELBO, alg, q, logπ)
    return eval_expec_logπ(alg, q, logπ) - eval_entropy(alg, q)
end

function evaluate(vo::ELBO, alg, q, θ, logπ)
    evaluate(vo, alg, q(θ), logπ)
end