struct ELBO <: VariationalObjective end

const FreeEnergy = ELBO

## Generic evaluation of the free energy
function evaluate(::ELBO, alg, q, logπ)
    return expec_logπ(alg, q, logπ) - entropy(alg, q)
end

function elbo(alg, q, logπ)
    return evaluate(ELBO(), alg, q, logπ)
end
