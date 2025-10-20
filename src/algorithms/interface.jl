
"""
This family of algorithms (`<:KLMinRepGradDescent`,`<:KLMinRepGradProxDescent`,`<:KLMinScoreGradDescent`) applies stochastic gradient descent (SGD) to the variational `objective` over the (Euclidean) space of variational parameters.
The trainable parameters in the variational approximation are expected to be extractable through `Optimisers.destructure`.
This requires the variational approximation to be marked as a functor through `Functors.@functor`.
"""
const ParamSpaceSGD = Union{
    <:KLMinRepGradDescent,<:KLMinRepGradProxDescent,<:KLMinScoreGradDescent
}

"""
    estimate_objective([rng,] alg, q, prob; n_samples, entropy)

Estimate the ELBO of the variational approximation `q` against the target log-density `prob`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `alg::Union{<:KLMinRepGradDescent,<:KLMinRepGradProxDescent,<:KLMinScoreGradDescent}`: Variational inference algorithm.
- `q`: Variational approximation.
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.

# Keyword Arguments
- `n_samples::Int`: Number of Monte Carlo samples for estimating the objective. (default: Same as the the number of samples used for estimating the gradient during optimization.)
- `entropy::AbstractEntropyEstimator`: Entropy estimator. (default: `MonteCarloEntropy()`)

# Returns
- `obj_est`: Estimate of the objective value.
"""
function estimate_objective(
    rng::Random.AbstractRNG,
    alg::Union{<:KLMinRepGradDescent,<:KLMinRepGradProxDescent,<:KLMinScoreGradDescent},
    q,
    prob;
    n_samples::Int=alg.objective.n_samples,
    entropy::AbstractEntropyEstimator=MonteCarloEntropy(),
)
    return estimate_objective(rng, RepGradELBO(n_samples; entropy=entropy), q, prob)
end

function init(rng::Random.AbstractRNG, alg::ParamSpaceSGD, q_init, prob)
    (; adtype, optimizer, averager, objective, operator) = alg
    if q_init isa AdvancedVI.MvLocationScale && operator isa AdvancedVI.IdentityOperator
        @warn(
            "IdentityOperator is used with a variational family <:MvLocationScale. Optimization can easily fail under this combination due to singular scale matrices. Consider using the operator `ClipScale` in the algorithm instead.",
        )
    end
    params, re = Optimisers.destructure(q_init)
    opt_st = Optimisers.setup(optimizer, params)
    obj_st = init(rng, objective, adtype, q_init, prob, params, re)
    avg_st = init(averager, params)
    grad_buf = DiffResults.DiffResult(zero(eltype(params)), similar(params))
    return (
        prob=prob,
        q=q_init,
        iteration=0,
        grad_buf=grad_buf,
        opt_st=opt_st,
        obj_st=obj_st,
        avg_st=avg_st,
    )
end

function output(alg::ParamSpaceSGD, state)
    params_avg = value(alg.averager, state.avg_st)
    _, re = Optimisers.destructure(state.q)
    return re(params_avg)
end

function step(
    rng::Random.AbstractRNG, alg::ParamSpaceSGD, state, callback, objargs...; kwargs...
)
    (; adtype, objective, operator, averager) = alg
    (; prob, q, iteration, grad_buf, opt_st, obj_st, avg_st) = state

    iteration += 1

    params, re = Optimisers.destructure(q)

    grad_buf, obj_st, info = estimate_gradient!(
        rng, objective, adtype, grad_buf, obj_st, params, re, objargs...
    )

    grad = DiffResults.gradient(grad_buf)
    opt_st, params = Optimisers.update!(opt_st, params, grad)
    params = apply(operator, typeof(q), opt_st, params, re)
    avg_st = apply(averager, avg_st, params)

    state = (
        prob=prob,
        q=re(params),
        iteration=iteration,
        grad_buf=grad_buf,
        opt_st=opt_st,
        obj_st=obj_st,
        avg_st=avg_st,
    )

    if !isnothing(callback)
        averaged_params = value(averager, avg_st)
        info′ = callback(;
            rng,
            iteration,
            restructure=re,
            params=params,
            averaged_params=averaged_params,
            gradient=grad,
            state=state,
        )
        info = !isnothing(info′) ? merge(info′, info) : info
    end
    state, false, info
end
