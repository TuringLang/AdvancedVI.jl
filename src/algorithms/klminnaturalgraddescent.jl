
"""
    KLMinNaturalGradDescent(stepsize, n_samples, ensure_posdef, subsampling)
    KLMinNaturalGradDescent(; stepsize, n_samples, ensure_posdef, subsampling)

KL divergence minimization by running natural gradient descent[^KL2017][^KR2023], also called variational online Newton.
This algorithm can be viewed as an instantiation of mirror descent, where the Bregman divergence is chosen to be the KL divergence.

If the `ensure_posdef` argument is true, the algorithm applies the technique by Lin *et al.*[^LSK2020], where the precision matrix update includes an additional term that guarantees positive definiteness.
This, however, involves an additional set of matrix-matrix system solves that could be costly.

The original algorithm requires estimating the quantity \$\$ \\mathbb{E}_q \\nabla^2 \\log \\pi \$\$, where \$\$ \\log \\pi \$\$ is the target log-density and \$\$q\$\$ is the current variational approximation.
If the target `LogDensityProblem` associated with \$\$ \\log \\pi \$\$ has second-order differentiation [capability](https://www.tamaspapp.eu/LogDensityProblems.jl/dev/#LogDensityProblems.capabilities), we use the sample average of the Hessian.
If the target has only first-order capability, we use Stein's identity.

# (Keyword) Arguments
- `stepsize::Float64`: Step size.
- `n_samples::Int`: Number of samples used to estimate the natural gradient. (default: `1`)
- `ensure_posdef::Bool`: Ensure that the updated precision preserves positive definiteness. (default: `true`)
- `subsampling::Union{Nothing,<:AbstractSubsampling}`: Optional subsampling strategy.

!!! note
    The `subsampling` strategy is only applied to the target `LogDensityProblem` but not to the variational approximation `q`. That is, `KLMinNaturalGradDescent` does not support amortization or structured variational families.

# Output
- `q`: The last iterate of the algorithm.

# Callback Signature
The `callback` function supplied to `optimize` needs to have the following signature:

    callback(; rng, iteration, q, info)

The keyword arguments are as follows:
- `rng`: Random number generator internally used by the algorithm.
- `iteration`: The index of the current iteration.
- `q`: Current variational approximation.
- `info`: `NamedTuple` containing the information generated during the current iteration.

# Requirements
- The variational family is [`FullRankGaussian`](@ref FullRankGaussian).
- The target distribution has unconstrained support (\$\$\\mathbb{R}^d\$\$).
- The target `LogDensityProblems.logdensity(prob, x)` has at least first-order differentiation capability.
"""
@kwdef struct KLMinNaturalGradDescent{Sub<:Union{Nothing,<:AbstractSubsampling}} <:
              AbstractVariationalAlgorithm
    stepsize::Float64
    n_samples::Int = 1
    ensure_posdef::Bool = true
    subsampling::Sub = nothing
end

struct KLMinNaturalGradDescentState{Q,P,S,Prec,QCov,GradBuf,HessBuf}
    q::Q
    prob::P
    prec::Prec
    qcov::QCov
    iteration::Int
    sub_st::S
    grad_buf::GradBuf
    hess_buf::HessBuf
end

function init(
    rng::Random.AbstractRNG,
    alg::KLMinNaturalGradDescent,
    q_init::MvLocationScale{<:LowerTriangular,<:Normal,L},
    prob,
) where {L}
    sub = alg.subsampling
    n_dims = LogDensityProblems.dimension(prob)
    capability = LogDensityProblems.capabilities(typeof(prob))
    if capability < LogDensityProblems.LogDensityOrder{1}()
        throw(
            ArgumentError(
                "`KLMinNaturalGradDescent` requires at least first-order differentiation capability. The capability of the supplied `LogDensityProblem` is $(capability).",
            ),
        )
    end
    sub_st = isnothing(sub) ? nothing : init(rng, sub)
    grad_buf = Vector{eltype(q_init.location)}(undef, n_dims)
    hess_buf = Matrix{eltype(q_init.location)}(undef, n_dims, n_dims)
    scale = q_init.scale
    qcov = Hermitian(scale*scale')
    scale_inv = inv(scale)
    prec_chol = scale_inv'
    prec = Hermitian(prec_chol*prec_chol')
    return KLMinNaturalGradDescentState(
        q_init, prob, prec, qcov, 0, sub_st, grad_buf, hess_buf
    )
end

output(::KLMinNaturalGradDescent, state) = state.q

function step(
    rng::Random.AbstractRNG,
    alg::KLMinNaturalGradDescent,
    state,
    callback,
    objargs...;
    kwargs...,
)
    (; ensure_posdef, n_samples, stepsize, subsampling) = alg
    (; q, prob, prec, qcov, iteration, sub_st, grad_buf, hess_buf) = state

    m = mean(q)
    S = prec
    η = convert(eltype(m), stepsize)
    iteration += 1

    # Maybe apply subsampling
    prob_sub, sub_st′, sub_inf = if isnothing(subsampling)
        prob, sub_st, NamedTuple()
    else
        batch, sub_st′, sub_inf = step(rng, subsampling, sub_st)
        prob_sub = subsample(prob, batch)
        prob_sub, sub_st′, sub_inf
    end

    logπ_avg, grad_buf, hess_buf = gaussian_expectation_gradient_and_hessian!(
        rng, q, n_samples, grad_buf, hess_buf, prob_sub
    )

    S′ = if ensure_posdef
        # Udpate rule guaranteeing positive definiteness in the proof of Theorem 1.
        # Lin, W., Schmidt, M., & Khan, M. E.
        # Handling the positive-definite constraint in the Bayesian learning rule.
        # In ICML 2020.
        G_hat = S - Symmetric(-hess_buf)
        Hermitian(S - η*G_hat + η^2/2*G_hat*qcov*G_hat)
    else
        Hermitian(((1 - η) * S + η * Symmetric(-hess_buf)))
    end
    m′ = m - η * (S′ \ (-grad_buf))

    prec_chol = cholesky(S′).L
    prec_chol_inv = inv(prec_chol)
    scale = prec_chol_inv'
    qcov = Hermitian(scale*scale')
    q′ = MvLocationScale(m′, scale, q.dist)

    state = KLMinNaturalGradDescentState(
        q′, prob, S′, qcov, iteration, sub_st′, grad_buf, hess_buf
    )
    elbo = logπ_avg + entropy(q′)
    info = merge((elbo=elbo,), sub_inf)

    if !isnothing(callback)
        info′ = callback(; rng, iteration, q=q′, info)
        info = !isnothing(info′) ? merge(info′, info) : info
    end
    state, false, info
end

"""
    estimate_objective([rng,] alg, q, prob; n_samples)

Estimate the ELBO of the variational approximation `q` against the target log-density `prob`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `alg::KLMinNaturalGradDescent`: Variational inference algorithm.
- `q::MvLocationScale{<:Any,<:Normal,<:Any}`: Gaussian variational approximation.
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.

# Keyword Arguments
- `n_samples::Int`: Number of Monte Carlo samples for estimating the objective. (default: Same as the the number of samples used for estimating the gradient during optimization.)

# Returns
- `obj_est`: Estimate of the objective value.
"""
function estimate_objective(
    rng::Random.AbstractRNG,
    alg::KLMinNaturalGradDescent,
    q::MvLocationScale{S,<:Normal,L},
    prob;
    n_samples::Int=alg.n_samples,
) where {S,L}
    obj = RepGradELBO(n_samples; entropy=MonteCarloEntropy())
    if isnothing(alg.subsampling)
        return estimate_objective(rng, obj, q, prob)
    else
        sub = alg.subsampling
        sub_st = init(rng, sub)
        return mapreduce(+, 1:length(sub)) do _
            batch, sub_st, _ = step(rng, sub, sub_st)
            prob_sub = subsample(prob, batch)
            estimate_objective(rng, obj, q, prob_sub) / length(sub)
        end
    end
end
