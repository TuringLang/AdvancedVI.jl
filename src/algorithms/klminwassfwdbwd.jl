
"""
    KLMinWassFwdBwd(n_samples, stepsize, subsampling)
    KLMinWassFwdBwd(; n_samples, stepsize, subsampling)

KL divergence minimization by running stochastic proximal gradient descent (forward-backward splitting) in Wasserstein space[^DBCS2023].

# (Keyword) Arguments
- `n_samples::Int`: Number of samples used to estimate the Wasserstein gradient. (default: `1`)
- `stepsize::Float64`: Step size of stochastic proximal gradient descent.
- `subsampling::Union{Nothing,<:AbstractSubsampling}`: Optional subsampling strategy.

!!! note
    The `subsampling` strategy is only applied to the target `LogDensityProblem` but not to the variational approximation `q`. That is, `KLMinWassFwdBwd` does not support amortization or structured variational families.

# Output
- `q`: The last iterate of the algorithm.

# Callback
The callback function `callback` has a signature of

    callback(; rng, iteration, q, info)

The arguments are as follows:
- `rng`: Random number generator internally used by the algorithm.
- `iteration`: The index of the current iteration.
- `q`: Current variational approximation.
- `info`: `NamedTuple` containing the information generated during the current iteration.

# Requirements
- The variational family is [`FullRankGaussian`](@ref FullRankGaussian).
- The target distribution has unconstrained support (\$\$\\mathbb{R}^d\$\$).
- The target `LogDensityProblems.logdensity(prob, x)` has second-order differentiation capability. (`KLMinWassFwdBwd` uses Hessians of the log-density.)
"""
@kwdef struct KLMinWassFwdBwd{Sub<:Union{Nothing,<:AbstractSubsampling}} <:
              AbstractVariationalAlgorithm
    n_samples::Int = 1
    stepsize::Float64
    subsampling::Sub = nothing
end

struct KLMinWassFwdBwdState{Q,P,S,Sigma,GradBuf,HessBuf}
    q::Q
    prob::P
    sigma::Sigma
    iteration::Int
    sub_st::S
    grad_buf::GradBuf
    hess_buf::HessBuf
end

function init(
    rng::Random.AbstractRNG,
    alg::KLMinWassFwdBwd,
    q_init::MvLocationScale{<:LowerTriangular,<:Normal,L},
    prob,
) where {L}
    sub = alg.subsampling
    n_dims = LogDensityProblems.dimension(prob)
    capability = LogDensityProblems.capabilities(typeof(prob))
    if capability < LogDensityProblems.LogDensityOrder{2}()
        throw(
            ArgumentError(
                "`KLMinWassFwdBwd` requires second-order differentiation capability. The capability of the supplied `LogDensityProblem` is $(capability).",
            ),
        )
    end
    sub_st = isnothing(sub) ? nothing : init(rng, sub)
    grad_buf = Vector{eltype(q_init.location)}(undef, n_dims)
    hess_buf = Matrix{eltype(q_init.location)}(undef, n_dims, n_dims)
    return KLMinWassFwdBwdState(q_init, prob, cov(q_init), 0, sub_st, grad_buf, hess_buf)
end

output(::KLMinWassFwdBwd, state) = state.q

function step(
    rng::Random.AbstractRNG, alg::KLMinWassFwdBwd, state, callback, objargs...; kwargs...
)
    (; n_samples, stepsize, subsampling) = alg
    (; q, prob, sigma, iteration, sub_st, grad_buf, hess_buf) = state

    m = mean(q)
    Σ = sigma
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

    # Estimate the moments required for computing the Wasserstein gradient
    z = rand(rng, q, n_samples)
    V_avg = 0
    fill!(grad_buf, zero(eltype(grad_buf)))
    fill!(hess_buf, zero(eltype(hess_buf)))
    for b in 1:n_samples
        negVb, neg∇Vb, neg∇2Vb = LogDensityProblems.logdensity_gradient_and_hessian(
            prob_sub, z[:, b]
        )
        V_avg += -negVb/n_samples
        grad_buf += -neg∇Vb/n_samples
        hess_buf += -neg∇2Vb/n_samples
    end

    m′ = m - η*grad_buf
    M = I - η*Hermitian(hess_buf)
    Σ_half = Hermitian(M*Σ*M)

    # Compute the JKO proximal operator
    Σ′ = (Σ_half + 2*η*I + sqrt(Hermitian(Σ_half*(Σ_half + 4*η*I))))/2
    q′ = MvLocationScale(m′, cholesky(Σ′).L, q.dist)

    state = KLMinWassFwdBwdState(q′, prob, Σ′, iteration, sub_st′, grad_buf, hess_buf)
    elbo = -V_avg + entropy(q′)
    info = merge((elbo=elbo,), sub_inf)

    if !isnothing(callback)
        info′ = callback(; rng, iteration, q, info)
        info = !isnothing(info′) ? merge(info′, info) : info
    end
    state, false, info
end

"""
    estimate_objective([rng,] alg, q, prob; n_samples)

Estimate the ELBO of the variational approximation `q` against the target log-density `prob`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `alg::KLMinWassFwdBwd`: Variational inference algorithm.
- `q::MvLocationScale{<:Any,<:Normal,<:Any}`: Gaussian variational approximation.
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.

# Keyword Arguments
- `n_samples::Int`: Number of Monte Carlo samples for estimating the objective. (default: Same as the the number of samples used for estimating the gradient during optimization.)

# Returns
- `obj_est`: Estimate of the objective value.
"""
function estimate_objective(
    rng::Random.AbstractRNG,
    alg::KLMinWassFwdBwd,
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
