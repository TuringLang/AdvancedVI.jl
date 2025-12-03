
"""
    FisherMinBatchMatch(n_samples, subsampling)
    FisherMinBatchMatch(; n_samples, subsampling)

Covariance-weighted Fisher divergence minimization via the batch-and-match algorithm, which is a proximal point-type optimization scheme.

# (Keyword) Arguments
- `n_samples::Int`: Number of samples (batchsize) used to compute the moments required for the batch-and-match update. (default: `32`)
- `subsampling::Union{Nothing,<:AbstractSubsampling}`: Optional subsampling strategy. (default: `nothing`)

!!! warning
    `FisherMinBatchMatch` with subsampling enabled results in a biased algorithm and may not properly optimize the covariance-weighted Fisher divergence.

!!! note
    `FisherMinBatchMatch` requires a sufficiently large `n_samples` to converge quickly.

!!! note
    The `subsampling` strategy is only applied to the target `LogDensityProblem` but not to the variational approximation `q`. That is, `FisherMinBatchMatch` does not support amortization or structured variational families.

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
- The target distribution has unconstrained support.
- The target `LogDensityProblems.logdensity(prob, x)` has at least first-order differentiation capability.
"""
@kwdef struct FisherMinBatchMatch{Sub<:Union{Nothing,<:AbstractSubsampling}} <:
              AbstractVariationalAlgorithm
    n_samples::Int = 32
    subsampling::Sub = nothing
end

struct BatchMatchState{Q,P,Sigma,Sub,UBuf,GradBuf}
    q::Q
    prob::P
    sigma::Sigma
    iteration::Int
    sub_st::Sub
    u_buf::UBuf
    grad_buf::GradBuf
end

function init(
    rng::Random.AbstractRNG,
    alg::FisherMinBatchMatch,
    q::MvLocationScale{<:LowerTriangular,<:Normal,L},
    prob,
) where {L}
    (; n_samples, subsampling) = alg
    capability = LogDensityProblems.capabilities(typeof(prob))
    if capability < LogDensityProblems.LogDensityOrder{1}()
        throw(
            ArgumentError(
                "`FisherMinBatchMatch` requires at least first-order differentiation capability. The capability of the supplied `LogDensityProblem` is $(capability).",
            ),
        )
    end
    sub_st = isnothing(subsampling) ? nothing : init(rng, subsampling)
    params, _ = Optimisers.destructure(q)
    n_dims = LogDensityProblems.dimension(prob)
    u_buf = Matrix{eltype(params)}(undef, n_dims, n_samples)
    grad_buf = Matrix{eltype(params)}(undef, n_dims, n_samples)
    return BatchMatchState(q, prob, cov(q), 0, sub_st, u_buf, grad_buf)
end

output(::FisherMinBatchMatch, state) = state.q

function rand_batch_match_samples_with_objective!(
    rng::Random.AbstractRNG,
    q::MvLocationScale,
    n_samples::Int,
    prob,
    u_buf=Matrix{eltype(q)}(undef, LogDensityProblems.dimension(prob), n_samples),
    grad_buf=Matrix{eltype(q)}(undef, LogDensityProblems.dimension(prob), n_samples),
)
    μ = q.location
    C = q.scale
    u = Random.randn!(rng, u_buf)
    z = C*u .+ μ
    logπ_sum = zero(eltype(μ))
    for b in 1:n_samples
        zb = if use_view_in_gradient(prob)
            view(z, :, b)
        else
            z[:, b]
        end
        logπb, gb = LogDensityProblems.logdensity_and_gradient(prob, zb)
        grad_buf[:, b] = gb
        logπ_sum += logπb
    end
    logπ_avg = logπ_sum/n_samples

    # Estimate objective values
    #
    # F = E[| ∇log(q/π) (z) |_{CC'}^2] (definition)
    #   = E[| C' (∇logq(z) - ∇logπ(z)) |^2] (Σ = CC')
    #   = E[| C' ( -(CC')\((Cu + μ) - μ) - ∇logπ(z)) |^2] (z = Cu + μ)
    #   = E[| C' ( -(CC')\(Cu) - ∇logπ(z)) |^2]
    #   = E[| -u - C'∇logπ(z)) |^2]
    fisher = sum(abs2, -u_buf - (C'*grad_buf))/n_samples

    return u_buf, z, grad_buf, fisher, logπ_avg
end

function step(
    rng::Random.AbstractRNG,
    alg::FisherMinBatchMatch,
    state,
    callback,
    objargs...;
    kwargs...,
)
    (; n_samples, subsampling) = alg
    (; q, prob, sigma, iteration, sub_st, u_buf, grad_buf) = state

    d = LogDensityProblems.dimension(prob)
    μ = q.location
    C = q.scale
    Σ = sigma
    iteration += 1

    # Maybe apply subsampling
    prob_sub, sub_st′, sub_inf = if isnothing(subsampling)
        prob, sub_st, NamedTuple()
    else
        batch, sub_st′, sub_inf = step(rng, subsampling, sub_st)
        prob_sub = subsample(prob, batch)
        prob_sub, sub_st′, sub_inf
    end

    u_buf, z, grad_buf, fisher, logπ_avg = rand_batch_match_samples_with_objective!(
        rng, q, n_samples, prob_sub, u_buf, grad_buf
    )

    # BaM updates
    zbar, C = mean_and_cov(z, 2)
    gbar, Γ = mean_and_cov(grad_buf, 2)

    μmz = μ - zbar
    λ = convert(eltype(μ), d*n_samples / iteration)

    U = Symmetric(λ*Γ + (λ/(1 + λ)*gbar)*gbar')
    V = Symmetric(Σ + λ*C + (λ/(1 + λ)*μmz)*μmz')

    Σ′ = Hermitian(2*V/(I + real(sqrt(I + 4*U*V))))
    μ′ = 1/(1 + λ)*μ + λ/(1 + λ)*(Σ′*gbar + zbar)
    q′ = MvLocationScale(μ′[:, 1], cholesky(Σ′).L, q.dist)

    elbo = logπ_avg + entropy(q)
    info = (iteration=iteration, covweighted_fisher=fisher, elbo=elbo)

    state = BatchMatchState(q′, prob, Σ′, iteration, sub_st′, u_buf, grad_buf)

    if !isnothing(callback)
        info′ = callback(; rng, iteration, q, state)
        info = !isnothing(info′) ? merge(info′, info) : info
    end
    state, false, info
end

"""
    estimate_objective([rng,] alg, q, prob; n_samples)

Estimate the covariance-weighted Fisher divergence of the variational approximation `q` against the target log-density `prob`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `alg::FisherMinBatchMatch`: Variational inference algorithm.
- `q::MvLocationScale{<:Any,<:Normal,<:Any}`: Gaussian variational approximation.
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.

# Keyword Arguments
- `n_samples::Int`: Number of Monte Carlo samples for estimating the objective. (default: Same as the the number of samples used for estimating the gradient during optimization.)

# Returns
- `obj_est`: Estimate of the objective value.
"""
function estimate_objective(
    rng::Random.AbstractRNG,
    alg::FisherMinBatchMatch,
    q::MvLocationScale{S,<:Normal,L},
    prob;
    n_samples::Int=alg.n_samples,
) where {S,L}
    _, _, _, fisher, _ = rand_batch_match_samples_with_objective!(rng, q, n_samples, prob)
    return fisher
end
