
"""
    KLMinWassFwdBwd(n_samples, stepsize, subsampling)
    KLMinWassFwdBwd(; n_samples, stepsize, subsampling)

KL divergence minimization by running stochastic proximal gradient descent (forward-backward splitting) in Wasserstein space[^DBCS2023].

Denoting the target log-density as \$\$ \\log \\pi \$\$ and the current variational approximation as \$\$q\$\$, the original algorithm requires estimating the quantity \$\$ \\mathbb{E}_q \\nabla^2 \\log \\pi \$\$. It the target `LogDensityProblem` associated with \$\$ \\log \\pi \$\$ has second-order differentiation [capability](https://www.tamaspapp.eu/LogDensityProblems.jl/dev/#LogDensityProblems.capabilities), we use the sample average of the Hessian. If the target has only first-order capability, we use Stein's identity.

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
- The target `LogDensityProblems.logdensity(prob, x)` has at least first-order differentiation capability.
"""
@kwdef struct KLMinWassFwdBwd{Sub<:Union{Nothing,<:AbstractSubsampling}} <:
              AbstractVariationalAlgorithm
    n_samples::Int = 1
    stepsize::Float64
    subsampling::Sub = nothing
end

"""
    gaussian_expectation_gradient_and_hessian!(rng, q, n_samples, grad_buf, hess_buf, prob)

Estimate the expectations of the gradient and Hessians of the log-density of `prob` taken over the Gaussian `q`. For estimating the expectation of the Hessian, if `prob` has second-order differentiation capability, this function uses the sample average of the Hessian. Otherwise, it uses Stein's identity.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `q::MvLocationScale{<:LowerTriangular,<:Normal,L}`: Gaussian to take expectation over.
- `n_samples::Int`: Number of samples used for estimation.
- `grad_buf::AbstractVector`: Buffer for the gradient estimate.
- `hess_buf::AbstractMatrix`: Buffer for the Hessian estimate.
- `prob`: `LogDensityProblem` associated with the log-density gradient and Hessian subject to expectation.
"""
function gaussian_expectation_gradient_and_hessian!(
    rng::Random.AbstractRNG,
    q::MvLocationScale{<:LowerTriangular,<:Normal,L},
    n_samples::Int,
    grad_buf::AbstractVector{T},
    hess_buf::AbstractMatrix{T},
    prob,
) where {T<:Real,L}
    logπ_avg = zero(T)
    fill!(grad_buf, zero(T))
    fill!(hess_buf, zero(T))

    if LogDensityProblems.capabilities(typeof(prob)) ≤
        LogDensityProblems.LogDensityOrder{1}()
        # Use Stein's identity
        d = LogDensityProblems.dimension(prob)
        u = randn(rng, T, d, n_samples)
        z = q.scale*u .+ q.location
        for b in 1:n_samples
            zb, ub = view(z, :, b), view(u, :, b)
            logπ, ∇logπ = LogDensityProblems.logdensity_and_gradient(prob, zb)
            logπ_avg += logπ/n_samples
            grad_buf += ∇logπ/n_samples
            hess_buf += ub*(∇logπ/n_samples)'
        end
        return logπ_avg, grad_buf, hess_buf
    else
        # Use sample average of the Hessian.
        z = rand(rng, q, n_samples)
        for b in 1:n_samples
            zb = view(z, :, b)
            logπ, ∇logπ, ∇2logπ = LogDensityProblems.logdensity_gradient_and_hessian(prob, zb)
            logπ_avg += logπ/n_samples
            grad_buf += ∇logπ/n_samples
            hess_buf += ∇2logπ/n_samples
        end
        return logπ_avg, grad_buf, hess_buf
    end
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
    if capability < LogDensityProblems.LogDensityOrder{1}()
        throw(
            ArgumentError(
                "`KLMinWassFwdBwd` requires at least first-order differentiation capability. The capability of the supplied `LogDensityProblem` is $(capability).",
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

    # Estimate the Wasserstein gradient
    logπ_avg, grad_buf, hess_buf = gaussian_expectation_gradient_and_hessian!(
        rng, q, n_samples, grad_buf, hess_buf, prob_sub
    )

    m′ = m - η*-grad_buf
    M = I - η*Hermitian(-hess_buf)
    Σ_half = Hermitian(M*Σ*M)

    # Compute the JKO proximal operator
    Σ′ = (Σ_half + 2*η*I + sqrt(Hermitian(Σ_half*(Σ_half + 4*η*I))))/2
    q′ = MvLocationScale(m′, cholesky(Σ′).L, q.dist)

    state = KLMinWassFwdBwdState(q′, prob, Σ′, iteration, sub_st′, grad_buf, hess_buf)
    elbo = logπ_avg + entropy(q′)
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
