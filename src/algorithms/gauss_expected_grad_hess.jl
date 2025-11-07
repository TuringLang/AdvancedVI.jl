
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
            logπ, ∇logπ, ∇2logπ = LogDensityProblems.logdensity_gradient_and_hessian(
                prob, zb
            )
            logπ_avg += logπ/n_samples
            grad_buf += ∇logπ/n_samples
            hess_buf += ∇2logπ/n_samples
        end
        return logπ_avg, grad_buf, hess_buf
    end
end
