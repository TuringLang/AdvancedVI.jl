
"""
    gaussian_expectation_gradient_and_hessian!(rng, q, n_samples, grad_buf, hess_buf, prob)

Estimate the expectations of the gradient and Hessians of the log-density of `prob` taken over the Gaussian `q`.
For estimating the expectation of the Hessian, if `prob` has second-order differentiation capability, this function uses the sample average of the Hessian.
Otherwise, it uses Stein's identity.

!!! warning
    The resulting `hess_buf` may not be perfectly symmetric due to numerical issues. It is therefore useful to wrap it in a `Symmetric` before usage.

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
    q::MvLocationScale{<:LinearAlgebra.AbstractTriangular,<:Normal,L},
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
        # First-order-only: use Stein/Price identity for the Hessian
        #
        #   E_{z ~ N(m, CC')} ∇2 log π(z)
        #   = E_{z ~ N(m, CC')} (CC')^{-1} (z - m) ∇ log π(z)T
        #   = E_{u ~ N(0, I)} C' \ (u ∇ log π(z)T) .
        # 
        # Algorithmically, draw u ~ N(0, I), z = C u + m, where C = q.scale.
        # Accumulate A = E[ u ∇ log π(z)T ], then map back: H = C \ A.
        d = LogDensityProblems.dimension(prob)
        u = randn(rng, T, d, n_samples)
        m, C = q.location, q.scale
        z = C*u .+ m
        for b in 1:n_samples
            zb, ub = if use_view_in_gradient(prob)
                view(z, :, b), view(u, :, b)
            else
                z[:, b], u[:, b]
            end
            logπ, ∇logπ = LogDensityProblems.logdensity_and_gradient(prob, zb)
            logπ_avg += logπ/n_samples

            rdiv!(∇logπ, n_samples)
            ∇logπ_div_nsamples = ∇logπ

            grad_buf[:] .+= ∇logπ_div_nsamples
            hess_buf[:, :] .+= ub*∇logπ_div_nsamples'
        end
        hess_buf[:, :] .= C' \ hess_buf
        return logπ_avg, grad_buf, hess_buf
    else
        # Second-order: use naive sample average
        z = rand(rng, q, n_samples)
        for b in 1:n_samples
            zb = if use_view_in_gradient(prob)
                view(z, :, b)
            else
                z[:, b]
            end
            logπ, ∇logπ, ∇2logπ = LogDensityProblems.logdensity_gradient_and_hessian(
                prob, zb
            )

            rdiv!(∇logπ, n_samples)
            ∇logπ_div_nsamples = ∇logπ

            rdiv!(∇2logπ, n_samples)
            ∇2logπ_div_nsamples = ∇2logπ

            logπ_avg += logπ/n_samples
            grad_buf[:] .+= ∇logπ_div_nsamples
            hess_buf[:, :] .+= ∇2logπ_div_nsamples
        end
        return logπ_avg, grad_buf, hess_buf
    end
end
