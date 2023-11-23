
"""
    RepGradELBO(n_samples; kwargs...)

Evidence lower-bound objective with the reparameterization gradient formulation[^TL2014][^RMW2014][^KW2014].
This computes the evidence lower-bound (ELBO) through the formulation:
```math
\\begin{aligned}
\\mathrm{ELBO}\\left(\\lambda\\right)
&\\triangleq
\\mathbb{E}_{z \\sim q_{\\lambda}}\\left[
  \\log \\pi\\left(z\\right)
\\right]
+ \\mathbb{H}\\left(q_{\\lambda}\\right),
\\end{aligned}
```

# Arguments
- `n_samples::Int`: Number of Monte Carlo samples used to estimate the ELBO.

# Keyword Arguments
- `entropy`: The estimator for the entropy term. (Type `<: AbstractEntropyEstimator`; Default: ClosedFormEntropy())

# Requirements
- ``q_{\\lambda}`` implements `rand`.
- The target `logdensity(prob, x)` must be differentiable wrt. `x` by the selected AD backend.

Depending on the options, additional requirements on ``q_{\\lambda}`` may apply.

# References
[^TL2014]: Titsias, M., & Lázaro-Gredilla, M. (2014, June). Doubly stochastic variational Bayes for non-conjugate inference. In ICML.
[^RMW2014]: Rezende, D. J., Mohamed, S., & Wierstra, D. (2014, June). Stochastic backpropagation and approximate inference in deep generative models. In ICML.
[^KW2014]: Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. In ICLR.
"""
struct RepGradELBO{EntropyEst <: AbstractEntropyEstimator} <: AbstractVariationalObjective
    entropy  ::EntropyEst
    n_samples::Int
end

RepGradELBO(
    n_samples::Int;
    entropy  ::AbstractEntropyEstimator = ClosedFormEntropy()
) = RepGradELBO(entropy, n_samples)

Base.show(io::IO, obj::RepGradELBO) =
    print(io, "RepGradELBO(entropy=$(obj.entropy), n_samples=$(obj.n_samples))")

function estimate_entropy_maybe_stl(entropy_estimator::AbstractEntropyEstimator, samples, q, q_stop)
    q_maybe_stop = maybe_stop_entropy_score(entropy_estimator, q, q_stop)
    estimate_entropy(entropy_estimator, samples, q_maybe_stop)
end

function estimate_energy_with_samples(::RepGradELBO, samples, prob)
    mean(Base.Fix1(LogDensityProblems.logdensity, prob), eachsample(samples))
end

function estimate_repgradelbo_maybe_stl_with_samples(
    obj::RepGradELBO, q, q_stop, samples::AbstractMatrix, prob
)
    energy  = estimate_energy_with_samples(obj, samples, prob)
    entropy = estimate_entropy_maybe_stl(obj.entropy, samples, q, q_stop)
    energy + entropy
end

function estimate_repgradelbo_maybe_stl(rng::Random.AbstractRNG, obj::RepGradELBO, q, q_stop, prob)
    samples = rand(rng, q, obj.n_samples)
    estimate_repgradelbo_maybe_stl_with_samples(obj, q, q_stop, samples, prob)
end

function estimate_objective(
    rng::Random.AbstractRNG,
    obj::RepGradELBO,
    q,
    prob;
    n_samples::Int = obj.n_samples
)
    samples = rand(rng, q, n_samples)
    estimate_repgradelbo_maybe_stl_with_samples(obj, q, q, samples, prob)
end

estimate_objective(obj::RepGradELBO, q, prob; n_samples::Int = obj.n_samples) =
    estimate_objective(Random.default_rng(), obj, q, prob; n_samples)

function estimate_gradient!(
    rng      ::Random.AbstractRNG,
    obj      ::RepGradELBO,
    adbackend::ADTypes.AbstractADType,
    out      ::DiffResults.MutableDiffResult,
    prob,
    λ,
    restructure,
    est_state,
)
    q_stop = restructure(λ)
    function f(λ′)
        q = restructure(λ′)
        elbo = estimate_repgradelbo_maybe_stl(rng, obj, q, q_stop, prob)
        -elbo
    end
    value_and_gradient!(adbackend, f, λ, out)

    nelbo = DiffResults.value(out)
    stat  = (elbo=-nelbo,)

    out, nothing, stat
end
