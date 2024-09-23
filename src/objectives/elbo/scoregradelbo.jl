"""
	ScoreGradELBO(n_samples; kwargs...)

Evidence lower-bound objective computed with score function gradients. 
```math
\\begin{aligned}
\\nabla_{\\lambda} \\mathrm{ELBO}\\left(\\lambda\\right)
&\\=
\\mathbb{E}_{z \\sim q_{\\lambda}}\\left[
  \\log \\pi\\left(z\\right) \\nabla_{\\lambda} \\log q_{\\lambda}(z)
\\right]
+ \\mathbb{H}\\left(q_{\\lambda}\\right),
\\end{aligned}
```

To reduce the variance of the gradient estimator, we use a baseline computed from a running average of the previous ELBO values and subtract it from the objective.

```math
\\mathbb{E}_{z \\sim q_{\\lambda}}\\left[
  \\nabla_{\\lambda} \\log q_{\\lambda}(z) \\left(\\pi\\left(z\\right) - \\beta\\right)
\\right]
```

# Arguments
- `n_samples::Int`: Number of Monte Carlo samples used to estimate the ELBO.

# Keyword Arguments
- `entropy`: The estimator for the entropy term. (Type `<: AbstractEntropyEstimator`; Default: `ClosedFormEntropy()`)
- `baseline_window_size::Int`: The window size to use to compute the baseline. (Default: `10`)
- `baseline_history::Vector{Float64}`: The history of the baseline. (Default: `Float64[]`)

# Requirements
- The variational approximation ``q_{\\lambda}`` implements `rand` and `logpdf`.
- `logpdf(q, x)` must be differentiable with respect to `q` by the selected AD backend.
- The target distribution and the variational approximation have the same support.

Depending on the options, additional requirements on ``q_{\\lambda}`` may apply.
"""
struct ScoreGradELBO{EntropyEst <: AbstractEntropyEstimator} <:
	   AdvancedVI.AbstractVariationalObjective
	entropy::EntropyEst
	n_samples::Int
    baseline_window_size::Int
    baseline_history::Vector{Float64}
end

function ScoreGradELBO(
	n_samples::Int;
	entropy::AbstractEntropyEstimator = ClosedFormEntropy(),
    baseline_window_size::Int = 10,
    baseline_history::Vector{Float64} = Float64[]
)
	ScoreGradELBO(entropy, n_samples, baseline_window_size, baseline_history)
end


function Base.show(io::IO, obj::ScoreGradELBO)
	print(io, "ScoreGradELBO(entropy=")
	print(io, obj.entropy)
	print(io, ", n_samples=")
	print(io, obj.n_samples)
    print(io, ", baseline_window_size=")
    print(io, obj.baseline_window_size)
	print(io, ")")
end

function compute_control_variate_baseline(history, window_size)
    if length(history) == 0
        return 1.0
    end
    min_index = max(1, length(history) - window_size)
    return mean(history[min_index:end])
end

function estimate_energy_with_samples(prob, samples_stop, samples_logprob, samples_logprob_stop, baseline)
	fv = Base.Fix1(LogDensityProblems.logdensity, prob).(eachsample(samples_stop))
	fv_mean = mean(fv) 
	score_grad = mean(@. samples_logprob * (fv - baseline))
	score_grad_stop = mean(@. samples_logprob_stop * (fv - baseline))
	return fv_mean + (score_grad - score_grad_stop)
end

function compute_elbo(q, q_stop, samples_stop, entropy, problem, baseline)
	samples_logprob = logpdf.(Ref(q), AdvancedVI.eachsample(samples_stop))
	samples_logprob_stop = logpdf.(Ref(q_stop), AdvancedVI.eachsample(samples_stop))
	energy = estimate_energy_with_samples(problem, samples_stop, samples_logprob, samples_logprob_stop, baseline)
	elbo = energy + entropy
	return elbo
end

function estimate_objective(
	rng::Random.AbstractRNG,
	obj::ScoreGradELBO,
	q,
	prob;
	n_samples::Int = obj.n_samples,
)
    samples, entropy = reparam_with_entropy(rng, q, q, obj.n_samples, obj.entropy)
	energy = map(Base.Fix1(LogDensityProblems.logdensity, prob), eachsample(samples))
	return mean(energy) + entropy
end

function estimate_objective(
	obj::ScoreGradELBO, q, prob; n_samples::Int = obj.n_samples)
	estimate_objective(Random.default_rng(), obj, q, prob; n_samples)
end

function estimate_scoregradelbo_ad_forward(params′, aux)
	@unpack rng, obj, problem, restructure, q_stop = aux
    baseline = compute_control_variate_baseline(obj.baseline_history, obj.baseline_window_size)
	q = restructure(params′)
    samples_stop = rand(rng, q_stop, obj.n_samples)
    entropy = estimate_entropy_maybe_stl(obj.entropy, samples_stop, q, q_stop)
	elbo = compute_elbo(q, q_stop, samples_stop, entropy, problem, baseline)
	return -elbo
end


function AdvancedVI.estimate_gradient!(
	rng::Random.AbstractRNG,
	obj::ScoreGradELBO,
	adtype::ADTypes.AbstractADType,
	out::DiffResults.MutableDiffResult,
	prob,
	params,
	restructure,
	state,
)
	q_stop = restructure(params)
	aux = (rng=rng, obj=obj, problem=prob, restructure=restructure, q_stop=q_stop)
	AdvancedVI.value_and_gradient!(
		adtype, estimate_scoregradelbo_ad_forward, params, aux, out,
	)
	nelbo = DiffResults.value(out)
	stat = (elbo = -nelbo,)
    push!(obj.baseline_history, -nelbo)
	out, nothing, stat
end
