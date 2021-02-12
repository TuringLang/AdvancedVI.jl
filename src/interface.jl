"""
    optimize!([alg::VariationalInference{AD}, q::VariationalPosterior, model::Model], θ]; optimizer = TruncatedADAGrad())

Iteratively updates parameters by calling `grad!` and using the given `optimizer` to compute
the steps.
"""
function optimize!(
    alg::VariationalInference,
    q,
    model,
    optimizer = TruncatedADAGrad();
    hyperparams = nothing,
    optimizer_hp = nothing,
)
    # TODO: should we always assume `samples_per_step` and `max_iters` for all algos?
    alg_name = alg_str(alg)
    samples_per_step = nsamples(alg)
    max_iters = alg.max_iters
    
    # TODO: really need a better way to warn the user about potentially
    # not using the correct accumulator
    if (optimizer isa TruncatedADAGrad) && (θ ∉ keys(optimizer.acc))
        # this message should only occurr once in the optimization process
        @info "[$alg_name] Should only be seen once: optimizer created for θ" objectid(θ)
    end
    x₀ = rand(q, samples_per_step) # Preallocating x₀
    x = similar(x₀) # Preallocating x
    diff_result = DiffResults.GradientResult(x)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    # add criterion? A running mean maybe?
    time_elapsed = @elapsed while (i < max_iters) # & converged
        logπ = makelogπ(model, hyperparams)
        step!(alg, q, logπ, x₀, x, diff_result)

        AdvancedVI.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return θ
end

## Generic evaluation of the free energy
function free_energy(alg, q, logπ)
    return eval_expec_logπ(alg, q, logπ) - eval_entropy(alg, q)
end

## Generic evaluation of the expectation
function eval_expec_logπ(alg, q, logπ)
    mean(logπ, eachcol(rand(q, nsamples(alg))))
end