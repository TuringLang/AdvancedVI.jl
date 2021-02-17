## First part is for optimization based on `q = f(θ)`

"""
    vi(model, alg::VariationalInference)
    vi(model, alg::VariationalInference, q::VariationalPosterior)
    vi(model, alg::VariationalInference, getq::Function, θ::AbstractArray)

Constructs the variational posterior from the `model` and performs the optimization
following the configuration of the given `VariationalInference` instance.
# Arguments
- `vo` : `VariationalObjective`, `ELBO()` by default
- `model`: `Turing.Model` or `Function` z ↦ log p(x, z) where `x` denotes the observations
- `alg`: the VI algorithm used
- `q`: a `VariationalPosterior` for which it is assumed a specialized implementation of the variational objective used exists.
- `getq`: function taking parameters `θ` as input and returns a `VariationalPosterior`
- `θ`: only required if `getq` is used, in which case it is the initial parameters for the variational posterior
"""
function vi(vo::VariationalObjective, model, alg::ADVI, q, θ_init; opt=TruncatedADAGrad(), hyperparams=nothing, opt_hyperparams=nothing)
    θ = copy(θ_init)
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    # If `q` is a mean-field approx we use the specialized `update` function
    if q isa Distribution
        return update(q, θ)
    else
        # Otherwise we assume it's a mapping θ → q
        return q(θ)
    end
end

function vi(model, alg::ADVI, q, θ_init; opt=TruncatedADAGrad(), hyperparams=nothing, opt_hyperparams=nothing)
    return vi(ELBO(), model, alg, q, θ_init; opt=opt, hyperparams=hyperparams, opt_hyperparams=opt_hyperparams)
end

## Second part is for optimization based on q.θ

function vi(vo::VariationalObjective, model, alg::ADVI, q; opt=TruncatedADAGrad(), hyperparams=nothing, opt_hyperparams=nothing)
    optimize!(vo, alg, q, model; opt=opt, hyperparams=hyperparams, opt_hyperparams=opt_hyperparams)
    return q
end

function vi(model, alg::ADVI, q; opt = TruncatedADAGrad())
    return vi(ELBO(), model, alg, q; opt=opt, hyperparams=hyperparams, opt_hyperparams=opt_hyperparams)
end


"""
    optimize!([vo::VariationalObjective, [alg::VariationalInference{AD}, q::VariationalPosterior, model::Model], θ], ]; optimizer = TruncatedADAGrad())

Iteratively updates parameters by calling `grad!` and using the given `optimizer` to compute
the steps.
"""
function optimize!(
    vo::VariationalObjective,
    alg::VariationalInference,
    q,
    model;
    opt = TruncatedADAGrad(),
    hyperparams = nothing,
    opt_hyperparams = nothing,
)
    # TODO: should we always assume `samples_per_step` and `max_iters` for all algos?
    alg_name = alg_str(alg)
    samples_per_step = nsamples(alg)
    max_iters = niters(alg)
    
    # TODO: really need a better way to warn the user about potentially
    # not using the correct accumulator
    if (opt isa TruncatedADAGrad) && (θ ∉ keys(opt.acc))
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
        step!(alg, q, logπ, x₀, x, diff_result, opt)

        AdvancedVI.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return θ
end
