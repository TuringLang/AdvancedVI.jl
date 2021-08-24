"""
    vi([rng::AbstractRNG, [vo::VariationalObjective]], model, alg::VariationalInference, q::VariationalPosterior; opt, hyperparams, opt_hyperparams)::VariationalPosterior

Constructs the variational posterior from the `model` and performs the optimization
following the configuration of the given `VariationalInference` instance.
## Arguments
- `vo` : `VariationalObjective`, `ELBO()` by default
- `model`: `Turing.Model` or `Function` z ↦ log p(x, z) where `x` denotes the observations
- `alg`: the VI algorithm used
- `q`: a `VariationalPosterior` for which it is assumed a specialized implementation of the variational objective used exists.
## Keyword Arguments
- `opt` : Optimiser (from `Flux.Optimise`) used to update the variational parameters
- `hyperparams` : Hyperparameters, if different than nothing, `model(hyperparams)` will be called to obatin the logjoint
- `opt_hyperparams` : Optimiser for the Hyperparameters

    vi([rng::AbstractRNG, [vo::VariationalObjective]], model, alg::VariationalInference, q::Function, θ::AbstractVector; opt, hyperparams, opt_hyperparams)::AbstractVector

Constructs the variational posterior from the `model` and performs the optimization
following the configuration of the given `VariationalInference` instance.
## Arguments
- `vo` : `VariationalObjective`, `ELBO()` by default
- `model`: `Turing.Model` or `Function` z ↦ log p(x, z) where `x` denotes the observations
- `alg`: the VI algorithm used
- `q`: a function creating a distribution from the parameters `θ`
- `θ`: the variational parameters
## Keyword Arguments
- `opt` : Optimiser (from `Flux.Optimise`) used to update the variational parameters
- `hyperparams` : Hyperparameters, if different than nothing, `model(hyperparams)` will be called to obatin the logjoint
- `opt_hyperparams` : Optimiser for the Hyperparameters

"""
function vi(
    rng::AbstractRNG,
    vo::VariationalObjective,
    model,
    alg::VariationalInference,
    q;
    opt=TruncatedADAGrad(),
    hyperparams=nothing,
    opt_hyperparams=nothing,
)
    θ, to_dist = flatten(q)
    θ = vi(
        rng,
        vo,
        alg,
        to_dist,
        θ,
        model;
        opt=opt,
        hyperparams=hyperparams,
        opt_hyperparams=opt_hyperparams,
    )
    return to_dist(θ)
end

function vi(
    vo::VariationalObjective,
    model,
    alg::VariationalInference,
    q;
    opt=TruncatedADAGrad(),
    hyperparams=nothing,
    opt_hyperparams=nothing,
)
    return vi(
        GLOBAL_RNG,
        vo,
        model,
        alg,
        q;
        opt=opt,
        hyperparams=hyperparams,
        opt_hyperparams=opt_hyperparams,
    )
end

function vi(
    model,
    alg::VariationalInference,
    q;
    opt=TruncatedADAGrad(),
    hyperparams=nothing,
    opt_hyperparams=nothing,
)
    return vi(
        ELBO(),
        model,
        alg,
        q;
        opt=opt,
        hyperparams=hyperparams,
        opt_hyperparams=opt_hyperparams,
    )
end

function vi(
    rng::AbstractRNG,
    vo::VariationalObjective,
    model,
    alg::VariationalInference,
    q,
    θ::AbstractVector;
    opt=TruncatedADAGrad(),
    hyperparams=nothing,
    opt_hyperparams=nothing,
)
    return optimize!(
        rng,
        vo,
        alg,
        q,
        θ,
        model;
        opt=opt,
        hyperparams=hyperparams,
        opt_hyperparams=opt_hyperparams,
    )
end

function vi(
    vo::VariationalObjective,
    model,
    alg::VariationalInference,
    q,
    θ::AbstractVector;
    opt=TruncatedADAGrad(),
    hyperparams=nothing,
    opt_hyperparams=nothing,
)
    return vi(
        GLOBAL_RNG,
        vo,
        model,
        alg,
        q,
        θ;
        opt=opt,
        hyperparams=hyperparams,
        opt_hyperparams=opt_hyperparams,
    )
end

function vi(
    model,
    alg::VariationalInference,
    q,
    θ;
    opt=TruncatedADAGrad(),
    hyperparams=nothing,
    opt_hyperparams=nothing,
)
    return vi(
        ELBO(),
        model,
        alg,
        q,
        θ;
        opt=opt,
        hyperparams=hyperparams,
        opt_hyperparams=opt_hyperparams,
    )
end



"""
    optimize!([vo::VariationalObjective, [alg::VariationalInference{AD}, q::VariationalPosterior, model::Model], θ], ]; optimizer = TruncatedADAGrad())

Iteratively updates parameters by calling `grad!` and using the given `optimizer` to compute
the steps.
"""
function optimize!(
    rng::AbstractRNG,
    vo::VariationalObjective,
    alg::VariationalInference,
    to_dist,
    θ,
    model;
    opt=TruncatedADAGrad(),
    hyperparams=nothing,
    opt_hyperparams=nothing,
)
    max_iters = maxiters(alg)

    state = init(rng, alg, to_dist, θ, opt) # opt is there to be used in the future

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$(alg_str(alg))] Optimizing...", 0)
    else
        0
    end

    # add criterion? A running mean maybe?
    time_elapsed = @elapsed while (i < max_iters) # & converged
        logπ = makelogπ(model, hyperparams)
        step!(rng, vo, alg, to_dist, θ, logπ, state, opt)

        # For debugging this would need to be updated somehow
        # AdvancedVI.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return θ
end