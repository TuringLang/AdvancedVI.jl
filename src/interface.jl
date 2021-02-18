"""
    vi([vo, [model, alg::VariationalInference]]; opt, hyperparams, opt_hyperparams)
    vi([vo, [model, alg::VariationalInference, q::VariationalPosterior]]; opt, hyperparams, opt_hyperparams)

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
"""
function vi(vo::VariationalObjective, model, alg::VariationalInference, q; opt=TruncatedADAGrad(), hyperparams=nothing, opt_hyperparams=nothing)
    optimize!(vo, alg, q, model; opt=opt, hyperparams=hyperparams, opt_hyperparams=opt_hyperparams)
    return q
end

function vi(model, alg::VariationalInference, q; opt=TruncatedADAGrad(), hyperparams=nothing, opt_hyperparams=nothing)
    check_compatibility(alg, q)
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
    max_iters = niters(alg)
    
    global state = init(alg, q, opt) # opt is there to be used in the future

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$(alg_str(alg))] Optimizing...", 0)
    else
        0
    end

    # add criterion? A running mean maybe?
    time_elapsed = @elapsed while (i < max_iters) # & converged
        logπ = makelogπ(model, hyperparams)
        step!(vo, alg, q, logπ, state, opt)

        # For debugging this would need to be updated somehow
        # AdvancedVI.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return q = finish(alg, q, state)
end

## Generic solution for most problems
finish(alg, q, state) = q

## Verify that the algorithm can work with the corresponding variational distribution
function check_compatibility(alg, q)
    if !compat(alg, q)
        throw(ArgumentError("Algorithm $(alg) cannot work with distributions of type $(typeof(q)), compatible distributions are: $(compats(alg))"))
    end
end

function compat(alg::VariationalInference, q)
    return q isa compats(alg)
end

function compats(::Any)
    return ()
end 
