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
function vi(vo::VariationalObjective, model, alg::VariationalInference, q, θ_init; opt=TruncatedADAGrad(), hyperparams=nothing, opt_hyperparams=nothing)
    θ = copy(θ_init)
    optimize!(vo, model, alg, q, θ; opt = opt, hyperparams=hyperparams, opt_hyperparams=opt_hyperparams)
    return q(θ)
end

function vi(model, alg::VariationalInference, q, θ_init; opt=TruncatedADAGrad(), hyperparams=nothing, opt_hyperparams=nothing)
    return vi(ELBO(), model, alg, q, θ_init; opt=opt, hyperparams=hyperparams, opt_hyperparams=opt_hyperparams)
end

function optimize!(
    vo::VariationalObjective,
    model,
    alg::VariationalInference,
    q,
    θ;
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
    diff_result = DiffResults.GradientResult(θ)
    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    # add criterion? A running mean maybe?
    time_elapsed = @elapsed while (i < max_iters) # & converged
        logπ = makelogπ(model, hyperparams)
        step!(alg, q, logπ, θ, diff_result, opt)

        AdvancedVI.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return θ
end


## Second part is for optimization based on `q.θ`

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
    alg_name = alg_str(alg)
    max_iters = niters(alg)
    
    state = init(alg, q, opt) # opt is there to be used in the future

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    # add criterion? A running mean maybe?
    time_elapsed = @elapsed while (i < max_iters) # & converged
        logπ = makelogπ(model, hyperparams)
        step!(alg, q, logπ, state, opt)

        AdvancedVI.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return q
end

## Verify that the algorithm can work with the corresponding variational distribution
function check_compatibility(alg, q)
    if !compat(alg, q)
        throw(ArgumentError("Algorithm $(alg) cannot work with distributions of type $(typeof(q)), compatible distributions are: $(compats(q))"))
    end
end

function compat(alg::VariationalInference, q)
    return q <: compats(alg)
end

function compats(::Any)
    return ()
end 
