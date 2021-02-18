
# default implementations
function gradobjective!(
    g::DiffResults.MutableDiffResult,
    vo::VariationalObjective,
    alg::VariationalInference{<:ForwardDiffAD},
    logπ, # Log joint distribution
    q, # Function to create the variational distribution
    x::AbstractVector{<:Real}, # Variational parameters
    args...,
)
    f(x) = evaluate(vo, alg, q, x, logπ)
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, x, chunk)
    return ForwardDiff.gradient!(g, f, x, config)
end

function gradlogπ!(
    g::DiffResults.MutableDiffResult,
    alg::VariationalInference{<:ForwardDiffAD},
    logπ, # Log joint distribution
    q, # Variational distribution
    x::AbstractMatrix, # Collection of samples
)
    f(X) = sum(eachcol(X)) do x
        return evaluate(logπ, q, x)
    end
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(x), chunk_size))
    config = ForwardDiff.GradientConfig(f, x, chunk)
    return ForwardDiff.gradient!(g, f, x, config)
end

function gradbbvi!(
    logπ,
    state,
    alg::VariationalInference{<:ForwardDiffAD},
    q, # Variational distribution
)
    Δlog = logπ.(eachcol(state.x)) .- logpdf(to_dist(q, state.θ), state.x)
    f(θ) = mean(logpdf(to_dist(q, θ), state.x) .* Δlog)
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(state.θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, state.θ, chunk)
    return ForwardDiff.gradient!(state.diff_result, f, state.θ, config)
end