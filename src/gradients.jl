
# default implementations
function grad!(
    vo,
    alg::VariationalInference{<:ForwardDiffAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...,
)
    f(θ_) =
        if (q isa Distribution)
            -vo(alg, update(q, θ_), model, args...)
        else
            -vo(alg, q(θ_), model, args...)
        end

    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, θ, chunk)
    return ForwardDiff.gradient!(out, f, θ, config)
end

function gradlogπ!(
    g::GradientResult,
    alg::VariationalInference{<:ForwardDiffAD},
    logπ,
    q,
    x,
)
    f(x) =
    sum(eachcol(x)) do z
        return evaluate(logπ, q, z)
    end
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(x), chunk_size))
    config = ForwardDiff.GradientConfig(f, x, chunk)
    return ForwardDiff.gradient!(g, f, x, config)
end