
# default implementations
function grad!(
    f::Function,
    adtype::AutoForwardDiff{chunksize},
    λ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult
) where {chunksize}
    # Set chunk size and do ForwardMode.
    config = if isnothing(chunksize)
        ForwardDiff.GradientConfig(f, λ)
    else
        ForwardDiff.GradientConfig(f, λ, ForwardDiff.Chunk(length(λ), chunksize))
    end
    ForwardDiff.gradient!(out, f, λ, config)
end

function grad!(
    f::Function,
    ::AutoTracker,
    λ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult
)
    λ_tracked = Tracker.param(λ)
    y = f(λ_tracked)
    Tracker.back!(y, 1.0)

    DiffResults.value!(out, Tracker.data(y))
    DiffResults.gradient!(out, Tracker.grad(λ_tracked))
end
