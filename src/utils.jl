
function pm_next!(pm, stats::NamedTuple)
    return ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

eachsample(samples::AbstractMatrix) = eachcol(samples)

function catsamples_and_acc(
    state_curr::Tuple{<:AbstractArray,<:Real}, state_new::Tuple{<:AbstractVector,<:Real}
)
    x = hcat(first(state_curr), first(state_new))
    ∑y = last(state_curr) + last(state_new)
    return (x, ∑y)
end

"""
    use_view_in_gradient(prob)::Bool

When calling `logdensity_and_gradient(prob, x)`, this determines whether `x` can be passed
as a view. This is usually better for efficiency and hence the default is `true`. However,
some `prob`s may not support views (e.g. if gradient preparation has already been done with
a full vector).
"""
use_view_in_gradient(@nospecialize(prob::Any)) = true
