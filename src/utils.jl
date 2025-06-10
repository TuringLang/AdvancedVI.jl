
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
