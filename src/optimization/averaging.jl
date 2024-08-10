
struct NoAveraging <: AbstractAverager end

init(::NoAveraging, x) = x

apply(::NoAveraging, state, x) = x

value(::NoAveraging, state) = state

struct PolynomialAveraging{F} <: AbstractAverager 
    eta::F
end

init(::PolynomialAveraging, x) = (x, 1)

function apply(avg::PolynomialAveraging, state, x::AbstractVector{T}) where {T}
    eta = T(avg.eta)
    x_bar, t = state

    w = (eta + 1)/(t + eta)
    x_bar = (1 - w)*x_bar + w*x
    
    return (x_bar, t+1)
end

value(::PolynomialAveraging, state) = first(state)
