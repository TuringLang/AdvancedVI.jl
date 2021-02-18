XXt(A) = A * A'
XtX(A) = A' * A

makelogπ(logπ, ::Nothing) = logπ
makelogπ(model, hp) = model(hp)

## Generic evaluation of the expectation
function expec_logπ(alg, q, logπ)
    mean(logπ, eachcol(rand(q, nsamples(alg))))
end

function evaluate(logπ, q::Bijectors.TransformedDistribution, x::AbstractVector)
    z, logjac = forward(q.transform, x)
    logπ(z) + logjac
end

function evaluate(logπ, ::Any, x::AbstractVector)
    logπ(x)
end