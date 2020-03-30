using .ReverseDiff: compile, GradientTape
using .ReverseDiff.DiffResults: GradientResult

struct ReverseDiffAD{cache} <: ADBackend end
const RDCache = Ref(false)
setcache(b::Bool) = RDCache[] = b
getcache() = RDCache[]
ADBackend(::Val{:reversediff}) = ReverseDiffAD{getcache()}
function setadbackend(::Val{:reversediff})
    ADBACKEND[] = :reversediff
end

tape(f, x) = GradientTape(f, x)
function taperesult(f, x)
    return tape(f, x), GradientResult(x)
end
