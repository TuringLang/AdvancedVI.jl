struct EnzymeAD <: ADBackend end
ADBackend(::Val{:enzyme}) = EnzymeAD
function setadbackend(::Val{:enzyme})
    ADBACKEND[] = :enzyme
end
