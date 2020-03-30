struct ZygoteAD <: ADBackend end
ADBackend(::Val{:zygote}) = ZygoteAD
function setadbackend(::Val{:zygote})
    ADBACKEND[] = :zygote
end
