##############################
# Global variables/constants #
##############################
const ADBACKEND = Ref(:forwarddiff)
setadbackend(backend_sym::Symbol) = setadbackend(Val(backend_sym))
function setadbackend(::Val{:forward_diff})
    Base.depwarn("`AdvancedVI.setadbackend(:forward_diff)` is deprecated. Please use `AdvancedVI.setadbackend(:forwarddiff)` to use `ForwardDiff`.", :setadbackend)
    setadbackend(Val(:forwarddiff))
end
function setadbackend(::Val{:forwarddiff})
    CHUNKSIZE[] == 0 && setchunksize(40)
    ADBACKEND[] = :forwarddiff
end

function setadbackend(::Val{:reverse_diff})
    Base.depwarn("`AdvancedVI.setadbackend(:reverse_diff)` is deprecated. Please use `AdvancedVI.setadbackend(:tracker)` to use `Tracker` or `AdvancedVI.setadbackend(:reversediff)` to use `ReverseDiff`. To use `ReverseDiff`, please make sure it is loaded separately with `using ReverseDiff`.",  :setadbackend)
    setadbackend(Val(:tracker))
end
function setadbackend(::Val{:tracker})
    ADBACKEND[] = :tracker
end

const ADSAFE = Ref(false)
function setadsafe(switch::Bool)
    @info("[AdvancedVI]: global ADSAFE is set as $switch")
    ADSAFE[] = switch
end

const CHUNKSIZE = Ref(40) # default chunksize used by AD

function setchunksize(chunk_size::Int)
    if ~(CHUNKSIZE[] == chunk_size)
        @info("[AdvancedVI]: AD chunk size is set as $chunk_size")
        CHUNKSIZE[] = chunk_size
    end
end

abstract type ADBackend end
struct ForwardDiffAD{chunk} <: ADBackend end
getchunksize(::Type{<:ForwardDiffAD{chunk}}) where chunk = chunk

struct TrackerAD <: ADBackend end

ADBackend() = ADBackend(ADBACKEND[])
ADBackend(T::Symbol) = ADBackend(Val(T))

ADBackend(::Val{:forwarddiff}) = ForwardDiffAD{CHUNKSIZE[]}
ADBackend(::Val{:tracker}) = TrackerAD
ADBackend(::Val) = error("The requested AD backend is not available. Make sure to load all required packages.")
