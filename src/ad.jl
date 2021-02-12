##############################
# Global variables/constants #
##############################
const ADBACKEND = Ref(:ForwardDiff)
setadbackend(backend_sym::Symbol) = setadbackend(Val(backend_sym))
function setadbackend(::Val{:ForwardDiff})
    Base.depwarn("`AdvancedVI.setadbackend(:ForwardDiff)` is deprecated. Please use `AdvancedVI.setadbackend(:forwarddiff)` to use `ForwardDiff`.", :setadbackend)
    setadbackend(Val(:forwarddiff))
end
function setadbackend(::Val{:ForwardDiff})
    CHUNKSIZE[] == 0 && setchunksize(40)
    ADBACKEND[] = :ForwardDiff
end

function setadbackend(::Val{:Tracker})
    ADBACKEND[] = :Tracker
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

ADBackend(::Val{:ForwardDiff}) = ForwardDiffAD{CHUNKSIZE[]}
ADBackend(::Val{:Tracker}) = TrackerAD
ADBackend(::Val) = error("The requested AD backend is not available. Make sure to load all required packages.")
