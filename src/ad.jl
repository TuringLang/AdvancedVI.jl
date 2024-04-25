# FIXME: All this should go away.
const ADBACKEND = Ref(:forwarddiff)
setadbackend(backend_sym::Symbol) = setadbackend(Val(backend_sym))
function setadbackend(::Val{:forward_diff})
    Base.depwarn("`AdvancedVI.setadbackend(:forward_diff)` is deprecated. Please use `AdvancedVI.setadbackend(:forwarddiff)` to use `ForwardDiff`.", :setadbackend)
    setadbackend(Val(:forwarddiff))
end
function setadbackend(::Val{:forwarddiff})
    Base.depwarn("`setadbackend` is deprecated. Please pass a `ADTypes.AbstractADType` as a keyword argument to the VI algorithm.", :setadbackend)
    ADBACKEND[] = :forwarddiff
end

function setadbackend(::Val{:reverse_diff})
    Base.depwarn("`AdvancedVI.setadbackend(:reverse_diff)` is deprecated. Please use `AdvancedVI.setadbackend(:tracker)` to use `Tracker` or `AdvancedVI.setadbackend(:reversediff)` to use `ReverseDiff`. To use `ReverseDiff`, please make sure it is loaded separately with `using ReverseDiff`.",  :setadbackend)
    setadbackend(Val(:tracker))
end
function setadbackend(::Val{:tracker})
    Base.depwarn("`setadbackend` is deprecated. Please pass a `ADTypes.AbstractADType` as a keyword argument to the VI algorithm.", :setadbackend)
    ADBACKEND[] = :tracker
end

const ADSAFE = Ref(false)
function setadsafe(switch::Bool)
    @info("[AdvancedVI]: global ADSAFE is set as $switch")
    ADSAFE[] = switch
end

const CHUNKSIZE = Ref(0) # 0 means letting ForwardDiff set it automatically

function setchunksize(chunk_size::Int)
    @info("[AdvancedVI]: AD chunk size is set as $chunk_size")
    CHUNKSIZE[] = chunk_size
end

getchunksize(::Type{<:ADTypes.AutoForwardDiff{chunk}}) where chunk = chunk

ADBackend() = ADBackend(ADBACKEND[])
ADBackend(T::Symbol) = ADBackend(Val(T))

ADBackend(::Val{:forwarddiff}) = ADTypes.AutoForwardDiff(chunksize=CHUNKSIZE[])
ADBackend(::Val{:tracker}) = ADTypes.AutoTracker()
ADBackend(::Val) = error("The requested AD backend is not available. Make sure to load all required packages.")
