module AdvancedVIDynamicPPLExt

using AdvancedVI: AdvancedVI
using DynamicPPL: DynamicPPL

function (g::AdvancedVI.WeightedLogJoint)(vi::DynamicPPL.AbstractVarInfo)
    loglike = DynamicPPL.getloglikelihood(vi)
    logprior = DynamicPPL.getlogprior(vi)
    logjac = DynamicPPL.getlogjac(vi)
    return g.scale * loglike + logprior - logjac
end

end
