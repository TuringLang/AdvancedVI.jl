module AdvancedVIDifferentiationInterfaceExt

using ADTypes: ADTypes
using AdvancedVI: AdvancedVI
import DifferentiationInterface as DI

function AdvancedVI._prepare_gradient(f, ad::ADTypes.AbstractADType, x, aux)
    return DI.prepare_gradient(f, ad, x, DI.Constant(aux))
end

function AdvancedVI._value_and_gradient(f, prep, ad::ADTypes.AbstractADType, x, aux)
    return DI.value_and_gradient(f, prep, ad, x, DI.Constant(aux))
end

end
