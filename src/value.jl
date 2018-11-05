
abstract type ValueModifier <: IntrusiveModifier end

specialize(::IF_eltype, mod::ValueModifier) = NoModification()
specialize(::IF_dimension, mod::ValueModifier) = NoModification()
specialize(::IF_size, mod::ValueModifier) = NoModification()
specialize(::IF_IndexStyle, mod::ValueModifier) = NoModification()
specialize(::IF_axes, mod::ValueModifier) = NoModification()
specialize(::IF_similar, mod::ValueModifier) = NoModification()

struct ScalarMultiple{T} <: ValueModifier
    scalar  ::  T
end

action(mod::ScalarMultiple, val) = mod.scalar * val
inverse(mod::ScalarMultiple, val) = val / mod.scalar

eltypemodifier(mod::ValueModifier) = mod
getindexmodifier(mod::ValueModifier) = mod
setindexmodifier(mod::ValueModifier) = mod
