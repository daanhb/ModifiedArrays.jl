
import LinearAlgebra: issymmetric, ishermitian, isdiag

struct Property{S} <: Modifiable end

modification(p::Property, ::NoModifier, mods, args...) =
    modification(p, mods, args...)
modification(p::Property, mods::Tuple, args...) =
    modification(p, mods[1], tail(mods), args...)

const P_issymmetric = Property{:issymmetric}
const P_ishermitian = Property{:ishermitian}
const P_isdiag = Property{:isdiag}

abstract type PropertyModifier <: NonintrusiveModifier end

struct SymmetricProperty <: PropertyModifier end

issymmetric(A::ModifiedArray) =
    modification(Property{:issymmetric}(), modifier(A), parent(A))
modification(::Property{:issymmetric}, ::Tuple{}, A) = issymmetric(A)

modification(::Property{:issymmetric}, ::SymmetricProperty, mods, A) = true


struct HermitianProperty <: PropertyModifier end

ishermitian(A::ModifiedArray) =
    modification(Property{:ishermitian}(), modifier(A), parent(A))
modification(::Property{:ishermitian}, ::Tuple{}, A) = ishermitian(A)

modification(::Property{:ishermitian}, ::HermitianProperty, mods, A) = true


struct DiagProperty <: PropertyModifier end

isdiag(A::ModifiedArray) =
    modification(Property{:isdiag}(), modifier(A), parent(A))
modification(::Property{:isdiag}, ::Tuple{}, A) = isdiag(A)

modification(::Property{:isdiag}, ::DiagProperty, mods, A) = true
