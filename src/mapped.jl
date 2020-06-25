
abstract type AbstractMapMod <: ArrayModifier end

struct ReadonlyMapMod{F} <: AbstractMapMod
    f   ::  F
end
struct MapMod{F,Finv} <: AbstractMapMod
    f   ::  F
    finv::  Finv
end

ModStyle(::AbstractMapMod, ::Interface) = ModNothing()
ModStyle(::Type{<:AbstractMapMod}, ::IF_IndexStyle) = ModNothing()

ModStyle(::AbstractMapMod, ::IF_eltype) = ModRecursive()
mod_eltype_post(mod::AbstractMapMod, T) = Base.promote_op(mod.f,T)


ModStyle(::AbstractMapMod, ::IF_getindex) = ModRecursive()
mod_getindex_post(mod::AbstractMapMod, Z, I...) = mod.f(Z)

ModStyle(::MapMod, ::IF_setindex!) = ModRecursive()
mod_setindex!_pre(mod::MapMod, val, I...) = (mod.finv(val), I...)

ModStyle(::ReadonlyMapMod, ::IF_setindex!) = ModFinal()
mod_setindex_final(mod::ReadonlyMapMod, I...) = throw(InexactError("Readonly map is not writeable!"))
