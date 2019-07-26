
"""
Modify an array by changing its element type on the fly to `T`, without changing
the storage of the underlying array.

Standard errors are thrown if conversion to or from `T` is not successful when
using `getindex` and `setindex!`.
"""
struct EltypeMod{T} <: ArrayModifier
end

ModStyle(::EltypeMod, ::Interface) = ModNothing()
ModStyle(::EltypeMod, ::IF_similar) = ModNothing()
ModStyle(::Type{<:EltypeMod}, ::IF_IndexStyle) = ModNothing()

ModStyle(::EltypeMod, ::IF_eltype) = ModFinal()
mod_eltype_final(::EltypeMod{T}) where {T} = T

ModStyle(::EltypeMod, ::IF_getindex) = ModRecursive()
mod_getindex_post(mod::EltypeMod{T}, Z, I...) where {T} = convert(T, Z)
