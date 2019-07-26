module ModifiedArrays

import Base: first, tail
using Base: @propagate_inbounds

export
    modify,
    ModifiedArray,
    # transpose.jl
    modified_transpose,
    modified_adjoint,
    TransposeMod,
    AdjointMod

## Main types

"The supertype of all array modifiers."
abstract type ArrayModifier end

"The special case of no modification at all."
struct NoModifier <: ArrayModifier end




"The main modified array type consists of a parent array and a modifier."
struct ModifiedArray{T,N,AA,M} <: AbstractArray{T,N}
    parent      ::  AA      # the array-like object being modified
    modifier    ::  M       # the modifier
end

ModifiedArray(A, mod::ArrayModifier) = ModifiedArray{mod_eltype(mod,A),mod_ndims(mod,A)}(A, mod)
ModifiedArray{T,N}(A, mod::ArrayModifier) where {T,N} =
    ModifiedArray{T,N,typeof(A),typeof(mod)}(A, mod)

modify(A, mod::ArrayModifier) = ModifiedArray(A, mod)

Base.parent(a::ModifiedArray) = a.parent
modifier(a::ModifiedArray) = a.modifier




## Array specific interface

abstract type Modifiable end

struct Interface{S} <: Modifiable end


# The two type variables
const IF_eltype = Interface{:eltype}
const IF_ndims = Interface{:ndims}

# Compulsory interface methods
const IF_size = Interface{:size}
const IF_getindex = Interface{:getindex}
const IF_setindex! = Interface{:setindex!}

# Optional methods
const IF_IndexStyle = Interface{:IndexStyle}
const IF_similar = Interface{:similar}
const IF_axes = Interface{:axes}


abstract type ModStyle end
struct ModRecursive <: ModStyle end
struct ModFinal <: ModStyle end
struct ModNothing <: ModStyle end


## eltype
mod_eltype(A::ModifiedArray) = mod_eltype(modifier(A), parent(A))
mod_eltype(mod::ArrayModifier, A) = mod_eltype(ModStyle(mod, IF_eltype()), mod, A)

mod_eltype(::ModNothing, mod, A) = eltype(A)
mod_eltype(::ModFinal, mod, A) = mod_eltype_final(mod)
mod_eltype(::ModRecursive, mod, A) = mod_eltype_post(mod, eltype(A))


## ndims
mod_ndims(A::ModifiedArray) = mod_ndims(modifier(A), parent(A))
mod_ndims(mod::ArrayModifier, A) = mod_ndims(ModStyle(mod, IF_ndims()), mod, A)

mod_ndims(::ModNothing, mod, A) = ndims(A)
mod_ndims(::ModFinal, mod, A) = mod_ndims_final(mod)
mod_ndims(::ModRecursive, mod, A) = mod_ndims_post(mod, ndims(A))


## size
Base.size(A::ModifiedArray) = mod_size(A)
mod_size(A::ModifiedArray) = mod_size(modifier(A), parent(A))
mod_size(mod::ArrayModifier, A) = mod_size(ModStyle(mod, IF_size()), mod, A)

mod_size(::ModNothing, mod, A) = size(A)
mod_size(::ModFinal, mod, A) = mod_size_final(mod)
mod_size(::ModRecursive, mod, A) = mod_size_post(mod, size(A))


## axes
Base.axes(A::ModifiedArray) = mod_axes(A)
mod_axes(A::ModifiedArray) = mod_axes(modifier(A), parent(A))
mod_axes(mod::ArrayModifier, A) = mod_axes(ModStyle(mod, IF_axes()), mod, A)

mod_axes(::ModNothing, mod, A) = axes(A)
mod_axes(::ModFinal, mod, A) = mod_axes_final(mod)
mod_axes(::ModRecursive, mod, A) = mod_axes_post(mod, axes(A))


## IndexStyle
Base.IndexStyle(Mod::Type{<:ModifiedArray}) = mod_IndexStyle(Mod)
mod_IndexStyle(::Type{ModifiedArray{T,N,AA,M}}) where {T,N,AA,M} =
    mod_IndexStyle(M, AA)
mod_IndexStyle(M::Type{<:ArrayModifier}, AA) = mod_IndexStyle(ModStyle(M, IF_IndexStyle()), M, AA)

mod_IndexStyle(::ModNothing, M, AA) = IndexStyle(AA)
mod_IndexStyle(::ModFinal, M, AA) = mod_IndexStyle_final(M)
mod_IndexStyle(::ModRecursive, M, AA) = mod_IndexStyle_post(M, IndexStyle(AA))


## getindex

# We only intercept calls with a linear index, or with a number of indices equal
# to the dimension of the array.
Base.getindex(A::ModifiedArray{T,1}, i::Int) where {T} = mod_getindex(A, i)
Base.getindex(A::ModifiedArray{T,N}, I::Vararg{Int,N}) where {T,N} =
    mod_getindex(A, I...)
# Linear indexing of arrays that are not vectors:
Base.getindex(A::ModifiedArray{T,N}, i::Int) where {T,N} =
    __getindex(IndexStyle(A), A, i)
# - the array supports linear indexing: invoke mod_getindex
__getindex(::IndexLinear, A, i) = mod_getindex(A, i)
# - the array supports cartesian indexing: translate the index to cartesian
__getindex(::IndexCartesian, A, i) = mod_getindex(A, Base._to_subscript_indices(A, i)...)

mod_getindex(A::ModifiedArray, I...) =
    mod_getindex(modifier(A), parent(A), I...)
mod_getindex(mod::ArrayModifier, A, I...) =
    mod_getindex(ModStyle(mod, IF_getindex()), mod, A, I...)

mod_getindex(::ModNothing, mod, A, I...) = getindex(A, I...)
mod_getindex(::ModFinal, mod, istyle, A, I...) = mod_getindex_final(mod, I...)
mod_getindex(::ModRecursive, mod, A, I...) =
    mod_getindex_post(mod, getindex(A, mod_getindex_pre(mod, I...)...), I...)

# default definition
mod_getindex_pre(mod::ArrayModifier, I...) = I

## setindex!
# Like with getindex above, we only intercept a few calls to setindex!.
Base.setindex!(A::ModifiedArray{T,1}, val, i::Int) where {T} =
    mod_setindex!(A, val, i)
Base.setindex!(A::ModifiedArray{T,N}, val, I::Vararg{Int,N}) where {T,N} =
    mod_setindex!(A, val, I...)
# Linear indexing of arrays that are not vectors, similar to getindex above:
Base.setindex!(A::ModifiedArray{T,N}, val, i::Int) where {T,N} =
    __setindex!(IndexStyle(A), A, val, i)
__setindex!(::IndexLinear, A, val, i) = mod_setindex!(A, val, i)
__setindex!(::IndexCartesian, A, val, i) =
    mod_setindex!(A, val, Base._to_subscript_indices(A, i)...)


# The return value of setindex! is not prescribed by the array interface.
# In most cases, the mutated object is returned, so we just return A.
# That means there is no need for a mod_setindex!_post function.
# See also Julia issue #31891
mod_setindex!(A::ModifiedArray, val, I...) =
    (mod_setindex!(modifier(A), parent(A), val, I...); A)
mod_setindex!(mod::ArrayModifier, A, val, I...) =
    mod_setindex!(ModStyle(mod, IF_setindex!()), mod, A, val, I...)

mod_setindex!(::ModNothing, mod, A, val, I...) = setindex!(A, val, I...)
mod_setindex!(::ModFinal, mod, A, val, I...) =
    mod_setindex!_final(mod, val, I...)
mod_setindex!(::ModRecursive, mod, A, val, I...) =
    setindex!(A, mod_setindex!_pre(mod, val, I...)...)

# default definition
mod_setindex!_pre(mod::ArrayModifier, val, I...) = (val, I...)


## similar
Base.similar(A::ModifiedArray, ::Type{T}, dims::Dims) where {T} =
    mod_similar(A, T, dims)
mod_similar(A::ModifiedArray, ::Type{T}, dims::Dims) where {T} =
    mod_similar(modifier(A), parent(A), T, dims)
mod_similar(mod::ArrayModifier, A, T, dims) =
    mod_similar(ModStyle(mod, IF_similar()), mod, A, T, dims)

mod_similar(::ModNothing, mod, A, T, dims) = similar(A, T, dims)
mod_similar(::ModFinal, mod, A, T, dims) = mod_similar_final(mod, T, dims)
mod_similar(::ModRecursive, mod, A, T, dims) =
    mod_similar_post(mod, similar(A, mod_similar_pre(mod, T, dims)...), T, dims)

# Since similar is an optional function, we can safely introduce a ModNothing default.
ModStyle(::ArrayModifier, ::IF_similar) = ModNothing()


include("composite.jl")
include("transpose.jl")
include("offset.jl")
include("eltype.jl")

end # module
