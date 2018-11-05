"""
`IndexingModifier` is the supertype of modifiers that only change the way
the array is indexed. These do not affect the `size` of the array. However,
they do affect `getindex`, `setindex` and `axes`.
"""
abstract type IndexingModifier <: IntrusiveModifier end

# Size, eltype and dimension are not affected by this class of modifiers
specialize(::IF_size, ::IndexingModifier) = NoModifier()
specialize(::IF_eltype, ::IndexingModifier) = NoModifier()
specialize(::IF_dimension, ::IndexingModifier) = NoModifier()

# `getindex` and `setindex` are invoked on the parent array with a modified index.
# Concrete modifiers should implement the `modify_index` routine.
recmod(IF::IF_getindex, mod::IndexingModifier, mods, A, I...) =
    recmod(IF, mods, A, modify_index(mod, I...)...)

recmod(IF::IF_setindex, mod::IndexingModifier, mods, A, val, I...) =
    recmod(IF, mods, A, val, modify_index(mod, I...)...)


"""
An `OffsetModifier` results in an array for which the indices start at a value
different from `1`. The modifier is functionally equivalent to an `OffsetArray`
"""
struct OffsetModifier{N} <: IndexingModifier
    offsets ::  NTuple{N,Int}
end

const SimpleOffsetArray{T,N} = TypedModifiedArray{Array{T,N},Tuple{OffsetModifier{N}},T,N}


# In general, modify_index for OffsetModifiers just invokes `offset`
# as defined in the OffsetArrays package.
# One particular case is that we simply return any linear index into a
# higher-dimensional array. Hence, we add two special cases below the general one.
modify_index(mod::OffsetModifier{N}, I::Vararg{Int,N}) where {N} = offset(mod.offsets, I)
modify_index(mod::OffsetModifier{1}, i::Int) = offset(mod.offsets, (i,))
modify_index(mod::OffsetModifier{N}, i::Int) where {N} = i

# These definitions are copied from OffsetArrays
indexoffset(r::AbstractRange) = first(r) - 1
indexoffset(i::Integer) = 0
indexlength(r::AbstractRange) = length(r)
indexlength(i::Integer) = i

offset(offsets::NTuple{N,Int}, inds::NTuple{N,Int}) where {N} =
    (inds[1]-offsets[1], offset(Base.tail(offsets), Base.tail(inds))...)
offset(::Tuple{}, ::Tuple{}) = ()

offset(offsets::Tuple{Vararg{Int}}, inds::Tuple{Vararg{Int}}) =
    (offset(offsets, Base.front(inds))..., inds[end])
offset(offsets::Tuple{Vararg{Int}}, inds::Tuple{}) = error("inds cannot be shorter than offsets")

# specialize(::IF_IndexStyle, ::OffsetModifier) = NoModifier()
recmod(IF::IF_IndexStyle, ::OffsetModifier, mods, A) = recmod(IF, mods, A)

recmod(IF::IF_axes, mod::OffsetModifier, mods, A) =
    _modified_axes(recmod(IF, mods, A), mod.offsets)

_modified_axes(inds, offsets) =
    (Base.Slice(inds[1] .+ offsets[1]), _modified_axes(tail(inds), tail(offsets))...)
_modified_axes(::Tuple{}, ::Tuple{}) = ()


# Everything below here is copied from OffsetArrays and translated to the current setting

using Base: Indices, Dims

const OffsetAxis = Union{Integer, UnitRange, Base.Slice{<:UnitRange}, Base.OneTo}

## similar

function Base.similar(A::AbstractArray, ::Type{T}, inds::Tuple{OffsetAxis,Vararg{OffsetAxis}}) where {T}
    B = similar(A, T, map(indexlength, inds))
    ModifiedOffsetArray(B, map(indexoffset, inds))
end

Base.similar(::Type{T}, shape::Tuple{OffsetAxis,Vararg{OffsetAxis}}) where {T<:AbstractArray} =
    ModifiedOffsetArray(T(undef, map(indexlength, shape)), map(indexoffset, shape))


## reshape

Base.reshape(A::AbstractArray, inds::Tuple{OffsetAxis,Vararg{OffsetAxis}}) =
    ModifiedOffsetArray(reshape(A, map(indexlength, inds)), map(indexoffset, inds))

# Reshaping OffsetArrays can "pop" the original OffsetArray wrapper and return
# an OffsetArray(reshape(...)) instead of an OffsetArray(reshape(OffsetArray(...)))
Base.reshape(A::SimpleOffsetArray, inds::Tuple{OffsetAxis,Vararg{OffsetAxis}}) =
    ModifiedOffsetArray(reshape(parent(A), map(indexlength, inds)), map(indexoffset, inds))
# And for non-offset axes, we can just return a reshape of the parent directly
Base.reshape(A::SimpleOffsetArray, inds::Tuple{Union{Integer,Base.OneTo},Vararg{Union{Integer,Base.OneTo}}}) = reshape(parent(A), inds)
Base.reshape(A::SimpleOffsetArray, inds::Dims) = reshape(parent(A), inds)


## fill

Base.fill(v, inds::NTuple{N, Union{Integer, AbstractUnitRange}}) where {N} =
    fill!(ModifiedOffsetArray(Array{typeof(v), N}(undef, map(indexlength, inds)), map(indexoffset, inds)), v)
Base.zeros(::Type{T}, inds::NTuple{N, Union{Integer, AbstractUnitRange}}) where {T, N} =
    fill!(ModifiedOffsetArray(Array{T, N}(undef, map(indexlength, inds)), map(indexoffset, inds)), zero(T))
Base.ones(::Type{T}, inds::NTuple{N, Union{Integer, AbstractUnitRange}}) where {T, N} =
    fill!(ModifiedOffsetArray(Array{T, N}(undef, map(indexlength, inds)), map(indexoffset, inds)), one(T))
Base.trues(inds::NTuple{N, Union{Integer, AbstractUnitRange}}) where {N} =
    fill!(ModifiedOffsetArray(BitArray{N}(undef, map(indexlength, inds)), map(indexoffset, inds)), true)
Base.falses(inds::NTuple{N, Union{Integer, AbstractUnitRange}}) where {N} =
    fill!(ModifiedOffsetArray(BitArray{N}(undef, map(indexlength, inds)), map(indexoffset, inds)), false)


# Compatibility with OffsetArray constructors

ModifiedOffsetArray(A::AbstractArray{T,N}, offsets::NTuple{N,Int}) where {T,N} =
    modify(A, OffsetModifier(offsets))
ModifiedOffsetArray(A::AbstractArray{T,N}, offsets::Vararg{Int,N}) where {T,N} =
    ModifiedOffsetArray(A, offsets)

ModifiedOffsetArray(::Type{T}, ::UndefInitializer, inds::Indices{N}) where {T,N} =
    ModifiedOffsetArray(Array{T,N}(undef, map(indexlength, inds)), map(indexoffset, inds))
ModifiedOffsetArray(::Type{T}, ::UndefInitializer, inds::Vararg{AbstractUnitRange,N}) where {T,N} = ModifiedOffsetArray(T, undef, inds)
ModifiedOffsetArray(A::AbstractArray{T,0}) where {T} = ModifiedOffsetArray(A, ())

ModifiedOffsetArray(A::AbstractArray{T,0}, inds::Tuple{}) where {T} =
    modify(A, OffsetModifier(Tuple{}()))
ModifiedOffsetArray(A::AbstractArray{T,N}, inds::Tuple{}) where {T,N} = error("this should never be called")
function ModifiedOffsetArray(A::AbstractArray{T,N}, inds::NTuple{N,AbstractUnitRange}) where {T,N}
    lA = map(indexlength, axes(A))
    lI = map(indexlength, inds)
    lA == lI || throw(DimensionMismatch("supplied axes do not agree with the size of the array (got size $lA for the array and $lI for the indices"))
    ModifiedOffsetArray(A, map(indexoffset, inds))
end
ModifiedOffsetArray(A::AbstractArray{T,N}, inds::Vararg{AbstractUnitRange,N}) where {T,N} =
    ModifiedOffsetArray(A, inds)


## IO routines

function Base.showarg(io::IO, a::SimpleOffsetArray, toplevel)
    print(io, "ModifiedOffsetArray(")
    Base.showarg(io, parent(a), false)
    if ndims(a) > 0
        print(io, ", ")
        printindices(io, axes(a)...)
    end
    print(io, ')')
    toplevel && print(io, " with eltype ", eltype(a))
end
printindices(io::IO, ind1, inds...) =
    (print(io, _unslice(ind1), ", "); printindices(io, inds...))
printindices(io::IO, ind1) = print(io, _unslice(ind1))
_unslice(x) = x
_unslice(x::Base.Slice) = x.indices
