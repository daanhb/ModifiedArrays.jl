
@static if !isdefined(Base, :IdentityUnitRange)
    const IdentityUnitRange = Base.Slice
else
    using Base: IdentityUnitRange
end

export ModifiedOffsetArray, ModifiedOffsetVector

"""
An `OffsetMod` results in an array for which the indices start at a value
different from `1`. The modifier is functionally equivalent to an `OffsetArray`
"""
struct OffsetMod{N} <: ArrayModifier
    offsets ::  NTuple{N,Int}
end

ModifiedOffsetArray{T,N,AA} = ModifiedArray{T,N,AA,OffsetMod{N}}
ModifiedOffsetVector{T,AA} = ModifiedOffsetArray{T,1,AA}

ModifiedOffsetArray(A::AbstractArray{T,N}, offsets::NTuple{N,Int}) where {T,N} =
    ModifiedArray{T,N}(A, OffsetMod(offsets))
ModifiedOffsetArray(A::AbstractArray{T,N}, offsets::Vararg{Int,N}) where {T,N} =
    ModifiedOffsetArray(A, offsets)

ModStyle(::OffsetMod, ::IF_eltype) = ModNothing()
ModStyle(::OffsetMod, ::IF_ndims) = ModNothing()
ModStyle(::OffsetMod, ::IF_size) = ModNothing()

ModStyle(::OffsetMod, ::IF_getindex) = ModRecursive()
ModStyle(::OffsetMod, ::IF_setindex!) = ModRecursive()
ModStyle(::Type{<:OffsetMod}, ::IF_IndexStyle) = ModNothing()
ModStyle(::OffsetMod, ::IF_axes) = ModRecursive()
ModStyle(::OffsetMod, ::IF_similar) = ModNothing()


mod_getindex_pre(mod::OffsetMod{N}, I::Vararg{Int,N}) where {N} = offset(mod.offsets, I)
mod_getindex_pre(mod::OffsetMod, i::Int) = (i,)

mod_setindex!_pre(mod::OffsetMod{N}, val, I::Vararg{Int,N}) where {N} = (val,offset(mod.offsets, I)...)
mod_setindex!_pre(mod::OffsetMod, val, i::Int) = (val, i)

mod_axes_post(mod::OffsetMod, Z) = _axes(Z, mod.offsets)
_axes(inds, offsets) =
    (_slice(inds[1], offsets[1]), _axes(tail(inds), tail(offsets))...)
_axes(::Tuple{}, ::Tuple{}) = ()

# Avoid the kw-arg on the range(r+x, length=length(r)) call in r .+ x
@inline _slice(r, x) = IdentityUnitRange(Base._range(first(r) + x, nothing, nothing, length(r)))

## Take from OffsetArrays source code:
# Computing a shifted index (subtracting the offset)
@inline offset(offsets::NTuple{N,Int}, inds::NTuple{N,Int}) where {N} =
    (inds[1]-offsets[1], offset(Base.tail(offsets), Base.tail(inds))...)
offset(::Tuple{}, ::Tuple{}) = ()

# Support trailing 1s
@inline offset(offsets::Tuple{Vararg{Int}}, inds::Tuple{Vararg{Int}}) =
    (offset(offsets, Base.front(inds))..., inds[end])
offset(offsets::Tuple{Vararg{Int}}, inds::Tuple{}) = error("inds cannot be shorter than offsets")

indexoffset(r::AbstractRange) = first(r) - 1
indexoffset(i::Integer) = 0
indexlength(r::AbstractRange) = length(r)
indexlength(i::Integer) = i


# Everything below here is copied from OffsetArrays and translated to the current setting

using Base: Indices, Dims

const OffsetAxis = Union{Integer, UnitRange, Base.Slice, Base.OneTo, IdentityUnitRange}

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
Base.reshape(A::ModifiedOffsetArray, inds::Tuple{OffsetAxis,Vararg{OffsetAxis}}) =
    ModifiedOffsetArray(reshape(parent(A), map(indexlength, inds)), map(indexoffset, inds))
# And for non-offset axes, we can just return a reshape of the parent directly
Base.reshape(A::ModifiedOffsetArray, inds::Tuple{Union{Integer,Base.OneTo},Vararg{Union{Integer,Base.OneTo}}}) = reshape(parent(A), inds)
Base.reshape(A::ModifiedOffsetArray, inds::Dims) = reshape(parent(A), inds)


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

const ArrayInitializer = Union{UndefInitializer, Missing, Nothing}
ModifiedOffsetArray{T,N}(init::ArrayInitializer, inds::Indices{N}) where {T,N} =
    ModifiedOffsetArray{T,N,Array{T,N}}(Array{T,N}(init, map(indexlength, inds)), map(indexoffset, inds))
ModifiedOffsetArray{T}(init::ArrayInitializer, inds::Indices{N}) where {T,N} = ModifiedOffsetArray{T,N}(init, inds)
ModifiedOffsetArray{T,N}(init::ArrayInitializer, inds::Vararg{AbstractUnitRange,N}) where {T,N} = ModifiedOffsetArray{T,N}(init, inds)
ModifiedOffsetArray{T}(init::ArrayInitializer, inds::Vararg{AbstractUnitRange,N}) where {T,N} = ModifiedOffsetArray{T,N}(init, inds)
ModifiedOffsetArray(A::AbstractArray{T,0}) where {T} = ModifiedOffsetArray{T,0,typeof(A)}(A, ())

# ModifiedOffsetVector constructors
ModifiedOffsetVector(A::AbstractVector, offset) = ModifiedOffsetArray(A, offset)
ModifiedOffsetVector{T}(init::ArrayInitializer, inds::AbstractUnitRange) where {T} = ModifiedOffsetArray{T}(init, inds)


ModifiedOffsetArray{T,N,AA}(A::AbstractArray{T,N}, offsets::NTuple{N,Int}) where {T,N,AA} =
    modify(A, OffsetMod(offsets))


# The next two are necessary for ambiguity resolution. Really, the
# second method should not be necessary.
ModifiedOffsetArray(A::AbstractArray{T,0}, inds::Tuple{}) where {T} = ModifiedOffsetArray{T,0,typeof(A)}(A, ())
ModifiedOffsetArray(A::AbstractArray{T,N}, inds::Tuple{}) where {T,N} = error("this should never be called")
function ModifiedOffsetArray(A::AbstractArray{T,N}, inds::NTuple{N,AbstractUnitRange}) where {T,N}
    lA = map(indexlength, axes(A))
    lI = map(indexlength, inds)
    lA == lI || throw(DimensionMismatch("supplied axes do not agree with the size of the array (got size $lA for the array and $lI for the indices"))
    ModifiedOffsetArray(A, map(indexoffset, inds))
end
ModifiedOffsetArray(A::AbstractArray{T,N}, inds::Vararg{AbstractUnitRange,N}) where {T,N} =
    ModifiedOffsetArray(A, inds)

# avoid a level of indirection when nesting ModifiedOffsetArrays
function ModifiedOffsetArray(A::ModifiedOffsetArray, inds::NTuple{N,AbstractUnitRange}) where {N}
    ModifiedOffsetArray(parent(A), inds)
end
ModifiedOffsetArray(A::ModifiedOffsetArray{T,0}, inds::Tuple{}) where {T} = ModifiedOffsetArray{T,0,typeof(A)}(parent(A), ())
ModifiedOffsetArray(A::ModifiedOffsetArray{T,N}, inds::Tuple{}) where {T,N} = error("this should never be called")


### Special handling for AbstractRange

const OffsetRange{T} = ModifiedOffsetArray{T,1,<:AbstractRange{T}}
const IIUR = IdentityUnitRange{S} where S<:AbstractUnitRange{T} where T<:Integer

Base.step(a::OffsetRange) = step(parent(a))

Base.getindex(a::OffsetRange, r::OffsetRange) = ModifiedOffsetArray(a[parent(r)], r.modifier.offsets)
Base.getindex(a::OffsetRange, r::AbstractRange) = a.parent[r .- a.modifier.offsets[1]]
Base.getindex(a::AbstractRange, r::OffsetRange) = ModifiedOffsetArray(a[parent(r)], r.modifier.offsets)

Base.getindex(r::UnitRange, s::IIUR) =
    ModifiedOffsetArray(r[s.indices], s)

Base.getindex(r::StepRange, s::IIUR) =
    ModifiedOffsetArray(r[s.indices], s)

Base.getindex(r::StepRangeLen{T,<:Base.TwicePrecision,<:Base.TwicePrecision}, s::IIUR) where T =
    ModifiedOffsetArray(r[s.indices], s)
Base.getindex(r::StepRangeLen{T}, s::IIUR) where {T} =
    ModifiedOffsetArray(r[s.indices], s)

Base.getindex(r::LinRange, s::IIUR) =
    ModifiedOffsetArray(r[s.indices], s)

function Base.show(io::IO, r::OffsetRange)
    show(io, r.parent)
    o = r.modifier.offsets[1]
    print(io, " with indices ", o+1:o+length(r))
end
Base.show(io::IO, ::MIME"text/plain", r::OffsetRange) = show(io, r)

### Convenience functions ###

Base.fill(x, inds::Tuple{UnitRange,Vararg{UnitRange}}) =
    fill!(ModifiedOffsetArray{typeof(x)}(undef, inds), x)
@inline Base.fill(x, ind1::UnitRange, inds::UnitRange...) = fill(x, (ind1, inds...))


## IO routines

function Base.showarg(io::IO, a::ModifiedOffsetArray, toplevel)
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
_unslice(x::IdentityUnitRange) = x.indices


### Some mutating functions defined only for OffsetVector ###

Base.resize!(A::ModifiedOffsetVector, nl::Integer) = (resize!(A.parent, nl); A)
Base.push!(A::ModifiedOffsetVector, x...) = (push!(A.parent, x...); A)
Base.pop!(A::ModifiedOffsetVector) = pop!(A.parent)
Base.empty!(A::ModifiedOffsetVector) = (empty!(A.parent); A)

function no_offset_view(A::AbstractArray)
    if Base.has_offset_axes(A)
        ModifiedOffsetArray(A, map(r->1-first(r), axes(A)))
    else
        A
    end
end

no_offset_view(A::ModifiedOffsetArray) = no_offset_view(parent(A))
