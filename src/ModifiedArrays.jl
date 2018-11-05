module ModifiedArrays

using Base: tail

export
# Types
# Functions
    modify,
    modified_transpose,
    modified_adjoint,
    ModifiedOffsetArray

## Main types

"The supertype of all array modifiers."
abstract type Modifier end

"Modifier that by default does not participate in any modification."
abstract type NonintrusiveModifier <: Modifier end
"Modifier that by default participates in every modification."
abstract type IntrusiveModifier <: Modifier end

"The special case of no modification at all."
struct NoModifier <: Modifier end



abstract type Modifiable end

specialize(::Modifiable, mod::NoModifier) = NoModifier()
specialize(::Modifiable, mod::NonintrusiveModifier) = NoModifier()
specialize(::Modifiable, mod::IntrusiveModifier) = mod



struct Interface{S} <: Modifiable end

# For any interface:
# - we skip NoModifier
recmod(IF::Interface, ::NoModifier, mods, args...) =
    recmod(IF, mods, args...)
# - given a tuple of modifiers we invoke recursion and specialize on the interface
recmod(IF::Interface, mods::Tuple, args...) =
    recmod(IF, specialize(IF, mods[1]), tail(mods), args...)



abstract type ModifiedArray{T,N} <: AbstractArray{T,N} end

Base.parent(a::ModifiedArray) = a.parent
modifier(a::ModifiedArray) = a.modifier

struct TypedModifiedArray{AA,M,T,N} <: ModifiedArray{T,N}
    parent      ::  AA      # the array-like object being modified
    modifier    ::  M       # tuple of modifiers
end

struct UntypedModifiedArray{T,N} <: ModifiedArray{T,N}
    parent
    modifier
end



## Array specific interface
#
# The methods we enable modifications for are:
#   size, getindex, setindex!, IndexStyle, axes, similar
#
# In addition, we provide `eltype` and `dimension` function, to compute the
# type parameters of the abstract array we inherit from at creation time,
# since ModifiedArray <: AbstractArray{T,N}

const IF_eltype = Interface{:eltype}
const IF_dimension = Interface{:dimension}

struct ElementTypeModifier{T} <: Modifier end
struct DimensionModifier{N} <: Modifier end

recmod(::IF_eltype, ::Tuple{}, A) = eltype(A)
recmod(::IF_eltype, ::ElementTypeModifier{T}, mods, A) where {T} = T

# This routine returns dimension as a type parameter to Val
recmod(::IF_dimension, ::Tuple{}, A::AbstractArray{T,N}) where {T,N} = Val{N}()
recmod(::IF_dimension, ::DimensionModifier{N}, mods, A) where {N} = Val{N}()


const IF_size = Interface{:size}
const IF_getindex = Interface{:getindex}
const IF_setindex = Interface{:setindex}
const IF_IndexStyle = Interface{:setindex}
const IF_axes = Interface{:axes}
const IF_similar = Interface{:similar}

sizemodifier(mod) = specialize(IF_size(), mod)
getindexmodifier(mod) = specialize(IF_getindex(), mod)
setindexmodifier(mod) = specialize(IF_setindex(), mod)
IndexStylemodifier(mod) = specialize(IF_IndexStyle(), mod)
axesmodifier(mod) = specialize(IF_axes(), mod)
similarmodifier(mod) = specialize(IF_similar(), mod)



Base.size(A::ModifiedArray) = recmod(IF_size(), modifier(A), parent(A))
recmod(::IF_size, ::Tuple{}, A) = size(A)

Base.getindex(A::ModifiedArray, i::Int) =
    recmod(IF_getindex(), modifier(A), parent(A), i)
Base.getindex(A::ModifiedArray{T,N}, I::Vararg{Int,N}) where {T,N} =
    recmod(IF_getindex(), modifier(A), parent(A), I...)
recmod(::IF_getindex, ::Tuple{}, A, I...) = getindex(A, I...)

Base.setindex!(A::ModifiedArray, val, i::Int) =
    (recmod(IF_setindex(), modifier(A), parent(A), val, i); A)
Base.setindex!(A::ModifiedArray{T,N}, val, I::Vararg{Int,N}) where {T,N} =
    (recmod(IF_setindex(), modifier(A), parent(A), val, I...); A)
recmod(::IF_setindex, ::Tuple{}, A, val, I...) = setindex!(A, val, I...)

Base.IndexStyle(A::ModifiedArray) =
    recmod(IF_IndexStyle(), modifier(A), parent(A))
recmod(::IF_IndexStyle, ::Tuple{}, A) = IndexStyle(A)

Base.axes(A::ModifiedArray) = recmod(IF_axes(), modifier(A), parent(A))
recmod(::IF_axes, ::Tuple{}, A) = axes(A)


function Base.similar(A::ModifiedArray, ::Type{T}, dims::Dims) where {T}
    B = similar(parent(A), T, dims)
end


## More convenient constructors

TypedModifiedArray(A, mods) = TypedModifiedArray(
    recmod(IF_eltype(), mods, A),
    recmod(IF_dimension(), mods, A),
    A, mods)

TypedModifiedArray(::Type{T}, ::Val{N}, A, mods) where {T,N} =
    TypedModifiedArray{typeof(A),typeof(mods),T,N}(A,mods)


modify(A, mod::Modifier) = modify(A, (mod,))
modify(A, mods::Modifier...) = modify(A, mods)
modify(A::ModifiedArray, mod::Modifier) =
    modify(parent(A), (mod, modifier(A)...))
modify(A::ModifiedArray, mods::Modifier...) =
    modify(parent(A), (mods..., modifier(A)...))

modify(A, mods::Tuple{Vararg{Modifier}}) = TypedModifiedArray(A, mods)




include("transpose.jl")
include("indexing.jl")
include("view.jl")
include("property.jl")

end # module
