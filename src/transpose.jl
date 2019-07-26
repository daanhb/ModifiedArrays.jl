
function check_dimension(::Val{N}) where {N}
    1 <= N <= 2 || error(string(
        "Dimension of adjoint and transpose must be between 1 and 2."))
    return nothing
end

struct TransposeMod{T,N} <: ArrayModifier
    function TransposeMod{T,N}() where {T,N}
        check_dimension(Val{N}())
        new()
    end
end

struct AdjointMod{T,N} <: ArrayModifier
    function AdjointMod{T,N}() where {T,N}
        check_dimension(Val{N}())
        new()
    end
end

const TransposeMod1{T} = TransposeMod{T,1}
const TransposeMod2{T} = TransposeMod{T,2}
const AdjointMod1{T} = AdjointMod{T,1}
const AdjointMod2{T} = AdjointMod{T,2}

const AdjOrTransModifier{T,N} = Union{AdjointMod{T,N},TransposeMod{T,N}}
const AdjOrTransModifier1{T} = AdjOrTransModifier{T,1}
const AdjOrTransModifier2{T} = AdjOrTransModifier{T,2}

const TransposeModifiedMatrix{T,AA} = ModifiedArray{T,2,AA,TransposeMod{T,2}}
const TransposeModifiedVector{T,AA} = ModifiedArray{T,2,AA,TransposeMod{T,1}}
const AdjointModifiedMatrix{T,AA} = ModifiedArray{T,2,AA,AdjointMod{T,2}}
const AdjointModifiedVector{T,AA} = ModifiedArray{T,2,AA,AdjointMod{T,1}}

const AdjOrTransModifiedVector{T,AA} = Union{TransposeModifiedVector{T,AA},AdjointModifiedVector{T,AA}}
const AdjOrTransModifiedMatrix{T,AA} = Union{TransposeModifiedMatrix{T,AA},AdjointModifiedMatrix{T,AA}}

# Logic to compute the type of the transpose and adjoint modifiers:
# we make sure to use the same element type as the transpose and adjoint
# methods applied to A.
transpose_modifier(A::AbstractVector) =
    TransposeMod{Base.promote_op(modified_transpose,eltype(A)),1}()
transpose_modifier(A::AbstractMatrix) =
    TransposeMod{Base.promote_op(modified_transpose,eltype(A)),2}()
adjoint_modifier(A::AbstractVector) =
    AdjointMod{Base.promote_op(modified_adjoint,eltype(A)),1}()
adjoint_modifier(A::AbstractMatrix) =
    AdjointMod{Base.promote_op(modified_adjoint,eltype(A)),2}()

modified_transpose(A::AbstractVector) = modify(A, transpose_modifier(A))
modified_transpose(A::AbstractMatrix) = modify(A, transpose_modifier(A))

modified_transpose(A::TransposeModifiedVector) = parent(A)
modified_transpose(A::TransposeModifiedMatrix) = parent(A)

modified_adjoint(A::AbstractVector) = modify(A, adjoint_modifier(A))
modified_adjoint(A::AbstractMatrix) = modify(A, adjoint_modifier(A))

modified_adjoint(A::AdjointModifiedVector) = parent(A)
modified_adjoint(A::AdjointModifiedMatrix) = parent(A)

modified_transpose(A::Number) = A
modified_adjoint(A::Number) = conj(A)


Base.vec(A::TransposeModifiedVector) = parent(A)
Base.vec(A::AdjointModifiedVector) = parent(A)


###
# The interface
###

## eltype
ModStyle(::AdjOrTransModifier, ::IF_eltype) = ModFinal()
mod_eltype_final(::AdjOrTransModifier{T,N}) where {T,N} = T

## ndims
ModStyle(::AdjOrTransModifier{T,2}, ::IF_ndims) where {T} = ModNothing()
ModStyle(::AdjOrTransModifier{T,1}, ::IF_ndims) where {T} = ModFinal()
mod_ndims_final(::AdjOrTransModifier{T,1}) where {T} = 2

## size
ModStyle(::AdjOrTransModifier, ::IF_size) = ModRecursive()
mod_size_post(mod::AdjOrTransModifier1, size) = (1, size...)
mod_size_post(mod::AdjOrTransModifier2, size) = reverse(size)

## getindex
ModStyle(::AdjOrTransModifier, ::IF_getindex) = ModRecursive()

# First, indexing in the 1D case
# - with a linear index
mod_getindex_pre(mod::AdjOrTransModifier1, i) = (i,)

# - or with two indices. In this case, we check that the first index is 1
checkone(i) = i==1 || throw(BoundsError())
function mod_getindex_pre(mod::AdjOrTransModifier1, i, j)
    @boundscheck checkone(i)
    (j,)
end

mod_getindex_post(mod::TransposeMod1, Z, args...) = modified_transpose(Z)
mod_getindex_post(mod::AdjointMod1, Z, args...) = modified_adjoint(Z)


# - indexing in the 2D case
mod_getindex_pre(mod::AdjOrTransModifier2, i, j) = (j, i)

mod_getindex_post(mod::TransposeMod2, Z, i, j) = modified_transpose(Z)
mod_getindex_post(mod::AdjointMod2, Z, i, j) = modified_adjoint(Z)

## setindex!
ModStyle(::AdjOrTransModifier, ::IF_setindex!) = ModRecursive()

# Like above, first the 1D case with one or two indices
mod_setindex!_pre(mod::TransposeMod1, val, i) = (modified_transpose(val), i)
mod_setindex!_pre(mod::AdjointMod1, val, i) = (modified_adjoint(val), i)

function mod_setindex!_pre(mod::TransposeMod1, val, i, j)
    @boundscheck checkone(i)
    modified_transpose(val), j
end
function mod_setindex!_pre(mod::AdjointMod1, val, i, j)
    @boundscheck checkone(i)
    modified_adjoint(val), j
end

# Then the 2d case
mod_setindex!_pre(mod::TransposeMod2, val, i, j) = (modified_transpose(val), j, i)
mod_setindex!_pre(mod::AdjointMod2, val, i, j) = (modified_adjoint(val), j, i)


## IndexStyle
ModStyle(::Type{<:AdjOrTransModifier}, ::IF_IndexStyle) = ModFinal()

mod_IndexStyle_final(::Type{<:AdjOrTransModifier1}) = IndexLinear()
mod_IndexStyle_final(::Type{<:AdjOrTransModifier2}) = IndexCartesian()


## axes
ModStyle(::AdjOrTransModifier, ::IF_axes) = ModRecursive()

mod_axes_post(mod::AdjOrTransModifier1, axes) = (Base.OneTo(1), axes...)
mod_axes_post(mod::AdjOrTransModifier2, axes) = reverse(axes)


# ## similar
# ModStyle(::AdjOrTransModifier, ::IF_similar) = ModRecursive()
#
# mod_similar_pre(::AdjOrTransModifier, T, dims) = (T, dims)
# mod_similar_post(::AdjointMod, Z, T, dims) = modified_adjoint(Z)
# mod_similar_post(::TransposeMod, Z, T, dims) = modified_transpose(Z)


### The following code is adapted from adjtrans.jl in Julia stdlib

## multiplication

import Base: *, \, /
using LinearAlgebra: dot
import LinearAlgebra: pinv

*(u::AdjointModifiedVector, v::AbstractVector) = dot(u.parent, v)
*(u::TransposeModifiedVector{T}, v::AbstractVector{T}) where {T<:Real} = dot(u.parent, v)
function *(u::TransposeModifiedVector, v::AbstractVector)
    # require_one_based_indexing(u, v)
    @boundscheck length(u) == length(v) || throw(DimensionMismatch())
    return sum(@inbounds(u[k]*v[k]) for k in 1:length(u))
end
# vector * Adjoint/Transpose-vector
*(u::AbstractVector, v::AdjOrTransModifiedVector) = broadcast(*, u, v)
# Adjoint/Transpose-vector * Adjoint/Transpose-vector
# (necessary for disambiguation with fallback methods in linalg/matmul)
*(u::AdjointModifiedVector, v::AdjointModifiedVector) = throw(MethodError(*, (u, v)))
*(u::TransposeModifiedVector, v::TransposeModifiedVector) = throw(MethodError(*, (u, v)))

# AdjOrTransModifiedVector{<:Any,<:AdjOrTransModifiedVector} is a lazy conj vectors
# We need to expand the combinations to avoid ambiguities
(*)(u::TransposeModifiedVector, v::AdjointModifiedVector{<:Any,<:TransposeModifiedVector}) =
    sum(uu*vv for (uu, vv) in zip(u, v))
(*)(u::AdjointModifiedVector,   v::AdjointModifiedVector{<:Any,<:TransposeModifiedVector}) =
    sum(uu*vv for (uu, vv) in zip(u, v))
(*)(u::TransposeModifiedVector, v::TransposeModifiedVector{<:Any,<:AdjointModifiedVector}) =
    sum(uu*vv for (uu, vv) in zip(u, v))
(*)(u::AdjointModifiedVector,   v::TransposeModifiedVector{<:Any,<:AdjointModifiedVector}) =
    sum(uu*vv for (uu, vv) in zip(u, v))

## pseudoinversion
pinv(v::AdjointModifiedVector, tol::Real = 0) = pinv(v.parent, tol).parent
pinv(v::TransposeModifiedVector, tol::Real = 0) = pinv(conj(v.parent)).parent

## left-division \
\(u::AdjOrTransModifiedVector, v::AdjOrTransModifiedVector) = pinv(u) * v


## right-division /
### Lines below disabled due to ambiguities
# /(u::AdjointModifiedVector, A::AbstractMatrix) = modified_adjoint(modified_adjoint(A) \ u.parent)
# /(u::TransposeModifiedVector, A::AbstractMatrix) = modified_transpose(modified_transpose(A) \ u.parent)
/(u::AdjointModifiedVector, A::TransposeModifiedMatrix) = modified_adjoint(conj(A.parent) \ u.parent) # technically should be adjoint(copy(adjoint(copy(A))) \ u.parent)
/(u::TransposeModifiedVector, A::AdjointModifiedMatrix) = modified_transpose(conj(A.parent) \ u.parent) # technically should be transpose(copy(transpose(copy(A))) \ u.parent)
