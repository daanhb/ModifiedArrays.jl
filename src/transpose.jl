
function check_dimension(::Val{N}) where {N}
    1 <= N <= 2 || error(string(
        "Dimension of adjoint and transpose must be between 1 and 2."))
    return nothing
end

struct TransposeModifier{T,N} <: IntrusiveModifier
    function TransposeModifier{T,N}() where {T,N}
        check_dimension(Val{N}())
        new()
    end
end

struct AdjointModifier{T,N} <: IntrusiveModifier
    function AdjointModifier{T,N}() where {T,N}
        check_dimension(Val{N}())
        new()
    end
end

const TransposeModifiers{T,N} = Union{AdjointModifier{T,N},TransposeModifier{T,N}}

# # Adjoint behaves like transpose regarding size, axes and index style
# specialize(::IF_size, ::AdjointModifier{T,N}) where {T,N} = TransposeModifier{T,N}()
# specialize(::IF_axes, ::AdjointModifier{T,N}) where {T,N} = TransposeModifier{T,N}()
# specialize(::IF_IndexStyle, ::AdjointModifier{T,N}) where {T,N} = TransposeModifier{T,N}()


specialize(::IF_eltype, ::TransposeModifiers{T,N}) where {T,N} = ElementTypeModifier{T}()

# dimension goes from 1 to 2 when the transpose of a vector is taken
specialize(::IF_dimension, ::TransposeModifiers{T,1}) where {T} = DimensionModifier{2}()
specialize(::IF_dimension, ::TransposeModifiers{T,2}) where {T} = NoModifier()

modified_transpose(a::AbstractVector) = modify(a, TransposeModifier{Base.promote_op(transpose,eltype(a)),1}())
modified_transpose(a::AbstractMatrix) = modify(a, TransposeModifier{Base.promote_op(transpose,eltype(a)),2}())
modified_adjoint(a::AbstractVector) = modify(a, AdjointModifier{Base.promote_op(adjoint,eltype(a)),1}())
modified_adjoint(a::AbstractMatrix) = modify(a, AdjointModifier{Base.promote_op(adjoint,eltype(a)),2}())



const TransposeModifier1{T} = TransposeModifier{T,1}
const TransposeModifier2{T} = TransposeModifier{T,2}
const AdjointModifier1{T} = AdjointModifier{T,1}
const AdjointModifier2{T} = AdjointModifier{T,2}


## size
# - The size of a transposed vector of length L becomes (1,L)
recmod(IF::IF_size, ::TransposeModifiers{T,1}, mods, A) where {T} =
    (1, recmod(IF, mods, A)[1])
# - The size of a matrix is simple reversed
recmod(IF::IF_size, ::TransposeModifiers{T,2}, mods, A) where {T} =
    reverse(recmod(IF, mods, A))


## getindex

# Note that we have to take the transpose/adjoint of the element itself, since
# transpose/adjoint are recursive in Julia v1. We use Julia's transpose
# (which may and up using the Transpose and Adjoint types of Base if the
# element type is a vector)

# - For matrices we only have to reverse the order of the indices
recmod(IF::IF_getindex, ::TransposeModifier2, mods, A, i, j) =
    transpose(recmod(IF, mods, A, j, i))
recmod(IF::IF_getindex, ::AdjointModifier2, mods, A, i, j) =
    adjoint(recmod(IF, mods, A, j, i))

# Vectors are more difficult.
# - If the transpose/adjoint of a vector is indexed with two indices, it must
#   be the case that the first index is 1. We hook into Julia's native
#   boundschecking system to enforce this.
#   We avoid calling checkbounds(A, I), since A can be modified by mods.
#   Instead, we call the downstream function `checkbounds_indices` and pass it
#   the correct modified axes. This means we have to throw the error ourselves.
function recmod(IF::IF_getindex, mod::TransposeModifiers{T,1}, mods, A, I::Vararg{Int,2}) where {T}
    @boundscheck Base.checkbounds_indices(Bool, recmod(IF_axes(), axesmodifier(mod), mods, A), I) || throw(BoundsError())
    recmod(IF, mod, mods, A, I[2])
end
# - if a single index is passed, we simply pass it through
recmod(IF::IF_getindex, ::TransposeModifier1, mods, A, i) =
    transpose(recmod(IF, mods, A, i))
recmod(IF::IF_getindex, ::AdjointModifier1, mods, A, i) =
    adjoint(recmod(IF, mods, A, i))


## setindex

# Same comments as for getindex apply.
# - indexing vectors with two indices
function recmod(IF::IF_setindex, mod::TransposeModifiers{T,1}, mods, A, val, I::Vararg{Int,2}) where {T}
    @boundscheck Base.checkbounds_indices(Bool, recmod(IF_axes(), specialize(IF_axes, mod), mods, A), I) || throw(BoundsError())
    recmod(IF, mod, mods, A, val, I[2])
end

# - indexing with a single index
recmod(IF::IF_setindex, ::TransposeModifier1, mods, A, val, i) =
    recmod(IF, mods, A, transpose(val), i)
recmod(IF::IF_setindex, ::AdjointModifier1, mods, A, val, i) =
    recmod(IF, mods, A, adjoint(val), i)

# - matrix case
recmod(IF::IF_setindex, ::TransposeModifier2, mods, A, val, i, j) =
    recmod(IF, mods, A, transpose(val), j, i)
recmod(IF::IF_setindex, ::AdjointModifier2, mods, A, val, i, j) =
    recmod(IF, mods, A, adjoint(val), j, i)

## IndexStyle
recmod(IF::IF_IndexStyle, ::TransposeModifiers{T,1}, mods, A) where {T} = IndexLinear()
recmod(IF::IF_IndexStyle, ::TransposeModifiers{T,2}, mods, A) where {T} = IndexCartesian()

## axes
recmod(IF::IF_axes, mod::TransposeModifiers{T,1}, mods, A) where {T} =
    (Base.OneTo(1), recmod(IF, mods, A)...)
recmod(IF::IF_axes, mod::TransposeModifiers{T,2}, mods, A) where {T} =
    reverse(recmod(IF, mods, A))
