
struct WrappedArray{AA,T,N} <: AbstractArray{T,N}
    parent  ::  AA
end

WrappedArray(A::AbstractArray{T,N}) where {T,N} = WrappedArray{typeof(A),T,N}(A)

parent(A::WrappedArray) = A.parent

Base.size(A::WrappedArray) = size(parent(A))

Base.getindex(A::WrappedArray, i::Int) = getindex(parent(A), i)
Base.getindex(A::WrappedArray{T,N}, I::Vararg{Int,N}) where {T,N} =
    getindex(parent(A), I...)

Base.setindex!(A::WrappedArray, val, i::Int) = setindex!(parent(A), val, i)
Base.setindex!(A::WrappedArray{T,N}, val, I::Vararg{Int,N}) where {T,N} =
    setindex!(parent(A), val, I...)

Base.IndexStyle(A::WrappedArray) = IndexStyle(parent(A))

Base.similar(A::WrappedArray, ::Type{S}, dims::Dims) where {S} =
    similar(parent(A), S, dims)

Base.axes(A::WrappedArray) = axes(parent(A))
