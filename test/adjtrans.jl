# This file is copied from Julia itself

module TestAdjointTranspose

using Test, LinearAlgebra, SparseArrays
using ModifiedArrays

# <--- removed inner constructor tests

# <--- removed outer constructor tests

@testset "Adjoint and Transpose basic AbstractArray functionality" begin
    # vectors and matrices with real scalar eltype, and their adjoints/transposes
    intvec, intmat = [1, 2], [1 2 3; 4 5 6]
    tintvec, tintmat = [1 2], [1 4; 2 5; 3 6]
    @testset "length methods" begin
        @test length(modified_adjoint(intvec)) == length(intvec)
        @test length(modified_adjoint(intmat)) == length(intmat)
        @test length(modified_transpose(intvec)) == length(intvec)
        @test length(modified_transpose(intmat)) == length(intmat)
    end
    @testset "size methods" begin
        @test size(modified_adjoint(intvec)) == (1, length(intvec))
        @test size(modified_adjoint(intmat)) == reverse(size(intmat))
        @test size(modified_transpose(intvec)) == (1, length(intvec))
        @test size(modified_transpose(intmat)) == reverse(size(intmat))
    end
    @testset "indices methods" begin
        @test axes(modified_adjoint(intvec)) == (Base.OneTo(1), Base.OneTo(length(intvec)))
        @test axes(modified_adjoint(intmat)) == reverse(axes(intmat))
        @test axes(modified_transpose(intvec)) == (Base.OneTo(1), Base.OneTo(length(intvec)))
        @test axes(modified_transpose(intmat)) == reverse(axes(intmat))
    end
    @testset "IndexStyle methods" begin
        @test IndexStyle(modified_adjoint(intvec)) == IndexLinear()
        @test IndexStyle(modified_adjoint(intmat)) == IndexCartesian()
        @test IndexStyle(modified_transpose(intvec)) == IndexLinear()
        @test IndexStyle(modified_transpose(intmat)) == IndexCartesian()
    end
    # vectors and matrices with complex scalar eltype, and their adjoints/transposes
    complexintvec, complexintmat = [1im, 2im], [1im 2im 3im; 4im 5im 6im]
    tcomplexintvec, tcomplexintmat = [1im 2im], [1im 4im; 2im 5im; 3im 6im]
    acomplexintvec, acomplexintmat = conj.(tcomplexintvec), conj.(tcomplexintmat)
    # vectors and matrices with real-vector and real-matrix eltype, and their adjoints/transposes
    intvecvec = [[1, 2], [3, 4]]
    tintvecvec = [[[1 2]] [[3 4]]]
    intmatmat = [[[1 2]] [[3  4]] [[ 5  6]];
                 [[7 8]] [[9 10]] [[11 12]]]
    tintmatmat = [[hcat([1, 2])] [hcat([7, 8])];
                  [hcat([3, 4])] [hcat([9, 10])];
                  [hcat([5, 6])] [hcat([11, 12])]]
    # vectors and matrices with complex-vector and complex-matrix eltype, and their adjoints/transposes
    complexintvecvec, complexintmatmat = im .* (intvecvec, intmatmat)
    tcomplexintvecvec, tcomplexintmatmat = im .* (tintvecvec, tintmatmat)
    acomplexintvecvec, acomplexintmatmat = conj.(tcomplexintvecvec), conj.(tcomplexintmatmat)
    @testset "getindex methods, elementary" begin
        # implicitly test elementary definitions, for arrays with concrete real scalar eltype
        @test modified_adjoint(intvec) == tintvec
        @test modified_adjoint(intmat) == tintmat
        @test modified_transpose(intvec) == tintvec
        @test modified_transpose(intmat) == tintmat
        # implicitly test elementary definitions, for arrays with concrete complex scalar eltype
        @test modified_adjoint(complexintvec) == acomplexintvec
        @test modified_adjoint(complexintmat) == acomplexintmat
        @test modified_transpose(complexintvec) == tcomplexintvec
        @test modified_transpose(complexintmat) == tcomplexintmat
        # implicitly test elementary definitions, for arrays with concrete real-array eltype
        @test modified_adjoint(intvecvec) == tintvecvec
        @test modified_adjoint(intmatmat) == tintmatmat
        @test modified_transpose(intvecvec) == tintvecvec
        @test modified_transpose(intmatmat) == tintmatmat
        # implicitly test elementary definitions, for arrays with concrete complex-array type
        @test modified_adjoint(complexintvecvec) == acomplexintvecvec
        @test modified_adjoint(complexintmatmat) == acomplexintmatmat
        @test modified_transpose(complexintvecvec) == tcomplexintvecvec
        @test modified_transpose(complexintmatmat) == tcomplexintmatmat
    end
    @testset "getindex(::AdjOrTransVec, ::Colon, ::AbstractArray{Int}) methods that preserve wrapper type" begin
        # for arrays with concrete scalar eltype
        @test modified_adjoint(intvec)[:, [1, 2]] == modified_adjoint(intvec)
        @test modified_transpose(intvec)[:, [1, 2]] == modified_transpose(intvec)
        @test modified_adjoint(complexintvec)[:, [1, 2]] == modified_adjoint(complexintvec)
        @test modified_transpose(complexintvec)[:, [1, 2]] == modified_transpose(complexintvec)
        # for arrays with concrete array eltype
        @test modified_adjoint(intvecvec)[:, [1, 2]] == modified_adjoint(intvecvec)
        @test modified_transpose(intvecvec)[:, [1, 2]] == modified_transpose(intvecvec)
        @test modified_adjoint(complexintvecvec)[:, [1, 2]] == modified_adjoint(complexintvecvec)
        @test modified_transpose(complexintvecvec)[:, [1, 2]] == modified_transpose(complexintvecvec)
    end
    @testset "getindex(::AdjOrTransVec, ::Colon, ::Colon) methods that preserve wrapper type" begin
        # for arrays with concrete scalar eltype
        @test modified_adjoint(intvec)[:, :] == modified_adjoint(intvec)
        @test modified_transpose(intvec)[:, :] == modified_transpose(intvec)
        @test modified_adjoint(complexintvec)[:, :] == modified_adjoint(complexintvec)
        @test modified_transpose(complexintvec)[:, :] == modified_transpose(complexintvec)
        # for arrays with concrete array elype
        @test modified_adjoint(intvecvec)[:, :] == modified_adjoint(intvecvec)
        @test modified_transpose(intvecvec)[:, :] == modified_transpose(intvecvec)
        @test modified_adjoint(complexintvecvec)[:, :] == modified_adjoint(complexintvecvec)
        @test modified_transpose(complexintvecvec)[:, :] == modified_transpose(complexintvecvec)
    end
    @testset "getindex(::AdjOrTransVec, ::Colon, ::Int) should preserve wrapper type on result entries" begin
        # for arrays with concrete scalar eltype
        @test modified_adjoint(intvec)[:, 2] == intvec[2:2]
        @test modified_transpose(intvec)[:, 2] == intvec[2:2]
        @test modified_adjoint(complexintvec)[:, 2] == conj.(complexintvec[2:2])
        @test modified_transpose(complexintvec)[:, 2] == complexintvec[2:2]
        # for arrays with concrete array eltype
        # @test modified_adjoint(intvecvec)[:, 2] == Adjoint.(intvecvec[2:2])
        # @test modified_transpose(intvecvec)[:, 2] == Transpose.(intvecvec[2:2])
        # @test modified_adjoint(complexintvecvec)[:, 2] == Adjoint.(complexintvecvec[2:2])
        # @test modified_transpose(complexintvecvec)[:, 2] == Transpose.(complexintvecvec[2:2])
    end
    @testset "setindex! methods" begin
        # for vectors with real scalar eltype
        @test (wv = modified_adjoint(copy(intvec));
                wv === setindex!(wv, 3, 2) &&
                 wv == setindex!(copy(tintvec), 3, 1, 2)    )
        @test (wv = modified_transpose(copy(intvec));
                wv === setindex!(wv, 4, 2) &&
                 wv == setindex!(copy(tintvec), 4, 1, 2)    )
        # for matrices with real scalar eltype
        @test (wA = modified_adjoint(copy(intmat));
                wA === setindex!(wA, 7, 3, 1) &&
                 wA == setindex!(copy(tintmat), 7, 3, 1)    )
        @test (wA = modified_transpose(copy(intmat));
                wA === setindex!(wA, 7, 3, 1) &&
                 wA == setindex!(copy(tintmat), 7, 3, 1)    )
        # for vectors with complex scalar eltype
        @test (wz = modified_adjoint(copy(complexintvec));
                wz === setindex!(wz, 3im, 2) &&
                 wz == setindex!(copy(acomplexintvec), 3im, 1, 2)   )
        @test (wz = modified_transpose(copy(complexintvec));
                wz === setindex!(wz, 4im, 2) &&
                 wz == setindex!(copy(tcomplexintvec), 4im, 1, 2)   )
        # for  matrices with complex scalar eltype
        @test (wZ = modified_adjoint(copy(complexintmat));
                wZ === setindex!(wZ, 7im, 3, 1) &&
                 wZ == setindex!(copy(acomplexintmat), 7im, 3, 1)   )
        @test (wZ = modified_transpose(copy(complexintmat));
                wZ === setindex!(wZ, 7im, 3, 1) &&
                 wZ == setindex!(copy(tcomplexintmat), 7im, 3, 1)   )
        # for vectors with concrete real-vector eltype
        @test (wv = modified_adjoint(copy(intvecvec));
                wv === setindex!(wv, modified_adjoint([5, 6]), 2) &&
                 wv == setindex!(copy(tintvecvec), [5 6], 2))
        @test (wv = modified_transpose(copy(intvecvec));
                wv === setindex!(wv, modified_transpose([5, 6]), 2) &&
                 wv == setindex!(copy(tintvecvec), [5 6], 2))
        # for matrices with concrete real-matrix eltype
        @test (wA = modified_adjoint(copy(intmatmat));
                wA === setindex!(wA, modified_adjoint([13 14]), 3, 1) &&
                 wA == setindex!(copy(tintmatmat), hcat([13, 14]), 3, 1))
        @test (wA = modified_transpose(copy(intmatmat));
                wA === setindex!(wA, modified_transpose([13 14]), 3, 1) &&
                 wA == setindex!(copy(tintmatmat), hcat([13, 14]), 3, 1))
        # for vectors with concrete complex-vector eltype
        @test (wz = modified_adjoint(copy(complexintvecvec));
                wz === setindex!(wz, modified_adjoint([5im, 6im]), 2) &&
                 wz == setindex!(copy(acomplexintvecvec), [-5im -6im], 2))
        @test (wz = modified_transpose(copy(complexintvecvec));
                wz === setindex!(wz, modified_transpose([5im, 6im]), 2) &&
                 wz == setindex!(copy(tcomplexintvecvec), [5im 6im], 2))
        # for matrices with concrete complex-matrix eltype
        @test (wZ = modified_adjoint(copy(complexintmatmat));
                wZ === setindex!(wZ, modified_adjoint([13im 14im]), 3, 1) &&
                 wZ == setindex!(copy(acomplexintmatmat), hcat([-13im, -14im]), 3, 1))
        @test (wZ = modified_transpose(copy(complexintmatmat));
                wZ === setindex!(wZ, modified_transpose([13im 14im]), 3, 1) &&
                 wZ == setindex!(copy(tcomplexintmatmat), hcat([13im, 14im]), 3, 1))
    end
end

# @testset "Adjoint and Transpose convert methods that convert underlying storage" begin
#     intvec, intmat = [1, 2], [1 2 3; 4 5 6]
#     @test convert(Adjoint{Float64,Vector{Float64}}, modified_adjoint(intvec))::Adjoint{Float64,Vector{Float64}} == modified_adjoint(intvec)
#     @test convert(Adjoint{Float64,Matrix{Float64}}, modified_adjoint(intmat))::Adjoint{Float64,Matrix{Float64}} == modified_adjoint(intmat)
#     @test convert(Transpose{Float64,Vector{Float64}}, modified_transpose(intvec))::Transpose{Float64,Vector{Float64}} == modified_transpose(intvec)
#     @test convert(Transpose{Float64,Matrix{Float64}}, modified_transpose(intmat))::Transpose{Float64,Matrix{Float64}} == modified_transpose(intmat)
# end

# @testset "Adjoint and Transpose similar methods" begin
#     intvec, intmat = [1, 2], [1 2 3; 4 5 6]
#     # similar with no additional specifications, vector (rewrapping) semantics
#     @test size(similar(modified_adjoint(intvec))::Adjoint{Int,Vector{Int}}) == size(modified_adjoint(intvec))
#     @test size(similar(modified_transpose(intvec))::Transpose{Int,Vector{Int}}) == size(modified_transpose(intvec))
#     # similar with no additional specifications, matrix (no-rewrapping) semantics
#     @test size(similar(modified_adjoint(intmat))::Matrix{Int}) == size(modified_adjoint(intmat))
#     @test size(similar(modified_transpose(intmat))::Matrix{Int}) == size(modified_transpose(intmat))
#     # similar with element type specification, vector (rewrapping) semantics
#     @test size(similar(modified_adjoint(intvec), Float64)::Adjoint{Float64,Vector{Float64}}) == size(modified_adjoint(intvec))
#     @test size(similar(modified_transpose(intvec), Float64)::Transpose{Float64,Vector{Float64}}) == size(modified_transpose(intvec))
#     # similar with element type specification, matrix (no-rewrapping) semantics
#     @test size(similar(modified_adjoint(intmat), Float64)::Matrix{Float64}) == size(modified_adjoint(intmat))
#     @test size(similar(modified_transpose(intmat), Float64)::Matrix{Float64}) == size(modified_transpose(intmat))
#     # similar with element type and arbitrary dims specifications
#     shape = (2, 2, 2)
#     @test size(similar(modified_adjoint(intvec), Float64, shape)::Array{Float64,3}) == shape
#     @test size(similar(modified_adjoint(intmat), Float64, shape)::Array{Float64,3}) == shape
#     @test size(similar(modified_transpose(intvec), Float64, shape)::Array{Float64,3}) == shape
#     @test size(similar(modified_transpose(intmat), Float64, shape)::Array{Float64,3}) == shape
# end

@testset "Adjoint and Transpose parent methods" begin
    intvec, intmat = [1, 2], [1 2 3; 4 5 6]
    @test parent(modified_adjoint(intvec)) === intvec
    @test parent(modified_adjoint(intmat)) === intmat
    @test parent(modified_transpose(intvec)) === intvec
    @test parent(modified_transpose(intmat)) === intmat
end

@testset "Adjoint and Transpose vector vec methods" begin
    intvec = [1, 2]
    @test vec(modified_adjoint(intvec)) == intvec
    @test vec(modified_transpose(intvec)) === intvec
    cvec = [1 + 1im]
    @test vec(cvec')[1] == cvec[1]'
end

# @testset "horizontal concatenation of Adjoint/Transpose-wrapped vectors and Numbers" begin
#     # horizontal concatenation of Adjoint/Transpose-wrapped vectors and Numbers
#     # should preserve the Adjoint/Transpose-wrapper to preserve semantics downstream
#     vec, tvec, avec = [1im, 2im], [1im 2im], [-1im -2im]
#     vecvec = [[1im, 2im], [3im, 4im]]
#     tvecvec = [[[1im 2im]] [[3im 4im]]]
#     avecvec = [[[-1im -2im]] [[-3im -4im]]]
#     # for arrays with concrete scalar eltype
#     @test hcat(modified_adjoint(vec), modified_adjoint(vec))::Adjoint{Complex{Int},Vector{Complex{Int}}} == hcat(avec, avec)
#     @test hcat(modified_adjoint(vec), 1, modified_adjoint(vec))::Adjoint{Complex{Int},Vector{Complex{Int}}} == hcat(avec, 1, avec)
#     @test hcat(modified_transpose(vec), modified_transpose(vec))::Transpose{Complex{Int},Vector{Complex{Int}}} == hcat(tvec, tvec)
#     @test hcat(modified_transpose(vec), 1, modified_transpose(vec))::Transpose{Complex{Int},Vector{Complex{Int}}} == hcat(tvec, 1, tvec)
#     # for arrays with concrete array eltype
#     @test hcat(modified_adjoint(vecvec), modified_adjoint(vecvec))::Adjoint{Adjoint{Complex{Int},Vector{Complex{Int}}},Vector{Vector{Complex{Int}}}} == hcat(avecvec, avecvec)
#     @test hcat(modified_transpose(vecvec), modified_transpose(vecvec))::Transpose{Transpose{Complex{Int},Vector{Complex{Int}}},Vector{Vector{Complex{Int}}}} == hcat(tvecvec, tvecvec)
# end

# @testset "map/broadcast over Adjoint/Transpose-wrapped vectors and Numbers" begin
#     # map and broadcast over Adjoint/Transpose-wrapped vectors and Numbers
#     # should preserve the Adjoint/Transpose-wrapper to preserve semantics downstream
#     vec, tvec, avec = [1im, 2im], [1im 2im], [-1im -2im]
#     vecvec = [[1im, 2im], [3im, 4im]]
#     tvecvec = [[[1im 2im]] [[3im 4im]]]
#     avecvec = [[[-1im -2im]] [[-3im -4im]]]
#     # unary map over wrapped vectors with concrete scalar eltype
#     @test map(-, modified_adjoint(vec))::Adjoint{Complex{Int},Vector{Complex{Int}}} == -avec
#     @test map(-, modified_transpose(vec))::Transpose{Complex{Int},Vector{Complex{Int}}} == -tvec
#     # unary map over wrapped vectors with concrete array eltype
#     @test map(-, modified_adjoint(vecvec))::Adjoint{Adjoint{Complex{Int},Vector{Complex{Int}}},Vector{Vector{Complex{Int}}}} == -avecvec
#     @test map(-, modified_transpose(vecvec))::Transpose{Transpose{Complex{Int},Vector{Complex{Int}}},Vector{Vector{Complex{Int}}}} == -tvecvec
#     # binary map over wrapped vectors with concrete scalar eltype
#     @test map(+, modified_adjoint(vec), modified_adjoint(vec))::Adjoint{Complex{Int},Vector{Complex{Int}}} == avec + avec
#     @test map(+, modified_transpose(vec), modified_transpose(vec))::Transpose{Complex{Int},Vector{Complex{Int}}} == tvec + tvec
#     # binary map over wrapped vectors with concrete array eltype
#     @test map(+, modified_adjoint(vecvec), modified_adjoint(vecvec))::Adjoint{Adjoint{Complex{Int},Vector{Complex{Int}}},Vector{Vector{Complex{Int}}}} == avecvec + avecvec
#     @test map(+, modified_transpose(vecvec), modified_transpose(vecvec))::Transpose{Transpose{Complex{Int},Vector{Complex{Int}}},Vector{Vector{Complex{Int}}}} == tvecvec + tvecvec
#     # unary broadcast over wrapped vectors with concrete scalar eltype
#     @test broadcast(-, modified_adjoint(vec))::Adjoint{Complex{Int},Vector{Complex{Int}}} == -avec
#     @test broadcast(-, modified_transpose(vec))::Transpose{Complex{Int},Vector{Complex{Int}}} == -tvec
#     # unary broadcast over wrapped vectors with concrete array eltype
#     @test broadcast(-, modified_adjoint(vecvec))::Adjoint{Adjoint{Complex{Int},Vector{Complex{Int}}},Vector{Vector{Complex{Int}}}} == -avecvec
#     @test broadcast(-, modified_transpose(vecvec))::Transpose{Transpose{Complex{Int},Vector{Complex{Int}}},Vector{Vector{Complex{Int}}}} == -tvecvec
#     # binary broadcast over wrapped vectors with concrete scalar eltype
#     @test broadcast(+, modified_adjoint(vec), modified_adjoint(vec))::Adjoint{Complex{Int},Vector{Complex{Int}}} == avec + avec
#     @test broadcast(+, modified_transpose(vec), modified_transpose(vec))::Transpose{Complex{Int},Vector{Complex{Int}}} == tvec + tvec
#     # binary broadcast over wrapped vectors with concrete array eltype
#     @test broadcast(+, modified_adjoint(vecvec), modified_adjoint(vecvec))::Adjoint{Adjoint{Complex{Int},Vector{Complex{Int}}},Vector{Vector{Complex{Int}}}} == avecvec + avecvec
#     @test broadcast(+, modified_transpose(vecvec), modified_transpose(vecvec))::Transpose{Transpose{Complex{Int},Vector{Complex{Int}}},Vector{Vector{Complex{Int}}}} == tvecvec + tvecvec
#     # trinary broadcast over wrapped vectors with concrete scalar eltype and numbers
#     @test broadcast(+, modified_adjoint(vec), 1, modified_adjoint(vec))::Adjoint{Complex{Int},Vector{Complex{Int}}} == avec + avec .+ 1
#     @test broadcast(+, modified_transpose(vec), 1, modified_transpose(vec))::Transpose{Complex{Int},Vector{Complex{Int}}} == tvec + tvec .+ 1
#     @test broadcast(+, modified_adjoint(vec), 1im, modified_adjoint(vec))::Adjoint{Complex{Int},Vector{Complex{Int}}} == avec + avec .+ 1im
#     @test broadcast(+, modified_transpose(vec), 1im, modified_transpose(vec))::Transpose{Complex{Int},Vector{Complex{Int}}} == tvec + tvec .+ 1im
#     # ascertain inference friendliness, ref. https://github.com/JuliaLang/julia/pull/25083#issuecomment-353031641
#     sparsevec = SparseVector([1.0, 2.0, 3.0])
#     @test map(-, modified_adjoint(sparsevec), modified_adjoint(sparsevec)) isa Adjoint{Float64,SparseVector{Float64,Int}}
#     @test map(-, modified_transpose(sparsevec), modified_transpose(sparsevec)) isa Transpose{Float64,SparseVector{Float64,Int}}
#     @test broadcast(-, modified_adjoint(sparsevec), modified_adjoint(sparsevec)) isa Adjoint{Float64,SparseVector{Float64,Int}}
#     @test broadcast(-, modified_transpose(sparsevec), modified_transpose(sparsevec)) isa Transpose{Float64,SparseVector{Float64,Int}}
#     @test broadcast(+, modified_adjoint(sparsevec), 1.0, modified_adjoint(sparsevec)) isa Adjoint{Float64,SparseVector{Float64,Int}}
#     @test broadcast(+, modified_transpose(sparsevec), 1.0, modified_transpose(sparsevec)) isa Transpose{Float64,SparseVector{Float64,Int}}
# end

@testset "Adjoint/Transpose-wrapped vector multiplication" begin
    realvec, realmat = [1, 2, 3], [1 2 3; 4 5 6; 7 8 9]
    complexvec, complexmat = [1im, 2, -3im], [1im 2 3; 4 5 -6im; 7im 8 9]
    # Adjoint/Transpose-vector * vector
    @test modified_adjoint(realvec) * realvec == dot(realvec, realvec)
    @test modified_transpose(realvec) * realvec == dot(realvec, realvec)
    @test modified_adjoint(complexvec) * complexvec == dot(complexvec, complexvec)
    @test modified_transpose(complexvec) * complexvec == dot(conj(complexvec), complexvec)
    # vector * Adjoint/Transpose-vector
    @test realvec * modified_adjoint(realvec) == broadcast(*, realvec, reshape(realvec, (1, 3)))
    @test realvec * modified_transpose(realvec) == broadcast(*, realvec, reshape(realvec, (1, 3)))
    @test complexvec * modified_adjoint(complexvec) == broadcast(*, complexvec, reshape(conj(complexvec), (1, 3)))
    @test complexvec * modified_transpose(complexvec) == broadcast(*, complexvec, reshape(complexvec, (1, 3)))
    # Adjoint/Transpose-vector * matrix
    ## Removed type assertions
    @test (modified_adjoint(realvec) * realmat) ==
        reshape(copy(modified_adjoint(realmat)) * realvec, (1, 3))
    @test (modified_transpose(realvec) * realmat) ==
        reshape(copy(modified_transpose(realmat)) * realvec, (1, 3))
    @test (modified_adjoint(complexvec) * complexmat) ==
        reshape(conj(copy(modified_adjoint(complexmat)) * complexvec), (1, 3))
    @test (modified_transpose(complexvec) * complexmat) ==
        reshape(copy(modified_transpose(complexmat)) * complexvec, (1, 3))
    # Adjoint/Transpose-vector * Adjoint/Transpose-matrix
    @test (modified_adjoint(realvec) * modified_adjoint(realmat)) ==
        reshape(realmat * realvec, (1, 3))
    @test (modified_transpose(realvec) * modified_transpose(realmat)) ==
        reshape(realmat * realvec, (1, 3))
    @test (modified_adjoint(complexvec) * modified_adjoint(complexmat)) ==
        reshape(conj(complexmat * complexvec), (1, 3))
    @test (modified_transpose(complexvec) * modified_transpose(complexmat)) ==
        reshape(complexmat * complexvec, (1, 3))
end

@testset "Adjoint/Transpose-wrapped vector pseudoinversion" begin
    realvec, complexvec = [1, 2, 3, 4], [1im, 2, 3im, 4]
    rowrealvec, rowcomplexvec = reshape(realvec, (1, 4)), reshape(complexvec, (1, 4))
    # pinv(Adjoint/Transpose-vector) should match matrix equivalents
    # TODO tighten type asserts once pinv yields Transpose/Adjoint
    @test pinv(modified_adjoint(realvec))::Vector{Float64} ≈ pinv(rowrealvec)
    @test pinv(modified_transpose(realvec))::Vector{Float64} ≈ pinv(rowrealvec)
    @test pinv(modified_adjoint(complexvec))::Vector{Complex{Float64}} ≈ pinv(conj(rowcomplexvec))
    @test pinv(modified_transpose(complexvec))::Vector{Complex{Float64}} ≈ pinv(rowcomplexvec)
end

@testset "Adjoint/Transpose-wrapped vector left-division" begin
    realvec, complexvec = [1., 2., 3., 4.,], [1.0im, 2., 3.0im, 4.]
    rowrealvec, rowcomplexvec = reshape(realvec, (1, 4)), reshape(complexvec, (1, 4))
    # \(Adjoint/Transpose-vector, Adjoint/Transpose-vector) should mat matrix equivalents
    @test modified_adjoint(realvec)\modified_adjoint(realvec) ≈ rowrealvec\rowrealvec
    @test modified_transpose(realvec)\modified_transpose(realvec) ≈ rowrealvec\rowrealvec
    @test modified_adjoint(complexvec)\modified_adjoint(complexvec) ≈ conj(rowcomplexvec)\conj(rowcomplexvec)
    @test modified_transpose(complexvec)\modified_transpose(complexvec) ≈ rowcomplexvec\rowcomplexvec
end

@testset "Adjoint/Transpose-wrapped vector right-division" begin
    realvec, realmat = [1, 2, 3], [1 0 0; 0 2 0; 0 0 3]
    complexvec, complexmat = [1im, 2, -3im], [2im 0 0; 0 3 0; 0 0 -5im]
    rowrealvec, rowcomplexvec = reshape(realvec, (1, 3)), reshape(complexvec, (1, 3))
    # /(Adjoint/Transpose-vector, matrix)
    @test (modified_adjoint(realvec) / realmat) ≈ rowrealvec / realmat
    @test (modified_adjoint(complexvec) / complexmat) ≈ conj(rowcomplexvec) / complexmat
    @test (modified_transpose(realvec) / realmat) ≈ rowrealvec / realmat
    @test (modified_transpose(complexvec) / complexmat) ≈ rowcomplexvec / complexmat
    # /(Adjoint/Transpose-vector, Adjoint matrix)
    @test (modified_adjoint(realvec) / modified_adjoint(realmat)) ≈ rowrealvec / copy(modified_adjoint(realmat))
    @test (modified_adjoint(complexvec) / modified_adjoint(complexmat)) ≈ conj(rowcomplexvec) / copy(modified_adjoint(complexmat))
    @test (modified_transpose(realvec) / modified_adjoint(realmat)) ≈ rowrealvec / copy(modified_adjoint(realmat))
    @test (modified_transpose(complexvec) / modified_adjoint(complexmat)) ≈ rowcomplexvec / copy(modified_adjoint(complexmat))
    # /(Adjoint/Transpose-vector, Transpose matrix)
    @test (modified_adjoint(realvec) / modified_transpose(realmat)) ≈ rowrealvec / copy(modified_transpose(realmat))
    @test (modified_adjoint(complexvec) / modified_transpose(complexmat)) ≈ conj(rowcomplexvec) / copy(modified_transpose(complexmat))
    @test (modified_transpose(realvec) / modified_transpose(realmat)) ≈ rowrealvec / copy(modified_transpose(realmat))
    @test (modified_transpose(complexvec) / modified_transpose(complexmat)) ≈ rowcomplexvec / copy(modified_transpose(complexmat))
end

@testset "norm and opnorm of Adjoint/Transpose-wrapped vectors" begin
    # definitions are in base/linalg/generic.jl
    realvec, complexvec = [3, -4], [3im, -4im]
    # one norm result should be sum(abs.(realvec)) == 7
    # two norm result should be sqrt(sum(abs.(realvec))) == 5
    # inf norm result should be maximum(abs.(realvec)) == 4
    for v in (realvec, complexvec)
        @test norm(modified_adjoint(v)) ≈ 5
        @test norm(modified_adjoint(v), 1) ≈ 7
        @test norm(modified_adjoint(v), Inf) ≈ 4
        @test norm(modified_transpose(v)) ≈ 5
        @test norm(modified_transpose(v), 1) ≈ 7
        @test norm(modified_transpose(v), Inf) ≈ 4
    end
    # one opnorm result should be maximum(abs.(realvec)) == 4
    # two opnorm result should be sqrt(sum(abs.(realvec))) == 5
    # inf opnorm result should be sum(abs.(realvec)) == 7
    for v in (realvec, complexvec)
        @test opnorm(modified_adjoint(v)) ≈ 5
        @test opnorm(modified_adjoint(v), 1) ≈ 4
        @test opnorm(modified_adjoint(v), Inf) ≈ 7
        @test opnorm(modified_transpose(v)) ≈ 5
        @test opnorm(modified_transpose(v), 1) ≈ 4
        @test opnorm(modified_transpose(v), Inf) ≈ 7
    end
end

@testset "adjoint and transpose of Numbers" begin
    @test adjoint(1) == 1
    @test adjoint(1.0) == 1.0
    @test adjoint(1im) == -1im
    @test adjoint(1.0im) == -1.0im
    @test transpose(1) == 1
    @test transpose(1.0) == 1.0
    @test transpose(1im) == 1im
    @test transpose(1.0im) == 1.0im
end

@testset "adjoint!(a, b) return a" begin
    a = fill(1.0+im, 5)
    b = fill(1.0+im, 1, 5)
    @test adjoint!(a, b) === a
    @test adjoint!(b, a) === b
end

@testset "aliasing with adjoint and transpose" begin
    A = collect(reshape(1:25, 5, 5)) .+ rand.().*im
    B = copy(A)
    B .= B'
    @test B == A'
    B = copy(A)
    B .= transpose(B)
    @test B == transpose(A)
    B = copy(A)
    B .= B .* B'
    @test B == A .* A'
end

@testset "test show methods for $t of Factorizations" for t in (Adjoint, Transpose)
    A = randn(4, 4)
    F = lu(A)
    Fop = t(F)
    @test "LinearAlgebra."*sprint(show, Fop) ==
                  "$t of "*sprint(show, parent(Fop))
    @test "LinearAlgebra."*sprint((io, t) -> show(io, MIME"text/plain"(), t), Fop) ==
                  "$t of "*sprint((io, t) -> show(io, MIME"text/plain"(), t), parent(Fop))
end

const BASE_TEST_PATH = joinpath(Sys.BINDIR, "..", "share", "julia", "test")
isdefined(Main, :OffsetArrays) || @eval Main include(joinpath($(BASE_TEST_PATH), "testhelpers", "OffsetArrays.jl"))
using .Main.OffsetArrays

@testset "offset axes" begin
    s = Base.Slice(-3:3)'
    @test axes(s) === (Base.OneTo(1), Base.IdentityUnitRange(-3:3))
    @test collect(LinearIndices(s)) == reshape(1:7, 1, 7)
    @test collect(CartesianIndices(s)) == reshape([CartesianIndex(1,i) for i = -3:3], 1, 7)
    @test s[1] == -3
    @test s[7] ==  3
    @test s[4] ==  0
    @test_throws BoundsError s[0]
    @test_throws BoundsError s[8]
    @test s[1,-3] == -3
    @test s[1, 3] ==  3
    @test s[1, 0] ==  0
    @test_throws BoundsError s[1,-4]
    @test_throws BoundsError s[1, 4]
end

end # module TestAdjointTranspose
