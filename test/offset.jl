
module TestOffsetArrays

using Test, DelimitedFiles
using ModifiedArrays
using ModifiedArrays:
    ModifiedOffsetArray,
    IdentityUnitRange, no_offset_view
# using CatIndices: BidirectionalVector

# These tests were copied verbatim from the OffsetArrays package,
# https://github.com/JuliaArrays/OffsetArrays.jl
# See the package for the various authors of the tests.
#
# The tests were adapted to the ModifiedArrays package in the following way:
# - occurrences of the OffsetArray constructor are modified to use ModifiedOffsetArray
# - explicit references to the offsets fields of an OffsetArray were changed to
#   refer to the offsets field of the OffsetMod in this package
# - various "isa" checks that check for the type OffsetArray now check for
#   the type SimpleOffsetArray

# @test isempty(detect_ambiguities(ModifiedArrays, Core))
# println(detect_ambiguities(ModifiedArrays, Base, Core))
@test isempty(detect_ambiguities(ModifiedArrays, Base, Core))

@testset "Single-entry arrays in dims 0:5" begin
    for n = 0:5
        for z in (ModifiedOffsetArray(ones(Int,ntuple(d->1,n)), ntuple(x->x-1,n)),
                  fill!(ModifiedOffsetArray{Float64}(undef, ntuple(x->x:x, n)), 1),
                  fill!(ModifiedOffsetArray{Float64}(undef, ntuple(x->x:x, n)...), 1),
                  fill!(ModifiedOffsetArray{Float64,n}(undef, ntuple(x->x:x, n)), 1),
                  fill!(ModifiedOffsetArray{Float64,n}(undef, ntuple(x->x:x, n)...), 1))
            @test length(LinearIndices(z)) == 1
            @test axes(z) == ntuple(x->x:x, n)
            @test z[1] == 1
        end
    end
    a0 = reshape([3])
    a = ModifiedOffsetArray(a0)
    @test axes(a) == ()
    @test ndims(a) == 0
    @test a[] == 3
end

@testset "OffsetVector constructors" begin
    local v = rand(5)
    @test ModifiedOffsetVector(v, -2) == ModifiedOffsetArray(v, -2)
    @test ModifiedOffsetVector(v, -2:2) == ModifiedOffsetArray(v, -2:2)
    @test typeof(ModifiedOffsetVector{Float64}(undef, -2:2)) == typeof(ModifiedOffsetArray{Float64}(undef, -2:2))
end

@testset "undef, missing, and nothing constructors" begin
    y = ModifiedOffsetArray{Float32}(undef, (IdentityUnitRange(-1:1),))
    @test axes(y) === (IdentityUnitRange(-1:1),)

    for (T, t) in ((Missing, missing), (Nothing, nothing))
        @test !isassigned(ModifiedOffsetArray{Union{T,Vector{Int}}}(undef, -1:1, -1:1), -1, -1)
        @test ModifiedOffsetArray{Union{T,Vector{Int}}}(t, -1:1, -1:1)[-1, -1] === t
        @test !isassigned(ModifiedOffsetVector{Union{T,Vector{Int}}}(undef, -1:1), -1)
        @test ModifiedOffsetVector{Union{T,Vector{Int}}}(t, -1:1)[-1] === t
    end
end

@testset "high dimensionality" begin
    y = ModifiedOffsetArray{Float64}(undef, -1:1, -7:7, -128:512, -5:5, -1:1, -3:3, -2:2, -1:1)
    @test axes(y) == (-1:1, -7:7, -128:512, -5:5, -1:1, -3:3, -2:2, -1:1)
    y[-1,-7,-128,-5,-1,-3,-2,-1] = 14
    y[-1,-7,-128,-5,-1,-3,-2,-1] += 5
    @test y[-1,-7,-128,-5,-1,-3,-2,-1] == 19
end

@testset "Offset range construction" begin
    r = -2:5
    y = ModifiedOffsetArray(r, r)
    @test axes(y) == (r,)
    y = ModifiedOffsetArray(r, (r,))
    @test axes(y) == (r,)
end

@testset "Traits" begin
    A0 = [1 3; 2 4]
    A = ModifiedOffsetArray(A0, (-1,2))                   # IndexLinear
    S = ModifiedOffsetArray(view(A0, 1:2, 1:2), (-1,2))   # IndexCartesian
    @test axes(A) == axes(S) == (0:1, 3:4)
    @test size(A) == size(A0)
    @test size(A, 1) == size(A0, 1)
    @test length(A) == length(A0)
    @test A == ModifiedOffsetArray(A0, 0:1, 3:4)
    @test_throws DimensionMismatch ModifiedOffsetArray(A0, 0:2, 3:4)
    @test_throws DimensionMismatch ModifiedOffsetArray(A0, 0:1, 2:4)
end

@testset "Scalar indexing" begin
    A0 = [1 3; 2 4]
    A = ModifiedOffsetArray(A0, (-1,2))
    S = ModifiedOffsetArray(view(A0, 1:2, 1:2), (-1,2))

    @test @inferred(A[0,3]) == @inferred(A[0,3,1]) == @inferred(A[1]) == @inferred(S[0,3]) == @inferred(S[0,3,1]) == @inferred(S[1]) == 1
    @test A[1,3] == A[1,3,1] == A[2] == S[1,3] == S[1,3,1] == S[2] == 2
    @test A[0,4] == A[0,4,1] == A[3] == S[0,4] == S[0,4,1] == S[3] == 3
    @test A[1,4] == A[1,4,1] == A[4] == S[1,4] == S[1,4,1] == S[4] == 4
    @test @inbounds(A[0,3]) == @inbounds(A[0,3,1]) == @inbounds(A[1]) == @inbounds(S[0,3]) == @inbounds(S[0,3,1]) == @inbounds(S[1]) == 1
    @test @inbounds(A[1,3]) == @inbounds(A[1,3,1]) == @inbounds(A[2]) == @inbounds(S[1,3]) == @inbounds(S[1,3,1]) == @inbounds(S[2]) == 2
    @test @inbounds(A[0,4]) == @inbounds(A[0,4,1]) == @inbounds(A[3]) == @inbounds(S[0,4]) == @inbounds(S[0,4,1]) == @inbounds(S[3]) == 3
    @test @inbounds(A[1,4]) == @inbounds(A[1,4,1]) == @inbounds(A[4]) == @inbounds(S[1,4]) == @inbounds(S[1,4,1]) == @inbounds(S[4]) == 4
    @test_throws BoundsError A[1,1]
    @test_throws BoundsError S[1,1]
    @test_throws BoundsError A[0,3,2]
    @test_throws BoundsError A[0,3,0]
    Ac = copy(A)
    Ac[0,3] = 10
    @test Ac[0,3] == 10
    Ac[0,3,1] = 11
    @test Ac[0,3] == 11
    @inbounds Ac[0,3,1] = 12
    @test Ac[0,3] == 12
end

@testset "Vector indexing" begin
    A0 = [1 3; 2 4]
    A = ModifiedOffsetArray(A0, (-1,2))
    S = ModifiedOffsetArray(view(A0, 1:2, 1:2), (-1,2))

    @test A[:, 3] == S[:, 3] == ModifiedOffsetArray([1,2], (A.modifier.offsets[1],))
    @test A[:, 4] == S[:, 4] == ModifiedOffsetArray([3,4], (A.modifier.offsets[1],))
    @test_throws BoundsError A[:, 1]
    @test_throws BoundsError S[:, 1]
    @test A[0, :] == S[0, :] == ModifiedOffsetArray([1,3], (A.modifier.offsets[2],))
    @test A[1, :] == S[1, :] == ModifiedOffsetArray([2,4], (A.modifier.offsets[2],))
    @test_throws BoundsError A[2, :]
    @test_throws BoundsError S[2, :]
    @test A[0:1, 3] == S[0:1, 3] == [1,2]
    @test A[[1,0], 3] == S[[1,0], 3] == [2,1]
    @test A[0, 3:4] == S[0, 3:4] == [1,3]
    @test A[1, [4,3]] == S[1, [4,3]] == [4,2]
    @test A[:, :] == S[:, :] == A
end

@testset "Vector indexing with offset ranges" begin
    r = ModifiedOffsetArray(8:10, -1:1)
    r1 = r[0:1]
    @test r1 === 9:10
    r1 = (8:10)[ModifiedOffsetArray(1:2, -5:-4)]
    @test axes(r1) === (IdentityUnitRange(-5:-4),)
    @test parent(r1) === 8:9
    r1 = ModifiedOffsetArray(8:10, -1:1)[ModifiedOffsetArray(0:1, -5:-4)]
    @test axes(r1) === (IdentityUnitRange(-5:-4),)
    @test parent(r1) === 9:10
end

@testset "CartesianIndexing" begin
    A0 = [1 3; 2 4]
    A = ModifiedOffsetArray(A0, (-1,2))
    S = ModifiedOffsetArray(view(A0, 1:2, 1:2), (-1,2))

    @test A[CartesianIndex((0,3))] == S[CartesianIndex((0,3))] == 1
    @test A[CartesianIndex((0,3)),1] == S[CartesianIndex((0,3)),1] == 1
    @test @inbounds(A[CartesianIndex((0,3))]) == @inbounds(S[CartesianIndex((0,3))]) == 1
    @test @inbounds(A[CartesianIndex((0,3)),1]) == @inbounds(S[CartesianIndex((0,3)),1]) == 1
    @test_throws BoundsError A[CartesianIndex(1,1)]
    @test_throws BoundsError A[CartesianIndex(1,1),0]
    @test_throws BoundsError A[CartesianIndex(1,1),2]
    @test_throws BoundsError S[CartesianIndex(1,1)]
    @test_throws BoundsError S[CartesianIndex(1,1),0]
    @test_throws BoundsError S[CartesianIndex(1,1),2]
    @test eachindex(A) == 1:4
    @test eachindex(S) == CartesianIndices(IdentityUnitRange.((0:1,3:4)))
end

@testset "view" begin
    A0 = [1 3; 2 4]
    A = ModifiedOffsetArray(A0, (-1,2))

    S = view(A, :, 3)
    @test S == ModifiedOffsetArray([1,2], (A.modifier.offsets[1],))
    @test S[0] == 1
    @test S[1] == 2
    @test_throws BoundsError S[2]
    @test axes(S) === (IdentityUnitRange(0:1),)
    S = view(A, 0, :)
    @test S == ModifiedOffsetArray([1,3], (A.modifier.offsets[2],))
    @test S[3] == 1
    @test S[4] == 3
    @test_throws BoundsError S[1]
    @test axes(S) === (IdentityUnitRange(3:4),)
    S = view(A, 0:0, 4)
    @test S == [3]
    @test S[1] == 3
    @test_throws BoundsError S[0]
    @test axes(S) === (Base.OneTo(1),)
    S = view(A, 1, 3:4)
    @test S == [2,4]
    @test S[1] == 2
    @test S[2] == 4
    @test_throws BoundsError S[3]
    @test axes(S) === (Base.OneTo(2),)
    S = view(A, :, :)
    @test S == A
    @test S[0,3] == S[1] == 1
    @test S[1,3] == S[2] == 2
    @test S[0,4] == S[3] == 3
    @test S[1,4] == S[4] == 4
    @test_throws BoundsError S[1,1]
    @test axes(S) === IdentityUnitRange.((0:1, 3:4))
end

@testset "iteration" begin
    A0 = [1 3; 2 4]
    A = ModifiedOffsetArray(A0, (-1,2))

    let a
        for (a,d) in zip(A, A0)
            @test a == d
        end
    end
end

@testset "show/summary" begin
    A0 = [1 3; 2 4]
    A = ModifiedOffsetArray(A0, (-1,2))
    S = ModifiedOffsetArray(view(A0, 1:2, 1:2), (-1,2))

    @test sprint(show, A) == "[1 3; 2 4]"
    @test sprint(show, S) == "[1 3; 2 4]"
    strs = split(strip(sprint(show, MIME("text/plain"), A)), '\n')
    @test strs[2] == " 1  3"
    @test strs[3] == " 2  4"
    v = ModifiedOffsetArray(rand(3), (-2,))
    @test sprint(show, v) == sprint(show, parent(v))
    io = IOBuffer()
    function cmp_showf(printfunc, io, A)
        ioc = IOContext(io, :limit=>true, :compact=>true)
        printfunc(ioc, A)
        str1 = String(take!(io))
        printfunc(ioc, parent(A))
        str2 = String(take!(io))
        @test str1 == str2
    end
    cmp_showf(Base.print_matrix, io, ModifiedOffsetArray(rand(5,5), (10,-9)))       # rows&cols fit
    cmp_showf(Base.print_matrix, io, ModifiedOffsetArray(rand(10^3,5), (10,-9)))    # columns fit
    cmp_showf(Base.print_matrix, io, ModifiedOffsetArray(rand(5,10^3), (10,-9)))    # rows fit
    cmp_showf(Base.print_matrix, io, ModifiedOffsetArray(rand(10^3,10^3), (10,-9))) # neither fits

    a = ModifiedOffsetArray([1 2; 3 4], -1:0, 5:6)
    shownsz = VERSION >= v"1.2.0-DEV.229" ? Base.dims2string(size(a))*' ' : ""
    @test summary(a) == "$(shownsz)ModifiedOffsetArray(::Array{$(Int),2}, -1:0, 5:6) with eltype $(Int) with indices -1:0×5:6"
    shownsz = VERSION >= v"1.2.0-DEV.229" ? Base.dims2string(size(view(a, :, 5)))*' ' : ""
    @test summary(view(a, :, 5)) == "$(shownsz)view(ModifiedOffsetArray(::Array{$(Int),2}, -1:0, 5:6), :, 5) with eltype $(Int) with indices -1:0"
    a = ModifiedOffsetArray(reshape([1]))
    @test summary(a) == "0-dimensional ModifiedOffsetArray(::Array{$(Int),0}) with eltype $(Int)"

    show(io, ModifiedOffsetArray(3:5, 0:2))
    @test String(take!(io)) == "3:5 with indices 0:2"
end

@testset "readdlm/writedlm" begin
    A0 = [1 3; 2 4]
    A = ModifiedOffsetArray(A0, (-1,2))

    io = IOBuffer()
    writedlm(io, A)
    seek(io, 0)
    @test readdlm(io, eltype(A)) == parent(A)
end

@testset "similar" begin
    A0 = [1 3; 2 4]
    A = ModifiedOffsetArray(A0, (-1,2))

    B = similar(A, Float32)
    @test isa(B, ModifiedArrays.ModifiedArray{Float32,2})
    @test isa(B, ModifiedArrays.ModifiedOffsetArray{Float32,2})
    @test axes(B) === axes(A)
    B = similar(A, (3,4))
    @test isa(B, Array{Int,2})
    @test size(B) == (3,4)
    @test axes(B) === (Base.OneTo(3), Base.OneTo(4))
    B = similar(A, (-3:3,1:4))
    @test isa(B, ModifiedOffsetArray{Int,2})
    @test axes(B) === IdentityUnitRange.((-3:3, 1:4))
    B = similar(parent(A), (-3:3,1:4))
    @test isa(B, ModifiedOffsetArray{Int,2})
    @test axes(B) === IdentityUnitRange.((-3:3, 1:4))
    @test isa([x for x in [1,2,3]], Vector{Int})
    @test similar(Array{Int}, (0:0, 0:0)) isa ModifiedOffsetArray{Int, 2}
    @test similar(Array{Int}, (1, 1)) isa Matrix{Int}
    @test similar(Array{Int}, (Base.OneTo(1), Base.OneTo(1))) isa Matrix{Int}
end

@testset "reshape" begin
    A0 = [1 3; 2 4]
    A = ModifiedOffsetArray(A0, (-1,2))

    B = reshape(A0, -10:-9, 9:10)
    @test isa(B, ModifiedOffsetArray{Int,2})
    @test parent(B) === A0
    @test axes(B) == IdentityUnitRange.((-10:-9, 9:10))
    B = reshape(A, -10:-9, 9:10)
    @test isa(B, ModifiedOffsetArray{Int,2})
    @test pointer(parent(B)) === pointer(A0)
    @test axes(B) == IdentityUnitRange.((-10:-9, 9:10))
    b = reshape(A, -7:-4)
    @test axes(b) == (IdentityUnitRange(-7:-4),)
    @test isa(parent(b), Vector{Int})
    @test pointer(parent(b)) === pointer(parent(A))
    @test parent(b) == A0[:]
    a = ModifiedOffsetArray(rand(3,3,3), -1:1, 0:2, 3:5)
    # Offset axes are required for reshape(::ModifiedOffsetArray, ::Val) support
    b = reshape(a, Val(2))
    @test isa(b, ModifiedOffsetArray{Float64,2})
    @test pointer(parent(b)) === pointer(parent(a))
    @test axes(b) == IdentityUnitRange.((-1:1, 1:9))
    b = reshape(a, Val(4))
    @test isa(b, ModifiedOffsetArray{Float64,4})
    @test pointer(parent(b)) === pointer(parent(a))
    @test axes(b) == (axes(a)..., IdentityUnitRange(1:1))
end

@testset "Indexing with ModifiedOffsetArray axes" begin
    A0 = [1 3; 2 4]

    i1 = ModifiedOffsetArray([2,1], (-5,))
    i1 = ModifiedOffsetArray([2,1], -5)
    b = A0[i1, 1]
    @test axes(b) === (IdentityUnitRange(-4:-3),)
    @test b[-4] == 2
    @test b[-3] == 1
    b = A0[1,i1]
    @test axes(b) === (IdentityUnitRange(-4:-3),)
    @test b[-4] == 3
    @test b[-3] == 1
    v = view(A0, i1, 1)
    @test axes(v) === (IdentityUnitRange(-4:-3),)
    v = view(A0, 1:1, i1)
    @test axes(v) === (Base.OneTo(1), IdentityUnitRange(-4:-3))

    for r in (1:10, 1:1:10, StepRangeLen(1, 1, 10), LinRange(1, 10, 10))
        for s in (IdentityUnitRange(2:3), ModifiedOffsetArray(2:3, 2:3))
            @test axes(r[s]) == axes(s)
        end
    end
end

@testset "logical indexing" begin
    A0 = [1 3; 2 4]
    A = ModifiedOffsetArray(A0, (-1,2))

    @test A[A .> 2] == [3,4]
end

@testset "copyto!" begin
    a = ModifiedOffsetArray{Int}(undef, (-3:-1,))
    fill!(a, -1)
    copyto!(a, (1,2))   # non-array iterables
    @test a[-3] == 1
    @test a[-2] == 2
    @test a[-1] == -1
    fill!(a, -1)
    copyto!(a, -2, (1,2))
    @test a[-3] == -1
    @test a[-2] == 1
    @test a[-1] == 2
    @test_throws BoundsError copyto!(a, 1, (1,2))
    fill!(a, -1)
    copyto!(a, -2, (1,2,3), 2)
    @test a[-3] == -1
    @test a[-2] == 2
    @test a[-1] == 3
    @test_throws BoundsError copyto!(a, -2, (1,2,3), 1)
    fill!(a, -1)
    copyto!(a, -2, (1,2,3), 1, 2)
    @test a[-3] == -1
    @test a[-2] == 1
    @test a[-1] == 2

    b = 1:2    # copy between AbstractArrays
    bo = ModifiedOffsetArray(1:2, (-3,))
    @test_throws BoundsError copyto!(a, b)
    fill!(a, -1)
    copyto!(a, bo)
    @test a[-3] == -1
    @test a[-2] == 1
    @test a[-1] == 2
    fill!(a, -1)
    copyto!(a, -2, bo)
    @test a[-3] == -1
    @test a[-2] == 1
    @test a[-1] == 2
    @test_throws BoundsError copyto!(a, -4, bo)
    @test_throws BoundsError copyto!(a, -1, bo)
    fill!(a, -1)
    copyto!(a, -3, b, 2)
    @test a[-3] == 2
    @test a[-2] == a[-1] == -1
    @test_throws BoundsError copyto!(a, -3, b, 1, 4)
    am = ModifiedOffsetArray{Int}(undef, (1:1, 7:9))  # for testing linear indexing
    fill!(am, -1)
    copyto!(am, b)
    @test am[1] == 1
    @test am[2] == 2
    @test am[3] == -1
    @test am[1,7] == 1
    @test am[1,8] == 2
    @test am[1,9] == -1
end

@testset "map" begin
    am = ModifiedOffsetArray{Int}(undef, (1:1, 7:9))  # for testing linear indexing
    fill!(am, -1)
    copyto!(am, 1:2)

    dest = similar(am)
    map!(+, dest, am, am)
    @test dest[1,7] == 2
    @test dest[1,8] == 4
    @test dest[1,9] == -2
end

@testset "reductions" begin
    A = ModifiedOffsetArray(rand(4,4), (-3,5))
    @test maximum(A) == maximum(parent(A))
    @test minimum(A) == minimum(parent(A))
    @test extrema(A) == extrema(parent(A))
    C = similar(A)
    cumsum!(C, A, dims = 1)
    @test parent(C) == cumsum(parent(A), dims = 1)
    @test parent(cumsum(A, dims = 1)) == cumsum(parent(A), dims = 1)
    cumsum!(C, A, dims = 2)
    @test parent(C) == cumsum(parent(A), dims = 2)
    R = similar(A, (1:1, 6:9))
    maximum!(R, A)
    @test parent(R) == maximum(parent(A), dims = 1)
    R = similar(A, (-2:1, 1:1))
    maximum!(R, A)
    @test parent(R) == maximum(parent(A), dims = 2)
    amin, iamin = findmin(A)
    pmin, ipmin = findmin(parent(A))
    @test amin == pmin
    @test A[iamin] == amin
    @test amin == parent(A)[ipmin]
    amax, iamax = findmax(A)
    pmax, ipmax = findmax(parent(A))
    @test amax == pmax
    @test A[iamax] == amax
    @test amax == parent(A)[ipmax]

    amin, amax = extrema(parent(A))
    @test clamp.(A, (amax+amin)/2, amax) == ModifiedOffsetArray(clamp.(parent(A), (amax+amin)/2, amax), axes(A))
end

# v  = ModifiedOffsetArray([1,1e100,1,-1e100], (-3,))*1000
# v2 = ModifiedOffsetArray([1,-1e100,1,1e100], (5,))*1000
# @test isa(v, ModifiedOffsetArray)
# cv  = ModifiedOffsetArray([1,1e100,1e100,2], (-3,))*1000
# cv2 = ModifiedOffsetArray([1,-1e100,-1e100,2], (5,))*1000
# @test isequal(cumsum_kbn(v), cv)
# @test isequal(cumsum_kbn(v2), cv2)
# @test isequal(sum_kbn(v), sum_kbn(parent(v)))

@testset "Collections" begin
    A = ModifiedOffsetArray(rand(4,4), (-3,5))

    @test unique(A, dims=1) == ModifiedOffsetArray(parent(A), 0, first(axes(A, 2)) - 1)
    @test unique(A, dims=2) == ModifiedOffsetArray(parent(A), first(axes(A, 1)) - 1, 0)
    v = ModifiedOffsetArray(rand(8), (-2,))
    @test sort(v) == ModifiedOffsetArray(sort(parent(v)), v.modifier.offsets)
    @test sortslices(A; dims=1) == ModifiedOffsetArray(sortslices(parent(A); dims=1), A.modifier.offsets)
    @test sortslices(A; dims=2) == ModifiedOffsetArray(sortslices(parent(A); dims=2), A.modifier.offsets)
    @test sort(A, dims = 1) == ModifiedOffsetArray(sort(parent(A), dims = 1), A.modifier.offsets)
    @test sort(A, dims = 2) == ModifiedOffsetArray(sort(parent(A), dims = 2), A.modifier.offsets)

    @test mapslices(v->sort(v), A, dims = 1) == ModifiedOffsetArray(mapslices(v->sort(v), parent(A), dims = 1), A.modifier.offsets)
    @test mapslices(v->sort(v), A, dims = 2) == ModifiedOffsetArray(mapslices(v->sort(v), parent(A), dims = 2), A.modifier.offsets)
end

@testset "rot/reverse" begin
    A = ModifiedOffsetArray(rand(4,4), (-3,5))

    @test rotl90(A) == ModifiedOffsetArray(rotl90(parent(A)), A.modifier.offsets[[2,1]])
    @test rotr90(A) == ModifiedOffsetArray(rotr90(parent(A)), A.modifier.offsets[[2,1]])
    @test reverse(A, dims = 1) == ModifiedOffsetArray(reverse(parent(A), dims = 1), A.modifier.offsets)
    @test reverse(A, dims = 2) == ModifiedOffsetArray(reverse(parent(A), dims = 2), A.modifier.offsets)
end

@testset "fill" begin
    B = fill(5, 1:3, -1:1)
    @test axes(B) == (1:3,-1:1)
    @test all(B.==5)
end

@testset "broadcasting" begin
    A = ModifiedOffsetArray(rand(4,4), (-3,5))

    @test A.+1 == ModifiedOffsetArray(parent(A).+1, A.modifier.offsets)
    @test 2*A == ModifiedOffsetArray(2*parent(A), A.modifier.offsets)
    @test A+A == ModifiedOffsetArray(parent(A)+parent(A), A.modifier.offsets)
    @test A.*A == ModifiedOffsetArray(parent(A).*parent(A), A.modifier.offsets)
end

@testset "@inbounds" begin
    a = ModifiedOffsetArray(zeros(7), -3:3)
    unsafe_fill!(x) = @inbounds(for i in axes(x,1); x[i] = i; end)
    function unsafe_sum(x)
        s = zero(eltype(x))
        @inbounds for i in axes(x,1)
            s += x[i]
        end
        s
    end
    unsafe_fill!(a)
    for i = -3:3
        @test a[i] == i
    end
    @test unsafe_sum(a) == 0
end

@testset "Resizing OffsetVectors" begin
    local a = ModifiedOffsetVector(rand(5),-3)
    axes(a,1) == -2:2
    length(a) == 5
    resize!(a,3)
    length(a) == 3
    axes(a,1) == -2:0
    @test_throws ArgumentError resize!(a,-3)
end

####
#### type defined for testing no_offset_view
####

struct NegativeArray{T,N,S <: AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::S
end

Base.axes(A::NegativeArray) = map(n -> (-n):(-1), size(A.parent))

Base.size(A::NegativeArray) = size(A.parent)

function Base.getindex(A::NegativeArray{T,N}, I::Vararg{Int,N}) where {T,N}
    getindex(A.parent, (I .+ size(A.parent) .+ 1)...)
end

@testset "no offset view" begin
    # ModifiedOffsetArray fallback
    A = randn(3, 3)
    O1 = ModifiedOffsetArray(A, -1:1, 0:2)
    O2 = ModifiedOffsetArray(O1, -2:0, -3:(-1))
    @test no_offset_view(O2) ≡ A

    # generic fallback
    A = collect(reshape(1:12, 3, 4))
    N = NegativeArray(A)
    @test N[-3, -4] == 1
    V = no_offset_view(N)
    @test collect(V) == A

    # bidirectional
    # B = BidirectionalVector([1, 2, 3])
    # pushfirst!(B, 0)
    # OB = ModifiedArrays.no_offset_view(B)
    # @test axes(OB, 1) == 1:4
    # @test collect(OB) == 0:3
end

@testset "no nesting" begin
    A = randn(2, 3)
    x = A[2, 2]
    O1 = ModifiedOffsetArray(A, -1:0, -1:1)
    O2 = ModifiedOffsetArray(O1, 0:1, 0:2)
    @test parent(O1) ≡ parent(O2)
    @test eltype(O1) ≡ eltype(O2)
    O2[1, 1] = x + 1            # just a sanity check
    @test A[2, 2] == x + 1
end

@testset "mutating functions for OffsetVector" begin
    # push!
    o = ModifiedOffsetVector(Int[], -1)
    @test push!(o) === o
    @test axes(o, 1) == 0:-1
    @test push!(o, 1) === o
    @test axes(o, 1) == 0:0
    @test o[end] == 1
    @test push!(o, 2, 3) === o
    @test axes(o, 1) == 0:2
    @test o[end-1:end] == [2, 3]
    # pop!
    o = ModifiedOffsetVector([1, 2, 3], -1)
    @test pop!(o) == 3
    @test axes(o, 1) == 0:1
    # empty!
    o = ModifiedOffsetVector([1, 2, 3], -1)
    @test empty!(o) === o
    @test axes(o, 1) == 0:-1
end

end #module
