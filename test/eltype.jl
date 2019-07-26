
module TestEltypeArrays

using Test
using ModifiedArrays
using ModifiedArrays: EltypeMod

@testset "Modifying element type" begin
    a = [1, 2, 3]
    A = modify(a, EltypeMod{Float64}())
    @test eltype(A) == Float64
    @test eltype(parent(a)) == Int64
    @test A[1] === 1.0
    A[2] = 4.0
    @test A[2] === 4.0
    @test parent(A)[2] === 4
    @test_throws InexactError A[2] = 4.5
    @test parent(A)[2] === 4
end

end # module
