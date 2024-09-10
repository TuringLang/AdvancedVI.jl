
@testset "metal" begin
    mtl(zeros(1,3); storage=Metal.SharedStorage)
    Metal.randn(Float32, 16, 32)
end
