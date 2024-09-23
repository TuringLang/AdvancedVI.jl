
@testset "amdgpu" begin
    AMDGPU.ones(Float32, 16)
    AMDGPU.randn(Float32, 16, 16)
end
