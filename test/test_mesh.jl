@testset "Mesh" begin
	m1 = icosphere(0)
	@test nvertices(m1) == 12
	@test length(m1.faces) == 20

	m2 = icosphere(1)
	@test nvertices(m2) == 42
	@test length(m2.faces) == 80

	for i in 1:nvertices(m2)
		@test isapprox(norm(m2.dirs[i,:]), 1.0; atol = 1e-10)
	end

	edge_count = Dict{Tuple{Int,Int},Int}()
	for (a,b,c) in m2.faces
		for (i,j) in ((a,b), (b,c), (c,a))
			key = i<j ? (i,j) : (j,i)
			edge_count[key] = get(edge_count, key, 0) + 1
		end
	end

	@test all(v==2 for v in values(edge_count))
end
