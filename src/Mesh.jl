struct IcosphereMesh
	dirs::Matrix{Float64}
	faces::Vector{NTuple{3,Int}}
	edges::Vector{Tuple{Int, Int}}
end

nvertices(m::IcosphereMesh) = size(m.dirs,1)

function icosphere(subdivision::Int)
	t = (1.0 + sqrt(5.0))/2.0
	raw = [
        	(-1.0,  t,  0.0), ( 1.0,  t,  0.0), (-1.0, -t,  0.0), ( 1.0, -t,  0.0),
        	( 0.0, -1.0,  t), ( 0.0,  1.0,  t), ( 0.0, -1.0, -t), ( 0.0,  1.0, -t),
		( t,  0.0, -1.0), ( t,  0.0,  1.0), (-t,  0.0, -1.0), (-t,  0.0,  1.0),
    	]
	verts = [collect(v)./norm(collect(v)) for v in raw]
	faces = [
        	(1,12,6), (1,6,2), (1,2,8), (1,8,11), (1,11,12),
        	(2,6,10), (6,12,5), (12,11,3), (11,8,7), (8,2,9),
        	(4,10,5), (4,5,3), (4,3,7), (4,7,9), (4,9,10),
        	(5,10,6), (3,5,12), (7,3,11), (9,7,8), (10,9,2),
    	]
	midpoint_cache = Dict{Tuple{Int,Int},Int}()

    	function _midpoint(i,j)
		key = i < j ? (i,j) : (j,i)
	     	haskey(midpoint_cache, key) && return midpoint_cache[key]
	     	m = (verts[i].+verts[j])./2.0
	     	push!(verts, m./norm(m))
	     	idx = length(verts)
	     	midpoint_cache[key] = idx
	     	return idx
     	end
		     
	for _ in 1:subdivision
		new_faces = NTuple{3,Int}[]
		for (a,b,c) in faces
			ab = _midpoint(a,b)
			bc = _midpoint(b,c)
			ca = _midpoint(c,a)
			push!(new_faces, (a,ab,ca))
			push!(new_faces, (b, bc, ab))
			push!(new_faces, (c, ca, bc))
			push!(new_faces, (ab, bc, ca))
		end
		faces = new_faces
	end

	dirs = permutedims(reduce(hcat, verts))
	edges = mesh_edges(faces)

	return IcosphereMesh(Matrix(dirs), faces, edges)
end

function mesh_edges(faces)
	es = Set{Tuple{Int,Int}}()
	for (a,b,c) in faces
		for (i,j) in ((a,b),(b,c),(c,a))
			push!(es, i<j ? (i,j) : (j,i))
		end
	end
	return collect(es)
end

function vertex_position(mesh::IcosphereMesh, logr::AbstractVector)
	r = exp.(logr)
	return mesh.dirs .+ r
end
