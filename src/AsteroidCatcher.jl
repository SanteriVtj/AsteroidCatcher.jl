module AsteroidCatcher

	using LinearAlgebra, Statistics, Printf, Random, Optim


	include("Mesh.jl")

	export IcosphereMesh, icosphere, nvertices, vertex_positions
end
