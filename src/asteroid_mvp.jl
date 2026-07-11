# ==================================================================
# asteroid_mvp.jl
#
# MVP for the Helsinki Asteroid Challenge 2026: recover a convex-ish
# shape from binary + intensity lightcurves, using a fixed-topology
# triangulated sphere (icosphere) whose vertex RADII are the only
# free parameters. This is convex lightcurve inversion (Kaasalainen &
# Torppa 2001) done directly on vertices instead of facet areas, with
# no self-occlusion/shadowing yet (that's the next stage, for
# concavities).
#
# Run with:   julia asteroid_mvp.jl
# Deps:       ] add Optim
#
# WHAT THIS FILE DOES
#   1. Builds an icosphere (fixed vertex directions + face connectivity).
#   2. Forward model: for known light/camera directions and known
#      rotation phase, computes the binary curve (silhouette area
#      proxy) and intensity curve (LS+Lambert brightness) as a sum
#      over facets with mu>0 and mu0>0 (valid exactly for convex
#      shapes; a reasonable approximation for mild non-convexity).
#   3. A synthetic self-test: build a known test ellipsoid, render its
#      curves, add noise, then recover the shape from the curves alone
#      via least squares + Tikhonov smoothness, using Optim.jl.
#   4. Exports the recovered mesh to STL.
#
# WHERE TO EXTEND (see comments marked EXTEND):
#   - load_curves(): swap in the real challenge data format.
#   - render_curves(): add ray-based self-occlusion/shadowing for
#     concavities (Stage 2 of the outlined pipeline).
#   - objective(): replace the Tikhonov penalty with a learned
#     regularizer / denoiser prior (Stage 3, the data-driven part).
#   - calibrate a,b (the LS/Lambert mix) on the public ground-truth
#     models instead of hardcoding them.
# ==================================================================

using LinearAlgebra
using Statistics
using Printf
using Random
using Optim

# ------------------------------------------------------------------
# 1. Icosphere: fixed vertex directions + triangle connectivity
# ------------------------------------------------------------------

"""
    icosphere(subdivisions::Int) -> (dirs::Matrix{Float64}, faces::Vector{NTuple{3,Int}})

`dirs` is N x 3 (unit vertex directions), `faces` are 1-based vertex
index triples. Only radii along these fixed directions are optimized;
the topology never changes.
"""
function icosphere(subdivisions::Int)
    t = (1.0 + sqrt(5.0)) / 2.0
    raw = [
        (-1.0,  t,  0.0), ( 1.0,  t,  0.0), (-1.0, -t,  0.0), ( 1.0, -t,  0.0),
        ( 0.0, -1.0,  t), ( 0.0,  1.0,  t), ( 0.0, -1.0, -t), ( 0.0,  1.0, -t),
        ( t,  0.0, -1.0), ( t,  0.0,  1.0), (-t,  0.0, -1.0), (-t,  0.0,  1.0),
    ]
    verts = [collect(v) ./ norm(collect(v)) for v in raw]

    faces = [
        (1,12,6), (1,6,2), (1,2,8), (1,8,11), (1,11,12),
        (2,6,10), (6,12,5), (12,11,3), (11,8,7), (8,2,9),
        (4,10,5), (4,5,3), (4,3,7), (4,7,9), (4,9,10),
        (5,10,6), (3,5,12), (7,3,11), (9,7,8), (10,9,2),
    ]

    midpoint_cache = Dict{Tuple{Int,Int},Int}()
    function midpoint!(i, j)
        key = i < j ? (i, j) : (j, i)
        haskey(midpoint_cache, key) && return midpoint_cache[key]
        m = (verts[i] .+ verts[j]) ./ 2.0
        push!(verts, m ./ norm(m))
        idx = length(verts)
        midpoint_cache[key] = idx
        return idx
    end

    for _ in 1:subdivisions
        new_faces = NTuple{3,Int}[]
        for (a, b, c) in faces
            ab = midpoint!(a, b)
            bc = midpoint!(b, c)
            ca = midpoint!(c, a)
            push!(new_faces, (a, ab, ca))
            push!(new_faces, (b, bc, ab))
            push!(new_faces, (c, ca, bc))
            push!(new_faces, (ab, bc, ca))
        end
        faces = new_faces
    end

    dirs = permutedims(reduce(hcat, verts))  # N x 3
    return Matrix(dirs), faces
end

"unique undirected edges of a triangle mesh, for the smoothness penalty"
function mesh_edges(faces)
    es = Set{Tuple{Int,Int}}()
    for (a, b, c) in faces
        for (i, j) in ((a, b), (b, c), (c, a))
            push!(es, i < j ? (i, j) : (j, i))
        end
    end
    return collect(es)
end

# ------------------------------------------------------------------
# 2. Forward model
# ------------------------------------------------------------------

Rz(phi) = [cos(phi) -sin(phi) 0.0; sin(phi) cos(phi) 0.0; 0.0 0.0 1.0]

"one fixed observation setup: light + camera direction in the LAB frame, plus sampled phases"
struct Geometry
    e0_lab::Vector{Float64}   # light direction, lab frame
    e_lab::Vector{Float64}    # camera direction, lab frame
    phases::Vector{Float64}   # rotation phases sampled (radians)
end

"""
    render_curves(logr, dirs, faces, geom, ab) -> (B, I)

logr: log-radius per vertex (positivity via exp, as in Kaasalainen &
Torppa's exponential parametrization). ab = (a, b) are the LS/Lambert
mixing coefficients: R(mu,mu0) = a/(mu+mu0) + b.

EXTEND: this sums mu>0 & mu0>0 facets with no occlusion test — exact
for convex shapes. For concavities, add a ray-triangle occlusion check
against the whole mesh before accepting a facet as visible/illuminated.
"""
function render_curves(logr::AbstractVector, dirs::Matrix{Float64},
                        faces::Vector{NTuple{3,Int}}, geom::Geometry,
                        ab::Tuple{<:Real,<:Real})
    a, b = ab
    r = exp.(logr)
    P = dirs .* r                      # N x 3 vertex positions
    K = length(geom.phases)
    B = zeros(eltype(logr), K)
    I = zeros(eltype(logr), K)

    for (k, phi) in enumerate(geom.phases)
        Rm = Rz(-phi)                  # body-frame directions at this phase
        e0 = Rm * geom.e0_lab
        e  = Rm * geom.e_lab
        Bsum = zero(eltype(logr))
        Isum = zero(eltype(logr))
        @inbounds for (i, j, l) in faces
            p_i = @view P[i, :]; p_j = @view P[j, :]; p_l = @view P[l, :]
            nvec = cross(p_j .- p_i, p_l .- p_i)
            nn = norm(nvec)
            nn == 0 && continue
            nhat = nvec ./ nn
            area = 0.5 * nn
            mu  = dot(nhat, e)
            mu0 = dot(nhat, e0)
            if mu > 0 && mu0 > 0
                Bsum += mu * area
                Isum += mu * mu0 * (a / (mu + mu0) + b) * area
            end
        end
        B[k] = Bsum
        I[k] = Isum
    end
    return B, I
end

normalize_curve(x) = x ./ mean(x)

# ------------------------------------------------------------------
# 3. Data loading (STUB — adjust to the real challenge format)
# ------------------------------------------------------------------

"""
    load_curves(path) -> Vector{Geometry}, Vector{Tuple{Vector,Vector}}

EXTEND: replace this with a real parser for the challenge's lightcurve
files. It must return, for each fixed (camera, tilt) observation
sequence: a `Geometry` (light + camera direction in the lab frame, and
the sampled phases) and the matching observed (binary, intensity)
curve pair, mean-normalized as the challenge provides them.
"""
function load_curves(path::AbstractString)
    error("load_curves: not implemented yet -- plug in the real data format here")
end

# ------------------------------------------------------------------
# 4. Objective: data misfit + smoothness prior
# ------------------------------------------------------------------

"""
    objective(logr, dirs, faces, geoms, obsB, obsI, edges, ab; lambda=1e-2)

Sum of squared errors between normalized model curves and normalized
observed curves, across all geometries, plus a Tikhonov smoothness
penalty on log-radii differences across mesh edges.

EXTEND: swap the `lambda * sum((logr[i]-logr[j])^2 ...)` term for a
learned regularizer J_theta(shape) (a trained denoiser prior) once you
have a synthetic training set -- see the design notes in the reply.
"""
function objective(logr, dirs, faces, geoms::Vector{Geometry},
                    obsB::Vector{Vector{Float64}}, obsI::Vector{Vector{Float64}},
                    edges, ab; lambda::Float64=1e-2)
    loss = zero(eltype(logr))
    for (g, ob, oi) in zip(geoms, obsB, obsI)
        Bm, Im = render_curves(logr, dirs, faces, g, ab)
        loss += sum(abs2, normalize_curve(Bm) .- ob)
        loss += sum(abs2, normalize_curve(Im) .- oi)
    end
    smooth = zero(eltype(logr))
    for (i, j) in edges
        smooth += (logr[i] - logr[j])^2
    end
    return loss + lambda * smooth
end

# ------------------------------------------------------------------
# 5. STL export
# ------------------------------------------------------------------

function write_stl(filename::AbstractString, P::Matrix{Float64}, faces)
    open(filename, "w") do io
        println(io, "solid asteroid_mvp")
        for (i, j, k) in faces
            p1, p2, p3 = P[i, :], P[j, :], P[k, :]
            n = cross(p2 .- p1, p3 .- p1)
            nn = norm(n)
            n = nn > 0 ? n ./ nn : n
            @printf(io, "  facet normal %.6e %.6e %.6e\n", n[1], n[2], n[3])
            println(io, "    outer loop")
            for p in (p1, p2, p3)
                @printf(io, "      vertex %.6e %.6e %.6e\n", p[1], p[2], p[3])
            end
            println(io, "    endloop")
            println(io, "  endfacet")
        end
        println(io, "endsolid asteroid_mvp")
    end
end

# ------------------------------------------------------------------
# 6. Synthetic self-test: recover a known ellipsoid from its curves
# ------------------------------------------------------------------

function ellipsoid_logr(dirs::Matrix{Float64}, A::Float64, B::Float64, C::Float64)
    N = size(dirs, 1)
    logr = zeros(N)
    for i in 1:N
        dx, dy, dz = dirs[i, 1], dirs[i, 2], dirs[i, 3]
        s = dx^2 / A^2 + dy^2 / B^2 + dz^2 / C^2
        logr[i] = -0.5 * log(s)
    end
    return logr
end

function run_self_test(; subdivisions::Int=1, noise_level::Float64=0.02, seed::Int=1)
    Random.seed!(seed)
    dirs, faces = icosphere(subdivisions)
    N = size(dirs, 1)
    edges = mesh_edges(faces)
    @printf("Mesh: %d vertices, %d faces\n", N, length(faces))

    # --- known test shape: an elongated ellipsoid ---
    logr_true = ellipsoid_logr(dirs, 1.0, 0.8, 0.6)

    # --- placeholder observation geometries ---
    # EXTEND: replace with the real rig's light/camera directions and
    # the tilt configurations actually used by the challenge.
    e0_lab = [1.0, 0.0, 0.0]                        # light along +x
    phases = collect(range(0, 2pi, length=73))[1:end-1]
    geoms = [
        Geometry(e0_lab, [0.0, -1.0, 0.0], phases),                 # side view, phase ~90 deg
        Geometry(e0_lab, [0.0, -cosd(45), sind(45)], phases),       # oblique view
        Geometry(e0_lab, [0.0, 0.0, 1.0], phases),                  # top view
    ]
    ab_true = (0.5, 0.5)   # true LS/Lambert mix used to generate synthetic data

    # --- generate noisy synthetic observations ---
    obsB = Vector{Vector{Float64}}()
    obsI = Vector{Vector{Float64}}()
    for g in geoms
        Bt, It = render_curves(logr_true, dirs, faces, g, ab_true)
        Bn = normalize_curve(Bt) .* (1 .+ noise_level .* randn(length(Bt)))
        In = normalize_curve(It) .* (1 .+ noise_level .* randn(length(It)))
        push!(obsB, Bn)
        push!(obsI, In)
    end

    # --- recover shape from curves alone ---
    ab_assumed = (0.5, 0.5)   # EXTEND: calibrate this on public ground-truth models
    logr0 = zeros(N)          # start from the unit sphere
    f(x) = objective(x, dirs, faces, geoms, obsB, obsI, edges, ab_assumed; lambda=5e-3)

    println("Optimizing (this uses finite-difference gradients; expect it to take a bit)...")
    res = _optimize(f, logr0)
    logr_hat = res

    # --- evaluate recovery quality ---
    r_true = exp.(logr_true)
    r_hat  = exp.(logr_hat)
    rel_err = mean(abs.(r_hat .- r_true) ./ r_true)
    @printf("Mean relative radius error: %.4f\n", rel_err)

    # --- export recovered mesh ---
    P_hat = dirs .* r_hat
    out = joinpath(@__DIR__, "recovered_shape.stl")
    write_stl(out, P_hat, faces)
    println("Wrote recovered shape to: ", out)

    return rel_err
end

# ------------------------------------------------------------------
# Optimizer wrapper (requires Optim.jl: `] add Optim`)
# ------------------------------------------------------------------

function _optimize(f, x0)
    # Uses finite-difference gradients by default (robust, a bit slow).
    # For a speed-up once you trust the model: optimize(f, x0, LBFGS(),
    # opts; autodiff = :forward) using ForwardDiff under the hood.
    res = optimize(f, x0, LBFGS(), Optim.Options(iterations=300, g_tol=1e-8))
    return Optim.minimizer(res)
end

# ------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    run_self_test()
end
