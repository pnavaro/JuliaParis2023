# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:nomarker
#     text_representation:
#       extension: .jl
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Julia 1.9.3
#     language: julia
#     name: julia-1.9
# ---

# # Ball mapper
#
# This notebook was prepared from an original idea of [Davide Gurnari](https://github.com/dgurnari). 

# ## Generate data

using LinearAlgebra
using Plots
using Random
using Statistics

# We will start by constructing a collection of points sampled from a unit circle.

function noisy_circle(rng, n, noise=0.05)
    x = zeros(n)
    y = zeros(n)
    θ = LinRange(0, 2π, n+1)[1:end-1]
    for i in 1:n
        x[i] = cos(θ[i]) + 2 * noise * (rand(rng) - 0.5)
        y[i] = sin(θ[i]) + 2 * noise * (rand(rng) - 0.5)
    end
    return vcat(x', y')
end

rng = MersenneTwister(72)

points = hcat(noisy_circle(rng, 1000), 0.5 .* noisy_circle(rng, 500) )
scatter(points[1,:], points[2,:]; aspect_ratio=1, legend=false, title="noisy circles")

function find_centers( points, ϵ )
    centers = Dict{Int, Int}() # dict of points {idx_v: idx_p, ... }
    centers_counter = 1
    
    for (idx_p, p) in enumerate(eachcol(points))
        
        is_covered = false

        for idx_v in keys(centers)
            distance = norm(p .- points[:, centers[idx_v]])
            if distance <= ϵ
                is_covered = true
                break
            end
        end

        if !is_covered
            centers[centers_counter] = idx_p
            centers_counter += 1
        end
        
    end
    return centers
end
ϵ = 0.2
centers = find_centers( points, ϵ )
idxs = collect(values(centers))
scatter(points[1,idxs], points[2,idxs]; aspect_ratio=1, label="centers")
function ball(h, k, r)
    θ = LinRange(0, 2π, 500)
    h .+ r * sin.(θ), k .+ r * cos.(θ)
end
for i in idxs
    plot!(ball(points[1,i], points[2,i], ϵ), seriestype = [:shape,], lw = 0.5, c = :blue, 
            linecolor = :black, legend = false, fillalpha = 0.1, aspect_ratio = 1)
end
scatter!(points[1,:], points[2,:]; aspect_ratio=1,  label = "points", ms = 2)

function compute_points_covered_by_landmarks( points, centers :: Dict{Int, Int}, ϵ)
    points_covered_by_landmarks = Dict{Int,Vector{Int}}()
    for idx_v in keys(centers)
        points_covered_by_landmarks[idx_v] = Int[]
        for (idx_p, p) in enumerate(eachcol(points))
            distance = norm(p .- points[:,centers[idx_v]])
            if distance <= ϵ
                push!(points_covered_by_landmarks[idx_v], idx_p)
            end
        end
    end
    return sort(points_covered_by_landmarks)
end
points_covered_by_landmarks = compute_points_covered_by_landmarks( points, centers, ϵ)

function compute_edges(points_covered_by_landmarks)
    edges = Tuple{Int,Int}[]
    idxs = collect(keys(points_covered_by_landmarks)) # centers
    for (i, idx_v) in enumerate(idxs[1:end-1])
        p_v = points_covered_by_landmarks[idx_v]
        for idx_u in idxs[i+1:end]
            if !isdisjoint( p_v, points_covered_by_landmarks[idx_u])
                push!(edges, (idx_v,idx_u))
            end
        end
    end
    edges
end


@show edges

using RecipesBase

@userplot EdgesPlot

@recipe function f(gp::EdgesPlot)
    points, centers, points_covered_by_landmarks = gp.args
    idxs = collect(values(centers))
    aspect_ratio := 1
    
    @series begin
        seriestype := :scatter
        points[1,idxs], points[2,idxs]
    end
    for (e1,e2) in compute_edges(points_covered_by_landmarks)
        x1, y1 = points[:,centers[e1]]
        x2, y2 = points[:,centers[e2]]
        @series begin
            color := :black
            legend := false
            [x1, x2], [y1, y2]
        end
    end
end

edgesplot(points, centers, points_covered_by_landmarks)

sort(centers)

function compute_colors( points, points_covered_by_landmarks)
    edges = compute_edges(points_covered_by_landmarks)
    nc = length(keys(points_covered_by_landmarks))
    center_colors = collect(1:nc)
    for i in 1:nc
        for (e1,e2) in edges
            center_colors[e2] = center_colors[e1]
        end
    end
    n = size(points, 2)
    colors = zeros(Int, n)
    for c in keys(centers)
        for (e1, e2) in edges
            colors[points_covered_by_landmarks[c]] .= center_colors[c]
        end
    end
    colors
end

colors = compute_colors( points, points_covered_by_landmarks)

scatter(points[1,:], points[2, :], group = colors, aspect_ratio=1)




