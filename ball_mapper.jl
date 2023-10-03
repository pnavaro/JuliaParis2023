# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:nomarker
#     text_representation:
#       extension: .jl
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.14.4
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
    for i in 1:n
        θ = 2π * rand(rng)
        x[i] = cos(θ) + 2 * noise * (rand(rng) - 0.5)
        y[i] = sin(θ) + 2 * noise * (rand(rng) - 0.5)
    end
    return vcat(x', y')
end

rng = MersenneTwister(72)

nc = noisy_circle(rng, 1000)
points = hcat(nc, 0.5 .* nc )
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
    points_covered_by_landmarks = Vector{Int}[]
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

function compute_graph(points_covered_by_landmarks)
    edges = Tuple{Int,Int}[]
    idxs = collect(keys(points_covered_by_landmarks)) # centers
    for (i, idx_v) in enumerate(idxs[1:end-1]), idx_u in idxs[i+1:end]
        if !isdisjoint(points_covered_by_landmarks[idx_v], points_covered_by_landmarks[idx_u])
            push!(edges, (idx_v,idx_u))
        end
    end
    edges
end
compute_graph(points_covered_by_landmarks)

using RecipesBase

@userplot EdgesPlot

@recipe function f(gp::EdgesPlot)
    points, centers, edges = gp.args
    idxs = values(centers)
    aspect_ratio := 1
    
    @series begin
        seriestype := :scatter
        points[1,idxs], points[2,idxs]
    end
    for (e1,e2) in edges
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

function compute_labels(points, points_covered_by_landmarks)
    
    labels = zeros(Int, size(points,2))
    idxs = collect(keys(points_covered_by_landmarks)) # centers
    color = 1
    for (i, idx_v) in enumerate(idxs[1:end-1]), idx_u in idxs[i+1:end]
        if !isdisjoint(points_covered_by_landmarks[idx_v], points_covered_by_landmarks[idx_u])
            labels[idx_v] = color
            labels[idx_u] = color
        else
            color += 1
        end
    end
    labels
end

labels = compute_labels(points, points_covered_by_landmarks)


idxs = collect(keys(points_covered_by_landmarks))
scatter(points[1,idxs], points[2,idxs], c = labels[idxs], aspect_ratio=1)

function compute_colors(points, points_covered_by_landmarks)
    n = size(points, 2)
    colors = zeros(Int, n)
    for (i, cluster) in enumerate(values(points_covered_by_landmarks))
        colors[cluster] .= i
    end
    return colors
end
colors = compute_colors(points, points_covered_by_landmarks)
scatter(points[1,:], points[2,:], group = colors, aspect_ratio=1)

using Graphs

function ball_mapper_graph(points_covered_by_landmarks)
    idxs = collect(keys(points_covered_by_landmarks))
    nv = length(idxs)
    graph = Graph(nv)
    for (i, idx_v) in enumerate(idxs[1:end-1]), idx_u in idxs[i+1:end]
        if !isdisjoint(points_covered_by_landmarks[idx_v], points_covered_by_landmarks[idx_u])
            add_edge!( graph, idx_v, idx_u )
        end
    end
    return graph
end   

g = ball_mapper_graph(points_covered_by_landmarks)

using GraphPlot
gplot(g)


