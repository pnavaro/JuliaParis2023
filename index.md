




# Who am I ?


  * My name is *Pierre Navaro*
  * Scientific Computing Engineer at Insitut de Recherche Mathématique de Rennes
  * Staff member of [Groupe Calcul](https://calcul.math.cnrs.fr) which promote exchanges within the scientific computing community.
  * **Fortran 77 + PVM** : during my PhD 1998-2002 (Université du Havre)
  * **Fortran 90-2003 + OpenMP-MPI** : Engineer in Strasbourg (2003-2015) at IRMA
  * **Numpy + Cython, R + Rcpp** : Engineer in Rennes (2015-now) at IRMAR
  * **Julia v1.0** since July 2018


French community newsletter about Julia language : https://pnavaro.github.io/NouvellesJulia


Slides : https://plmlab.math.cnrs.fr/navaro/JuliaParis2023


This is a joint work with [*Claire Brécheteau*](https://brecheteau.perso.math.cnrs.fr/page/index.html)  from Ecole Centrale de Nantes.


---






# The $k$-means method


$P$ distribution on $\mathbb{R}^d$


$$
\mathbf{c}= (c_1,c_2,\ldots,c_k) \in (\mathbb{R}^d)^k
$$


The optimal codebook $\mathbf{c}^*$ minimizes the $k$-means loss function 


$$
R : \mathbf{c}\mapsto P\min_{i = 1..k}\|\cdot-c_i\|^2.
$$


---






# Algorithm


  * Initialize k centroids.
  * Calculate the distance of every point to every centroid.
  * Assign every point to a cluster, by choosing the centroid with the minimum distance to the point.
  * Recalculate the centroids using the mean of the assigned points.
  * Repeat the steps until reaching convergence.


---


class: center, middle






# Lloyd’s algorithm method


![](assets/kmeans_example_step00bis.png)


---


class: center, middle




# Lloyd’s algorithm method


![](assets/kmeans_example_step01bis.png)


---


class: center, middle




# Lloyd’s algorithm method


![](assets/kmeans_example_step11bis.png)


---


class: center, middle




# Lloyd’s algorithm method


![](assets/kmeans_example_step12bis.png)


---


class: center, middle




# Lloyd’s algorithm method


![](assets/kmeans_example_step22bis.png)


---






# Compute the distance


```julia
function euclidean(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}

    s = zero(T)

    for i in eachindex(a)
        s += (a[i] - b[i])^2
    end

    return sqrt(s)

end
```


```
euclidean (generic function with 1 method)
```






## Distances.jl


```julia
using Distances

euclidean = Euclidean()
```


---


class: middle






# Initialize centers


```julia
using StatsBase

function initialize_centers(data, k)

    n = size(data, 1)

    return [data[i, :] for i in sample(1:n, k, replace=false)]

end
```


```
initialize_centers (generic function with 1 method)
```


---


class: middle






# Estimate cluster to all observations


```julia
function update_labels!( labels, data, centers)

    for (i, obs) in enumerate(eachrow(data))

        dist = [euclidean(obs, c) for c in centers]

        labels[i] = argmin(dist)

    end

end
```


```
update_labels! (generic function with 1 method)
```


---






## Update centers using the mean


```julia
function update_centers!(centers, data, labels)

    for k in eachindex(centers)

        centers[k] = vec(mean(view(data, labels .== k, :), dims = 1))

    end

end
```


```
update_centers! (generic function with 1 method)
```


---






## Compute inertia


```julia
function compute_inertia(centers, labels, data)

   inertia = 0.0

   for k in eachindex(centers)

       cluster = view(data, labels .== k, :)

       inertia += sum(euclidean(p, centers[k])^2 for p in eachrow(cluster))

   end

   return inertia

end
```


```
compute_inertia (generic function with 1 method)
```


---






# $k$-means


```julia
function kmeans( data, k; maxiter = 100, nstart = 10)

    n, d = size(data)
    opt_centers = [zeros(d) for i in 1:k]  # allocate optimal centers
    labels = zeros(Int, n) # initialize labels
    opt_inertia = Inf
    for istart in 1:nstart
        centers = initialize_centers(data, k)
        for istep in 1:maxiter
            old_centers = deepcopy(centers)
            update_labels!( labels, data, centers)
            update_centers!(centers, data, labels)
            centers ≈ old_centers && break
        end
        inertia = compute_inertia(centers, labels, data)
        if inertia < opt_inertia
            opt_inertia = inertia
            opt_centers .= deepcopy(centers)
        end
    end
    update_labels!( labels, data, opt_centers)
    return opt_centers, labels

end
```


```
kmeans (generic function with 1 method)
```


---


```julia
using Plots, CluGen

o = clugen(2, 3, 1000, [1, 1], pi / 8, [10, 10], 10, 2, 1)
centers, labels = kmeans(o.points, 3)
scatter( o.points[:,1], o.points[:,2], group=labels)
scatter!( Tuple.(centers), m = :star, ms = 10, c = :yellow, label = "centers")
```


![](plot1.svg)


---


class: middle






# Noisy circle


```julia
using Random

rng = MersenneTwister(72)

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
```


```
noisy_circle (generic function with 2 methods)
```


---


```julia
points = hcat(noisy_circle(rng, 1000) , 0.5 .* noisy_circle(rng, 500) )
centers, labels = kmeans(points', 2)
scatter( points[1,:], points[2,:], group=labels)
scatter!( Tuple.(centers), m = :star, ms = 10, c = :yellow, aspect_ratio=1)
```


![](plot2.svg)


---






# Topological data analysis






# *Data have shape, shape has meaning, meaning brings value.*






# Point cloud => Topological descriptor => Inference


---


class: center, middle






# Build filtered complex of the point cloud


![](assets/filtration1.png)


---


class: center, middle




# Build filtered complex of the point cloud


![](assets/filtration2.png)


---


class: center, middle




# Build filtered complex of the point cloud


![](assets/filtration3.png)


---


class: center, middle






# Example: The Ball-Mapper algorithm


[Davide Gurnari](https://dioscuri-tda.org/Paris_TDA_Tutorial_2021.html)


---


```julia
import LinearAlgebra: norm

function find_centers( points, ϵ )
    centers = Dict{Int, Int}() # dict of points
    centers_counter = 1

    for (idx_p, p) in enumerate(eachcol(points)) # Loop over points

        is_covered = false

        for idx_v in keys(centers) # Loop over centers
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
```


```
Dict{Int64, Int64} with 44 entries:
  5  => 129
  35 => 1169
  30 => 1001
  32 => 1068
  6  => 163
  4  => 94
  13 => 402
  12 => 367
  28 => 906
  23 => 739
  41 => 1370
  43 => 1437
  11 => 333
  36 => 1203
  39 => 1304
  7  => 199
  25 => 805
  34 => 1139
  2  => 33
  ⋮  => ⋮
```


---


```julia
function ball(h, k, r)
    θ = LinRange(0, 2π, 500)
    h .+ r * sin.(θ), k .+ r * cos.(θ)
end
scatter(points[1,:], points[2,:]; aspect_ratio=1,  label = "points", ms = 2)
for i in values(centers)
    plot!(ball(points[1,i], points[2,i], ϵ), seriestype = [:shape,], lw = 0.5, c = :blue,
            linecolor = :black, legend = false, fillalpha = 0.1)
end
```


![](plot3.svg)


---


```julia
function compute_points_covered_by_landmarks( points, centers, ϵ)

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
```


```
OrderedCollections.OrderedDict{Int64, Vector{Int64}} with 44 entries:
  1  => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  991, 992, 993, 994, 995, 996, 997, 9…
  2  => [4, 6, 8, 10, 11, 12, 13, 14, 15, 16  …  58, 59, 60, 61, 62, 63, 65, 66…
  3  => [34, 35, 42, 43, 44, 45, 46, 47, 48, 49  …  91, 92, 93, 95, 96, 97, 98,…
  4  => [69, 70, 73, 74, 75, 76, 77, 78, 79, 80  …  127, 128, 130, 132, 133, 13…
  5  => [107, 108, 109, 110, 111, 112, 113, 114, 115, 116  …  158, 159, 160, 16…
  6  => [139, 140, 142, 143, 146, 147, 148, 149, 150, 151  …  191, 192, 193, 19…
  7  => [171, 172, 173, 174, 175, 177, 178, 179, 180, 181  …  225, 226, 227, 23…
  8  => [200, 201, 207, 208, 209, 210, 211, 212, 213, 214  …  258, 259, 260, 26…
  9  => [237, 238, 240, 242, 243, 244, 245, 246, 247, 248  …  289, 290, 291, 29…
  10 => [271, 275, 276, 277, 278, 279, 280, 281, 282, 283  …  325, 326, 327, 32…
  11 => [297, 300, 304, 305, 306, 307, 309, 310, 311, 312  …  359, 360, 361, 36…
  12 => [340, 343, 345, 346, 348, 349, 350, 351, 352, 353  …  394, 395, 396, 39…
  13 => [375, 378, 379, 380, 381, 382, 385, 387, 388, 389  …  433, 434, 435, 43…
  14 => [410, 414, 415, 417, 418, 419, 420, 421, 422, 423  …  468, 469, 470, 47…
  15 => [444, 445, 447, 448, 449, 450, 451, 452, 453, 454  …  493, 494, 495, 49…
  16 => [482, 484, 485, 486, 487, 488, 489, 490, 491, 492  …  528, 529, 530, 53…
  17 => [500, 505, 507, 508, 509, 510, 511, 512, 513, 514  …  561, 562, 563, 56…
  18 => [537, 539, 540, 541, 542, 543, 544, 546, 547, 548  …  595, 596, 597, 59…
  19 => [575, 576, 578, 579, 580, 581, 582, 583, 584, 586  …  633, 634, 635, 63…
  ⋮  => ⋮
```


---


```julia
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

edges = compute_edges(points_covered_by_landmarks)
```


```
46-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (1, 29)
 (2, 3)
 (3, 4)
 (4, 5)
 (5, 6)
 (6, 7)
 (7, 8)
 (8, 9)
 (9, 10)
 ⋮
 (36, 37)
 (37, 38)
 (38, 39)
 (39, 40)
 (40, 41)
 (41, 42)
 (41, 43)
 (42, 43)
 (43, 44)
```


---


```julia
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
            color --> :black
            legend := false
            [x1, x2], [y1, y2]
        end

    end
end
```


.footnote[[Daniel Schwabeneder - How do Recipes actually work?](https://daschw.github.io/recipes/)]


---


class: center, middle


```julia
edgesplot(points, centers, points_covered_by_landmarks)
```


![](plot4.svg)


---


```julia
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
```


```
1500-element Vector{Int64}:
  1
  1
  1
  1
  1
  1
  1
  1
  1
  1
  ⋮
 30
 30
 30
 30
 30
 30
 30
 30
 30
```


---


```julia
scatter(points[1,:], points[2, :], group = colors, aspect_ratio=1, legend = false)
```


![](plot5.svg)


---






# TDA algorithms


  * Cluster merging phase using density map.
  * Use of topological persistence to guide the merging of clusters.


[Ripserer.jl](https://mtsch.github.io/Ripserer.jl/dev/generated/stability/)


[TDA example in Julia](https://github.com/pnavaro/IntroToTDA.jl/blob/main/Trees_In_Philly_Old_City.ipynb)


[Steve Oudot et al - Topological Mode Analysis Tool](https://geometrica.saclay.inria.fr/data/Steve.Oudot/clustering/)


---


.cols[ .fifty[


```julia
n = length(f)
v = sortperm(f, rev = true) # sort vertices using f
sort!(f, rev = true) # sort f
v_inv = Dict(zip(v, 1:n)) 
G = [[v_inv[i] for i in subset] for subset in graph[v]]
𝒰 = IntDisjointSets(n)
for i = eachindex(v)
    𝒩 = [j for j in G[i] if j < i]
    if length(𝒩) != 0
        g = 𝒩[argmax(view(f, 𝒩))] 
        e_i = find_root!(𝒰, g) 
        e_i = union!(𝒰, e_i, i) 
        for j in 𝒩 
            e_j = find_root!(𝒰, j) 
            if e_i != e_j && min(f[e_i], f[e_j]) <= f[i] + τ 
                if f[e_j] <= f[e_i]
                    e_i = union!(𝒰, e_i, e_j)
                else
                    e_i = union!(𝒰, e_j, e_i)
                end
            end
        end
    end
end
```


]


.fifty[


![tomato](assets/algorithm-tomato.png)


] ]






## .footnote[[ClusteringToMaTo.jl](https://github.com/pnavaro/ClusteringToMATo.jl)]


class: center, middle






# Clustering with unions of ellipsoids


![](assets/evol_ssniv.gif)


.footnote[[GeometricClusterAnalysis.jl](https://github.com/pnavaro/GeometricClusterAnalysis.jl)]
