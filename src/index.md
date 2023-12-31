# Who am I ?

 - My name is *Pierre Navaro*

 - Scientific Computing Engineer at Institut de Recherche Mathématique de Rennes

 - Staff member of [Groupe Calcul](https://calcul.math.cnrs.fr) which promote exchanges within the scientific computing community.

 - **Fortran 77 + PVM** : during my PhD 1998-2002 (Université du Havre)

 - **Fortran 90-2003 + OpenMP-MPI** : Engineer in Strasbourg (2003-2015) at IRMA

 - **Numpy + Cython, R + Rcpp** : Engineer in Rennes (2015-now) at IRMAR

 - **Julia v1.0** since July 2018

 French community newsletter about Julia language : https://pnavaro.github.io/NouvellesJulia

 Slides : https://github.com/pnavaro/JuliaParis2023

 This is a joint work with [*Claire Brécheteau*](https://brecheteau.perso.math.cnrs.fr/page/index.html)
 from Ecole Centrale de Nantes.


---

# The $k$-means method

Dataset ``X_1, X_2, \cdots, X_n`` distribution on ``\mathbb{R}^d``

```math
\mathbf{c}= (c_1,c_2,\ldots,c_k) \in (\mathbb{R}^d)^k
```

We look for the centers ``\mathbf{c}*`` that minimize the $k$-means loss function 


```math
\frac{1}{n} \sum\_{i=1}^n \min\_{j=1..k} \|\|X\_i-c\_j\|\|^2
```

---

# Algorithm

- Initialize k centroids.


- Calculate the distance of every point to every centroid.


- Assign every point to a cluster, by choosing the centroid with the minimum distance to the point.


- Recalculate the centroids using the mean of the assigned points.


- Repeat the steps until reaching convergence. 

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

```@example paris
function euclidean(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}

    s = zero(T)

    for i in eachindex(a)
        s += (a[i] - b[i])^2
    end

    return sqrt(s)

end
```

## Distances.jl

```julia
using Distances

euclidean = Euclidean()
```

---

class: middle

# Initialize centers

```@example paris
using StatsBase

function initialize_centers(data, k) 

    n = size(data, 1)

    return [data[i, :] for i in sample(1:n, k, replace=false)]

end
```

---

class: middle

# Estimate cluster to all observations

```@example paris
function update_labels!( labels, data, centers)

    for (i, obs) in enumerate(eachrow(data))

        dist = [euclidean(obs, c) for c in centers]

        labels[i] = argmin(dist)

    end

end
```

---

## Update centers using the mean


```@example paris
function update_centers!(centers, data, labels)
    
    for k in eachindex(centers)

        centers[k] = vec(mean(view(data, labels .== k, :), dims = 1))

    end

end
```

---

##  Compute inertia

```@example paris
function compute_inertia(centers, labels, data)

   inertia = 0.0

   for k in eachindex(centers)

       cluster = view(data, labels .== k, :)

       inertia += sum(euclidean(p, centers[k])^2 for p in eachrow(cluster))

   end

   return inertia

end
```

---

# $k$-means

```@example paris
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

---

```@example paris
using Plots, CluGen 

o = clugen(2, 3, 1000, [1, 1], pi / 8, [10, 10], 10, 2, 1) # cluster generation
centers, labels = kmeans(o.points, 3)
scatter( o.points[:,1], o.points[:,2], group=labels)
scatter!( Tuple.(centers), m = :star, ms = 10, c = :yellow, label = "centers")
savefig("plot1.svg") # hide
nothing # hide
```

![](plot1.svg)

---

class: middle

# Noisy circle

```@example paris
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
---

```@example paris
points = hcat(noisy_circle(rng, 1000) , 0.5 .* noisy_circle(rng, 500) )
centers, labels = kmeans(points', 2)
scatter( points[1,:], points[2,:], group=labels)
scatter!( Tuple.(centers), m = :star, ms = 10, c = :yellow, aspect_ratio=1)
savefig("plot2.svg") # hide
nothing # hide
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

.footnote[figures : [Fredéric Chazal](https://geometrica.saclay.inria.fr/team/Fred.Chazal/)]

---

class: center, middle

# Build filtered complex of the point cloud

![](assets/filtration2.png)

.footnote[figures : [Fredéric Chazal](https://geometrica.saclay.inria.fr/team/Fred.Chazal/)]

---

class: center, middle

# Build filtered complex of the point cloud

![](assets/filtration3.png)

.footnote[figures : [Fredéric Chazal](https://geometrica.saclay.inria.fr/team/Fred.Chazal/)]

---

class: center, middle

# Example: The Ball-Mapper algorithm

[Davide Gurnari](https://dioscuri-tda.org/Paris_TDA_Tutorial_2021.html)

---


```@example paris
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
nothing # hide
```

---


```@example paris
function ball(h, k, r)
    θ = LinRange(0, 2π, 500)
    h .+ r * sin.(θ), k .+ r * cos.(θ)
end
scatter(points[1,:], points[2,:]; aspect_ratio=1,  label = "points", ms = 2)
for i in values(centers)
    plot!(ball(points[1,i], points[2,i], ϵ), seriestype = [:shape,], lw = 0.5, c = :blue, 
            linecolor = :black, legend = false, fillalpha = 0.1)
end
savefig("plot3.svg") # hide
nothing # hide
```

![](plot3.svg)


---

```@example paris
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
nothing # hide
```

----

```@example paris
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
nothing # hide
```

----

```@example paris
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

```@example paris
edgesplot(points, centers, points_covered_by_landmarks)
savefig("plot4.svg") # hide
nothing # hide
```

![](plot4.svg)

---

```@example paris
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
nothing # hide
```

---

```@example paris
scatter(points[1,:], points[2, :], group = colors, aspect_ratio=1, legend = false)
savefig("plot5.svg") # hide
nothing # hide
```

![](plot5.svg)

---

# TDA algorithms 

- Cluster merging phase using density map. 

- Use of topological persistence to guide the merging of clusters. 

[Ripserer.jl](https://mtsch.github.io/Ripserer.jl/dev/generated/stability/)

[TDA example in Julia](https://github.com/pnavaro/IntroToTDA.jl/blob/main/Trees_In_Philly_Old_City.ipynb)

[Steve Oudot et al - Topological Mode Analysis Tool](https://geometrica.saclay.inria.fr/data/Steve.Oudot/clustering/)

---

.cols[
.fifty[

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

[Chazal, F., Guibas, L. J., Oudot, S. Y., and Skraba, P. 2011. Persistence-Based Clustering in Riemannian Manifolds. J. ACM 60, 6, Article A (January 2013)](https://geometrica.saclay.inria.fr/data/Steve.Oudot/clustering/jacm_oudot.pdf)

]
]

.footnote[[ClusteringToMaTo.jl](https://github.com/pnavaro/ClusteringToMATo.jl)]

---

class: center, middle

# Noisy data clustering with unions of ellipsoids 

![](assets/evol_ssniv.gif)

C. Brécheteau & P. Navaro - [GeometricClusterAnalysis.jl](https://github.com/pnavaro/GeometricClusterAnalysis.jl) (work in progress)


---
