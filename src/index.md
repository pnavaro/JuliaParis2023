# Who am I ?

 - My name is *Pierre Navaro*

 - Scientific Computing Engineer at Insitut de Recherche Mathématique de Rennes

 - **Fortran 77 + PVM** : during my PhD 1998-2002 (Université du Havre)

 - **Fortran 90-2003 + OpenMP-MPI** : Engineer in Strasbourg (2003-2015) at IRMA

 - **Numpy + Cython, R + Rcpp** : Engineer in Rennes (2015-now) at IRMAR

 - **Julia v1.0** since July 2018

 Slides : https://plmlab.math.cnrs.fr/navaro/JuliaParis2023

 This is a joint work with [*Claire Brécheteau*](https://brecheteau.perso.math.cnrs.fr/page/index.html)
 from Ecole Centrale de Nantes.

---

# The $k$-means method

``P`` distribution on ``\mathbb{R}^d``

```math
\mathbf{c}= (c_1,c_2,\ldots,c_k) \in (\mathbb{R}^d)^k
```

The optimal codebook $\mathbf{c}^*$ minimizes the $k$-means loss function 

```math
R : \mathbf{c}\mapsto P\min_{i = 1..k}\|\cdot-c_i\|^2.
```

## Algorithm

- Initialize k centroids.
- Calculate the distance of every point to every centroid.
- Assign every point to a cluster, by choosing the centroid with the minimum distance to the point.
- Recalculate the centroids using the mean of the assigned points.
- Repeat the steps until reaching convergence. 

---

# Lloyd’s algorithm method 

![](assets/kmeans_example_step00bis.png)

---

# Lloyd’s algorithm method 

![](assets/kmeans_example_step01bis.png)

---

# Lloyd’s algorithm method 

![](assets/kmeans_example_step11bis.png)

---

# Lloyd’s algorithm method 

![](assets/kmeans_example_step12bis.png)

---

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

## Initialize centers

```@example paris
using StatsBase

function initialize_centers(data, k) 
    n = size(data, 1)
    return [data[i, :] for i in sample(1:n, k, replace=false)]
end
```

## Estimate cluster to all observations

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

o = clugen(2, 3, 1000, [1, 1], pi / 8, [10, 10], 10, 2, 1)
centers, labels = kmeans(o.points, 3)
scatter( o.points[:,1], o.points[:,2], group=labels)
scatter!( Tuple.(centers), m = :star, ms = 10, c = :yellow, label = "centers")
savefig("plot1.svg") # hide
nothing # hide
```

![](plot1.svg)

---

# Outline

How to approximate a manifold with a set of $k$ points, from a noisy sample ?

---

![](assets/filtration1.png)

---

![](assets/filtration2.png)

---

![](assets/filtration3.png)

---

# Clustering with unions of ellipsoids 

![](assets/evol_ssniv.gif)

---

# Package

https://github.com/pnavaro/GeometricClusterAnalysis.jl
