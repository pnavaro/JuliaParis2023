# Talk at CNAM Paris October 2023

Link to [slides](https://pnavaro.github.io/JuliaParis2023).

To open the notebooks run them locally:

```bash
git clone https://github.com/pnavaro/JuliaParis2023
cd JuliaParis2023
julia --project
```

```julia
julia> using Pkg
julia> Pkg.instantiate()
julia> include("generate_nb.jl")
julia> using IJulia
julia> notebook(dir=joinpath(pwd(),"notebooks"))
[ Info: running ...
```
