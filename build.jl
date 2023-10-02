ENV["GKSwstype"]="100"

using Plots
using Remark

Remark.slideshow(@__DIR__; options = Dict("ratio" => "16:9"), title = "Julia Paris 2023")
