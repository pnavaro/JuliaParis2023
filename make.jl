ENV["GKSwstype"]="100"

using Plots
using Remark
using FileWatching

while true
    Remark.slideshow(@__DIR__; options = Dict("ratio" => "16:9"), title = "Julia Paris 2023")
    @info "Rebuilt"
    FileWatching.watch_folder(joinpath(@__DIR__, "src"))
end
