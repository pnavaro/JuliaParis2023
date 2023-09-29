ENV["GKSwstype"]="100"

using Literate
using Plots
using Remark
using FileWatching

while true
    slides = Remark.slideshow(@__DIR__,
                          options = Dict("ratio" => "16:9"),
                          title = "Julia Agrocampus 2023")
    @info "Rebuilt"
    FileWatching.watch_folder(joinpath(@__DIR__, "src"))
end
