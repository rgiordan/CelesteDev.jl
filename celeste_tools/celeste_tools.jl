using Celeste: Model, DeterministicVI
using StaticArrays

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

import Synthetic
using SampleData

using Distributions
using PyPlot


import Base.print
function print(ce::CatalogEntry)
  for field in fieldnames(ce)
    println(field, ": ", getfield(ce, field))
  end
end


type SubImage
    min_h::Int
    max_h::Int
    min_w::Int
    max_w::Int
    source_tiles::Array{Celeste.Model.ImageTile}
end


function get_source_pixel_range(sa::Int, b::Int, ea::Celeste.DeterministicVI.ElboArgs)
    source_tiles = ea.images[b].tiles[find([sa in tile for tile in ea.tile_source_map[b] ])];
    min_h = minimum([tile.h_range.start for tile in source_tiles])
    max_h = maximum([tile.h_range.stop for tile in source_tiles])
    min_w = minimum([tile.w_range.start for tile in source_tiles])
    max_w = maximum([tile.w_range.stop for tile in source_tiles])

    return SubImage(min_h, max_h, min_w, max_w, source_tiles)
end

function get_blank_image(si::SubImage)
    return Float64[NaN for h=1:(si.max_h - si.min_h + 1), w=1:(si.max_w - si.min_w + 1)];
end


function render_source(ea::ElboArgs, sa::Int, si::SubImage, include_epsilon::Bool)
    image = get_blank_image(si)
    for tile in si.source_tiles
      pix = DeterministicVI.tile_predicted_image(
        tile, ea, Int64[sa], include_epsilon=include_epsilon)
      image[tile.h_range - si.min_h + 1, tile.w_range - si.min_w + 1] = pix
    end
    return image
end


function show_source_image(ea::ElboArgs, sa::Int, si::SubImage)
    image = get_blank_image(si)
    for tile in si.source_tiles
        image[tile.h_range - si.min_h + 1, tile.w_range - si.min_w + 1] = tile.pixels
    end
    return image
end


function show_active_pixels(ea::ElboArgs, si::SubImage, b::Int)
    image = get_blank_image(si)
    for pixel in ea.active_pixels
        if pixel.n == b
            tile = ea.images[b].tiles[pixel.tile_ind]
            image[tile.h_range.start - si.min_h + pixel.h,
                  tile.w_range.start - si.min_w + pixel.w] = 1
        end
    end
    return image
end
