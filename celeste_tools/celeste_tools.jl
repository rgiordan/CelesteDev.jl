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

using DataFrames

function print_vp(vp::Array{Float64, 1})
    df = DataFrame(ids=ids_names)
    for s in 1:length(vp)
        df[symbol(string("v", s))] = vp[s]
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


using Celeste.DeterministicVI.load_source_brightnesses
using Celeste.Model.load_bvn_mixtures
using Celeste.DeterministicVI.add_pixel_term!

function render_source(ea::ElboArgs, s::Int, n::Int; include_epsilon=false)
    p = ea.patches[s, n]

    image = fill(NaN, size(p.active_pixel_bitmap))
    sbs = load_source_brightnesses(
        ea, calculate_derivs=false, calculate_hessian=false)

    img = ea.images[n]
    star_mcs, gal_mcs = load_bvn_mixtures(ea.S, ea.patches,
                                ea.vp, ea.active_sources,
                                ea.psf_K, n,
                                calculate_derivs=false,
                                calculate_hessian=false)

    H2, W2 = size(p.active_pixel_bitmap)
    for w2 in 1:W2, h2 in 1:H2
        # (h2, w2) index the local patch, while (h, w) index the image
        h = p.bitmap_corner[1] + h2
        w = p.bitmap_corner[2] + w2

        if !p.active_pixel_bitmap[h2, w2]
            continue
        end

        # if we're here it's a unique active pixel
        add_pixel_term!(ea, n, h, w, star_mcs, gal_mcs, sbs;
                        calculate_derivs=false,
                        calculate_hessian=false)
        image[h2, w2] = img.iota_vec[h] * ea.elbo_vars.E_G.v[1]
        if include_epsilon
            image[h2, w2] += img.epsilon_mat[h, w]
        end

    end

    return image
end


function show_source_image(ea::ElboArgs, s::Int, n::Int)
    p = ea.patches[s, n]
    H2, W2 = size(p.active_pixel_bitmap)
    image = fill(NaN, H2, W2);
    for w2 in 1:W2, h2 in 1:H2
        # (h2, w2) index the local patch, while (h, w) index the image
        h = p.bitmap_corner[1] + h2
        w = p.bitmap_corner[2] + w2

        if !p.active_pixel_bitmap[h2, w2]
            continue
        end

        image[h2, w2] = images[n].pixels[h, w]
    end
    return image
end
