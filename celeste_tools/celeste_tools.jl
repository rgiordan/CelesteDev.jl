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

function print_vp(vp::Vector{Float64})
    df = DataFrame(ids=ids_names)
    s = 1
    df[Symbol(string("v", s))] = vp
    println(df)
end



using Celeste.DeterministicVI.load_source_brightnesses
using Celeste.Model.load_bvn_mixtures
using Celeste.DeterministicVI.add_pixel_term!
using Celeste.Model.lidx
import Celeste.Model.linear_world_to_pix


function source_pixel_location(ea::ElboArgs, s::Int, n::Int)
    p = ea.patches[s, n]

    pix_loc = linear_world_to_pix(
        p.wcs_jacobian,
        p.center,
        p.pixel_center,
        ea.vp[s][lidx.u])
    return pix_loc - p.bitmap_corner
end


function render_source(ea::ElboArgs, s::Int, n::Int;
                       include_epsilon=true, field=:E_G,
                       include_iota=true)
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

        add_pixel_term!(ea, n, h, w, star_mcs, gal_mcs, sbs,
                        calculate_derivs=false,
                        calculate_hessian=false)
        if field == :E_G
            image[h2, w2] = ea.elbo_vars.E_G.v[1]
        elseif field == :fs0m
            image[h2, w2] = ea.elbo_vars.fs0m_vec[s].v[1]
        elseif field == :fs1m
            image[h2, w2] = ea.elbo_vars.fs1m_vec[s].v[1]
        else
            error("Unknown field ", field)
        end
        if include_iota
            image[h2, w2] *= img.iota_vec[h]
        end
        if include_epsilon
            image[h2, w2] += img.epsilon_mat[h, w]
        end
    end

    return image
end


function render_source_fft(
    ea::ElboArgs,
    fsm_vec::Array{ELBOPixelatedPSF.FSMSensitiveFloatMatrices,1},
    s::Int, n::Int;
    include_epsilon=true, lanczos_width=1,
    field=:E_G, include_iota=true)

    p = ea.patches[s, n]
    image = fill(NaN, size(p.active_pixel_bitmap))
    sbs = load_source_brightnesses(
        ea, calculate_derivs=false, calculate_hessian=false)

    img = ea.images[n]
    fsms = fsm_vec[n]

    gal_mcs = ELBOPixelatedPSF.load_gal_bvn_mixtures(
            ea.S, ea.patches, ea.vp, ea.active_sources, n,
            calculate_derivs=false,
            calculate_hessian=false);

    clear_brightness!(fsms)
    ELBOPixelatedPSF.populate_star_fsm_image!(
        ea, s, n, fsms.psf_vec[s], fsms.fs0m_conv,
        fsms.h_lower, fsms.w_lower, lanczos_width)
    ELBOPixelatedPSF.populate_gal_fsm_image!(ea, s, n, gal_mcs, fsms)
    ELBOPixelatedPSF.populate_source_band_brightness!(ea, s, n, fsms, sbs[s])

    H2, W2 = size(p.active_pixel_bitmap)
    for w2 in 1:W2, h2 in 1:H2
        # (h2, w2) index the local patch, while (h, w) index the image
        h = p.bitmap_corner[1] + h2
        w = p.bitmap_corner[2] + w2

        if !p.active_pixel_bitmap[h2, w2]
            continue
        end

        h_fsm = h - fsms.h_lower + 1
        w_fsm = w - fsms.w_lower + 1

        # if we're here it's a unique active pixel
        image[h2, w2] = getfield(fsms, field)[h_fsm, w_fsm].v[1]
        if include_iota
            image[h2, w2] *= img.iota_vec[h]
        end
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
