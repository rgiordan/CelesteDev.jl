#!/usr/bin/env julia

# using Celeste: Model, ElboDeriv
#
# import Celeste: Infer, ElboDeriv
# import Celeste: PSF, OptimizeElbo, SDSSIO, SensitiveFloats, Transform
# import SensitiveFloats.clear!
# import SDSSIO: RunCamcolField
#
# const testdir = joinpath(Pkg.dir("Celeste"), "test")
# cd(testdir)
#
# include("Synthetic.jl")
# include("SampleData.jl")
# include("DerivativeTestUtils.jl")
#
# import Synthetic
# using SampleData
#
# using Base.Test
# using Distributions
#
#
#
# using Celeste: ElboDeriv, SensitiveFloats
#
# using Base.Test
# using Distributions
# using DerivativeTestUtils
#


import DataFrames # TODO remove after temporary code goes
import FITSIO
import WCS

import Celeste: ElboDeriv, Infer, Model, OptimizeElbo
using PyPlot


function make_psf()
    alphaBar = [1.; 0.; 0.]
    xiBar = [0.; 0.]
    tauBar = [1. 0.; 0. 1.]
    [Model.PsfComponent(alphaBar[k], xiBar, tauBar) for k in 1:3]
end

function read_fits(filename)
    @assert isfile(filename)
    fits = FITSIO.FITS(filename)
    pixels = read(fits[1])
    header_str = FITSIO.read_header(fits[1], String)
    wcs = WCS.from_header(header_str)[1]
    psf = make_psf()
    close(fits)
    pixels, psf, wcs
end

function make_band_images(band_pixels, band_psfs, wcs)
    H, W = size(band_pixels[1])
    [
        Model.TiledImage(
            Model.Image(
                H,
                W,
                band_pixels[band],
                band,
                wcs,
                band_psfs[band],
                0, # SDSS run
                0, # SDSS camcol
                0, # SDSS field
                fill(0., H, W), #epsilon_mat,
                fill(1000., H), #iota_vec,
                Model.RawPSF(Array(Float64, 0, 0), 0, 0, Array(Float64, 0, 0, 0)),
            ),
            tile_width=48,
        )
        for band in 1:5
    ]
end

# TODO: from test/SampleData.jl
# sample_star_fluxes = [
#     4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]
# sample_galaxy_fluxes = [
#     1.377666E+01, 5.635334E+01, 1.258656E+02,
#     1.884264E+02, 2.351820E+02] * 100  # 1x wasn't bright enough

sample_star_fluxes = Float64[ 7 * maximum(pixels) for b=1:5 ] / 1000
sample_galaxy_fluxes = sample_star_fluxes

true_pix_loc = Float64[ ind2sub(size(pixels), findmax(pixels)[2])...]
true_world_loc = WCS.pix_to_world(wcs, true_pix_loc)

function make_catalog_entry()
    Model.CatalogEntry(
        # true_world_loc,
        [18., 16.], # pos
        false, # is_star
        sample_star_fluxes,
        sample_galaxy_fluxes,
        0.1, # gal_frac_dev
        0.7, # gal_ab
        pi / 4, # gal_angle
        4., # gal_scale
        "sample", # objid
        0, # thing_id
    )
end

pixels, psf, wcs = read_fits("/home/rgiordan/Downloads/galsim_test_image.fits");
band_images = make_band_images(fill(pixels, 5), fill(psf, 5), wcs);
catalog_entry = make_catalog_entry()

using Celeste.Model.ids

vp = Vector{Float64}[Model.init_source(catalog_entry)]
pix_loc = WCS.world_to_pix(wcs, vp[1][Celeste.ids.u])
pix_loc - true_pix_loc

# vp[1][ids.a] = [ 1 - 0.001, 0.001 ]
# vp[1][ids.a] = [ 0.1, 0.9 ]
# If e_scale is too low, the likelihood is NaN.
# vp[1][ids.e_scale] = 0.7 # This is about the threshold.
# vp[1][ids.e_scale] = 4

DataFrame(id_name=Celeste.Model.get_id_names(Celeste.Model.CanonicalParams()), v=vp[1], v_copy=vp_copy[1])

patches, tile_source_map = Infer.get_tile_source_map(band_images, [catalog_entry]);
active_sources = [1]
ea = ElboDeriv.ElboArgs(band_images, vp, tile_source_map, patches, active_sources);

sf = ElboDeriv.elbo_likelihood(ea, calculate_derivs=true, calculate_hessian=true);
sf.h
sf_elbo = ElboDeriv.elbo(ea, calculate_derivs=true, calculate_hessian=true);
sf_elbo.h

for b in 1:5
  pred_pix = Celeste.ElboDeriv.tile_predicted_image(ea.images[b].tiles[1, 1], ea, [1], include_epsilon=true);
  # println(sum(isnan(pred_pix)))
  println(maximum(pred_pix))
end

maximum(pixels)


pix_loc = WCS.world_to_pix(wcs, vp[1][Celeste.ids.u])
matshow(pixels); colorbar(); plot(pix_loc[2] - 1, pix_loc[1] - 1, "ro"); title("Actual")

pred_pix = Celeste.ElboDeriv.tile_predicted_image(ea.images[5].tiles[1, 1], ea, [1], include_epsilon=true);
matshow(pred_pix); colorbar(); plot(pix_loc[2] - 1, pix_loc[1] - 1, "ro"); title("Predicted")

ElboDeriv.elbo(ea, calculate_derivs=true, calculate_hessian=true).h
ElboDeriv.elbo_likelihood(ea, calculate_derivs=true, calculate_hessian=true).h

# The verbose option of maximize
OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, ea, loc_width=3.0, verbose=true)

# View variational parameters with names
DataFrame(id_name=Celeste.Model.get_id_names(Celeste.Model.CanonicalParams()), vp=vp[1])

# View a predicted and actual image using PyPlot
pred_pix = Celeste.ElboDeriv.tile_predicted_image(ea.images[5].tiles[1, 1], ea, [1], include_epsilon=true);
pix_loc = WCS.world_to_pix(wcs, vp[1][Celeste.ids.u])
matshow(pred_pix); colorbar(); plot(pix_loc[2] - 1, pix_loc[1] - 1, "ro"); title("Predicted")
matshow(pixels); colorbar(); plot(pix_loc[2] - 1, pix_loc[1] - 1, "ro"); title("Actual")
