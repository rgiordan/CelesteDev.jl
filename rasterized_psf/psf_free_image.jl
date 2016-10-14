


# Requires branch fsm_matrix of Celeste.jl
# Run from the CelesteDev.jl directory
"""
To reproduce segfault:
Check out branch fsm_matrix of Celeste.jl (commit d7aa6)
Run the below script in the CelesteDev.jl directory
"""



using Celeste: Model, DeterministicVI

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

import Synthetic
using SampleData

using Base.Test
using Distributions

using PyPlot


##########
# Ensure that test images are available.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")

include("rasterized_psf/predicted_image.jl")

using PSFConvolution

run, camcol, field = (3900, 6, 269)

images = SDSSIO.load_field_images(RunCamcolField(run, camcol, field), datadir);
tiled_images = TiledImage[TiledImage(img) for img in images];
dir = joinpath(datadir, "$run/$camcol/$field")
fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir run camcol field
catalog = SDSSIO.read_photoobj_celeste(fname);

# Pick an object.
objid = "1237662226208063499"
objids = [ce.objid for ce in catalog];
sa = findfirst(objids, objid);
neighbors = Infer.find_neighbors([sa], catalog, tiled_images)[1];

cat_local = vcat(catalog[sa], catalog[neighbors]);
vp = Vector{Float64}[init_source(ce) for ce in cat_local];
patches, tile_source_map = Infer.get_tile_source_map(tiled_images, cat_local);
ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [sa]; psf_K=1);
Infer.fit_object_psfs!(ea, ea.active_sources);
Infer.load_active_pixels!(ea);

elbo_time = time()
elbo = DeterministicVI.elbo(ea);
elbo_time = time() - elbo_time


##############

# This PSF effectively interpolates the pixelated PSF
psf_spread = 0.5
psf_comp = Model.PsfComponent(1.0, Float64[0, 0],
    Float64[ psf_spread 0.0; 0.0 psf_spread ]);
psf_image = PSF.get_psf_at_point(PsfComponent[psf_comp])


######################################
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
using PSFConvolution.FSMSensitiveFloatMatrices
using PSFConvolution.initialize_fsm_sf_matrices!

s = ea.active_sources[1]

elbo_vars = DeterministicVI.ElboIntermediateVariables(
    Float64, ea.S, length(ea.active_sources));

psf_image_vec = Matrix{Float64}[ psf_image for n in 1:ea.N ];

fsm_vec = FSMSensitiveFloatMatrices[FSMSensitiveFloatMatrices() for b in 1:ea.N];
PSFConvolution.initialize_fsm_sf_matrices!(fsm_vec, ea, psf_image_vec);


############### Process active pixels
using DeterministicVI.BvnComponent
using DeterministicVI.GalaxyCacheComponent
using DeterministicVI.load_bvn_mixtures
using DeterministicVI.load_source_brightnesses
using DeterministicVI.populate_fsm!


elbo_time = time()


sbs = load_source_brightnesses(ea,
    calculate_derivs=elbo_vars.calculate_derivs,
    calculate_hessian=elbo_vars.calculate_hessian);

star_mcs_vec = Array(Array{BvnComponent{Float64}, 2}, ea.N);
gal_mcs_vec = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N);

foo = time() - elbo_time
for b=1:ea.N
    star_mcs_vec[b], gal_mcs_vec[b] =
        load_bvn_mixtures(ea, b,
            calculate_derivs=elbo_vars.calculate_derivs,
            calculate_hessian=elbo_vars.calculate_hessian)
end

b = 1
# for b=1:ea.N

    gal_mcs = gal_mcs_vec[b];
    star_mcs = star_mcs_vec[b];

    pixel_ind = findfirst([ pixel.n == b for pixel in ea.active_pixels ])
    pixel = ea.active_pixels[pixel_ind]
    # for pixel in ea.active_pixels
        # if pixel.n == b
            # TODO: do this for all the sources.
            tile = ea.images[pixel.n].tiles[pixel.tile_ind]
            tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
            h = pixel.h
            w = pixel.w
            x = Float64[tile.h_range[h], tile.w_range[w]]
            populate_fsm!(elbo_vars, ea,
                          fsm_vec[b].fs0m_image[h, w],
                          fsm_vec[b].fs1m_image[h, w],
                          sa, b, x, true, gal_mcs, star_mcs)
        # end
    # end


    ########################
    # Works:
    using PSFConvolution.convolve_sensitive_float_matrix!

    fsms = fsm_vec[b];

    for h in 1:size(fsms.fs0m_image, 1), w in 1:size(fsms.fs0m_image, 2)
        fsms.fs0m_image_padded[h, w] = fsms.fs0m_image[h, w];
        fsms.fs1m_image_padded[h, w] = fsms.fs1m_image[h, w];
    end

    conv_time = time()
    convolve_sensitive_float_matrix!(
        fsms.fs0m_image_padded, fsms.psf_fft, fsms.fs0m_conv_padded);
    convolve_sensitive_float_matrix!(
        fsms.fs1m_image_padded, fsms.psf_fft, fsms.fs1m_conv_padded);
    conv_time = time() - conv_time
    println("Convolution time: ", conv_time)

    pad_pix_h = Integer((size(fsms.fs0m_image_padded, 1) - size(fsms.fs0m_image, 1)) / 2)
    pad_pix_w = Integer((size(fsms.fs0m_image_padded, 2) - size(fsms.fs0m_image, 2)) / 2)

    fsms.fs0m_conv = fsms.fs0m_conv_padded[(pad_pix_h + 1):(end - pad_pix_h),
                                           (pad_pix_w + 1):(end - pad_pix_w)];
    fsms.fs1m_conv = fsms.fs1m_conv_padded[(pad_pix_h + 1):(end - pad_pix_h),
                                           (pad_pix_w + 1):(end - pad_pix_w)];



    # Segfaults:
    PSFConvolution.convolve_fsm_images!(fsm_vec[b]);



    #################################
    # iterate over the pixels
    PSFConvolution.get_expected_brightness_from_image!(
        ea, elbo_vars, ea.active_pixels, b, s,
        sbs, star_mcs, gal_mcs,
        fsm_vec[b].fs0m_conv, fsm_vec[b].fs1m_conv,
        h_lower, w_lower, false)

# end

elbo_time = time() - elbo_time







######################################


import Base.print
function print(ce::CatalogEntry)
  for field in fieldnames(ce)
    println(field, ": ", getfield(ce, field))
  end
end


ea.vp[sa][ids.a] = [ 1, 0 ]
ea.vp[sa][ids.u] = WCS.pix_to_world(ea.images[b].wcs,
  floor(WCS.world_to_pix(ea.images[b].wcs, ea.vp[sa][ids.u])) + 0.5)

source_tiles = ea.images[b].tiles[find([sa in tile for tile in tile_source_map[b] ])];
min_h = minimum([tile.h_range.start for tile in source_tiles])
max_h = maximum([tile.h_range.stop for tile in source_tiles])
min_w = minimum([tile.w_range.start for tile in source_tiles])
max_w = maximum([tile.w_range.stop for tile in source_tiles])
image = Float64[NaN for h=1:(max_h - min_h + 1), w=1:(max_w - min_w + 1)];

# Show the active pixels
for pixel in ea.active_pixels
    if pixel.n == b
        tile = ea.images[b].tiles[pixel.tile_ind]
        image[tile.h_range.start - min_h + pixel.h,
              tile.w_range.start - min_w + pixel.w] = 1
    end
end


for tile in source_tiles
  pix = DeterministicVI.tile_predicted_image(tile, ea, Int64[sa], include_epsilon=false)
  image[tile.h_range - min_h + 1, tile.w_range - min_w + 1] = pix
end
image[abs(image) .< 1e-8 ] = 0;
matshow(image); title(psf_spread); colorbar()





############
