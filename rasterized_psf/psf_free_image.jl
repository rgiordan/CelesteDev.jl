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
wd = pwd()

include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/predicted_image.jl")
using PSFConvolution

run, camcol, field = (3900, 6, 269)

images = SDSSIO.load_field_images(RunCamcolField(run, camcol, field), datadir);
tiled_images = TiledImage[TiledImage(img) for img in images];
dir = joinpath(datadir, "$run/$camcol/$field");
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
ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [sa]; psf_K=2, num_allowed_sd=3.0);
active_pixels = DeterministicVI.get_active_pixels(ea);

elbo_time = time()
DeterministicVI.elbo(ea);
elbo_time = time() - elbo_time


##############

psf_spread = 0.5
psf_comp = Model.PsfComponent(1.0, Float64[0, 0],
    Float64[ psf_spread 0.0; 0.0 psf_spread ]);
psf_image = PSF.get_psf_at_point(PsfComponent[psf_comp])

######################################



b = 3
NumType = Float64
s = ea.active_sources[1]


elbo_vars = DeterministicVI.ElboIntermediateVariables(Float64, ea.S, length(ea.active_sources));

h_lower = Int[typemax(Int) for b in ea.images ]
w_lower = Int[typemax(Int) for b in ea.images ]
h_upper = Int[0 for b in ea.images ]
w_upper = Int[0 for b in ea.images ]
for pixel in active_pixels
    tile = ea.images[pixel.n].tiles[pixel.tile_ind]
    h_lower[pixel.n] = min(h_lower[pixel.n], tile.h_range.start)
    h_upper[pixel.n] = max(h_upper[pixel.n], tile.h_range.stop)
    w_lower[pixel.n] = min(w_lower[pixel.n], tile.w_range.start)
    w_upper[pixel.n] = max(w_upper[pixel.n], tile.w_range.stop)
end

h_width = h_upper[b] - h_lower[b] + 1
w_width = w_upper[b] - w_lower[b] + 1

using SensitiveFloats.zero_sensitive_float_array

# Pre-allocate arrays.
fs0m_image = zero_sensitive_float_array(
    StarPosParams, Float64, length(ea.active_sources), h_width, w_width);
fs1m_image = zero_sensitive_float_array(
    GalaxyPosParams, Float64, length(ea.active_sources), h_width, w_width);

fs0m_conv = zero_sensitive_float_array(
    StarPosParams, Float64, length(ea.active_sources), h_width, w_width);
fs1m_conv = zero_sensitive_float_array(
    GalaxyPosParams, Float64, length(ea.active_sources), h_width, w_width);

(fft_size1, fft_size2) =
    (size(fs0m_image, 1) + size(psf_image, 1) - 1,
     size(fs0m_image, 2) + size(psf_image, 2) - 1)
psf_fft = zeros(Complex{Float64}, fft_size1, fft_size2);
psf_fft[1:size(psf_image, 1), 1:size(psf_image, 2)] = psf_image;
fft!(psf_fft);

fs0m_image_padded =
    zero_sensitive_float_array(StarPosParams, Float64, length(ea.active_sources),
    size(psf_fft)...);
fs1m_image_padded =
    zero_sensitive_float_array(GalaxyPosParams, Float64, length(ea.active_sources),
    size(psf_fft)...);

fs0m_conv_padded =
    zero_sensitive_float_array(StarPosParams, Float64, length(ea.active_sources),
    size(psf_fft)...);
fs1m_conv_padded =
    zero_sensitive_float_array(GalaxyPosParams, Float64, length(ea.active_sources),
    size(psf_fft)...);



############### Process active pixels
using DeterministicVI.BvnComponent
using DeterministicVI.GalaxyCacheComponent
using DeterministicVI.load_bvn_mixtures
using DeterministicVI.load_source_brightnesses
using DeterministicVI.populate_fsm_vecs!


elbo_time = time()

star_mcs_vec = Array(Array{BvnComponent{NumType}, 2}, ea.N);
gal_mcs_vec = Array(Array{GalaxyCacheComponent{NumType}, 4}, ea.N);

foo = time() - elbo_time
for b=1:ea.N
    star_mcs_vec[b], gal_mcs_vec[b] =
        load_bvn_mixtures(ea, b,
            calculate_derivs=elbo_vars.calculate_derivs,
            calculate_hessian=elbo_vars.calculate_hessian)
end

sbs = load_source_brightnesses(ea,
    calculate_derivs=elbo_vars.calculate_derivs,
    calculate_hessian=elbo_vars.calculate_hessian);

b = 3
# Just compute one band N times for now
for foo=1:ea.N
    # TODO: In general, the image ranges could be different for each band.
    gal_mcs = gal_mcs_vec[b];
    star_mcs = star_mcs_vec[b];

    for pixel in active_pixels
        if pixel.n == b
            tile = ea.images[pixel.n].tiles[pixel.tile_ind]
            tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
            populate_fsm_vecs!(elbo_vars, ea, tile_sources, tile, pixel.h, pixel.w,
                               gal_mcs, star_mcs)
            h = tile.h_range[pixel.h] - h_lower[b] + 1
            w = tile.w_range[pixel.w] - w_lower[b] + 1

            # TODO: probably this deepcopy is what is so slow.
            fs0m_image[h, w] = deepcopy(elbo_vars.fs0m_vec[s])
            fs1m_image[h, w] = deepcopy(elbo_vars.fs1m_vec[s])
        end
    end


    ########################
    PSFConvolution.convolve_fsm_images!(
        fs0m_image, fs1m_image,
        fs0m_image_padded, fs1m_image_padded,
        fs0m_conv, fs1m_conv,
        fs0m_conv_padded, fs1m_conv_padded,
        psf_fft);

    #################################
    # iterate over the pixels
    PSFConvolution.get_expected_brightness_from_image!(
        ea, elbo_vars, active_pixels, b, s,
        sbs, star_mcs, gal_mcs, fs0m_conv, fs1m_conv, h_lower, w_lower, false)
end

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

for tile in source_tiles
  pix = DeterministicVI.tile_predicted_image(tile, ea, Int64[sa], include_epsilon=false)
  image[tile.h_range - min_h + 1, tile.w_range - min_w + 1] = pix
end
image[abs(image) .< 1e-8 ] = 0;
matshow(image); title(point_psf_width); colorbar()
image_sparse = sparse(image)





# Convolution of sensitive floats

NumType = Float64
tile = source_tiles[4];
elbo_vars = DeterministicVI.ElboIntermediateVariables(Float64, ea.S, length(ea.active_sources));
tile_sources = Int[sa]

star_mcs, gal_mcs = DeterministicVI.load_bvn_mixtures(ea, tile.b, calculate_derivs=true);

# Sanity check:
star_mcs[1, sa].the_mean
obj_loc_pix = WCS.world_to_pix(ea.images[b].wcs, ea.vp[sa][ids.u])
Model.linear_world_to_pix(ea.patches[sa, b].wcs_jacobian,
                          ea.patches[sa, b].center,
                          ea.patches[sa, b].pixel_center, ea.vp[sa][ids.u])


fs0m_tile, fs1m_tile =
    get_source_tile_fsm_matrix(tile, ea, sa, elbo_vars, star_mcs, gal_mcs,
                               tile_sources, true, true);
(fft_size1, fft_size2) =
    (size(fs0m_tile, 1) + size(psf_image, 1) - 1,
     size(fs0m_tile, 2) + size(psf_image, 2) - 1)
psf_fft = zeros(Complex{Float64}, fft_size1, fft_size2);
psf_fft[1:size(psf_image, 1), 1:size(psf_image, 2)] = psf_image;
fft!(psf_fft);

ParamType = StarPosParams

fs0m_tile_padded =
    zero_sensitive_float_array(StarPosParams, Float64, length(ea.active_sources),
    size(psf_fft)...);
fs1m_tile_padded =
    zero_sensitive_float_array(GalaxyPosParams, Float64, length(ea.active_sources),
    size(psf_fft)...);

fs0m_tile_padded[1:size(fs0m_tile, 1), 1:size(fs0m_tile, 2)] = fs0m_tile;
fs1m_tile_padded[1:size(fs1m_tile, 1), 1:size(fs1m_tile, 2)] = fs1m_tile;

conv_time = time()
fs0m_conv_padded = convolve_sensitive_float_matrix(fs0m_tile_padded, psf_fft);
fs1m_conv_padded = convolve_sensitive_float_matrix(fs1m_tile_padded, psf_fft);
conv_time = time() - conv_time

pad_pix_h = Integer((size(psf_image, 1) - 1) / 2)
pad_pix_w = Integer((size(psf_image, 2) - 1) / 2)

fs0m_conv = fs0m_conv_padded[(pad_pix_h + 1):(end - pad_pix_h), (pad_pix_w + 1):(end - pad_pix_w)];
fs1m_conv = fs1m_conv_padded[(pad_pix_h + 1):(end - pad_pix_h), (pad_pix_w + 1):(end - pad_pix_w)];

PyPlot.close("all")
# A point gets mapped to its location in foo + location in bar - 1.  Since the psf
# image is centered at (26, 26), spatial locations are increased by 25.
obj_loc_tile_pix =
    obj_loc_pix -
    [ tile.h_range.start - 1, tile.w_range.start - 1]

matshow([sf.v[1] for sf in fs0m_conv]); title("star conv");
plot(obj_loc_tile_pix[2] - 1, obj_loc_tile_pix[1] - 1, "ro");
colorbar()

matshow([sf.v[1] for sf in fs0m_tile]); title("star tile"); colorbar()
plot(obj_loc_tile_pix[2] - 1, obj_loc_tile_pix[1] - 1, "ro");

matshow([sf.v[1] for sf in fs1m_tile]); title("gal tile"); colorbar()
plot(obj_loc_tile_pix[2] - 1, obj_loc_tile_pix[1] - 1, "ro");
#matshow(conv2(Float64[ sf.v[1] for sf in fs0m_tile ], psf_image)); colorbar(); title("conv2")















############
