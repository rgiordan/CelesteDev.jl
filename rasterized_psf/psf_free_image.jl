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

run, camcol, field = (3900, 6, 269)

images = SDSSIO.load_field_images(RunCamcolField(run, camcol, field), datadir);
tiled_images = TiledImage[TiledImage(img) for img in images];
dir = joinpath(datadir, "$run/$camcol/$field");
fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir run camcol field
catalog = SDSSIO.read_photoobj_celeste(fname);

# Pick an object.
objid = "1237662226208063499"
objids = [ce.objid for ce in catalog]
sa = findfirst(objids, objid);
neighbors = Infer.find_neighbors([sa], catalog, tiled_images)[1];

cat_local = vcat(catalog[sa], catalog[neighbors]);
vp = Vector{Float64}[init_source(ce) for ce in cat_local];
patches, tile_source_map = Infer.get_tile_source_map(tiled_images, cat_local);
ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [sa]; psf_K=2, num_allowed_sd=3.0);
active_pixels = DeterministicVI.get_active_pixels(ea);




######################################


point_psf_width = 0.5
point_psf = Model.PsfComponent(1.0, Float64[0, 0],
    Float64[ point_psf_width 0.0; 0.0 point_psf_width ])
for s in 1:size(ea.patches)[1], b in 1:size(ea.patches)[2]
  ea.patches[s, b] = SkyPatch(ea.patches[s, b], Model.PsfComponent[ point_psf ]);
end
ea.psf_K = 1

b = 3


############### Process active pixels

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

NumType = Float64
star_mcs_vec = Array(Array{BvnComponent{NumType}, 2}, ea.N);
gal_mcs_vec = Array(Array{GalaxyCacheComponent{NumType}, 4}, ea.N);

for b=1:ea.N
    star_mcs_vec[b], gal_mcs_vec[b] =
        load_bvn_mixtures(ea, b,
            calculate_derivs=elbo_vars.calculate_derivs,
            calculate_hessian=elbo_vars.calculate_hessian)
end

b = 3
s = ea.active_sources[1]

h_width = h_upper[b] - h_lower[b] + 1
w_width = w_upper[b] - w_lower[b] + 1

# These are all pointers to the same SF, which is zero.  Make sure to set
# with deepcopy.
zero_sf0 = zero_sensitive_float(StarPosParams, Float64, length(ea.active_sources))
fs0m_image = fill(zero_sf0, h_width, w_width);

zero_sf1 = zero_sensitive_float(GalaxyPosParams, Float64, length(ea.active_sources))
fs1m_image = fill(zero_sf1, h_width, w_width);

gal_mcs = gal_mcs_vec[b];
star_mcs = star_mcs_vec[b];

for pixel in active_pixels
    if pixel.n == b
        tile = ea.images[pixel.n].tiles[pixel.tile_ind]
        tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
        populate_fsm_vecs!(elbo_vars, ea, tile_sources, tile, pixel.h, pixel.w, gal_mcs, star_mcs)
        h = tile.h_range[pixel.h] - h_lower[b] + 1
        w = tile.w_range[pixel.w] - w_lower[b] + 1
        fs0m_image[h, w] = deepcopy(elbo_vars.fs0m_vec[s])
        fs1m_image[h, w] = deepcopy(elbo_vars.fs1m_vec[s])
    end
end


########################

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

fs0m_image_padded[1:size(fs0m_image, 1), 1:size(fs0m_image, 2)] = fs0m_image;
fs1m_image_padded[1:size(fs1m_image, 1), 1:size(fs1m_image, 2)] = fs1m_image;

conv_time = time()
fs0m_conv_padded = convolve_sensitive_float_matrix(fs0m_image_padded, psf_fft);
fs1m_conv_padded = convolve_sensitive_float_matrix(fs1m_image_padded, psf_fft);
conv_time = time() - conv_time

pad_pix_h = Integer((size(psf_image, 1) - 1) / 2)
pad_pix_w = Integer((size(psf_image, 2) - 1) / 2)

fs0m_conv = fs0m_conv_padded[(pad_pix_h + 1):(end - pad_pix_h), (pad_pix_w + 1):(end - pad_pix_w)];
fs1m_conv = fs1m_conv_padded[(pad_pix_h + 1):(end - pad_pix_h), (pad_pix_w + 1):(end - pad_pix_w)];


#################################

using DeterministicVI.accumulate_source_brightness!
using DeterministicVI.add_sources_sf!
using DeterministicVI.add_elbo_log_term!
using DeterministicVI.add_scaled_sfs!

include_epsilon = false

sbs = load_source_brightnesses(ea,
    calculate_derivs=elbo_vars.calculate_derivs,
    calculate_hessian=elbo_vars.calculate_hessian);

# iterate over the pixels

pixel = active_pixels[find([ pix.n ==b for pix in active_pixels ])[1]]
for pixel in active_pixels
    if pixel.n == b
        tile = ea.images[pixel.n].tiles[pixel.tile_ind]
        tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
        this_pixel = tile.pixels[pixel.h, pixel.w]

        # Get the brightness.

        # get_expected_pixel_brightness!(
        #     elbo_vars, pixel.h, pixel.w, sbs,
        #     star_mcs_vec[pixel.n], gal_mcs_vec[pixel.n], tile,
        #     ea, tile_sources, include_epsilon=true)

        # This combines the sources into a single brightness value for the pixel.
        # combine_pixel_sources!(elbo_vars, ea, tile_sources, tile, sbs)

        clear!(elbo_vars.E_G,
            elbo_vars.calculate_hessian && elbo_vars.calculate_derivs)
        clear!(elbo_vars.var_G,
            elbo_vars.calculate_hessian && elbo_vars.calculate_derivs)

        # For now s is fixed, and only doing one band.
        # for s in tile_sources
        active_source = s in ea.active_sources
        calculate_hessian =
            elbo_vars.calculate_hessian && elbo_vars.calculate_derivs &&
            active_source

        # The key is this: ea.fs?m_vec[s] must contain the appropriate fsm
        # sensitive floats.

        # These are indices within the fs?m image.
        h = tile.h_range[pixel.h] - h_lower[b] + 1
        w = tile.w_range[pixel.w] - w_lower[b] + 1

        elbo_vars.fs0m_vec[s] = fs0m_image[h, w];
        elbo_vars.fs1m_vec[s] = fs1m_image[h, w];
        accumulate_source_brightness!(elbo_vars, ea, sbs, s, tile.b);

        if active_source
            sa = findfirst(ea.active_sources, s)
            add_sources_sf!(elbo_vars.E_G, elbo_vars.E_G_s, sa, calculate_hessian)
            add_sources_sf!(elbo_vars.var_G, elbo_vars.var_G_s, sa, calculate_hessian)
        else
            # If the sources is inactives, simply accumulate the values.
            elbo_vars.E_G.v[1] += elbo_vars.E_G_s.v[1]
            elbo_vars.var_G.v[1] += elbo_vars.var_G_s.v[1]
        end
    end


    if include_epsilon
        # There are no derivatives with respect to epsilon, so can safely add
        # to the value.
        elbo_vars.E_G.v[1] += tile.epsilon_mat[h, w]
    end


    # Add the terms to the elbo given the brightness.
    iota = tile.iota_vec[pixel.h]
    add_elbo_log_term!(elbo_vars, this_pixel, iota)
    add_scaled_sfs!(elbo_vars.elbo,
                    elbo_vars.E_G, -iota,
                    elbo_vars.calculate_hessian &&
                    elbo_vars.calculate_derivs)

    # Subtract the log factorial term. This is not a function of the
    # parameters so the derivatives don't need to be updated. Note that
    # even though this does not affect the ELBO's maximum, it affects
    # the optimization convergence criterion, so I will leave it in for now.
    elbo_vars.elbo.v[1] -= lfact(this_pixel)
end








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


psf_spread = 5.0
psf_comp = Model.PsfComponent(1.0, Float64[0, 0],
    Float64[ psf_spread 0.0; 0.0 psf_spread ]);
psf_image = PSF.get_psf_at_point(PsfComponent[psf_comp])
# psf_nonzero = ind2sub(size(psf_image), find(abs(psf_image) .> 1e-16))
# psf_h_range = minimum(psf_nonzero[1]):maximum(psf_nonzero[1])
# psf_w_range = minimum(psf_nonzero[2]):maximum(psf_nonzero[2])
# psf_image = psf_image[psf_h_range, psf_w_range]

psf_image[abs(psf_image) .< 1e-8] = 0

image_convolved_psf = conv2(psf_image, image);
# matshow(image_convolved_psf); title("convolved"); colorbar()
# PyPlot.close("all")



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
