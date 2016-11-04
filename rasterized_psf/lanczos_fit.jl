using Celeste: Model, DeterministicVI
using StaticArrays

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/predicted_image.jl")
include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/celeste_tools/celeste_tools.jl")
include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/lanczos.jl")


# using PSFConvolution


import Synthetic
using SampleData

using Base.Test
using Distributions

using PyPlot


##########
# Ensure that test images are available.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")

run, camcol, field = (4263, 5,119)

images = SDSSIO.load_field_images(RunCamcolField(run, camcol, field), datadir);
tiled_images = TiledImage[TiledImage(img) for img in images];
dir = joinpath(datadir, "$run/$camcol/$field")
fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir run camcol field
catalog = SDSSIO.read_photoobj_celeste(fname);

# Pick an object.
for cat in catalog
    if minimum(cat.star_fluxes) > 300
        print(cat)
        print("\n")
    end
end
objid = "1237663784734490800"
objids = [ce.objid for ce in catalog];
sa = findfirst(objids, objid);
neighbors = Infer.find_neighbors([sa], catalog, tiled_images)[1];
cat_local = vcat(catalog[sa], catalog[neighbors]);

vp = Vector{Float64}[init_source(ce) for ce in cat_local];
patches, tile_source_map = Infer.get_tile_source_map(tiled_images, cat_local);
ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [1]; psf_K=2);
Infer.fit_object_psfs!(ea, ea.active_sources);
load_active_pixels!(ea, false);
length(ea.active_pixels)

# For compiling
elbo = DeterministicVI.elbo(ea);

current_elbo_time = time()
elbo = DeterministicVI.elbo(ea);
current_elbo_time = time() - current_elbo_time

##############

b = 3

# Get the actual PSF images using the /first/ source.
psf_image_vec =
    Matrix{Float64}[ PSF.get_psf_at_point(ea.patches[1, b].psf) for b in 1:ea.N ];

star_sf_image = zero_sensitive_float_array(StarPosParams, Float64, 1, 60, 60);
wcs_jacobian = ea.patches[1, b].wcs_jacobian

include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/lanczos.jl")
star_sf_image = zero_sensitive_float_array(StarPosParams, Float64, 1, 60, 60);
lanczos_interpolate!(star_sf_image, psf_image, star_loc, 3.0, wcs_jacobian, true)
matshow([ sf.v[1] for sf in star_sf_image])
plot(star_loc[2] - 1, star_loc[1] - 1, "ro")

# matshow([ sf.d[star_ids.u[1], 1] for sf in star_sf_image])
matshow([ sf.h[star_ids.u[1], star_ids.u[1]] for sf in star_sf_image])
plot(star_loc[2] - 1, star_loc[1] - 1, "ro")


sinc_with_derivatives(2.5)
sinc(2.5)


include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/lanczos.jl")
lh_v, lh_d, lh_h = lanczos_kernel_with_derivatives(2.5, 3.0)

x = 2.5
a = 3.0
sinc_x, sinc_x_d, sinc_x_h = sinc_with_derivatives(x)
sinc_xa, sinc_xa_d, sinc_xa_h = sinc_with_derivatives(x / a)
lh_v, lh_d, lh_h = lanczos_kernel_with_derivatives(x, a)
lanczos_kernel(x, a)



b = 3

include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/lanczos.jl")
gal_mcs = load_gal_bvn_mixtures(
        ea.S, ea.patches, ea.vp, ea.active_sources, b,
        calculate_derivs=true,
        calculate_hessian=true);


################################
#

using StaticArrays
import Celeste.Model
import Celeste.Model.SkyPatch

ea_fft = ElboArgs(tiled_images, deepcopy(vp), tile_source_map,
                  deepcopy(patches), [1]; psf_K=1);
load_active_pixels!(ea_fft, false);


# Then set the fft ea "psf" to a small width to interpolate the pixelated PSF.
point_psf_width = 0.5;
point_psf = Model.PsfComponent(1.0, SVector{2,Float64}([0, 0]),
    SMatrix{2, 2, Float64, 4}([ point_psf_width 0.0; 0.0 point_psf_width ]));
for s in 1:size(ea_fft.patches)[1], b in 1:size(ea_fft.patches)[2]
    ea_fft.patches[s, b] = SkyPatch(ea_fft.patches[s, b], Model.PsfComponent[ point_psf ]);
end

# matshow(PSF.get_psf_at_point([point_psf]))
# matshow(psf_image_vec[3])
# sum(PSF.get_psf_at_point([point_psf]))
# sum(psf_image_vec[3])


######################################

elbo_vars_fft = DeterministicVI.ElboIntermediateVariables(
    Float64, ea_fft.S, length(ea_fft.active_sources));

fsm_vec = FSMSensitiveFloatMatrices[FSMSensitiveFloatMatrices() for b in 1:ea_fft.N];
initialize_fsm_sf_matrices!(fsm_vec, ea_fft, psf_image_vec);

# For compilation
elbo_likelihood_with_fft!(ea_fft, elbo_vars_fft, fsm_vec);

elbo_time = time()
elbo_likelihood_with_fft!(ea_fft, elbo_vars_fft, fsm_vec);
DeterministicVI.subtract_kl!(ea, elbo_vars_fft.elbo, calculate_derivs=true);
elbo_time = time() - elbo_time

println("Time ratio: ", elbo_time / current_elbo_time)

elbo_fft = deepcopy(elbo_vars_fft.elbo);

println(elbo_fft.v[1], " ", elbo.v[1])
hcat(elbo_fft.d, elbo.d)
# plot((elbo_fft.d + 1e-6) ./ (elbo.d + 1e-6), "o")
# plot((elbo_fft.h + 1e-6) ./ (elbo.h + 1e-6), "o")




######################################

# Debugging
populate_fsm_vec!(ea_fft, elbo_vars_fft, fsm_vec);

PyPlot.close("all")
b = 3
s = ea.active_sources[1]
sub_image = get_source_pixel_range(s, b, ea);
pix_loc =
    WCS.world_to_pix(ea_fft.images[b].wcs, ea_fft.vp[s][ids.u]) -
    Float64[sub_image.min_h - 1, sub_image.min_w - 1];
plot_loc() = plot(pix_loc[2] - 1, pix_loc[1] - 1,  "ro")

# Display fft image
fft_image = Float64[ sf.v[1] for sf in fsm_vec[b].E_G ];
for pixel in ea.active_pixels
    if pixel.n == b
        tile = ea.images[pixel.n].tiles[pixel.tile_ind]
        this_pixel = tile.pixels[pixel.h, pixel.w]

        # These are indices within the fs?m image.
        h_fsm = tile.h_range[pixel.h] - fsm_vec[b].h_lower + 1
        w_fsm = tile.w_range[pixel.w] - fsm_vec[b].w_lower + 1

        fft_image[h_fsm, w_fsm] *= tile.iota_vec[pixel.w]
    end
end
matshow(fft_image); title("FFT image"); colorbar(); plot_loc()

# Display rendered image
rendered_image = render_source(ea, s, sub_image, false);
matshow(rendered_image); title("Rendered image"); colorbar(); plot_loc()

# Display original image
original_image = show_source_image(ea, s, sub_image);
matshow(original_image); title("Original image"); colorbar(); plot_loc()

# Active pixel map
active_image = show_active_pixels(ea, sub_image, b);
matshow(active_image); title("Active pixels"); plot_loc()

active_image = show_active_pixels(ea_fft, sub_image, b);
matshow(active_image); title("FFT Active pixels"); plot_loc()



# Look at the mcs.  This is ok.
star_mcs, gal_mcs =
    load_bvn_mixtures(ea, b,
        calculate_derivs=false,
        calculate_hessian=false);

star_mcs_fft, gal_mcs_fft =
    load_bvn_mixtures(ea_fft, b,
        calculate_derivs=false,
        calculate_hessian=false);

###############
# Debugging
# Look at fsm.
populate_fsm_vec!(ea_fft, elbo_vars_fft, fsm_vec);
s = ea.active_sources[1]

PyPlot.close("all")
b = 3

star_mcs, gal_mcs =
    load_bvn_mixtures(ea, b,
        calculate_derivs=false,
        calculate_hessian=false);


x = SVector{2, Float64}(WCS.world_to_pix(ea_fft.images[b].wcs, ea_fft.vp[s][ids.u]))
wcs_jacobian = ea.patches[s, b].wcs_jacobian;
elbo_vars = DeterministicVI.ElboIntermediateVariables(
    Float64, ea.S, length(ea.active_sources));

fsms = fsm_vec[b];
ea_fs0m_image = fill(0.0, size(fsms.fs0m_image));
ea_fs1m_image = fill(0.0, size(fsms.fs1m_image));
for pixel in ea.active_pixels
  tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
  if pixel.n == b && s in tile_sources
      tile = ea.images[pixel.n].tiles[pixel.tile_ind]
      h_fsm = tile.h_range[pixel.h] - fsms.h_lower + 1
      w_fsm = tile.w_range[pixel.w] - fsms.w_lower + 1

      x = SVector{2, Float64}([tile.h_range[pixel.h], tile.w_range[pixel.w]])
      populate_fsm!(elbo_vars.bvn_derivs,
                    elbo_vars.fs0m_vec[s], elbo_vars.fs1m_vec[s],
                    false, false,
                    s, x, true, Inf,
                    wcs_jacobian, gal_mcs, star_mcs);
      ea_fs0m_image[h_fsm, w_fsm] = elbo_vars.fs0m_vec[s].v[1]
      ea_fs1m_image[h_fsm, w_fsm] = elbo_vars.fs1m_vec[s].v[1]
  end
end

fs0m_image = Float64[ sf.v[1] for sf in fsm_vec[b].fs0m_image ];
fs1m_image = Float64[ sf.v[1] for sf in fsm_vec[b].fs1m_image ];
fs0m_conv = Float64[ sf.v[1] for sf in fsm_vec[b].fs0m_conv ];
fs1m_conv = Float64[ sf.v[1] for sf in fsm_vec[b].fs1m_conv ];

# matshow(ea_fs0m_image); colorbar()
sum(ea_fs0m_image)
sum(fs0m_image)

sum(ea_fs1m_image)
sum(fs1m_image)

matshow(ea_fs0m_image); colorbar(); title("fs0m model")
matshow(fs0m_conv); colorbar(); title("fs0m FFT")

matshow(ea_fs1m_image); colorbar(); title("fs1m model")
matshow(fs1m_conv); colorbar(); title("fs1m FFT")



########################################
# Profiling:

# The FFT is taking almost all the time.
@profile begin
    for i in 1:10
        add_source_to_elbo!(ea, elbo_vars, s, sbs,
                            star_mcs_vec, gal_mcs_vec, fsm_vec);
    end
end

Profile.print()



##########
