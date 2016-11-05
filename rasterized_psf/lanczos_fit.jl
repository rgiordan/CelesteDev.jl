using Celeste: Model, DeterministicVI
using StaticArrays

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

include("celeste_tools/celeste_tools.jl")
const dir = "/home/rgiordan/Documents/git_repos/CelesteDev.jl/"

# include(joinpath(dir, "rasterized_psf/lanczos.jl"))
include(joinpath(dir, "rasterized_psf/elbo_pixelated_psf.jl"))

using ELBOPixelatedPSF

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
Celeste.Infer.load_active_pixels!(ea, false);
length(ea.active_pixels)

# For compiling
elbo = DeterministicVI.elbo(ea);

current_elbo_time = time()
elbo = DeterministicVI.elbo(ea);
current_elbo_time = time() - current_elbo_time



###########################
using StaticArrays
import Celeste.Model
import Celeste.Model.SkyPatch

ea_fft = ElboArgs(tiled_images, deepcopy(vp), tile_source_map,
                  deepcopy(patches), [1]; psf_K=1);
Celeste.Infer.load_active_pixels!(ea_fft, false);

psf_image_mat = Matrix{Float64}[
    PSF.get_psf_at_point(ea.patches[s, b].psf) for s in 1:ea.S, b in 1:ea.N];

elbo_vars_fft = DeterministicVI.ElboIntermediateVariables(
    Float64, ea_fft.S, length(ea_fft.active_sources));

fsm_vec = ELBOPixelatedPSF.FSMSensitiveFloatMatrices[
    ELBOPixelatedPSF.FSMSensitiveFloatMatrices() for b in 1:ea_fft.N];
ELBOPixelatedPSF.initialize_fsm_sf_matrices!(fsm_vec, ea_fft, psf_image_mat);

# For compilation
function elbo_fft_opt{NumType <: Number}(
                    ea::ElboArgs{NumType};
                    calculate_derivs=true,
                    calculate_hessian=true)
    elbo_vars_fft = DeterministicVI.ElboIntermediateVariables(
        Float64, ea.S, length(ea.active_sources));
    @assert ea.psf_K == 1
    ELBOPixelatedPSF.elbo_likelihood_with_fft!(ea, elbo_vars_fft, 1, fsm_vec);
    DeterministicVI.subtract_kl!(ea, elbo, calculate_derivs=calculate_derivs)
    return deepcopy(elbo_vars_fft.elbo)
end

elbo_fft = elbo_fft_opt(ea_fft);

elbo_time = time()
elbo_fft_opt(ea_fft);
elbo_time = time() - elbo_time

println("Time ratio: ", elbo_time / current_elbo_time)

println(elbo_fft.v[1], " ", elbo.v[1])
hcat(elbo_fft.d, elbo.d)
# plot((elbo_fft.d + 1e-6) ./ (elbo.d + 1e-6), "o")
# plot((elbo_fft.h + 1e-6) ./ (elbo.h + 1e-6), "o")

f_evals_fft, max_f_fft, max_x_fft, nm_result_fft =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft, verbose=true);
f_evals, max_f, max_x, nm_result =
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea, verbose=true);

max_f_fft
max_f

hcat(max_x_fft, max_x)


######################################

# Debugging
# populate_star_fsm_image!(ea, elbo_vars, s, b, fsms.psf_vec[s], fsms.fs0m_conv, lanczos_width)
# populate_gal_fsm_image!(ea, elbo_vars, s, b, gal_mcs_vec[b], fsms)
# populate_source_band_brightness!(ea, elbo_vars, s, b, fsms, sbs[s])

include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/predicted_image.jl")
populate_fsm_vec!(ea_fft, elbo_vars_fft, fsm_vec, 1);

PyPlot.close("all")
b = 5
s = ea.active_sources[1]
sub_image = get_source_pixel_range(s, b, ea);
pix_loc =
    WCS.world_to_pix(ea_fft.images[b].wcs, ea_fft.vp[s][ids.u]) -
    Float64[sub_image.min_h - 1, sub_image.min_w - 1];
plot_loc() = plot(pix_loc[2] - 1, pix_loc[1] - 1,  "ro")

# Display fft image
minimum([ sf.v[1] for sf in fsm_vec[b].fs0m_conv ])
minimum([ sf.v[1] for sf in fsm_vec[b].fs1m_conv ])
maximum([ sf.v[1] for sf in fsm_vec[b].fs0m_conv ])
maximum([ sf.v[1] for sf in fsm_vec[b].fs1m_conv ])

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

# include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/lanczos.jl")
# include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/predicted_image.jl")
#
# fft_image = Float64[ sf.v[1] for sf in fsm_vec[b].fs0m_conv ];
# minimum(fft_image)
# maximum(fft_image)
# matshow(fft_image); title("FFT image"); colorbar(); plot_loc()
# sum(fft_image[fft_image .< 0])
# sum(fft_image[fft_image .> 0])
# sum(fft_image)
# sum(fsm_vec[b].psf_vec[s])
# matshow(fsm_vec[b].psf_vec[s]); colorbar(); title("PSF")

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
