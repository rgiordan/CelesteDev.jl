using Celeste: Model, DeterministicVI
using StaticArrays

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

const dir = "/home/rgiordan/Documents/git_repos/CelesteDev.jl/"

include(joinpath(dir, "rasterized_psf/elbo_pixelated_psf.jl"))
include(joinpath(dir, "celeste_tools/celeste_tools.jl"))

using ELBOPixelatedPSF

import Synthetic
using SampleData

using Base.Test
using Distributions

using PyPlot


##########
# Ensure that test images are available.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
rcf = Celeste.SDSSIO.RunCamcolField(4263, 5,119)
images = Celeste.SDSSIO.load_field_images(rcf, datadir);
catalog = SDSSIO.read_photoobj_files([rcf], datadir, duplicate_policy=:first);

# Pick an object.
for cat in catalog
    if minimum(cat.star_fluxes) > 300
        print(cat.objid)
        print("\n")
    end
end
objid = "1237663784734490677"
objids = [ce.objid for ce in catalog];
sa = findfirst(objids, objid);
neighbors = Infer.find_neighbors([sa], catalog, images)[1];
cat_local = vcat(catalog[sa], catalog[neighbors]);
vp = Vector{Float64}[Celeste.Infer.init_source(ce) for ce in cat_local];
patches = Celeste.Infer.get_sky_patches(images, cat_local);
ea = ElboArgs(images, vp, patches, [1]);
Celeste.Infer.load_active_pixels!(ea);

# For compiling
elbo = Celeste.DeterministicVI.elbo(ea);

current_elbo_time = time()
elbo = Celeste.DeterministicVI.elbo(ea);
current_elbo_time = time() - current_elbo_time



###########################
# FFT elbo

using StaticArrays
import Celeste.Model
import Celeste.Model.SkyPatch

ea_fft = ElboArgs(images, deepcopy(vp), patches, [1], psf_K=1);
Celeste.Infer.load_active_pixels!(ea_fft, exclude_nan=false);

psf_image_mat = Matrix{Float64}[
    PSF.get_psf_at_point(ea.patches[s, b].psf) for s in 1:ea.S, b in 1:ea.N];
fsm_vec = ELBOPixelatedPSF.FSMSensitiveFloatMatrices[
    ELBOPixelatedPSF.FSMSensitiveFloatMatrices() for b in 1:ea_fft.N];
ELBOPixelatedPSF.initialize_fsm_sf_matrices!(fsm_vec, ea_fft, psf_image_mat);

function elbo_fft_opt{NumType <: Number}(
                    ea::ElboArgs{NumType};
                    calculate_derivs=true,
                    calculate_hessian=true)
    @assert ea.psf_K == 1
    ELBOPixelatedPSF.elbo_likelihood_with_fft!(ea, 2, fsm_vec);
    DeterministicVI.subtract_kl!(ea, elbo, calculate_derivs=calculate_derivs)
    return deepcopy(ea.elbo_vars.elbo)
end

# For compilation
#include(joinpath(dir, "rasterized_psf/elbo_pixelated_psf.jl"))
ELBOPixelatedPSF.elbo_likelihood_with_fft!(ea_fft, 2, fsm_vec);

ea_fft.elbo_vars.elbo

elbo_fft = elbo_fft_opt(ea_fft);

elbo_time = time()
elbo_fft = elbo_fft_opt(ea_fft);
elbo_time = time() - elbo_time

println("Time ratio: ", elbo_time / current_elbo_time)


#####################
# images

s = 1
n = 3

graph_vp = deepcopy(vp)
graph_vp[s][ids.a] = [1, 0]
# graph_vp[s][ids.e_scale] = 2
ea.vp = deepcopy(graph_vp)
ea_fft.vp = deepcopy(graph_vp)

image_fft = render_source_fft(ea, fsm_vec, s, n, include_iota=false, field=:E_G);
image_orig = render_source(ea, s, n, include_iota=false, field=:E_G);

PyPlot.close("all")
matshow(image_fft); colorbar(); title("fft")
matshow(image_orig); colorbar(); title("original")
matshow(image_fft - image_orig); colorbar(); title("diff")
PyPlot.figure(); plot(image_fft[:], image_orig[:], "ko"); plot(maximum(image_fft), maximum(image_fft), "rx")


image_diff = image_fft - image_orig

sum(abs(image_diff[!isnan(image_diff)])) / sum(abs(image_orig[!isnan(image_orig)]))
mean(image_diff[!isnan(image_diff)])


######################
# Optimize


ea_fft = ElboArgs(images, deepcopy(vp), patches, [1], psf_K=1);
Celeste.Infer.load_active_pixels!(ea_fft, exclude_nan=false);
f_evals_fft, max_f_fft, max_x_fft, nm_result_fft =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft,
                               verbose=true, max_iters=200);

ea = ElboArgs(images, deepcopy(vp), patches, [1]);
Celeste.Infer.load_active_pixels!(ea);
f_evals, max_f, max_x, nm_result =
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea, verbose=true);

max_f_fft
max_f

s = 1
b = 3

orig_pix_loc = source_pixel_location(ea, s, b)
fft_pix_loc = source_pixel_location(ea_fft, s, b)


PyPlot.close("all")
matshow(render_source(ea_fft, s, b));
plot(fft_pix_loc[1] - 1, fft_pix_loc[2] - 1, "ro"); colorbar();
title("fft rendered")

matshow(render_source(ea, s, b));
plot(orig_pix_loc[1] - 1, orig_pix_loc[2] - 1, "ro"); colorbar();
title("orig rendered")

matshow(show_source_image(ea, s, b)); colorbar(); title("original")
fft_diff = show_source_image(ea, s, b) - render_source(ea_fft, s, b);
fft_diff[isnan(fft_diff)] = 0;

orig_diff = show_source_image(ea, s, b) - render_source(ea, s, b);
orig_diff[isnan(orig_diff)] = 0;

matshow(fft_diff); colorbar(); title("fft diff")
matshow(orig_diff); colorbar(); title("orig diff")

sum(abs(fft_diff)) / sum(abs(orig_diff))
mean(fft_diff)
mean(orig_diff)

###################

sbs = Celeste.DeterministicVI.load_source_brightnesses(ea,
    calculate_derivs=ea.elbo_vars.calculate_derivs,
    calculate_hessian=ea.elbo_vars.calculate_hessian);

using DeterministicVI.GalaxyCacheComponent
gal_mcs_vec = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N);
# for b=1:ea.N
b = 3
s = 1
gal_mcs_vec[b] = ELBOPixelatedPSF.load_gal_bvn_mixtures(
        ea.S, ea.patches, ea.vp, ea.active_sources, b,
        calculate_derivs=ea.elbo_vars.calculate_derivs,
        calculate_hessian=ea.elbo_vars.calculate_hessian);
# end

import Celeste.Model.lidx
import Celeste.Model.GalaxySigmaDerivs
import Celeste.Model.get_bvn_cov
import Celeste.Model.galaxy_prototypes
import Celeste.Model.linear_world_to_pix

gal_mcs = gal_mcs_vec[b]
NumType = Float64
sp = ea.vp[1];
world_loc = sp[lidx.u]
m_pos = linear_world_to_pix(
    ea.patches[s, b].wcs_jacobian,
    ea.patches[s, b].center,
    ea.patches[s, b].pixel_center, world_loc)

i = 1
e_dev_dir = (i == 1) ? 1. : -1.
e_dev_i = (i == 1) ? sp[lidx.e_dev] : 1. - sp[lidx.e_dev]
j = 1
gal_mcs[1, j, i, s] = GalaxyCacheComponent(
    e_dev_dir, e_dev_i, galaxy_prototypes[i][j], m_pos,
    sp[lidx.e_axis], sp[lidx.e_angle], sp[lidx.e_scale],
    true, true)

gc = galaxy_prototypes[i][j];
e_axis = sp[lidx.e_axis]
e_angle = sp[lidx.e_angle]
e_scale = sp[lidx.e_scale]

import Celeste.DeterministicVI.BvnComponent
gal_mcs = ELBOPixelatedPSF.load_gal_bvn_mixtures(
        ea.S, ea.patches, ea.vp, ea.active_sources, b,
        calculate_derivs=ea.elbo_vars.calculate_derivs,
        calculate_hessian=ea.elbo_vars.calculate_hessian);

XiXi = get_bvn_cov(e_axis, e_angle, e_scale)
var_s = gc.nuBar * XiXi

bmc = BvnComponent{NumType}(
    SVector{2, NumType}([ 05, 5.0 ]), var_s, gc.etaBar, true)

sig_sf = GalaxySigmaDerivs(e_angle, e_axis, e_scale, XiXi, true)
GalaxyCacheComponent(e_dev_dir, e_dev_i, bmc, sig_sf)


using ELBOPixelatedPSF.populate_star_fsm_image!
using ELBOPixelatedPSF.populate_gal_fsm_image!
using ELBOPixelatedPSF.populate_source_band_brightness!

n = 3
fsms = fsm_vec[n]
lanczos_width = 2
for s in 1:ea.S
    populate_star_fsm_image!(
        ea, s, n, fsms.psf_vec[s], fsms.fs0m_conv,
        fsms.h_lower, fsms.w_lower, lanczos_width)
    populate_gal_fsm_image!(ea, s, n, gal_mcs, fsms)
    populate_source_band_brightness!(ea, s, n, fsms, sbs[s])
end

[ sf.d[1] for sf in fsms.fs1m_conv ]

h_fsm, w_fsm = 10, 10
Celeste.DeterministicVI.accumulate_source_pixel_brightness!(
                    ea.elbo_vars,
                    ea,
                    fsms.E_G[h_fsm, w_fsm],
                    fsms.var_G[h_fsm, w_fsm],
                    fsms.fs0m_conv[h_fsm, w_fsm],
                    fsms.fs1m_conv[h_fsm, w_fsm],
                    sbs[s], b, s, true)
fsms.E_G[h_fsm, w_fsm].d

######################################

# Debugging
# populate_star_fsm_image!(ea, elbo_vars, s, b, fsms.psf_vec[s], fsms.fs0m_conv, lanczos_width)
# populate_gal_fsm_image!(ea, elbo_vars, s, b, gal_mcs_vec[b], fsms)
# populate_source_band_brightness!(ea, elbo_vars, s, b, fsms, sbs[s])


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
