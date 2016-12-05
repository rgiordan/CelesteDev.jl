using Celeste: Model, DeterministicVI
using StaticArrays

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField
import Celeste.DeterministicVIImagePSF
import Celeste: CelesteEDA

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

const dir = "/home/rgiordan/Documents/git_repos/CelesteDev.jl/"

# include(joinpath(dir, "rasterized_psf/elbo_pixelated_psf.jl"))
# include(joinpath(dir, "celeste_tools/celeste_tools.jl"))

# using DeterministicVIImagePSF

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
# objid = "1237663784734490677"
objid = "1237663784734490643"
objids = [ce.objid for ce in catalog];
sa = findfirst(objids, objid);
neighbors = Infer.find_neighbors([sa], catalog, images)[1];
cat_local = vcat(catalog[sa], catalog[neighbors]);
vp = Vector{Float64}[Celeste.Infer.init_source(ce) for ce in cat_local];
patches = Celeste.Infer.get_sky_patches(images, cat_local);
Celeste.Infer.load_active_pixels!(images, patches; exclude_nan=false);
ea = ElboArgs(images, vp, patches, [1]);


######################
# Optimize

# For now set this to false so we are comparing apples to apples
use_raw_psf = false
ea_fft, fsm_vec = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(vp), patches, [1], use_raw_psf=use_raw_psf);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_vec, 2);

ea = ElboArgs(images, deepcopy(vp), patches, [1]);

f_evals_fft, max_f_fft, max_x_fft, nm_result_fft =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft,
                               verbose=true, max_iters=200);

f_evals, max_f, max_x, nm_result =
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea, verbose=true);

max_f_fft
max_f

############################
# Render

s = 1
b = 5

orig_pix_loc = Celeste.CelesteEDA.source_pixel_location(ea, s, b)
fft_pix_loc = Celeste.CelesteEDA.source_pixel_location(ea_fft, s, b)

fft_rendered = CelesteEDA.render_source_fft(ea_fft, fsm_vec, s, b);
orig_rendered = CelesteEDA.render_source(ea, s, b);
raw_image = CelesteEDA.show_source_image(ea, s, b);
vmax = maximum([ maximum(fft_rendered), 
                 maximum(orig_rendered),
                 maximum(raw_image) ])

PyPlot.close("all")
matshow(fft_rendered, vmin=0, vmax=vmax);
plot(fft_pix_loc[1] - 1, fft_pix_loc[2] - 1, "ro"); colorbar();
title("fft rendered")

matshow(orig_rendered, vmin=0, vmax=vmax);
plot(orig_pix_loc[1] - 1, orig_pix_loc[2] - 1, "ro"); colorbar();
title("orig rendered")

matshow(raw_image, vmin=0, vmax=vmax); colorbar(); title("raw image")


fft_diff = CelesteEDA.show_source_image(ea, s, b) -
           CelesteEDA.render_source(ea_fft, s, b);
fft_diff[isnan(fft_diff)] = 0;

orig_diff = CelesteEDA.show_source_image(ea, s, b) -
            CelesteEDA.render_source(ea, s, b);
orig_diff[isnan(orig_diff)] = 0;

matshow(fft_diff); colorbar(); title("fft residuals")
matshow(orig_diff); colorbar(); title("orig residuals")

sum(abs(fft_diff)) / sum(abs(orig_diff))
mean(fft_diff)
mean(orig_diff)


########################
# Look at PSF differences

PyPlot.close("all")
s = 1
for b in 1:5
    orig_psf = Celeste.PSF.get_psf_at_point(ea.patches[s, b].psf);
    matshow(fsm_vec[b].psf_vec[s] - orig_psf); colorbar(); title(b)
end


###########################
# Timing comparison

# For compiling
elbo = Celeste.DeterministicVI.elbo(ea);

current_elbo_time = time()
elbo = Celeste.DeterministicVI.elbo(ea);
current_elbo_time = time() - current_elbo_time


ea_fft, fsm_vec = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(vp), patches, [1]);

elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_vec, 2);

elbo_fft = elbo_fft_opt(ea_fft);

elbo_time = time()
elbo_fft = elbo_fft_opt(ea_fft);
elbo_time = time() - elbo_time

println("Time ratio: ", elbo_time / current_elbo_time)

