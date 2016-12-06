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

using DataFrames

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
# Set initial parameters

lanczos_width = 3

# For now set this to false so we are comparing apples to apples
use_raw_psf = false
ea_fft, fsm_vec = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(vp), patches, [1], use_raw_psf=use_raw_psf);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(
    ea_fft, fsm_vec, lanczos_width);

ea = ElboArgs(images, deepcopy(vp), patches, [1]);

#######################
# Optimize
f_evals_fft, max_f_fft, max_x_fft, nm_result_fft =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft,
                               verbose=true, max_iters=200);
vp_opt_fft = deepcopy(ea_fft.vp[1]);

f_evals, max_f, max_x, nm_result =
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea, verbose=true);
vp_opt = deepcopy(ea.vp[1]);

max_f_fft
max_f

df = DataFrame(ids=ids_names, vp_fft=vp_opt_fft, vp_orig=vp_opt,
               pdiff=(vp_opt_fft - vp_opt) ./ vp_opt)

# There is evidently an optimization problem with the FFT stuff.
[ tr.metadata["delta"] for tr in nm_result_fft.trace ]
[ tr.metadata["delta"] for tr in nm_result.trace ]


############################
# Render

s = 1
b = 3

# Set parameters to view if you want to
# vp_render = deepcopy(vp[s]);
# vp_render[ids.a] = [0, 1];
# ea.vp[s] = deepcopy(vp_render);
# ea_fft.vp[s] = deepcopy(vp_render);
# df = DataFrame(ids=ids_names, vp_fft=ea_fft.vp[s], vp_orig=ea.vp[s]);
# df[:pdiff] = (df[:vp_fft] - df[:vp_orig]) ./ df[:vp_orig];

orig_pix_loc = Celeste.CelesteEDA.source_pixel_location(ea, s, b)
fft_pix_loc = Celeste.CelesteEDA.source_pixel_location(ea_fft, s, b)

fft_rendered = CelesteEDA.render_source_fft(
    ea_fft, fsm_vec, s, b, lanczos_width=lanczos_width);
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
matshow(raw_image); colorbar(); title("raw image")

fft_diff = fft_rendered - raw_image;
fft_diff[isnan(fft_diff)] = 0;

orig_diff = orig_rendered - raw_image;
orig_diff[isnan(orig_diff)] = 0;

matshow(fft_diff); colorbar(); title("fft residuals")
matshow(orig_diff); colorbar(); title("orig residuals")
matshow(fft_rendered - orig_rendered); colorbar(); title("fft - orig")

sum(abs(fft_diff)) / sum(abs(orig_diff))
mean(fft_diff)
mean(orig_diff)


########################
# Look at PSF differences

PyPlot.close("all")
s = 1
for b in 1:5
    orig_psf = Celeste.PSF.get_psf_at_point(ea.patches[s, b].psf);
    # matshow(fsm_vec[b].psf_vec[s] - orig_psf); colorbar(); title(b)
    matshow(fsm_vec[b].psf_vec[s]); colorbar(); title(b)
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



###########################
# GSL derivative Check

using GSL

ea_fft.vp = deepcopy(vp);
ea.vp = deepcopy(vp);

x_ind = 1
fft_elbo = true
function elbo_component(x::Float64, calculate_derivs::Bool)
    if fft_elbo
        ea_fft.vp[1][x_ind] = x
        return elbo_fft_opt(ea_fft, calculate_derivs=calculate_derivs)
    else
        ea.vp[1][x_ind] = x
        return DeterministicVI.elbo(ea)        
    end
end

function elbo_component_val(x::Float64)
    return elbo_component(x, false).v[]
end


# Test the gradients
diffs = fill(NaN, length(ea_fft.vp[1]));
for x_ind in 1:length(ea_fft.vp[1])
    val = ea_fft.vp[1][x_ind]
    result, abserr = deriv_central(elbo_component_val, val, 1e-8)
    elbo = elbo_component(val, true);
    diffs[x_ind] = (elbo.d[x_ind] - result + 1e-6) / (result + 1e-6)
    println(ids_names[x_ind], ": ", diffs[x_ind])
end

df = DataFrame(names=ids_names, diffs=diffs)


x_grad_ind = 1
function elbo_component_grad_val(x::Float64)
    return elbo_component(x, false).d[x_grad_ind, 1]
end


# Test the gradients
n = length(ea_fft.vp[1]);
result_mat = fill(NaN, n, n);
for x_ind in 1:n
    println(x_ind)
    val = ea_fft.vp[1][x_ind]
    elbo = elbo_component(val, true);
    for x_grad_ind in 1:n
        print(".")
        result, abserr = deriv_central(elbo_component_grad_val, val, 1e-8)
        result_mat[x_ind, x_grad_ind] = result
    end
    println("Done.")
end

elbo = DeterministicVI.elbo(ea);
elbo_fft = elbo_fft_opt(ea_fft, calculate_derivs=true);

eigvals(elbo.h)
eigvals(elbo_fft.h)

function condition_number(mat)
    ev = eigvals(mat)
    return maximum(abs(ev)) / minimum(abs(ev))
end

condition_number(elbo.h)
condition_number(elbo_fft.h)

diff = (result_mat - elbo_fft.h) ./ (result_mat  + 1e-6);
x_ind, y_ind = ind2sub(size(result_mat), find(abs(diff) .> 0.01))
hcat(ids_names[x_ind], ids_names[y_ind])