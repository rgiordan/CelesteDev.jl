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
#objid = "1237663784734490643"
objids = [ cat.objid for cat in catalog ];
objid = "1237663784734491069"
sa = findfirst(objids, objid);
neighbors = Infer.find_neighbors([sa], catalog, images)[1];
cat_local = vcat(catalog[sa], catalog[neighbors]);
vp = DeterministicVI.init_sources([1], cat_local)
patches = Infer.get_sky_patches(images, cat_local);

ea = ElboArgs(images, deepcopy(vp), patches, [1]);
Celeste.Infer.load_active_pixels!(ea.images, ea.patches);

ea_fft = ElboArgs(images, deepcopy(vp), patches, [1], psf_K=1);
Celeste.Infer.load_active_pixels!(ea_fft.images, ea.patches; exclude_nan=false);

psf_image_mat = Matrix{Float64}[
    PSF.get_psf_at_point(ea.patches[s, b].psf) for s in 1:ea_fft.S, b in 1:ea_fft.N];
fsm_mat = DeterministicVIImagePSF.FSMSensitiveFloatMatrices[
    DeterministicVIImagePSF.FSMSensitiveFloatMatrices() for
    s in 1:ea_fft.S, b in 1:ea_fft.N];
DeterministicVIImagePSF.initialize_fsm_sf_matrices!(fsm_mat, ea_fft, psf_image_mat);

s = 1
n = 3

image_fft = CelesteEDA.render_source_fft(ea, fsm_mat, s, n, include_iota=false, field=:E_G);
image_orig = CelesteEDA.render_source(ea, s, n, include_iota=false, field=:E_G);

PyPlot.close("all")
matshow(image_fft); colorbar(); title("fft Celeste")
matshow(image_orig); colorbar(); title("original Celeste")


######################
# Set initial parameters

# For now set this to false so we are comparing apples to apples
use_raw_psf = true
ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(vp), patches, [1], use_raw_psf=use_raw_psf);
for s in 1:ea_fft.S, n in 1:ea_fft.N
    fsm_mat[s, n].kernel_width = 2
    fsm_mat[s, n].kernel_fun =
        x -> DeterministicVIImagePSF.cubic_kernel_with_derivatives(x, 0.0)
    # fsm_mat[n].kernel_fun =
    #     x -> DeterministicVIImagePSF.bspline_kernel_with_derivatives(x)
end
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);

ea = ElboArgs(images, deepcopy(vp), patches, [1]);

#######################
# Optimize

f_evals_fft, max_f_fft, max_x_fft, nm_result_fft, transform_fft =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft,
                               verbose=true, max_iters=100);
vp_opt_fft = deepcopy(ea_fft.vp[1]);

f_evals, max_f, max_x, nm_result, transform =
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea, verbose=true);
vp_opt = deepcopy(ea.vp[1]);

max_f_fft
max_f

println("FFT ELBO improvement")
(max_f_fft - max_f)  / abs(max_f)

println("FFT Extra iterations")
(nm_result_fft.iterations - nm_result.iterations) / nm_result.iterations

# df = DataFrame(ids=ids_names, vp_fft=vp_opt_fft, vp_orig=vp_opt,
#                pdiff=(vp_opt_fft - vp_opt) ./ vp_opt)

