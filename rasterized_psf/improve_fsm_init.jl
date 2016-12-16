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
objid = "1237663784734490800"
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

# For now set this to false so we are comparing apples to apples
use_raw_psf = true
ea_fft, fsm_vec = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(vp), deepcopy(patches), [1], use_raw_psf=use_raw_psf);
for n in 1:ea_fft.N
    fsm_vec[n].kernel_width = 2
    # fsm_vec[n].kernel_fun =
    #     x -> DeterministicVIImagePSF.cubic_kernel_with_derivatives(x, -0.75)
    fsm_vec[n].kernel_fun =
        x -> DeterministicVIImagePSF.bspline_kernel_with_derivatives(x)
end
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_vec);

ea = ElboArgs(images, deepcopy(vp), patches, [1]);


# ##################
# # Trim PSF


elbo_fft_opt(ea_fft);
fft_time = time()
elbo_fft_opt(ea_fft);
fft_time = time() - fft_time

DeterministicVI.elbo(ea);
orig_time = time()
DeterministicVI.elbo(ea);
orig_time = time() - orig_time

println("FFT extra time:")
(fft_time - orig_time) / orig_time



#######################
# Optimize

fft_time = time()
f_evals_fft, max_f_fft, max_x_fft, nm_result_fft, transform_fft =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft,
                               verbose=true, max_iters=100);
fft_time = time() - fft_time
vp_opt_fft = deepcopy(ea_fft.vp[1]);

orig_time = time()
f_evals, max_f, max_x, nm_result, transform =
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea, verbose=true);
orig_time = time() - orig_time
vp_opt = deepcopy(ea.vp[1]);


max_f_fft
max_f

println("FFT ELBO improvement")
(max_f_fft - max_f)  / abs(max_f)

println("FFT Extra iterations")
(nm_result_fft.iterations - nm_result.iterations) / nm_result.iterations

df = DataFrame(ids=ids_names, vp_fft=vp_opt_fft, vp_orig=vp_opt,
               pdiff=(vp_opt_fft - vp_opt) ./ vp_opt)







