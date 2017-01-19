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
ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(vp), patches, [1], use_raw_psf=use_raw_psf);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);

ea = ElboArgs(images, deepcopy(vp), patches, [1]);


#######################
# Optimize

gal_only_ids_free = vcat(ids_free.e_dev, ids_free.e_axis,
                    ids_free.e_angle, ids_free.e_scale,
                    ids_free.r1[2], ids_free.r2[2],
                    ids_free.c1[:, 2][:], ids_free.c2[:, 2][:],
                    ids_free.k[:, 2][:])

star_only_ids_free = vcat(ids_free.r1[1], ids_free.r2[1],
                     ids_free.c1[:, 1][:], ids_free.c2[:, 1][:],
                     ids_free.k[:, 1][:])


gal_only_ids = vcat(ids.e_dev, ids.e_axis, ids.e_angle, ids.e_scale,
                 ids.r1[2], ids.r2[2],
                 ids.c1[:, 2][:], ids.c2[:, 2][:],
                 ids.k[:, 2][:])

star_only_ids = vcat(ids.r1[1], ids.r2[1],
                  ids.c1[:, 1][:], ids.c2[:, 1][:],
                  ids.k[:, 1][:])


ea_fft.vp[1][ids.a] = [1, 0]
f_evals_fft, max_f_fft, max_x_fft, nm_result_fft, transform_fft =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft,
                               omitted_ids=[1],
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
