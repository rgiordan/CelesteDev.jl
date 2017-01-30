using JLD
using Celeste
using PyPlot


import Celeste: Infer, DeterministicVI, ParallelRun, Model, DeterministicVI,
    PSF, SDSSIO, SensitiveFloats, Transform, DeterministicVIImagePSF,
    CelesteEDA

using DerivativeTestUtils

using Base.Test

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

import Celeste.DeterministicVI:
    ElboArgs, ElboIntermediateVariables, maximize_f, maximize_f_two_steps,
    StarPosParams, GalaxyPosParams, CanonicalParams, VariationalParams,
    SourceBrightness, GalaxyComponent, SkyPatch,
    load_source_brightnesses, add_elbo_log_term!,
    accumulate_source_pixel_brightness!,
    KLDivergence, init_sources

import Celeste.Infer: load_active_pixels!, get_sky_patches, is_pixel_in_patch,
    get_active_pixel_range

const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")



using Celeste.Model.ids
using Celeste.Model.ids_free

run, camcol, field = (3900, 6, 269)

images = SDSSIO.load_field_images(RunCamcolField(run, camcol, field), datadir);
dir = joinpath(datadir, "$run/$camcol/$field")
fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir run camcol field
catalog = SDSSIO.read_photoobj_celeste(fname);

# Pick an object.
objid = "1237662226208063499"
objids = [ce.objid for ce in catalog];
sa = findfirst(objids, objid)
neighbors = Infer.find_neighbors([sa], catalog, images)[1]

cat_local = vcat(catalog[sa:sa], catalog[neighbors])
vp_init = Vector{Float64}[DeterministicVI.catalog_init_source(ce) for ce in cat_local];
patches = Infer.get_sky_patches(images, cat_local);
# ea = ElboArgs(images, vp, patches, [1]);



ea = ElboArgs(images, deepcopy(vp_init), patches, [1], psf_K=1);
Infer.load_active_pixels!(images, ea.patches; exclude_nan=false);
fsm_mat = DeterministicVIImagePSF.load_fsm_mat(ea, images; use_raw_psf=false, use_trimmed_psf=true);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea, fsm_mat);


transform = Transform.get_mp_transform(ea.vp, ea.active_sources);
omitted_ids = Int[]
star_ids_free = vcat(ids_free.u,
                     ids_free.r1[1], ids_free.r2[1],
                     ids_free.c1[:, 1][:], ids_free.c2[:, 1][:],
                     ids_free.k[:, 1][:])

star_omitted_ids = union(setdiff(1:length(ids_free), star_ids_free),
                         omitted_ids)


ea.vp = deepcopy(vp_init);
ea.active_source_star_only = false
f_evals_star, max_f_star, max_x_star, nm_result =
 maximize_f(elbo_fft_opt, ea, transform;
         verbose=true,
         max_iters=200);
vp_opt_joint = deepcopy(ea.vp);
nm_result_joint = deepcopy(nm_result)
print(vp_opt_joint)
x_vals = reduce(hcat, [ tr.metadata["x"] for tr in nm_result_joint.trace ])
delta_vals = [ tr.metadata["delta"] for tr in nm_result_joint.trace ]
f_vals = [ tr.value for tr in nm_result_joint.trace ];
plot(log(f_vals))


ea.vp = deepcopy(vp_init);
ea.active_source_star_only = false
ea.vp[1][ids.a] = [1, 0]
time_full = Sys.time()
f_evals_star, max_f_star, max_x_star, nm_result =
    maximize_f(elbo_fft_opt, ea, transform;
            omitted_ids=star_omitted_ids,
            verbose=true,
            max_iters=200);
time_full = Sys.time() - time_full
vp_opt_full = deepcopy(ea.vp);
print(vp_opt_full)
nm_result_full = deepcopy(nm_result);
x_vals = reduce(hcat, [ tr.metadata["x"] for tr in nm_result_full.trace ])
delta_vals = [ tr.metadata["delta"] for tr in nm_result_full.trace ]
f_vals = [ tr.value for tr in nm_result_full.trace ];
plot(log(f_vals))
plot(delta_vals)


ea.vp = deepcopy(vp_init);
ea.active_source_star_only = true
ea.vp[1][ids.a] = [1, 0]
time_star = Sys.time()
f_evals_star, max_f_star, max_x_star, nm_result =
    maximize_f(elbo_fft_opt, ea, transform;
               omitted_ids=star_omitted_ids,
               verbose=true,
               max_iters=200);
time_star = Sys.time() - time_star
vp_opt_star = deepcopy(ea.vp);
nm_result_star = deepcopy(nm_result);
x_vals = reduce(hcat, [ tr.metadata["x"] for tr in nm_result_star.trace ])
delta_vals = [ tr.metadata["delta"] for tr in nm_result_star.trace ]
f_vals = [ tr.value for tr in nm_result_star.trace ];
plot(log(f_vals))
plot(log10(delta_vals))


vp_df = CelesteEDA.print_vp(vp_init[1])
vp_df[:star] = vp_opt_star[1] 
vp_df[:full] = vp_opt_full[1]
vp_df[:joint] = vp_opt_joint[1]
println(vp_df)
 


