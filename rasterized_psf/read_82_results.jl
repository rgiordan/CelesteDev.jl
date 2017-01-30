using JLD
using Celeste
using PyPlot


import Celeste: Infer, DeterministicVI, ParallelRun, Model, DeterministicVI,
    PSF, SDSSIO, SensitiveFloats, Transform, DeterministicVIImagePSF,
    CelesteEDA

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


#############################
# Load stripe82 results.

stripe82_dir = joinpath(Pkg.dir("Celeste"), "benchmark/stripe82/")
stripe_82_results = JLD.load(joinpath(stripe82_dir, "results_and_errors_small_box_fft-004263-5-0119.jld"))

keep_cols = [:objid, :is_star, :gal_mag_r, :star_mag_r ]
results = hcat(stripe_82_results["celeste_df"][keep_cols],
               stripe_82_results["coadd_df"][keep_cols])
results[:gal_mag_error] = results[:gal_mag_r] - results[:gal_mag_r_1] 
both_gal = (results[:is_star] .< 0.5) & !results[:is_star_1]
foo = (results[both_gal, :]);
sort!(foo, cols=:gal_mag_error);
foo

stripe_82_results["celeste_err"]

#########################
# Load images

const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
rcf = Celeste.SDSSIO.RunCamcolField(4263, 5,119);
images = Celeste.SDSSIO.load_field_images(rcf, datadir);
catalog = SDSSIO.read_photoobj_files([rcf], datadir, duplicate_policy=:first);
objids = [ cat.objid for cat in catalog ];

# Get the RA, DEC coordinates of the box.
for n in 1:length(images)
    println(n)
    println(WCS.pix_to_world(images[n].wcs, Float64[1., 1.]))
    println(WCS.pix_to_world(images[n].wcs, Float64[images[n].H, 1]))
    println(WCS.pix_to_world(images[n].wcs, Float64[1., images[n].W]))
    println(WCS.pix_to_world(images[n].wcs, Float64[images[n].H, images[n].W]))
end

# Pick a source and optimize
#coadd_objid = 8647474692482139458
# coadd_objid = 8647474692482138819
# objid = convert(String, stripe_82_results["celeste_df"][stripe_82_results["coadd_df"][:objid] .== coadd_objid, :objid][1])

#objid = "1237663784734491701"
objid = "1237663784734490687" # A NaN object on the MOG code
results[ results[:objid] .== objid, :]

target_sources =  [ findfirst(objids, objid) ];
neighbor_map = Infer.find_neighbors(target_sources, catalog, images);

ts = 1
s = target_sources[ts]
entry = catalog[s]
print(entry)
neighbors = catalog[neighbor_map[ts]]

# vs_opt = DeterministicVIImagePSF.infer_source_fft_two_step(
#     images, neighbors, entry);
cat_local = vcat([entry], neighbors)
vp_init = init_sources([1], cat_local);
patches = get_sky_patches(images, cat_local);
load_active_pixels!(images, patches);

ea = ElboArgs(images, deepcopy(vp_init), patches, [1]);
f_evals, max_f, max_x, nm_result =
    Celeste.DeterministicVI.maximize_f(
        Celeste.DeterministicVI.elbo, ea, verbose=true, max_iters=200);
ea.vp
mog_vp_opt = deepcopy(ea.vp);

# Fix the NaN
using Model.ids
mog_vp_opt[1][ids.k[2, 1]] = 0.00015
mog_vp_opt[1][ids.k[:, 1]] = mog_vp_opt[1][ids.k[:, 1]] / sum(mog_vp_opt[1][ids.k[:, 1]])

ea = ElboArgs(images, deepcopy(mog_vp_opt), patches, [1]);
f_evals, max_f, max_x, nm_result =
    Celeste.DeterministicVI.maximize_f(
        Celeste.DeterministicVI.elbo, ea, verbose=true, max_iters=200);
ea.vp
mog_vp_opt = deepcopy(ea.vp);


ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(vp_init), patches, [1], use_raw_psf=true);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);
f_evals, max_f, max_x, nm_result, transform =
    maximize_f_two_steps(elbo_fft_opt, ea_fft, verbose=false);
vp_opt = deepcopy(ea_fft.vp);



println("Mog, then FFT")
hcat(CelesteEDA.print_vp(mog_vp_opt[1]), CelesteEDA.print_vp(vp_opt[1]))
results[ results[:objid] .== objid, :]
stripe_82_results["celeste_df"][ results[:objid] .== objid, :]
stripe_82_results["coadd_df"][ results[:objid] .== objid, :]
stripe_82_results["celeste_err"][ results[:objid] .== objid, :]

# Look at it
n = 3;
sources = collect(1:ea_fft.S);
ea_fft.vp = deepcopy(vp_init);
start_image_fft = CelesteEDA.render_sources_fft(
    ea_fft, fsm_mat, sources, n,
    include_iota=true, include_epsilon=true, field=:E_G);
ea_fft.vp = deepcopy(vp_opt);
image_fft = CelesteEDA.render_sources_fft(
        ea_fft, fsm_mat, sources, n,
        include_iota=true, include_epsilon=true, field=:E_G);
ea.vp = deepcopy(mog_vp_opt);
image_mog = CelesteEDA.render_sources(
        ea, sources, n, include_iota=true, include_epsilon=true, field=:E_G);
raw_image = CelesteEDA.show_sources_image(ea_fft, sources, n);

PyPlot.close("all")
matshow(start_image_fft); colorbar(); title("initial fft Celeste ")
matshow(image_fft); colorbar(); title("fft Celeste ")
matshow(image_mog); colorbar(); title("mog Celeste ")
matshow(raw_image); colorbar(); title("Raw image ")
matshow(image_fft - raw_image); colorbar(); title("Final residual ")

