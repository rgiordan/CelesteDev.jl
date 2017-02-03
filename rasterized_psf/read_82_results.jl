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
stripe_82_results = JLD.load(joinpath(stripe82_dir, "results_and_errors_full_box_fft_two_step-004263-5-0119.jld"))

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
bad_ids = [
8647474692482203916,
8647474692482203881,
8647474692482203819,
8647474692482203817,
8647474692482138341,
8647474692482203737,
];

coadd_objid = bad_ids[2]
objid = convert(String, stripe_82_results["celeste_df"][stripe_82_results["coadd_df"][:objid] .== coadd_objid, :objid][1])

#objid = "1237663784734491701"
results[ results[:objid] .== objid, :]
vcat(
    stripe_82_results["celeste_df"][ results[:objid] .== objid, :],
    stripe_82_results["coadd_df"][ results[:objid] .== objid, :],
    stripe_82_results["celeste_err"][ results[:objid] .== objid, :])

for id in bad_ids
    objid = convert(String, stripe_82_results["celeste_df"][stripe_82_results["coadd_df"][:objid] .== id, :objid][1])
    println(stripe_82_results["celeste_err"][ results[:objid] .== objid, :])
end


for id in bad_ids
    objid = convert(String, stripe_82_results["celeste_df"][stripe_82_results["coadd_df"][:objid] .== id, :objid][1])
    println(stripe_82_results["celeste_df"][ results[:objid] .== objid, :])
end


objid = "1237663784734490980"

# objid = "1237663784734490824" # This is bad
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

# Fit MOG
ea = ElboArgs(images, deepcopy(vp_init), patches, [1]);
f_evals, max_f, max_x, nm_result =
    Celeste.DeterministicVI.maximize_f(
        Celeste.DeterministicVI.elbo, ea, verbose=true, max_iters=200, ftol_rel=1e-10);
ea.vp
mog_vp_opt = deepcopy(ea.vp);

# Fit FFT
ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(vp_init), patches, [1], use_raw_psf=true);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);
f_evals, max_f, max_x, nm_result_fft, transform =
    maximize_f_two_steps(elbo_fft_opt, ea_fft, verbose=true, ftol_rel=1e-10, max_iters=200);
vp_opt = deepcopy(ea_fft.vp);

view_s = 1
hcat(CelesteEDA.print_vp(vp_opt[view_s]),
     CelesteEDA.print_vp(mog_vp_opt[view_s]),
     CelesteEDA.print_vp(vp_init[view_s]))

b = 3
rendered_images, rendered_images_star, rendered_images_gal, vp_array = render_optimization_steps(
     ea_fft, fsm_mat, nm_result_fft, transform, [1], b);

show_images(rendered_images_gal)

println("Mog, then FFT, then init")
hcat(CelesteEDA.print_vp(mog_vp_opt[1]), CelesteEDA.print_vp(vp_opt[1]), CelesteEDA.print_vp(vp_init[1]))

# Look at it
n = 3;
# sources = collect(1:ea_fft.S);
sources = [1]
ea_fft.vp = deepcopy(vp_init);
start_image_fft = CelesteEDA.render_sources_fft(
    ea_fft, fsm_mat, sources, n,
    include_iota=true, include_epsilon=true, field=:E_G);
ea_fft.vp = deepcopy(vp_opt);
# ea_fft.vp[1][ids.a] = [0, 1]
image_fft = CelesteEDA.render_sources_fft(
        ea_fft, fsm_mat, sources, n,
        include_iota=true, include_epsilon=true, field=:E_G);
ea.vp = deepcopy(mog_vp_opt);
# ea.vp[1][ids.a] = [0, 1]
image_mog = CelesteEDA.render_sources(
        ea, sources, n, include_iota=true, include_epsilon=true, field=:E_G);
ea.vp = deepcopy(vp_init);
start_image_mog = CelesteEDA.render_sources(
        ea, sources, n, include_iota=true, include_epsilon=true, field=:E_G);
raw_image = CelesteEDA.show_sources_image(ea_fft, sources, n);
# raw_image[1:12, :] = NaN


PyPlot.close("all")
# matshow(start_image_fft); colorbar(); title("initial fft Celeste ")
# matshow(start_image_fft); colorbar(); title("initial mog Celeste ")
PyPlot.close("all")
matshow(image_fft); colorbar(); title("fft Celeste ")
matshow(image_mog); colorbar(); title("mog Celeste ")
matshow(raw_image); colorbar(); title("Raw image ")
matshow(image_fft - raw_image); colorbar(); title("fft residual ")
matshow(image_mog - raw_image); colorbar(); title("mog residual ")
matshow(image_mog - image_fft); colorbar(); title("fft - mog ")

