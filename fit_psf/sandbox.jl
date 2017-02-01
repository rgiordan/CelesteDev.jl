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
coords = reduce(hcat,
    [ hcat(WCS.pix_to_world(images[n].wcs, Float64[1., 1.]),
           WCS.pix_to_world(images[n].wcs, Float64[images[n].H, 1]),
           WCS.pix_to_world(images[n].wcs, Float64[1., images[n].W]),
           WCS.pix_to_world(images[n].wcs, Float64[images[n].H, images[n].W])) for n in 1:length(images) ] )

# RA:
minimum(coords[1, :])
maximum(coords[1, :])

# DEC:
minimum(coords[2, :])
maximum(coords[2, :])

(0.44161597899601157, 0.6064338079096621)
(0.4103441862324931, 0.6357206703337571)