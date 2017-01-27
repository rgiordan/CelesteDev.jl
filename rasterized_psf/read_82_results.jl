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



images, ea, two_body = SampleData.gen_two_body_dataset(perturb=false);
patches = Infer.get_sky_patches(images, two_body; radius_override_pix=50);
Infer.load_active_pixels!(
    images, patches, noise_fraction=Inf, min_radius_pix=Nullable(10));
n = 3

using Base.Test
img_1 = CelesteEDA.render_sources(ea, [1], n, include_epsilon=false, include_iota=false);
img_2 = CelesteEDA.render_sources(ea, [2], n, include_epsilon=false, include_iota=false);
img_12 = CelesteEDA.render_sources(ea, [1, 2], n, include_epsilon=false, include_iota=false);
@test_approx_eq_eps(maximum(abs(img_12 - img_1 - img_2)), 0.0, 1e-12)

ea_fft = DeterministicVIImagePSF.ElboArgs(
    images, deepcopy(ea.vp), patches, collect(1:ea.S), psf_K=1);
psf_image_mat = Matrix{Float64}[
    PSF.get_psf_at_point(patches[s, b].psf) for s in 1:ea_fft.S, b in 1:ea_fft.N];
fsm_mat = DeterministicVIImagePSF.FSMSensitiveFloatMatrices[
    DeterministicVIImagePSF.FSMSensitiveFloatMatrices() for
    s in 1:ea_fft.S, b in 1:ea_fft.N];
DeterministicVIImagePSF.initialize_fsm_sf_matrices!(fsm_mat, ea_fft, psf_image_mat);

fft_img_1 = CelesteEDA.render_sources_fft(ea_fft, fsm_mat, [1], n, include_epsilon=false, include_iota=false);
fft_img_2 = CelesteEDA.render_sources_fft(ea_fft, fsm_mat, [2], n, include_epsilon=false, include_iota=false);
fft_img_12 = CelesteEDA.render_sources_fft(ea_fft, fsm_mat, [1, 2], n, include_epsilon=false, include_iota=false);
@test_approx_eq_eps(maximum(abs(fft_img_12 - fft_img_1 - fft_img_2)), 0.0, 1e-12)

full_img_12 = CelesteEDA.render_sources(ea, [1, 2], n, include_epsilon=true, include_iota=true);
full_fft_img_12 = CelesteEDA.render_sources_fft(ea_fft, fsm_mat, [1, 2], n, include_epsilon=true, include_iota=true);
orig_img_12 = CelesteEDA.show_sources_image(ea, [1, 2], n);

# Check that the two images are roughly the same, and roughly the same as the original..
@test median(abs(full_fft_img_12 - full_img_12)) / maximum(abs(full_img_12)) < 0.01
@test median(abs(full_fft_img_12 - orig_img_12)) / maximum(abs(orig_img_12)) < 0.02



file_dir = joinpath(Pkg.dir("Celeste"), "benchmark/stripe82")

res = JLD.load(joinpath(file_dir, "results_and_errors_fft_twostep_200117.jld"));

celeste_df = res["celeste_df"];
coadd_df = res["coadd_df"];
res["celeste_err"]

comb_df =
    hcat(celeste_df[:, [:objid, :is_star, :star_mag_r, :gal_mag_r]],
         coadd_df[:, [:objid, :is_star, :star_mag_r, :gal_mag_r]]);

both_star = (celeste_df[:is_star] .> 0.5) & coadd_df[:is_star]
both_gal = (celeste_df[:is_star] .< 0.5) & (!coadd_df[:is_star])
gal_err = abs(celeste_df[:gal_mag_r] .- coadd_df[:gal_mag_r])

comb_df[both_gal & (gal_err .> 5), :]

PyPlot.close("all")
plot(comb_df[both_gal, :gal_mag_r_1], comb_df[both_gal, :gal_mag_r], "k+")
plot(maximum(comb_df[both_gal, :gal_mag_r_1]), maximum(comb_df[both_gal, :gal_mag_r_1]), "ro")


# Load images
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
rcf = Celeste.SDSSIO.RunCamcolField(4263, 5,119);
images = Celeste.SDSSIO.load_field_images(rcf, datadir);
catalog = SDSSIO.read_photoobj_files([rcf], datadir, duplicate_policy=:first);
objids = [ cat.objid for cat in catalog ];

# Pick a source and optimize
objid = "1237663784734490824"
target_sources =  [ findfirst(objids, objid) ];
neighbor_map = Infer.find_neighbors(target_sources, catalog, images);

ts = 1
s = target_sources[ts]
entry = catalog[s]
neighbors = catalog[neighbor_map[ts]]

# vs_opt = DeterministicVIImagePSF.infer_source_fft_two_step(
#     images, neighbors, entry);
cat_local = vcat([entry], neighbors)
vp_init = init_sources([1], cat_local);
patches = get_sky_patches(images, cat_local);
load_active_pixels!(images, patches);

ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(vp_init), patches, [1], use_raw_psf=false);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);
f_evals, max_f, max_x, nm_result, transform =
    maximize_f_two_steps(elbo_fft_opt, ea_fft, verbose=true);
vp_opt = deepcopy(ea_fft.vp);

# Look at it
s = 1; n = 3;
ea_fft.vp = deepcopy(vp_init);
start_image_fft = CelesteEDA.render_sources_fft(
    ea_fft, fsm_mat, [ s ], n,
    include_iota=true, include_epsilon=true, field=:E_G);
ea_fft.vp = deepcopy(vp_opt);
image_fft = CelesteEDA.render_sources_fft(
        ea_fft, fsm_mat, [ s ], n,
        include_iota=true, include_epsilon=true, field=:E_G);
raw_image = CelesteEDA.show_source_image(ea_fft, s, n);


PyPlot.close("all")
matshow(start_image_fft); colorbar(); title("initial fft Celeste")
matshow(image_fft); colorbar(); title("fft Celeste")
matshow(raw_image); colorbar(); title("Raw image")
matshow(image_fft - raw_image); colorbar(); title("Final residual")




p = ea_fft.patches[1, 3];
foo = is_pixel_in_patch(608, 1005, p)

H_min, W_min, H_max, W_max = get_active_pixel_range(ea_fft.patches, sources, 3);
image = fill(0.0, (H_max - H_min + 1, W_max - W_min + 1));

image_fft_vec = [
    CelesteEDA.render_source_fft(
        ea_fft, fsm_mat, s, n,
        include_iota=false, include_epsilon=false, field=:E_G) for s in sources ];

for h in H_min:H_max, w in W_min:W_max, s in sources
    p = ea_fft.patches[s, n]
    if is_pixel_in_patch(h, w, p)
        image[h - H_min + 1, w - W_min + 1] +=
            image_fft_vec[s][h - p.bitmap_offset[1], w - p.bitmap_offset[2]]
    end
end


# This double counts!
sources = Int[1, 2]
for s in sources
    p = ea_fft.patches[s, n];
    Hs, Ws = size(p.active_pixel_bitmap);
    image_fft = CelesteEDA.render_source_fft(
            ea_fft, fsm_mat, s, n,
            include_iota=true, include_epsilon=true, field=:E_G)
    image_fft[isnan(image_fft)] = 0
    image[ p.bitmap_offset[1] + 1 - H_min + (1:size(image_fft, 1)),
           p.bitmap_offset[2] + 1 - W_min + (1:size(image_fft, 2))] += image_fft
end