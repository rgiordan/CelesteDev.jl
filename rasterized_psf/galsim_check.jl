using Celeste: Infer, DeterministicVI, Model

current_path = joinpath(ENV["GIT_REPO_LOC"], "CelesteDev.jl/rasterized_psf")
include(joinpath(Pkg.dir("Celeste"), "benchmark/galsim/GalsimBenchmark.jl"))
include(joinpath(current_path, "galsim_lib.jl"))

using GalsimBenchmark:
    make_psf, read_fits, assert_counts_match_expected_flux,
    make_images, make_catalog_entries, benchmark_comparison_data

using Celeste.DeterministicVIImagePSF
using Celeste.CelesteEDA
using PyPlot
using DataFrames

############

galsim_filename_orig =
    joinpath(Pkg.dir("Celeste"),
             "benchmark/galsim/output/galsim_test_images.fits")
galsim_filename = joinpath(current_path, "galsim_test_images_more_psf.fits")

# extensions, wcs = read_fits(galsim_filename_orig);
extensions, wcs = read_fits(galsim_filename);
@assert length(extensions) % 5 == 0 # one extension per band for each test case

# PSF sizes are in this order:
# 0.2, 0.5, 1, 1.5, and 2.
# There are 24 test cases per psf size.
print_test_names(extensions)

psf_size_ind = 3
images, patches, vp, header = load_test_case(10 + (psf_size_ind - 1) * 24);
# images, patches, vp, header = load_test_case(1);
vp[1][ids.a] = [0.8, 0.2]

# matshow(images[1].pixels); colorbar()
ea = ElboArgs(images, deepcopy(vp), patches, [1]);
ea_fft, fsm_vec = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(vp), ea.patches, [1], use_raw_psf=false);


#############################
# Check the initial points, which should be very similar.

b = 3
fft_rendered = CelesteEDA.render_source_fft(ea_fft, fsm_vec, 1, b, include_epsilon=false);
orig_rendered = CelesteEDA.render_source(ea, 1, b, include_epsilon=false);
raw_image = CelesteEDA.show_source_image(ea, 1, b);
fft_rendered[isnan(fft_rendered)] = 0;
orig_rendered[isnan(orig_rendered)] = 0;

# Note that galaxies are very close but stars are not.
using(GLM)
ols_df = DataFrame(orig=orig_rendered[:], fft=fft_rendered[:]);
glm(orig ~ fft, ols_df, Normal(), IdentityLink())
# plot(orig_rendered[:], fft_rendered[:], "k.");  plot(maximum(fft_rendered[:]), maximum(fft_rendered[:]), "ro")

################
# Optimize

f_evals, max_f, max_x, nm_result =
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea, verbose=true);
vp_opt = deepcopy(ea.vp[1]);

elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_vec, 2);
f_evals_fft, max_f_fft, max_x_fft, nm_result_fft =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft, verbose=true);
vp_opt_fft = deepcopy(ea_fft.vp[1]);

#############
# Print

# Check the TR sizes.
[ tr.metadata["delta"] for tr in nm_result_fft.trace ]
[ tr.metadata["delta"] for tr in nm_result.trace ]

star_galaxy_index = header["CL_TYPE1"] == "star" ? 1 : 2
comp_df = GalsimBenchmark.get_expected_dataframe(header);
comp_df[:orig] = GalsimBenchmark.actual_values(ids, star_galaxy_index, vp_opt);
comp_df[:fft] = GalsimBenchmark.actual_values(ids, star_galaxy_index, vp_opt_fft);
comp_df[:orig_diff] = (comp_df[:orig] - comp_df[:expected]) ./ comp_df[:expected];
comp_df[:fft_diff] = (comp_df[:fft] - comp_df[:expected]) ./ comp_df[:expected];
print(comp_df)

using DataFrames
df = DataFrame(ids=ids_names, vp_orig=vp_opt, vp_fft=vp_opt_fft,
               pdiff=(vp_opt_fft - vp_opt) ./ vp_opt);
print(df)

#############
# Check that each is finding its respective optimum

ea_fft_check = deepcopy(ea_fft);
ea_fft_check.vp[1] = deepcopy(vp_opt);
ea_fft.vp[1] = vp_opt_fft;
println("FFT improvement:")
elbo_fft_opt(ea_fft).v[] - elbo_fft_opt(ea_fft_check).v[]
elbo_fft_opt(ea_fft).v[]

ea_check = deepcopy(ea);
ea_check.vp[1] = deepcopy(vp_opt_fft);
ea.vp[1] = deepcopy(vp_opt);
println("Ordinary improvement:")
DeterministicVI.elbo(ea).v[] - DeterministicVI.elbo(ea_check).v[]
DeterministicVI.elbo(ea).v[]


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


#################
# Check the PSFs

PyPlot.close("all")
s = 1
for b in 1:5
    orig_psf = Celeste.PSF.get_psf_at_point(ea.patches[s, b].psf);
    # matshow(fsm_vec[b].psf_vec[s] - orig_psf); colorbar(); title(b)
    matshow(fsm_vec[b].psf_vec[s]); colorbar(); title(b)
end

