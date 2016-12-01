using Celeste: Infer, DeterministicVI, Model
include(joinpath(Pkg.dir("Celeste"), "benchmark/galsim/GalsimBenchmark.jl"))
galsim_filename = joinpath(Pkg.dir("Celeste"), "benchmark/galsim/output/galsim_test_images.fits")
using GalsimBenchmark:
    make_psf, read_fits, assert_counts_match_expected_flux,
    make_images, make_catalog_entries, benchmark_comparison_data


psf = make_psf();
extensions, wcs = read_fits(galsim_filename);
@assert length(extensions) % 5 == 0 # one extension per band for each test case

for test_case_index in 1:div(length(extensions), 5)
    first_band_index = (test_case_index - 1) * 5 + 1
    println(test_case_index, " ",
            extensions[first_band_index].header["CL_DESCR"])
end

test_case_index = 10
first_band_index = (test_case_index - 1) * 5 + 1
header = extensions[first_band_index].header
this_test_case_name = header["CL_DESCR"]
println("Running test case '$this_test_case_name'")
iota = header["CL_IOTA"]

band_pixels = [
    extensions[index].pixels for index in first_band_index:(first_band_index + 4)
];
assert_counts_match_expected_flux(band_pixels, header, iota);
images = make_images(band_pixels, psf, wcs, header["CL_SKY"], iota);
catalog_entries = make_catalog_entries(header);

target_source = 1
target_entry = catalog_entries[target_source]

neighbor_map = Infer.find_neighbors([ target_source ], catalog_entries, images);
cat_local = vcat(target_entry, catalog_entries[neighbor_map[target_source]]);
vp = Vector{Float64}[init_source(ce) for ce in cat_local];
patches = Infer.get_sky_patches(images, cat_local);
Infer.load_active_pixels!(images, patches);

ea = ElboArgs(images, vp, patches, [1]);
f_evals, max_f, max_x, nm_result =
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea, verbose=false);
vp_opt = deepcopy(ea.vp[1]);


################
# FFT version

# using StaticArrays
# import Celeste.Model
# import Celeste.Model.SkyPatch
using Celeste.PSF
using Celeste.DeterministicVIImagePSF
using PyPlot

ea_fft = ElboArgs(images, deepcopy(vp), patches, [1], psf_K=1);
Celeste.Infer.load_active_pixels!(images, patches; exclude_nan=false);

psf_image_mat = Matrix{Float64}[
    PSF.get_psf_at_point(ea.patches[s, b].psf) for s in 1:ea.S, b in 1:ea.N];
fsm_vec = DeterministicVIImagePSF.FSMSensitiveFloatMatrices[
    DeterministicVIImagePSF.FSMSensitiveFloatMatrices() for b in 1:ea_fft.N];
DeterministicVIImagePSF.initialize_fsm_sf_matrices!(fsm_vec, ea_fft, psf_image_mat);

function elbo_fft_opt{NumType <: Number}(
                    ea::ElboArgs{NumType};
                    calculate_derivs=true,
                    calculate_hessian=true)
    @assert ea.psf_K == 1
    DeterministicVIImagePSF.elbo_likelihood_with_fft!(ea, 2, fsm_vec);
    DeterministicVI.subtract_kl!(ea, ea.elbo_vars.elbo)
    return deepcopy(ea.elbo_vars.elbo)
end

# For compilation
#include(joinpath(dir, "rasterized_psf/elbo_pixelated_psf.jl"))
DeterministicVIImagePSF.elbo_likelihood_with_fft!(ea_fft, 2, fsm_vec);

ea_fft.elbo_vars.elbo

elbo_fft = elbo_fft_opt(ea_fft);

ea_fft = ElboArgs(images, deepcopy(vp), patches, [1], psf_K=1);
Celeste.Infer.load_active_pixels!(images, patches; exclude_nan=false);
f_evals_fft, max_f_fft, max_x_fft, nm_result_fft =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft,
                               verbose=true, max_iters=200);
vp_opt_fft = deepcopy(ea_fft.vp[1])

#############
# Print

star_galaxy_index = header["CL_TYPE1"] == "star" ? 1 : 2
comp_df = GalsimBenchmark.get_expected_dataframe(header);
comp_df[:orig] = GalsimBenchmark.actual_values(ids, star_galaxy_index, vp_opt)
comp_df[:fft] = GalsimBenchmark.actual_values(ids, star_galaxy_index, vp_opt_fft)
comp_df[:orig_diff] = (comp_df[:orig] - comp_df[:expected]) ./ comp_df[:expected]
comp_df[:fft_diff] = (comp_df[:fft] - comp_df[:expected]) ./ comp_df[:expected]

using DataFrames
df = DataFrame(ids=ids_names, vp_fft=vp_opt_fft, vp_orig=vp_opt)


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
