using Celeste: Infer, DeterministicVI, Model
include(joinpath(Pkg.dir("Celeste"), "benchmark/galsim/GalsimBenchmark.jl"))
include("galsim_lib.jl")

using GalsimBenchmark:
    make_psf, read_fits, assert_counts_match_expected_flux,
    make_images, make_catalog_entries, benchmark_comparison_data

using Celeste.DeterministicVIImagePSF
using PyPlot

############

galsim_filename =
    joinpath(Pkg.dir("Celeste"), "benchmark/galsim/output/galsim_test_images.fits")

extensions, wcs = read_fits(galsim_filename);
@assert length(extensions) % 5 == 0 # one extension per band for each test case

print_test_names(extensions)

# Small PSF
images, patches, vp, header = load_test_case(22);

# matshow(images[1].pixels); colorbar()
ea = ElboArgs(images, vp, patches, [1]);
# ea.patches[1,1].psf[1].tauBar

f_evals, max_f, max_x, nm_result =
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea, verbose=false);
vp_opt = deepcopy(ea.vp[1]);


################
# FFT version

ea_fft, fsm_vec = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(ea.vp), ea.patches, [1], use_raw_psf=false);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_vec, 2);
# matshow(fsm_vec[3].psf_vec[1])

elbo_fft_opt(ea_fft);
f_evals, max_f, max_x, nm_result = DeterministicVI.maximize_f(elbo_fft_opt, ea_fft);
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
df = DataFrame(ids=ids_names, vp_fft=vp_opt_fft, vp_orig=vp_opt,
               pdiff=(vp_opt_fft - vp_opt) ./ vp_opt)

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
