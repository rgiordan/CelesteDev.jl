using Celeste: Infer, DeterministicVI, Model
include(joinpath(Pkg.dir("Celeste"), "benchmark/galsim/GalsimBenchmark.jl"))
galsim_filename = joinpath(Pkg.dir("Celeste"), "benchmark/galsim/output/galsim_test_images.fits")
using GalsimBenchmark:
    make_psf, read_fits, assert_counts_match_expected_flux,
    make_images, make_catalog_entries, benchmark_comparison_data


psf = make_psf();
extensions, wcs = read_fits(galsim_filename);
@assert length(extensions) % 5 == 0 # one extension per band for each test case

test_case_index = 1

first_band_index = (test_case_index - 1) * 5 + 1
header = extensions[first_band_index].header
this_test_case_name = header["CL_DESCR"]
println("Running test case '$this_test_case_name'")
iota = header["CL_IOTA"]

band_pixels = [
    extensions[index].pixels for index in first_band_index:(first_band_index + 4)
];
assert_counts_match_expected_flux(band_pixels, header, iota);
band_images = make_images(band_pixels, psf, wcs, header["CL_SKY"], iota);
catalog_entries = make_catalog_entries(header);

target_source = 1
target_entry = catalog_entries[target_source]

neighbor_map = Infer.find_neighbors([ target_source ], catalog_entries, band_images);
cat_local = vcat(target_entry, catalog_entries[neighbor_map[target_source]]);
vp = Vector{Float64}[init_source(ce) for ce in cat_local];
patches = Infer.get_sky_patches(band_images, cat_local);
Infer.load_active_pixels!(band_images, patches);

ea = ElboArgs(band_images, vp, patches, [1]);
f_evals, max_f, max_x, nm_result = DeterministicVI.maximize_f(DeterministicVI.elbo, ea);











single_infer_variational_parameters = parallel_inference(band_images, catalog_entries,
                                                         joint_infer=false);
joint_infer_variational_parameters = parallel_inference(band_images, catalog_entries,
                                                        joint_infer=true);

benchmark_data = benchmark_comparison_data(single_infer_variational_parameters,
                                           joint_infer_variational_parameters,
                                           header)

target_sources = collect(1:length(catalog_entries));

# Create the neighbor map (everyone is a neighbor of each other)
neighbor_map = Infer.find_neighbors(target_sources, catalog_entries, band_images)

cat_local = vcat(entry, neighbors)
vp = Vector{Float64}[init_source(ce) for ce in cat_local]
patches = Infer.get_sky_patches(images, cat_local)
Infer.load_active_pixels!(images, patches)

ea = ElboArgs(images, vp, patches, [1])
f_evals, max_f, max_x, nm_result = maximize_f(elbo, ea)
vp[1]
