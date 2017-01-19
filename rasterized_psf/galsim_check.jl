using Celeste: Infer, DeterministicVI, Model

using Celeste.GalsimBenchmark:
    make_psf, make_catalog, read_fits, assert_counts_match_expected_flux,
    make_images, make_catalog

current_path = joinpath(ENV["GIT_REPO_LOC"], "CelesteDev.jl/rasterized_psf")
# include(joinpath(Pkg.dir("Celeste"), "src/GalsimBenchmark.jl"))
include(joinpath(current_path, "galsim_lib.jl"))

using Celeste.ParallelRun: one_node_single_infer

using Celeste.DeterministicVIImagePSF
using Celeste.CelesteEDA
using PyPlot
using DataFrames




function condition_number(mat)
    ev = eigvals(mat)
    return maximum(abs(ev)) / minimum(abs(ev))
end

############

const GALSIM_BENCHMARK_DIR = joinpath(Pkg.dir("Celeste"), "benchmark", "galsim")
const LATEST_FITS_FILENAME_HOLDER = joinpath(
    GALSIM_BENCHMARK_DIR, "latest_filenames", "latest_galsim_benchmarks.txt")

latest_fits_filename = Celeste.GalsimBenchmark.get_latest_fits_filename("galsim_benchmarks")
extensions, wcs = read_fits(joinpath(GALSIM_BENCHMARK_DIR, "output", latest_fits_filename));

@assert length(extensions) % 5 == 0 # one extension per band for each test case

catalog, target_sources, neighbor_map, images, header = load_test_case(2);
header["CLDESCR"]
ts = target_sources[1]

s = target_sources[ts]
entry = catalog[s]

# could subset images to images_local here too.
neighbors = catalog[neighbor_map[ts]]

cat_local = vcat([entry], neighbors);
vp_init = init_sources([1], cat_local);
patches = Celeste.Infer.get_sky_patches(images, cat_local);
Celeste.Infer.load_active_pixels!(images, patches);


# Fit the original
ea = ElboArgs(images, deepcopy(vp_init), patches, [1]);
f_evals, max_f, max_x, nm_result =
    Celeste.DeterministicVI.maximize_f(
        Celeste.DeterministicVI.elbo, ea, verbose=true, max_iters=200);
vp_opt = deepcopy(ea.vp);


# Fit the FFT.
# Initialize 
ea_fft, fsm_mat = initialize_fft_elbo_parameters(
    images, deepcopy(vp_init), patches, [1], use_raw_psf=false);
# Initialize at the other other optimum
# ea_fft, fsm_mat = initialize_fft_elbo_parameters(
#     images, deepcopy(vp_opt), patches, [1], use_raw_psf=false);
elbo_fft_opt = get_fft_elbo_function(ea_fft, fsm_mat);

f_evals, max_f, max_x, nm_result_fft, transform =
    Celeste.DeterministicVI.maximize_f(
        elbo_fft_opt, ea_fft, verbose=true, max_iters=200);
vp_opt_fft = deepcopy(ea_fft.vp);

#############################
# Compare to ground truth

comparison_dataframe = Celeste.GalsimBenchmark.get_ground_truth_dataframe(header);
star_galaxy_index = header["CLTYP001"] == "star" ? 1 : 2
comparison_dataframe[:fft] =
Celeste.GalsimBenchmark.inferred_values(star_galaxy_index, ea_fft.vp[1]);
comparison_dataframe[:mog] =
    Celeste.GalsimBenchmark.inferred_values(star_galaxy_index, ea.vp[1]);

# exp(ea_fft.vp[1][ids.r1[star_galaxy_index]] + 0.5 * ea_fft.vp[1][ids.r2[star_galaxy_index]])
# exp(ea.vp[1][ids.r1[star_galaxy_index]] + 0.5 * ea.vp[1][ids.r2[star_galaxy_index]])

for col in [ "fft", "mog" ]
    comparison_dataframe[Symbol(col * "_perr")] =
        (comparison_dataframe[Symbol(col)] .-
         comparison_dataframe[:ground_truth]) ./
        comparison_dataframe[:ground_truth]
end

comparison_dataframe

using Celeste.CelesteEDA.print_vp
vp_opt_df = print_vp(vp_opt[1]);
vp_opt_df[:vp_fft] = vp_opt_fft[1];
vp_opt_df


#######################
# NM diagnostics

elbo = [ tr.value for tr in nm_result.trace ];
x = reduce(hcat, [ tr.metadata["x"] for tr in nm_result.trace ]);
gr = reduce(hcat, [ tr.metadata["g(x)"] for tr in nm_result.trace ]);
h = nm_result.trace[end].metadata["h(x)"];
delta = [ tr.metadata["delta"] for tr in nm_result.trace ];

#############
# Check that each is finding its respective optimum

ea_fft_check = deepcopy(ea_fft);
ea_fft_check.vp[1] = deepcopy(vp_opt[1]);
ea_fft.vp[1] = deepcopy(vp_opt_fft[1]);
println("FFT improvement (should be positive):")
elbo_fft_opt(ea_fft).v[] - elbo_fft_opt(ea_fft_check).v[]
elbo_fft_opt(ea_fft).v[]

ea_check = deepcopy(ea);
ea_check.vp[1] = deepcopy(vp_opt_fft[1]);
ea.vp[1] = deepcopy(vp_opt[1]);
println("Ordinary improvement (should be positive):")
DeterministicVI.elbo(ea).v[] - DeterministicVI.elbo(ea_check).v[]
DeterministicVI.elbo(ea).v[]

println("FFT over ordinary:")
(elbo_fft_opt(ea_fft).v[] - DeterministicVI.elbo(ea).v[]) /
    abs(DeterministicVI.elbo(ea).v[])


#############################
# Check the same value, which should be very similar.

import Celeste.CelesteEDA.print_vp

b = 3
same_vp = deepcopy(vp_opt_fft);
ea_fft.vp = deepcopy(same_vp); ea.vp = deepcopy(same_vp);

fft_rendered = CelesteEDA.render_source_fft(
    ea_fft, fsm_mat, 1, b, include_epsilon=true, include_iota=true);
orig_rendered = CelesteEDA.render_source(
    ea, 1, b, include_epsilon=true, include_iota=true);
raw_image = CelesteEDA.show_source_image(ea, 1, b);
fft_rendered[isnan(fft_rendered)] = 0;
orig_rendered[isnan(orig_rendered)] = 0;

# matshow(raw_image); colorbar(); title("raw")
# matshow(fft_rendered); colorbar(); title("fft")
# matshow(orig_rendered); colorbar(); title("orig")

PyPlot.close("all")
plot(orig_rendered[:], fft_rendered[:], "k.");
plot(maximum(fft_rendered[:]), maximum(fft_rendered[:]), "ro")
PyPlot.close("all")
plot(raw_image[:], fft_rendered[:], "k."); 
plot(maximum(fft_rendered[:]), maximum(fft_rendered[:]), "ro")
PyPlot.close("all")
plot(raw_image[:], orig_rendered[:], "k.");
plot(maximum(fft_rendered[:]), maximum(fft_rendered[:]), "ro")

# Note that galaxies are very close but stars are not.
using(GLM)
ols_df = DataFrame(orig=orig_rendered[:], fft=fft_rendered[:]);
glm(orig ~ fft, ols_df, Normal(), IdentityLink())
if false
    plot(orig_rendered[:], fft_rendered[:], "k.");
    plot(maximum(fft_rendered[:]), maximum(fft_rendered[:]), "ro")
end
if false
    plot(log(orig_rendered[:]), log(fft_rendered[:]), "k."); 
    plot(log(orig_rendered[:]), log(orig_rendered[:]), "r."); 
end



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

ea_fft.vp[s] = deepcopy(vp_opt_fft);
ea.vp[s] = deepcopy(vp_opt);

orig_pix_loc = Celeste.CelesteEDA.source_pixel_location(ea, s, b)
fft_pix_loc = Celeste.CelesteEDA.source_pixel_location(ea_fft, s, b)

fft_rendered = CelesteEDA.render_source_fft(ea_fft, fsm_mat, s, b);
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
    # matshow(fsm_mat[b].psf_vec[s] - orig_psf); colorbar(); title(b)
    matshow(fsm_mat[b].psf_vec[s]); colorbar(); title(b)
end



###########################
# Use the unit test to check out the Lanczos interpolation

import Celeste: SensitiveFloats.zero_sensitive_float_array
using Base.Test

psf_image = zeros(Float64, 5, 5);
psf_image[3, 3] = 0.5
psf_image[2, 3] = psf_image[3, 2] = psf_image[4, 3] = psf_image[3, 4] = 0.125

wcs_jacobian = Float64[0.9 0.2; 0.1 0.8]
world_loc = Float64[5.1, 5.2]
lanczos_width = 3

image_size = (11, 11)
function lanczos_interpolate_loc{T <: Number}(
    world_loc::Vector{T}, calculate_gradient::Bool)
    local image = zero_sensitive_float_array(T, length(StarPosParams), 1,
                                                        image_size...)
    local pixel_loc = Celeste.Model.linear_world_to_pix(
        wcs_jacobian, Float64[0., 0.], Float64[1.0, 0.5], world_loc)
    DeterministicVIImagePSF.lanczos_interpolate!(
        image, psf_image, pixel_loc, lanczos_width, wcs_jacobian,
        calculate_gradient, calculate_gradient)
    return image
end

for eps1 in 0:0.2:2, eps2 in 0:0.2:2
    world_loc = Float64[5 + eps1, 5 + eps2]
    image = lanczos_interpolate_loc(world_loc, true)
    println(sum([ sf.v[] for sf in image ]))
end


