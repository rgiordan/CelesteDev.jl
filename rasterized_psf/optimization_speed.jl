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


gal_only_ids_free = vcat(ids_free.e_dev, ids_free.e_axis, ids_free.e_angle, ids_free.e_scale,
                    ids_free.r1[2], ids_free.r2[2],
                    ids_free.c1[:, 2][:], ids_free.c2[:, 2][:],
                    ids_free.k[:, 2][:])

star_only_ids_free = vcat(ids_free.r1[1], ids_free.r2[1],
                         ids_free.c1[:, 1][:], ids_free.c2[:, 1][:],
                         ids_free.k[:, 1][:])


gal_only_ids = vcat(ids.e_dev, ids.e_axis, ids.e_angle, ids.e_scale,
                     ids.r1[2], ids.r2[2],
                     ids.c1[:, 2][:], ids.c2[:, 2][:],
                     ids.k[:, 2][:])

star_only_ids = vcat(ids.r1[1], ids.r2[1],
                     ids.c1[:, 1][:], ids.c2[:, 1][:],
                      ids.k[:, 1][:])


function fit_elbo_two_steps(callback, ea::ElboArgs, verbose=true)
    ea.vp[1][ids.a] = [1, 0]
    kept_ids = vcat(ids_free.u, star_only_ids_free)
    f_evals_star, max_f_star, max_x_star, nm_result_star =
        Celeste.DeterministicVI.maximize_f(
            callback, ea,
            omitted_ids=setdiff(1:length(ids_free), kept_ids),
            verbose=verbose, max_iters=200);

    # ea.vp[1][ids.a] = [0, 1]
    f_evals_gal, max_f_gal, max_x_gal, nm_result_gal =
        0, 0, 0, 0
        # Celeste.DeterministicVI.maximize_f(
        #     callback, ea,
        #     omitted_ids=setdiff(1:length(ids_free), gal_only_ids_free),
        #     verbose=verbose, max_iters=200);

    ea.vp[1][ids.a] = [0.8, 0.2]
    f_evals_both, max_f_both, max_x_both, nm_result_both =
            Celeste.DeterministicVI.maximize_f(
            callback, ea,
            verbose=verbose, max_iters=200);
    vp_opt_2step = deepcopy(ea.vp);
    
    f_evals_star, max_f_star, max_x_star, nm_result_star,
    f_evals_gal, max_f_gal, max_x_gal, nm_result_gal,
    f_evals_both, max_f_both, max_x_both, nm_result_both,
    deepcopy(ea.vp)
end



#########################
# Galaxy test

using Celeste: Model, Transform, SensitiveFloats, DeterministicVIImagePSF
include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

import Synthetic
using SampleData

using Base.Test

function verify_sample_star(vs, pos)
    @test vs[ids.a[2, 1]] <= 0.01

    @test_approx_eq_eps vs[ids.u[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.u[2]] pos[2] 0.1

    brightness_hat = exp(vs[ids.r1[1]] + 0.5 * vs[ids.r2[1]])
    @test_approx_eq_eps brightness_hat / sample_star_fluxes[3] 1. 0.01

    true_colors = log.(sample_star_fluxes[2:5] ./ sample_star_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 1]] true_colors[b] 0.2
    end
end

function verify_sample_galaxy(vs, pos)
    @test vs[ids.a[2, 1]] >= 0.99

    @test_approx_eq_eps vs[ids.u[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.u[2]] pos[2] 0.1

    @test_approx_eq_eps vs[ids.e_axis] .7 0.05
    @test_approx_eq_eps vs[ids.e_dev] 0.1 0.08
    @test_approx_eq_eps vs[ids.e_scale] 4. 0.2

    phi_hat = vs[ids.e_angle]
    phi_hat -= floor(phi_hat / pi) * pi
    five_deg = 5 * pi/180
    @test_approx_eq_eps phi_hat pi/4 five_deg

    brightness_hat = exp(vs[ids.r1[2]] + 0.5 * vs[ids.r2[2]])
    @test_approx_eq_eps brightness_hat / sample_galaxy_fluxes[3] 1. 0.01

    true_colors = log.(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 2]] true_colors[b] 0.2
    end
end

images, ea, body = gen_sample_galaxy_dataset();
ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(ea.vp), ea.patches, [1], use_raw_psf=false);
elbo_fft_opt =
    DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);
f_evals, max_f, max_x, nm_result, transform = 
    DeterministicVI.maximize_f(
        elbo_fft_opt, ea_fft; loc_width=3.0, verbose=true, ftol_rel=0, max_iters=200);

# DeterministicVI.maximize_f_two_steps(
#     elbo_fft_opt, ea_fft; loc_width=1.0, verbose=true);

# TODO: Currently failing since it misses the brighness by 3%, which is
# greater than the 1% permitted by the test.  However, the ELBO of the
# FFT optimum is lower than that of the MOG optimum.
verify_sample_galaxy(ea_fft.vp[1], [8.5, 9.6])



############

const GALSIM_BENCHMARK_DIR = joinpath(Pkg.dir("Celeste"), "benchmark", "galsim")
const LATEST_FITS_FILENAME_HOLDER = joinpath(
    GALSIM_BENCHMARK_DIR, "latest_filenames", "latest_galsim_benchmarks.txt")

latest_fits_filename = Celeste.GalsimBenchmark.get_latest_fits_filename("galsim_benchmarks")
extensions, wcs = read_fits(joinpath(GALSIM_BENCHMARK_DIR, "output", latest_fits_filename));

@assert length(extensions) % 5 == 0 # one extension per band for each test case

print_test_names(extensions)


for test_id in 1:(Int(length(extensions) / 5))
    catalog, target_sources, neighbor_map, images, header = load_test_case(test_id);
    println("\n", header["CLDESCR"])
    ts = target_sources[1]

    s = target_sources[ts]
    entry = catalog[s]

    # could subset images to images_local here too.
    neighbors = catalog[neighbor_map[ts]]

    cat_local = vcat([entry], neighbors);
    vp_init = init_sources([1], cat_local);
    patches = Celeste.Infer.get_sky_patches(images, cat_local);
    Celeste.Infer.load_active_pixels!(images, patches);

    # Fit in two steps
    ea = ElboArgs(images, deepcopy(vp_init), patches, [1]);

    f_evals_both, max_f_both, max_x_both, nm_result_both = 
        Celeste.DeterministicVI.maximize_f_two_steps(Celeste.DeterministicVI.elbo, ea, verbose=false)

    ea = ElboArgs(images, deepcopy(vp_init), patches, [1]);
    f_evals, max_f, max_x, nm_result =
        Celeste.DeterministicVI.maximize_f(
            Celeste.DeterministicVI.elbo, ea,
            verbose=false, max_iters=200);
    vp_opt = deepcopy(ea.vp);

    println(
        "   ELBO ratio: ", max_f / max_f_both,
        "   eval ratio: ", f_evals / f_evals_both,
        "\n\n")
end



for test_id in 1:(Int(length(extensions) / 5))
    catalog, target_sources, neighbor_map, images, header = load_test_case(test_id);
    println("\n", header["CLDESCR"])
    ts = target_sources[1]

    s = target_sources[ts]
    entry = catalog[s]

    # could subset images to images_local here too.
    neighbors = catalog[neighbor_map[ts]]

    cat_local = vcat([entry], neighbors);
    vp_init = init_sources([1], cat_local);
    patches = Celeste.Infer.get_sky_patches(images, cat_local);
    Celeste.Infer.load_active_pixels!(images, patches);

    # Fit in two steps
    ea_fft, fsm_mat = initialize_fft_elbo_parameters(
        images, deepcopy(vp_init), patches, [1], use_raw_psf=false);
    elbo_fft_opt = get_fft_elbo_function(ea_fft, fsm_mat);
    f_evals_star, max_f_star, max_x_star, nm_result_star,
    f_evals_gal, max_f_gal, max_x_gal, nm_result_gal,
    f_evals_both, max_f_both, max_x_both, nm_result_both,
    vp_opt_2step = fit_elbo_two_steps(elbo_fft_opt, ea_fft, false);
    f_evals_combined = f_evals_star + f_evals_both + f_evals_gal

    ea_fft, fsm_mat = initialize_fft_elbo_parameters(
        images, deepcopy(vp_init), patches, [1], use_raw_psf=false);
    elbo_fft_opt = get_fft_elbo_function(ea_fft, fsm_mat);
    f_evals, max_f, max_x, nm_result =
        Celeste.DeterministicVI.maximize_f(
            elbo_fft_opt, ea_fft, verbose=false, max_iters=200);
    vp_opt = deepcopy(ea_fft.vp);

    comparison_dataframe = Celeste.GalsimBenchmark.get_ground_truth_dataframe(header, 1);
    star_galaxy_index = header["CLTYP001"] == "star" ? 1 : 2
    comparison_dataframe[:two_step] =
    Celeste.GalsimBenchmark.inferred_values(star_galaxy_index, vp_opt_2step[1]);
    comparison_dataframe[:one_step] =
        Celeste.GalsimBenchmark.inferred_values(star_galaxy_index, vp_opt[1]);

    for col in [ "two_step", "one_step" ]
        comparison_dataframe[Symbol(col * "_perr")] =
            (comparison_dataframe[Symbol(col)] .-
             comparison_dataframe[:ground_truth]) ./
            comparison_dataframe[:ground_truth]
    end

    println(comparison_dataframe)
    println("  star prop: ", f_evals_star / f_evals_combined,
            "  gal prop: ", f_evals_gal / f_evals_combined)
    println("Combined evals: ", f_evals_combined)
    println(
        "   ELBO ratio (lower is better for joint): ", max_f / max_f_both,
        "   eval ratio: (higher is better for joint): ", f_evals / f_evals_combined,
        "\n\n")
end







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


