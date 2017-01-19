using Celeste: Model, DeterministicVI
using StaticArrays

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField
import Celeste.DeterministicVIImagePSF
import Celeste: CelesteEDA

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

const dir = "/home/rgiordan/Documents/git_repos/CelesteDev.jl/"
include(joinpath(dir, "rasterized_psf/eda_sandbox.jl"))

import Synthetic
using SampleData

using Base.Test
using Distributions

using PyPlot

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


s = 1
b = 3

images, ea, body = gen_sample_galaxy_dataset();
ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
  images, deepcopy(ea.vp), ea.patches, [s], use_raw_psf=false);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);

ea_debug = deepcopy(ea_fft);
vp_set = deepcopy(ea.vp);
vp_set[1][ids.a] = [0.5, 0.5]
ea_debug.vp = deepcopy(vp_set);
DeterministicVIImagePSF.debug_populate_fsm_mat!(ea_debug, fsm_mat, 2);

# The expectation is decoupled but the variance is not.
CelesteEDA.print_vp(fsm_mat[1].E_G[10,10].d[:, 1])
fsm_mat[1].E_G[10,10].d[star_only_ids]
fsm_mat[1].E_G[10,10].h[star_only_ids, gal_only_ids]
fsm_mat[1].var_G[10,10].h[star_only_ids, gal_only_ids]

# h = fsm_mat[1].E_G[9,10].h;
elbo_fft_opt(ea_fft);
h = ea_fft.elbo_vars.elbo.h;
eigvals, eigvecs = eig(h);
for k in 1:length(eigvals)
    if abs(eigvals[k] > 1e-8)
        println("\neigval ", k, ": ", eigvals[k])
        v = eigvecs[:, k]
        tot = sum(abs(v))
        star_tot = sum(abs(v[star_only_ids]))
        gal_tot = sum(abs(v[gal_only_ids]))
        println("Star: ",  star_tot / tot)
        println("Gal: ",  gal_tot / tot)
        println("Both: ",  (tot - star_tot - gal_tot) / tot)
    end
end







function test_two_body()
    images, ea, bodies = gen_two_body_dataset();
    s = 1

    ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
        images, deepcopy(ea.vp), ea.patches, [s], use_raw_psf=false);
    elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat, 2);
    elbo_fft_opt(ea_fft);
    f_evals, max_f, max_x, nm_result = DeterministicVI.maximize_f(elbo_fft_opt, ea_fft);
end


function verify_sample_star(vs, pos)
    @test_approx_eq_eps vs[ids.u[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.u[2]] pos[2] 0.1

    brightness_hat = exp(vs[ids.r1[1]] + 0.5 * vs[ids.r2[1]])
    @test_approx_eq_eps brightness_hat / sample_star_fluxes[3] 1. 0.01

    true_colors = log.(sample_star_fluxes[2:5] ./ sample_star_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 1]] true_colors[b] 0.2
    end

    @test vs[ids.a[2, 1]] <= 0.01
end

function verify_sample_galaxy(vs, pos)
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
    @test_approx_eq_eps brightness_hat / sample_galaxy_fluxes[3] 1. 0.03

    true_colors = log.(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 2]] true_colors[b] 0.2
    end
    
    @test vs[ids.a[2, 1]] >= 0.99
end


##################
# Galaxies

s = 1
b = 3

images, ea, body = gen_sample_galaxy_dataset();
ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(ea.vp), ea.patches, [s], use_raw_psf=false);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);
f_evals, max_f, max_x, nm_result, transform =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft; loc_width=3.0, verbose=true);
vp_opt = deepcopy(ea_fft.vp);
verify_sample_galaxy(ea_fft.vp[1], [8.5, 9.6]);
brightness_vals, brightness_squares = DeterministicVI.get_brightness(ea_fft)
brightness_vals[1][3, 2] / sample_galaxy_fluxes[3]

images_fft, images_star_fft, images_gal_fft, vp_array_fft =
    render_optimization_steps(ea_fft, fsm_mat, nm_result, transform, 1, 3);

images_resid = [ im - images[3].pixels for im in images_fft ];
show_images(images_resid)


images, ea, body = gen_sample_galaxy_dataset();
f_evals_orig, max_f_orig, max_x_orig, nm_result_orig, transform_orig =
    DeterministicVI.maximize_f(DeterministicVI.elbo_likelihood, ea;
                               loc_width=3.0, verbose=true);
vp_opt_orig = deepcopy(ea.vp);
verify_sample_galaxy(ea.vp[1], [8.5, 9.6]);

images_orig, images_star_orig, images_gal_orig, vp_array_orig =
    render_optimization_steps(ea, nm_result_orig, transform_orig, 1, 3);

images_resid_orig = [ im - images[3].pixels for im in images_orig ];

sum(images_resid_orig[length(images_resid_orig)] .^ 2)


orig_residuals = images_resid_orig[length(images_resid_orig)];
fft_residuals = images_resid[length(images_resid)];
orig_residuals[isnan(orig_residuals)] = 0
fft_residuals[isnan(fft_residuals)] = 0

# FFT gets a better ELBO but worse residuals.
nm_result.minimum / nm_result_orig.minimum
ea_fft.vp = deepcopy(vp_opt);
ea.vp = deepcopy(vp_opt_orig);
(elbo_fft_opt(ea_fft).v[] - DeterministicVI.elbo(ea).v[]) / abs(DeterministicVI.elbo(ea).v[])
mean(orig_residuals .^ 2) / mean(fft_residuals .^ 2)



######################
# Is the lognormal term the problem?

function avg_frobenius_norm(mat)
    sum(mat .* mat) / prod(size(mat))
end


ea_debug = deepcopy(ea_fft);
vp_set = deepcopy(vp_opt);
vp_set[1][ids.a] = [0, 1]
ea_debug.vp = deepcopy(vp_set);
DeterministicVIImagePSF.debug_populate_fsm_mat!(ea_debug, fsm_mat, 2);

# The expectation is decoupled but the variance is not.
# Am I crazy?  If a[1] = 0, shouldn't the derivatives with respect to the
# star ids be zero?
CelesteEDA.print_vp(fsm_mat[1].E_G[10,10].d[:, 1])
fsm_mat[1].E_G[10,10].d[star_only_ids]
fsm_mat[1].E_G[10,10].h[star_only_ids, gal_only_ids]
fsm_mat[1].var_G[10,10].h[star_only_ids, gal_only_ids]

# The contribution from the log term correction is so small...
E_G_mat = [ sf.v[] for sf in fsm_mat[b].E_G ];
var_G_mat = [ sf.v[] for sf in fsm_mat[b].var_G ];
log_term_mat = 0.5 * var_G_mat ./ (E_G_mat .^ 2);


vp_set = deepcopy(vp_opt);
vp_set[1][ids.a] = [0.5, 0.5]
ea_debug.vp = deepcopy(vp_set);
DeterministicVIImagePSF.elbo_likelihood_with_fft!(ea_debug, fsm_mat);
avg_frobenius_norm(ea_debug.elbo_vars.elbo.h[star_only_ids, star_only_ids])
avg_frobenius_norm(ea_debug.elbo_vars.elbo.h[gal_only_ids, gal_only_ids])
avg_frobenius_norm(ea_debug.elbo_vars.elbo.h[gal_only_ids, star_only_ids])


##################
# Stars


images, ea, body = gen_sample_star_dataset();
ea.vp[1][ids.a[:, 1]] = [0.8, 0.2]
ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(ea.vp), ea.patches, [1], use_raw_psf=false);
for n in 1:length(fsm_mat)
    fsm_mat[n].kernel_fun =
        x -> DeterministicVIImagePSF.cubic_kernel_with_derivatives(x, -0.75)
end
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);
f_evals, max_f, max_x, nm_result, transform =
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft, loc_width=1.0,
                               verbose=true, max_iters=500);
verify_sample_star(ea_fft.vp[1], [10.1, 12.2]);

images_fft, images_star_fft, images_gal_fft, vp_array_fft =
    render_optimization_steps(ea_fft, fsm_mat, nm_result, transform, 1, 3);


# img_array = [ images_fft[iter] - images[3].pixels for iter in 1:length(images_fft) ];
img_array = [ images_star_fft[iter] - images_gal_fft[iter] for iter in 1:length(images_fft) ];

PyPlot.close("all")
PyPlot.hold(false)
img_array = [ images_star_fft[iter] - images_gal_fft[iter] for iter in 1:length(images_fft) ];
# img_array = [ images_fft[iter] - images[3].pixels for iter in 1:length(images_fft) ];
max_pix = 1.1 * maximum([ maximum(img) for img in img_array ])
min_pix = minimum([ minimum(img) for img in img_array ])
for iter in 1:length(images_fft)
    im = img_array[iter]
    imshow(im, interpolation="nearest", vmin=min_pix, vmax=max_pix)
    if iter == 1
        colorbar()
    end
    title("Iteration " * string(iter))
    sleep(0.03)
    PyPlot.draw()
end

[vp[1][ids.a][1] for vp in vp_array_fft]
plot(1:length(nm_result.trace), [ log10(tr.value) for tr in nm_result.trace ])
plot(1:length(nm_result.trace), [ tr.metadata["delta"] for tr in nm_result.trace ])

hessians = [ tr.metadata["h(x)"] for tr in nm_result.trace ];
plot(1:length(nm_result.trace), [ condition_number(h) for h in hessians ])


function avg_frobenius_norm(mat)
    sum(mat .* mat) / prod(size(mat))
end

PyPlot.hold(true)
plot(1:length(hessians),
     [ log10(avg_frobenius_norm(h[gal_only_ids_free, star_only_ids_free])) for h in hessians ], "k.")
plot(1:length(hessians),
    [ log10(avg_frobenius_norm(h[gal_only_ids_free, gal_only_ids_free])) for h in hessians ], "b.")
plot(1:length(hessians),
    [ log10(avg_frobenius_norm(h[star_only_ids_free, star_only_ids_free])) for h in hessians ], "r.")

# Newton's method converges on a small galaxy unless we start with
# a high star probability.
ea.vp[1][ids.a[:, 1]] = [0.8, 0.2]
f_evals0, max_f0, max_x0, nm_result0, transform0 =
    DeterministicVI.maximize_f(DeterministicVI.elbo_likelihood, ea; loc_width=1.0)
verify_sample_star(ea.vp[1], [10.1, 12.2])
images_orig, images_star_orig, images_gal_orig, vp_array_orig =
    render_optimization_steps(ea_fft, nm_result0, transform0, 1, 3);




image_fft = CelesteEDA.render_source_fft(ea_fft, fsm_mat, s, b, include_epsilon=false);
image_orig = CelesteEDA.render_source(ea, s, b, include_epsilon=false);

# They are off by a constant factor.
matshow(image_fft - image_orig);
matshow(image_fft ./ image_orig);
plot(image_fft[:], image_orig[:], "k.")






