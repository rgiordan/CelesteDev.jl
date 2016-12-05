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

import Synthetic
using SampleData

using Base.Test
using Distributions

using PyPlot


images, ea, bodies = gen_two_body_dataset();
s = 1

ea_fft, fsm_vec = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(ea.vp), ea.patches, [s], use_raw_psf=false);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_vec, 2);

elbo_fft_opt(ea_fft);
f_evals, max_f, max_x, nm_result = DeterministicVI.maximize_f(elbo_fft_opt, ea_fft);


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


images, ea, body = gen_sample_star_dataset();
ea.vp[1][ids.a[:, 1]] = [0.8, 0.2]
ea_fft, fsm_vec = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(ea.vp), ea.patches, [s], use_raw_psf=false);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_vec, 1);
DeterministicVI.maximize_f(elbo_fft_opt, ea_fft; loc_width=1.0);
verify_sample_star(ea_fft.vp[1], [10.1, 12.2]);


images, ea, body = gen_sample_galaxy_dataset();
ea_fft, fsm_vec = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(ea.vp), ea.patches, [s], use_raw_psf=false);
elbo_fft_opt = DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_vec, 2);
DeterministicVI.maximize_f(elbo_fft_opt, ea_fft; loc_width=1.0);
verify_sample_galaxy(ea_fft.vp[1], [8.5, 9.6]);














