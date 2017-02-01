#!/usr/bin/env julia

using Celeste: Model, DeterministicVI, DeterministicVIImagePSF

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

import Synthetic
using SampleData

using Base.Test
using Distributions


using Celeste: DeterministicVI, SensitiveFloats

using Base.Test
using Distributions
using DerivativeTestUtils


function verify_sample_galaxy(vs, pos)
    @test vs[ids.a[2, 1]] >= 0.99

    @test isapprox(vs[ids.u[1]], pos[1], atol=0.1)
    @test isapprox(vs[ids.u[2]], pos[2], atol=0.1)

    @test isapprox(vs[ids.e_axis] , 0.7, atol=0.05)
    @test isapprox(vs[ids.e_dev]  , 0.1, atol=0.08)
    @test isapprox(vs[ids.e_scale], 4.0, atol=0.2)

    phi_hat = vs[ids.e_angle]
    phi_hat -= floor(phi_hat / pi) * pi
    five_deg = 5 * pi/180
    @test isapprox(phi_hat, pi/4, atol=five_deg)

    brightness_hat = exp(vs[ids.r1[2]] + 0.5 * vs[ids.r2[2]])
    @test isapprox(brightness_hat / sample_galaxy_fluxes[3], 1.0, atol=0.01)

    true_colors = log.(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
    for b in 1:4
        @test isapprox(vs[ids.c1[b, 2]], true_colors[b], atol=0.2)
    end
end


function verify_sample_star(vs, pos)
    @test vs[ids.a[2, 1]] <= 0.01

    @test isapprox(vs[ids.u[1]], pos[1], atol=0.1)
    @test isapprox(vs[ids.u[2]], pos[2], atol=0.1)

    brightness_hat = exp(vs[ids.r1[1]] + 0.5 * vs[ids.r2[1]])
    @test isapprox(brightness_hat / sample_star_fluxes[3], 1.0, atol=0.01)

    true_colors = log.(sample_star_fluxes[2:5] ./ sample_star_fluxes[1:4])
    for b in 1:4
        @test isapprox(vs[ids.c1[b, 1]], true_colors[b], atol=0.2)
    end
end


println("Testing star fft optimization.")

# clear allocations in case julia is running with --track-allocations=user

max_time = Sys.time()
images, ea, body = gen_sample_star_dataset();
ea.vp[1][ids.a[:, 1]] = [0.8, 0.2];
ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(ea.vp), ea.patches, [1], use_raw_psf=false);
elbo_fft_opt =
    DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);
    
Profile.clear_malloc_data()
Profile.init(delay=0.01)
@profile DeterministicVI.maximize_f(elbo_fft_opt, ea_fft; loc_width=1.0);
# Profile.print(format=:flat, sortedby=:count)
max_time = Sys.time() - max_time

max_time = Sys.time()
DeterministicVI.maximize_f(DeterministicVI.elbo, ea; loc_width=1.0);
max_time = Sys.time() - max_time



verify_sample_star(ea_fft.vp[1], [10.1, 12.2]);



println("Testing galaxy fft optimization.")

images, ea, body = gen_sample_galaxy_dataset();
ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
    images, deepcopy(ea.vp), ea.patches, [1], use_raw_psf=false);
elbo_fft_opt =
    DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat);
DeterministicVI.maximize_f_two_steps(elbo_fft_opt, ea_fft; loc_width=1.0);

