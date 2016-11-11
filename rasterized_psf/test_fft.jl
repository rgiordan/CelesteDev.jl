using Celeste: Model, DeterministicVI
using StaticArrays

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

include("celeste_tools/celeste_tools.jl")
const dir = "/home/rgiordan/Documents/git_repos/CelesteDev.jl/"

# include(joinpath(dir, "rasterized_psf/lanczos.jl"))
include(joinpath(dir, "rasterized_psf/elbo_pixelated_psf.jl"))

using ELBOPixelatedPSF
import Synthetic
using SampleData

using Base.Test
using Distributions

using DeterministicVI.CanonicalParams
using SensitiveFloats.zero_sensitive_float
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
using SensitiveFloats.clear!


fsms = ELBOPixelatedPSF.FSMSensitiveFloatMatrices();
psf_image_mat = Matrix{Matrix{Float64}}(1, 1);
psf_image = zeros(3, 3);
psf_image[2, 2] = 0.5;
psf_image[2, 1] = psf_image[1, 2] = 0.25;
psf_image_mat[1, 1] = psf_image;
ELBOPixelatedPSF.initialize_fsm_sf_matrices_band!(
    fsms, 1, 1, 1, 1, 3, 3, psf_image_mat)

sf = zero_sensitive_float(GalaxyPosParams, Float64)
sf.v[1] = 3;
sf.d[:, 1] = rand(size(sf.d, 1))
h = rand(size(sf.h))
sf.h = h * h';
fsms.fs1m_image_padded[2, 2] = deepcopy(sf);
ELBOPixelatedPSF.convolve_sensitive_float_matrix!(
    fsms.fs1m_image_padded, fsms.psf_fft_vec[1], fsms.fs1m_conv_padded);
h_indices = (1:3) + fsms.pad_pix_h
w_indices = (1:3) + fsms.pad_pix_w
conv_image =
    Float64[ sf.v[1] for sf in fsms.fs1m_conv_padded ][h_indices, w_indices];
@test_approx_eq(sf.v[1] * psf_image, conv_image)

for ind in 1:size(sf.d, 1)
    conv_image =
        Float64[ sf.d[ind] for sf in fsms.fs1m_conv_padded ][h_indices, w_indices];
    @test_approx_eq(sf.d[ind] * psf_image, conv_image)
end

for ind1 in 1:size(sf.h, 1), ind2 in 1:size(sf.h, 2)
    conv_image =
        Float64[ sf.h[ind1, ind2] for sf in fsms.fs1m_conv_padded ][h_indices, w_indices];
    @test_approx_eq(sf.h[ind1, ind2] * psf_image, conv_image)
end
