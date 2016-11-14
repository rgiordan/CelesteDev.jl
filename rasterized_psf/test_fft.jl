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


function test_convolve_sensitive_float_matrix()
    # Use the FSMSensitiveFloatMatrices because it initializes all the
    # sizes for us automatically.
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
end


# Test lanczos_interpolate!
using ForwardDiff
using ELBOPixelatedPSF

psf_image = zeros(Float64, 5, 5);
psf_image[3, 3] = 0.5
psf_image[2, 3] = psf_image[3, 2] = psf_image[4, 3] = psf_image[3, 4] = 0.125

# function lanczos_interpolate_loc{T <: Number}(star_loc::Vector{T})
#     image = zeros(T, 11, 11);
#     ELBOPixelatedPSF.lanczos_interpolate!(image, psf_image, star_loc, 2)
#     return image
# end
#
# image = lanczos_interpolate_loc(Float64[5, 5])
# @test_approx_eq psf_image image[3:7, 3:7]
#
# image_pixel = sub2ind(size(image), 5, 5)
# function lanczos_interpolate_loc_pixel{T <: Number}(star_loc::Vector{T})
#     image = lanczos_interpolate_loc(star_loc)
#     return image[image_pixel]
# end
#
# star_loc = Float64[5, 5]
# lanczos_interpolate_loc_pixel(star_loc)
#
import Base.floor
function floor(x::ForwardDiff.Dual{2,Float64})
    return floor(x.value)
end

import Base.sinc
function sinc(x::ForwardDiff.Dual{2,Float64})
    # Note that this won't work with x = 0.
    return sin(x * pi) / (x * pi)
end

function sinc(x::ForwardDiff.Dual{1,Float64})
    # Note that this won't work with x = 0.
    return sin(x * pi) / (x * pi)
end

function sinc(x::ForwardDiff.Dual{1,ForwardDiff.Dual{1,Float64}})
    # Note that this won't work with x = 0.
    return sin(x * pi) / (x * pi)
end

function floor(x::ForwardDiff.Dual{1,Float64})
    return floor(x.value)
end

function floor(x::ForwardDiff.Dual{1,ForwardDiff.Dual{1,Float64}})
    return floor(x.value)
end


x = ForwardDiff.Dual{1, Float64}(1.5)
sinc(x).value
sinc(x.value)

sin(1.5 * pi) / (1.5 * pi)
sinc(1.5)

#
# star_loc = Float64[5.1, 5.2]
# grad = ForwardDiff.gradient(lanczos_interpolate_loc_pixel, star_loc)
# sf = lanczos_interpolate_loc_pixel(star_loc)
#
# sf.d[ids.u, 1]


T = Float64
star_loc = Float64[5.1, 5.2]
lanczos_width = 2.0
function lanczos_interpolate_loc{T <: Number}(star_loc::Vector{T})
    image = zero_sensitive_float_array(StarPosParams, T, 1, 11, 11);
    ELBOPixelatedPSF.lanczos_interpolate!(image, psf_image, star_loc, 2)
    return image
end

function lanczos_kernel_fd{NumType <: Number}(x_vec::Vector{NumType})
    x = x_vec[1]
    v, d, h = ELBOPixelatedPSF.lanczos_kernel_with_derivatives_nocheck(
        x, lanczos_width)
    return v
end

x = 0.7
fd_v = lanczos_kernel_fd([x])
fd_d = ForwardDiff.gradient(lanczos_kernel_fd, Float64[ x ])[1]
fd_h = ForwardDiff.hessian(lanczos_kernel_fd, Float64[ x ])[1, 1]

v, d, h = ELBOPixelatedPSF.lanczos_kernel_with_derivatives_nocheck(x, lanczos_width)

@test_approx_eq fd_v v
@test_approx_eq fd_d d
@test_approx_eq fd_h h




function sinc_with_derivatives_fd{T <: Number}(x::Vector{T})
    v, d, h = ELBOPixelatedPSF.sinc_with_derivatives(x[1])
    return v
end

x = 0.7
fd_v = sinc_with_derivatives_fd(Float64[ x ])
fd_d = ForwardDiff.gradient(sinc_with_derivatives_fd, Float64[ x ])[1]
fd_h = ForwardDiff.hessian(sinc_with_derivatives_fd, Float64[ x ])[1, 1]

v, d, h = ELBOPixelatedPSF.sinc_with_derivatives(x)

@test_approx_eq sinc(x) v
@test_approx_eq fd_v v
@test_approx_eq fd_d d
@test_approx_eq fd_h h



######################
