

function lanczos_kernel{NumType <: Number}(x::NumType, a::Float64)
    abs(x) < a ? sinc(x) * sinc(x / a): zero(NumType)
end


function sinc_with_derivatives{NumType <: Number}(x::NumType)
    x_pi = pi * x
    sinc_x = sinc(x)
    sinc_x_d = (cos(x_pi) - sinc_x) / x
    sinc_x_h = -pi * (pi * sinc_x + 2 * sinc_x_d / x_pi)
    return sinc_x, sinc_x_d, sinc_x_h
end


# A function without checking a.  Factored out for testing with ForwardDiff..
function lanczos_kernel_with_derivatives_nocheck{NumType <: Number}(
    x::NumType, a::Float64)
    sinc_x, sinc_x_d, sinc_x_h = sinc_with_derivatives(x)
    sinc_xa, sinc_xa_d, sinc_xa_h = sinc_with_derivatives(x / a)

    return sinc_x * sinc_xa,
           sinc_x_d * sinc_xa + sinc_x * sinc_xa_d / a,
           sinc_x_h * sinc_xa + 2 * sinc_x_d * sinc_xa_d / a +
              sinc_x * sinc_xa_h / (a ^ 2)
end


function lanczos_kernel_with_derivatives{NumType <: Number}(x::NumType, a::Float64)
    if abs(x) > a
        return 0, 0, 0
    end
    return lanczos_kernel_with_derivatives_nocheck(x, a)
end


# # Interpolate the PSF to the pixel values.  Stores the result in image.
# function lanczos_interpolate!{NumType <: Number}(
#         image::Matrix{NumType}, psf_image::Matrix{Float64},
#         star_loc::Vector{NumType}, lanczos_a::Int64)
#
#     h_psf_width = (size(psf_image, 1) + 1) / 2.0
#     w_psf_width = (size(psf_image, 2) + 1) / 2.0
#     # h, w are pixel coordinates.
#     for h = 1:size(image, 1), w = 1:size(image, 2)
#
#         # h_psf, w_psf are in psf coordinates.
#         # The PSF is centered at star_loc + psf_width.
#         h_psf = h - star_loc[1] + h_psf_width
#         w_psf = w - star_loc[2] + w_psf_width
#
#         # Indices into the psf matrix, i.e., integer psf coordinates.
#         h_ind0, w_ind0 = Int(floor(h_psf)), Int(floor(w_psf))
#         h_min = max(h_ind0 - lanczos_a + 1, 1)
#         h_max = min(h_ind0 + lanczos_a, size(psf_image, 1))
#         for h_ind = h_min:h_max
#             lh = lanczos_kernel(h_psf - h_ind, Float64(lanczos_a))
#             if lh != 0
#                 w_min = max(w_ind0 - lanczos_a + 1, 1)
#                 w_max = min(w_ind0 + lanczos_a, size(psf_image, 2))
#                 for w_ind = w_min:w_max
#                     lw = lanczos_kernel(w_psf - w_ind, Float64(lanczos_a))
#                     if lw != 0
#                         image[h, w] += psf_image[h_ind, w_ind] * lh * lw
#                     end
#                 end
#             end
#         end
#     end
# end

using SensitiveFloats.SensitiveFloat
using SensitiveFloats.zero_sensitive_float
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
using SensitiveFloats.clear!
using StaticArrays


import Celeste.Model.getids
import Celeste.SensitiveFloats.multiply_sfs!
import Celeste.SensitiveFloats.add_scaled_sfs!

# Interpolate the PSF to the pixel values.
function lanczos_interpolate!{NumType <: Number, ParamType <: ParamSet}(
        image::Matrix{SensitiveFloat{ParamType, NumType}},
        psf_image::Matrix{Float64},
        object_loc::Vector{NumType},
        a_int::Int,
        wcs_jacobian::Matrix{Float64},
        calculate_hessian::Bool)

    a = Float64(a_int)
    h_psf_width = (size(psf_image, 1) + 1) / 2.0
    w_psf_width = (size(psf_image, 2) + 1) / 2.0

    param_ids = getids(ParamType)

    # These are sensitive floats representing derviatives of the Lanczos kernel.
    kernel = zero_sensitive_float(ParamType, NumType, 1)
    kernel_h = zero_sensitive_float(ParamType, NumType, 1)

    # Pre-compute terms for transforming derivatives to world coordinates.
    k_h_grad = wcs_jacobian' * Float64[1, 0]
    k_h_hess = wcs_jacobian' * Float64[1 0; 0 0] * wcs_jacobian

    k_w_grad = wcs_jacobian' * Float64[0, 1]
    k_w_hess = wcs_jacobian' * Float64[0 0; 0 1] * wcs_jacobian

    # h, w are pixel coordinates.
    for h = 1:size(image, 1), w = 1:size(image, 2)

        # h_psf, w_psf are in psf coordinates.
        # The PSF is centered at object_loc + psf_width.
        h_psf = h - object_loc[1] + h_psf_width
        w_psf = w - object_loc[2] + w_psf_width

        # Centers of indices of the psf matrix, i.e., integer psf coordinates.
        h_ind0, w_ind0 = Int(floor(h_psf)), Int(floor(w_psf))
        h_lower = max(h_ind0 - a_int + 1, 1)
        h_upper = min(h_ind0 + a_int, size(psf_image, 1))
        for h_ind = (h_lower:h_upper)
            lh_v, lh_d, lh_h = lanczos_kernel_with_derivatives(h_psf - h_ind, a)
            if lh_v != 0
                clear!(kernel_h)
                kernel_h.v[1] = lh_v
                # This is -1 * wcs_jacobian' * [lh_d, 0]
                # and -1 * wcs_jacobian' * [lh_h 0; 0 0] * wcs_jacobian
                kernel_h.d[param_ids.u] = -1 * k_h_grad * lh_d
                kernel_h.h[param_ids.u, param_ids.u] = -1 * k_h_hess * lh_h;
                w_lower = max(w_ind0 - a_int + 1, 1)
                w_upper = min(w_ind0 + a_int, size(psf_image, 2))
                for w_ind = (w_lower:w_upper)
                    lw_v, lw_d, lw_h =
                        lanczos_kernel_with_derivatives(w_psf - w_ind, a)
                    if lw_v != 0
                        clear!(kernel)
                        kernel.v[1] = lw_v
                        # This is -1 * wcs_jacobian' * [0, lw_d]
                        # and -1 * wcs_jacobian' * [0 0; 0 lw_h] * wcs_jacobian
                        kernel.d[param_ids.u] = -1 * k_w_grad * lw_d;
                        kernel.h[param_ids.u, param_ids.u] = -1 * k_w_hess * lw_h;
                        multiply_sfs!(kernel, kernel_h, calculate_hessian)
                        add_scaled_sfs!(
                            image[h, w], kernel, psf_image[h_ind, w_ind],
                            calculate_hessian)
                        # println(image[h, w].v[1])
                    end
                end
            end
        end
    end
end


import Celeste.Model.GalaxyCacheComponent

# Get a GalaxyCacheComponent with no PSF
function GalaxyCacheComponent{NumType <: Number}(
    e_dev_dir::Float64, e_dev_i::NumType,
    gc::GalaxyComponent, u::Vector{NumType},
    e_axis::NumType, e_angle::NumType, e_scale::NumType,
    calculate_derivs::Bool, calculate_hessian::Bool)

    # Declare in advance to save memory allocation.
    const empty_sig_sf =
    GalaxySigmaDerivs(Array(NumType, 0, 0), Array(NumType, 0, 0, 0))

    XiXi = get_bvn_cov(e_axis, e_angle, e_scale)
    var_s = gc.nuBar * XiXi

    # d siginv / dsigma is only necessary for the Hessian.
    bmc = BvnComponent{NumType}(
        SVector{2, NumType}(u), var_s, gc.etaBar,
        calculate_siginv_deriv=calculate_derivs && calculate_hessian)

    if calculate_derivs
        sig_sf = GalaxySigmaDerivs(
            e_angle, e_axis, e_scale, XiXi, calculate_tensor=calculate_hessian)
        sig_sf.j .*= gc.nuBar
        if calculate_hessian
            # The tensor is only needed for the Hessian.
            sig_sf.t .*= gc.nuBar
        end
    else
        sig_sf = empty_sig_sf
    end

    GalaxyCacheComponent(e_dev_dir, e_dev_i, bmc, sig_sf)
end


import Celeste.Model.lidx
import Celeste.Model.GalaxySigmaDerivs
import Celeste.Model.get_bvn_cov
import Celeste.Model.galaxy_prototypes
import Celeste.Model.linear_world_to_pix

"""
No PSF.
"""
function load_gal_bvn_mixtures{NumType <: Number}(
                    S::Int64,
                    patches::Matrix{SkyPatch},
                    source_params::Vector{Vector{NumType}},
                    active_sources::Vector{Int},
                    b::Int;
                    calculate_derivs::Bool=true,
                    calculate_hessian::Bool=true)

    # To maintain consistency with the rest of the code, use a 4d
    # array.  The first dimension was previously the PSF component.
    gal_mcs = Array(GalaxyCacheComponent{NumType}, 1, 8, 2, S)

    # TODO: do not keep any derviative information if the sources are not in
    # active_sources.
    for s in 1:S
        sp  = source_params[s]
        world_loc = sp[lidx.u]
        m_pos = linear_world_to_pix(
            patches[s, b].wcs_jacobian,
            patches[s, b].center,
            patches[s, b].pixel_center, world_loc)

        for i = 1:2 # i indexes dev vs exp galaxy types.
            e_dev_dir = (i == 1) ? 1. : -1.
            e_dev_i = (i == 1) ? sp[lidx.e_dev] : 1. - sp[lidx.e_dev]

            # Galaxies of type 1 have 8 components, and type 2 have 6 components.
            for j in 1:[8,6][i]
                gal_mcs[1, j, i, s] = GalaxyCacheComponent(
                    e_dev_dir, e_dev_i, galaxy_prototypes[i][j], m_pos,
                    sp[lidx.e_axis], sp[lidx.e_angle], sp[lidx.e_scale],
                    calculate_derivs && (s in active_sources),
                    calculate_hessian)
            end
        end
    end

    gal_mcs
end


#############
