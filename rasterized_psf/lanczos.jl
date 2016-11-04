

function lanczos_kernel{NumType <: Number}(x::NumType, a::Float64)
    abs(x) < a ? sinc(x) * sinc(x / a): 0.0
end


function sinc_with_derivatives{NumType <: Number}(x::NumType)
    sinc_x = sinc(x)
    sinc_x_d = (cos(x) - sinc_x) / x
    sinc_x_h = sinc_x - 2 * sinc_x_d / x
    return sinc_x, sinc_x_d, sinc_x_h
end


function lanczos_kernel_with_derivatives{NumType <: Number}(x::NumType, a::Float64)
    if abs(x) > a
        return 0, 0, 0
    end

    sinc_x, sinc_x_d, sinc_x_h = sinc_with_derivatives(x)
    sinc_xa, sinc_xa_d, sinc_xa_h = sinc_with_derivatives(x / a)

    return sinc_x * sinc_xa,
           sinc_x_d * sinc_xa + sinc_x * sinc_xa_d / a,
           sinc_x_h * sinc_xa + 2 * sinc_x_d * sinc_xa_d / a +
              sinc_x * sinc_xa_h / (a ^ 2)
end


# Interpolate the PSF to the pixel values.
function lanczos_interpolate!{NumType <: Number}(
        image::Matrix{NumType}, psf_image::Matrix{Float64},
        star_loc::Vector{NumType}, a::Float64)

    a_int = Int(a)
    h_psf_width = (size(psf_image, 1) + 1) / 2.0
    w_psf_width = (size(psf_image, 2) + 1) / 2.0
    # h, w are pixel coordinates.
    for h = 1:size(star_image, 1), w = 1:size(star_image, 2)

        # h_psf, w_psf are in psf coordinates.
        # The PSF is centered at star_loc + psf_width.
        h_psf = h - star_loc[1] + h_psf_width
        w_psf = w - star_loc[2] + w_psf_width

        # Indices into the psf matrix, i.e., integer psf coordinates.
        h_ind0, w_ind0 = Int(floor(h_psf)), Int(floor(w_psf))
        for h_ind = max(h_ind0 - a_int + 1, 1):min(h_ind0 + a_int, size(psf_image, 1))
            lh = lanczos_kernel(h_psf - h_ind, a)
            if lh != 0
                for w_ind = max(w_ind0 - a_int + 1, 1):min(w_ind0 + a_int, size(psf_image, 2))
                    lw = lanczos_kernel(w_psf - w_ind, a)
                    if lw != 0
                        star_image[h, w] += psf_image[h_ind, w_ind] * lh * lw
                    end
                end
            end
        end
    end
end

import Celeste.Model.getids
import Celeste.SensitiveFloats.multiply_sfs!
import Celeste.SensitiveFloats.add_scaled_sfs!

# Interpolate the PSF to the pixel values.
function lanczos_interpolate!{NumType <: Number, ParamType <: ParamSet}(
        image::Matrix{SensitiveFloat{ParamType, NumType}},
        psf_image::Matrix{Float64},
        object_loc::Vector{NumType},
        a::Float64,
        calculate_hessian::Bool)

    a_int = Int(a)
    h_psf_width = (size(psf_image, 1) + 1) / 2.0
    w_psf_width = (size(psf_image, 2) + 1) / 2.0

    param_ids = getids(ParamType)

    # Floats sensitive to object_loc
    h_sf = zero_sensitive_float(ParamType, NumType, 1)
    h_sf.v[1] = object_loc[1]
    h_sf.d[param_ids.u[1]] = 1

    w_sf = zero_sensitive_float(ParamType, NumType, 1)
    w_sf.v[1] = object_loc[2]
    w_sf.d[param_ids.u[2]] = 1

    kernel = zero_sensitive_float(ParamType, NumType, 1)
    kernel_h = zero_sensitive_float(ParamType, NumType, 1)

    # h, w are pixel coordinates.
    for h = 1:size(star_image, 1), w = 1:size(star_image, 2)

        # h_psf, w_psf are in psf coordinates.
        # The PSF is centered at object_loc + psf_width.
        h_psf = h - object_loc[1] + h_psf_width
        w_psf = w - object_loc[2] + w_psf_width

        # Centers of indices of the psf matrix, i.e., integer psf coordinates.
        h_ind0, w_ind0 = Int(floor(h_psf)), Int(floor(w_psf))
        for h_ind = max(h_ind0 - a_int + 1, 1):min(h_ind0 + a_int, size(psf_image, 1))
            lh_v, lh_d, lh_h = lanczos_kernel_with_derivatives(h_psf - h_ind, a)
            if lh_v != 0
                clear!(kernel_h)
                kernel_h.v[1] = lh_v
                kernel_h.d[param_ids.u[1]] = -1 * lh_d;
                kernel_h.h[param_ids.u[1], param_ids.u[1]] = -1 * lh_h;
                for w_ind = max(w_ind0 - a_int + 1, 1):min(w_ind0 + a_int, size(psf_image, 2))
                    lw_v, lw_d, lw_h = lanczos_kernel_with_derivatives(w_psf - w_ind, a)
                    if lw_v != 0
                        clear!(kernel)
                        kernel.v[1] = lw_v
                        kernel.d[param_ids.u[2]] = -1 * lw_d;
                        kernel.h[param_ids.u[2], param_ids.u[2]] = -1 * lw_h;
                        multiply_sfs!(kernel, kernel_h, calculate_hessian)
                        add_scaled_sfs!(image[h, w], kernel, psf_image[h_ind, w_ind],
                                        calculate_hessian)
                    end
                end
            end
        end
    end
end



#############
