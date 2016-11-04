


function lanczos_kernel{NumType <: Number}(x::NumType, a::Float64)
    abs(x) < a ? sinc(x) * sinc(x / a): 0.0
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

# matshow(star_image[90:110, 90:110])

# star_loc is h, w
star_loc = Float64[25.5, 30.5]
psf_image = psf_image_vec[3];
star_image = fill(0.0, 60, 60);
lanczos_interpolate!(star_image, psf_image, star_loc, 3.0)

matshow(star_image)
plot(star_loc[2] - 1, star_loc[1] - 1, "ro")



#############
