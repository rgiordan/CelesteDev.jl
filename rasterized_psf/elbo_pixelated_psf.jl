module ELBOPixelatedPSF

using DeterministicVI.ElboArgs
using DeterministicVI.ElboIntermediateVariables
using DeterministicVI.BvnComponent
using DeterministicVI.GalaxyCacheComponent
using DeterministicVI.SourceBrightness
using DeterministicVI.StarPosParams
using DeterministicVI.GalaxyPosParams
using DeterministicVI.CanonicalParams
using DeterministicVI.GalaxyComponent
using DeterministicVI.SkyPatch

using Celeste.Model.populate_gal_fsm!
using Celeste.Model.ParamSet
using Celeste.Model.linear_world_to_pix

using SensitiveFloats.zero_sensitive_float
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
using SensitiveFloats.clear!

using DeterministicVI.load_source_brightnesses
using DeterministicVI.add_sources_sf!
using DeterministicVI.add_elbo_log_term!
using DeterministicVI.add_scaled_sfs!
using DeterministicVI.accumulate_source_pixel_brightness!

using StaticArrays

typealias GMatrix Matrix{SensitiveFloat{CanonicalParams, Float64}}
typealias fs0mMatrix Matrix{SensitiveFloat{StarPosParams, Float64}}
typealias fs1mMatrix Matrix{SensitiveFloat{GalaxyPosParams, Float64}}


import Celeste.Model.GalaxyCacheComponent
import Celeste.DeterministicVI.BvnComponent

# To be included in Celeste.


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
        calculate_derivs && calculate_hessian)

    if calculate_derivs
        sig_sf = GalaxySigmaDerivs(
            e_angle, e_axis, e_scale, XiXi, calculate_hessian)
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




const dir = "/home/rgiordan/Documents/git_repos/CelesteDev.jl/"
include(joinpath(dir, "rasterized_psf/sensitive_float_fft.jl"))
include(joinpath(dir, "rasterized_psf/lanczos.jl"))

type FSMSensitiveFloatMatrices
    # The lower corner of the image (in terms of index values)
    h_lower::Int
    w_lower::Int

    fs1m_image::fs1mMatrix;
    fs1m_image_padded::fs1mMatrix;
    fs1m_conv::fs1mMatrix;
    fs1m_conv_padded::fs1mMatrix;

    # We convolve the star fs0m directly using Lanczos interpolation.
    fs0m_conv::fs0mMatrix;

    E_G::GMatrix
    var_G::GMatrix

    # A vector of psfs, one for each source.
    psf_fft_vec::Vector{Matrix{Complex{Float64}}}
    psf_vec::Vector{Matrix{Float64}}

    # The amount of padding introduced by the convolution on one side of the
    # image (the total pixels added in each dimension are twice this)
    pad_pix_h::Int
    pad_pix_w::Int

    FSMSensitiveFloatMatrices() = begin
        new(1, 1,
            fs1mMatrix(), fs1mMatrix(), fs1mMatrix(), fs1mMatrix(), fs0mMatrix(),
            GMatrix(), GMatrix(),
            Vector{Matrix{Complex{Float64}}}(),
            Vector{Matrix{Float64}}(),
            0, 0)
    end
end


function initialize_fsm_sf_matrices_band!(
    fsms::FSMSensitiveFloatMatrices,
    b::Int,
    num_active_sources::Int,
    h_lower::Int, w_lower::Int,
    h_upper::Int, w_upper::Int,
    psf_image_mat::Matrix{Matrix{Float64}})

    # Require that every PSF is the same size, since we are only
    # keeping one padded matrix for the whole band.
    psf_sizes = Set([ size(im) for im in psf_image_mat[:, b] ])
    @assert length(psf_sizes) == 1
    psf_size = pop!(psf_sizes)

    fsms.h_lower = h_lower
    fsms.w_lower = w_lower

    h_width = h_upper - h_lower + 1
    w_width = w_upper - w_lower + 1

    # An fsm value is only sensitive to one source's parameters.
    fsms.fs1m_image = zero_sensitive_float_array(
        GalaxyPosParams, Float64, 1, h_width, w_width);
    fsms.fs1m_conv = zero_sensitive_float_array(
        GalaxyPosParams, Float64, 1, h_width, w_width);

    fsms.fs0m_conv = zero_sensitive_float_array(
        StarPosParams, Float64, 1, h_width, w_width);

    # The amount of padding introduced by the convolution
    (fft_size1, fft_size2) =
        (h_width + psf_size[1] - 1, w_width + psf_size[2] - 1)
    # Make sure that the PSF has an odd dimension.
    @assert psf_size[1] % 2 == 1
    @assert psf_size[2] % 2 == 1
    fsms.pad_pix_h = Integer((psf_size[1] - 1) / 2)
    fsms.pad_pix_w = Integer((psf_size[2] - 1) / 2)

    fsms.fs1m_image_padded = zero_sensitive_float_array(
        GalaxyPosParams, Float64, 1, fft_size1, fft_size2);
    fsms.fs1m_conv_padded = zero_sensitive_float_array(
        GalaxyPosParams, Float64, 1, fft_size1, fft_size2);

    # Brightness images
    fsms.E_G = zero_sensitive_float_array(
        CanonicalParams, Float64, num_active_sources, h_width, w_width);
    fsms.var_G = zero_sensitive_float_array(
        CanonicalParams, Float64, num_active_sources, h_width, w_width);

    # Store the psf image and its FFT.
    S = size(psf_image_mat, 1)
    fsms.psf_fft_vec = Array(Matrix{Complex{Float64}}, S)
    fsms.psf_vec = Array(Matrix{Float64}, S)
    for s in 1:size(psf_image_mat, 1)
        fsms.psf_fft_vec[s] =
            zeros(Complex{Float64}, fft_size1, fft_size2);
        fsms.psf_fft_vec[s][1:psf_size[1], 1:psf_size[2]] =
            psf_image_mat[s, b];
        fft!(fsms.psf_fft_vec[s]);
        fsms.psf_vec[s] = psf_image_mat[s, b]
    end
end


function initialize_fsm_sf_matrices!(
    fsm_vec::Vector{FSMSensitiveFloatMatrices},
    ea::ElboArgs{Float64},
    psf_image_mat::Matrix{Matrix{Float64}})

    # Get the extreme active pixels in each band.
    h_lower_vec = Int[typemax(Int) for b in ea.images ]
    w_lower_vec = Int[typemax(Int) for b in ea.images ]
    h_upper_vec = Int[0 for b in ea.images ]
    w_upper_vec = Int[0 for b in ea.images ]

    for s in ea.active_sources, n in 1:ea.N
        p = ea.patches[s, n]
        h1 = p.bitmap_corner[1] + 1
        w1 = p.bitmap_corner[2] + 1
        h2 = h1 + size(p.active_pixel_bitmap, 1) - 1
        w2 = w1 + size(p.active_pixel_bitmap, 2) - 1
        h_lower_vec[n] = min(h_lower_vec[n], h1)
        h_upper_vec[n] = max(h_upper_vec[n], h2)
        w_lower_vec[n] = min(w_lower_vec[n], w1)
        w_upper_vec[n] = max(w_upper_vec[n], w2)
    end

    num_active_sources = length(ea.active_sources)

    # Pre-allocate arrays.
    for b in 1:ea.N
        initialize_fsm_sf_matrices_band!(
            fsm_vec[b], b, num_active_sources,
            h_lower_vec[b], w_lower_vec[b],
            h_upper_vec[b], w_upper_vec[b],
            psf_image_mat)
    end
end


# This is just for debugging.
function populate_fsm_vec!(
    ea::ElboArgs,
    fsm_vec::Array{FSMSensitiveFloatMatrices},
    lanczos_width::Int)

    sbs = load_source_brightnesses(ea,
        calculate_derivs=ea.elbo_vars.calculate_derivs,
        calculate_hessian=ea.elbo_vars.calculate_hessian);

    gal_mcs_vec = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N);
    for b=1:ea.N
        gal_mcs_vec[b] = load_gal_bvn_mixtures(
                ea.S, ea.patches, ea.vp, ea.active_sources, b,
                calculate_derivs=ea.elbo_vars.calculate_derivs,
                calculate_hessian=ea.elbo_vars.calculate_hessian);
    end

    for s in 1:ea.S
    end

    for b=1:ea.N
        for s in 1:ea.S
            populate_star_fsm_image!(
                ea, s, b, fsm_vec[b].psf_vec[s], fsm_vec[b].fs0m_conv,
                fsm_vec[b].h_lower, fsm_vec[b].w_lower, lanczos_width)
            populate_gal_fsm_image!(
                ea, s, b, gal_mcs_vec[b], fsm_vec[b])
            populate_source_band_brightness!(
                ea, s, b, fsm_vec[b], sbs[s])
        end
    end
end



"""
Convolve a populated set of SensitiveFloat matrices in fsms with the PSF
and store them in the matching fs*m_conv SensitiveFloat matrices.
"""
function convolve_fs1m_image!(fsms::FSMSensitiveFloatMatrices, s::Int)
    for h in 1:size(fsms.fs1m_image, 1), w in 1:size(fsms.fs1m_image, 2)
        fsms.fs1m_image_padded[h, w] = fsms.fs1m_image[h, w];
    end

    convolve_sensitive_float_matrix!(
        fsms.fs1m_image_padded, fsms.psf_fft_vec[s], fsms.fs1m_conv_padded);

    for h in 1:size(fsms.fs1m_image, 1), w in 1:size(fsms.fs1m_image, 2)
        fsms.fs1m_conv[h, w] =
            fsms.fs1m_conv_padded[fsms.pad_pix_h + h, fsms.pad_pix_w + w];
    end

    # Set return type
    return true
end


function clear_fs1m!(fsms::FSMSensitiveFloatMatrices)
    for sf in fsms.fs1m_image clear!(sf) end
    for sf in fsms.fs1m_image_padded clear!(sf) end
    for sf in fsms.fs1m_conv clear!(sf) end
    for sf in fsms.fs1m_conv_padded clear!(sf) end
end


function clear_brightness!(fsms::FSMSensitiveFloatMatrices)
    for sf in fsms.E_G clear!(sf) end
    for sf in fsms.var_G clear!(sf) end
end


"""
Populate the fs1m shape matrices and convolve with the PSF for a given
source and band.  Assumes that fsms.psf_fft has already been set.

The result is the sources shapes, convolved with the PSF, stored in fsms.fs1m_conv.

TODO: pass in derivative flags
"""
function populate_gal_fsm_image!(
            ea::ElboArgs{Float64},
            s::Int,
            n::Int,
            gal_mcs::Array{GalaxyCacheComponent{Float64}, 4},
            fsms::FSMSensitiveFloatMatrices)
    clear_fs1m!(fsms)
    is_active_source = s in ea.active_sources
    p = ea.patches[s, n]
    H_patch, W_patch = size(p.active_pixel_bitmap)
    for w_patch in 1:W_patch, h_patch in 1:H_patch
        h_image = h_patch + p.bitmap_corner[1]
        w_image = w_patch + p.bitmap_corner[2]

        h_fsm = h_image - fsms.h_lower + 1
        w_fsm = w_image - fsms.w_lower + 1

        x = SVector{2, Float64}([h_image, w_image])
        populate_gal_fsm!(ea.elbo_vars.bvn_derivs,
                          fsms.fs1m_image[h_fsm, w_fsm],
                          ea.elbo_vars.calculate_derivs,
                          ea.elbo_vars.calculate_hessian,
                          s, x, is_active_source, Inf,
                          p.wcs_jacobian,
                          gal_mcs)
    end
    convolve_fs1m_image!(fsms, s);
end


"""
Populate the fs1m shape matrices and convolve with the PSF for a given
source and band.  Assumes that fsms.psf_fft has already been set.

The result is the sources shapes, convolved with the PSF, stored in fsms.fs1m_conv.

TODO: pass in derivative flags
"""
function populate_star_fsm_image!(
            ea::ElboArgs{Float64},
            s::Int,
            b::Int,
            psf_image::Matrix{Float64},
            fs0m_conv::fs0mMatrix,
            h_lower::Int, w_lower::Int,
            lanczos_width::Int)
    for sf in fs0m_conv clear!(sf) end
    # The pixel location of the star.
    star_loc_pix =
        linear_world_to_pix(ea.patches[s, b].wcs_jacobian,
                            ea.patches[s, b].center,
                            ea.patches[s, b].pixel_center,
                            ea.vp[s][lidx.u]) -
        Float64[ h_lower - 1, w_lower - 1]
    lanczos_interpolate!(fs0m_conv, psf_image, star_loc_pix, lanczos_width,
                         ea.patches[s, b].wcs_jacobian,
                         ea.elbo_vars.calculate_derivs,
                         ea.elbo_vars.calculate_hessian);
end


"""
Updates fsms.E_G and fsms.var_G in place with the contributions from this
source in this band.
"""
function populate_source_band_brightness!(
    ea::ElboArgs{Float64},
    s::Int,
    n::Int,
    fsms::FSMSensitiveFloatMatrices,
    sb::SourceBrightness{Float64})

    is_active_source = s in ea.active_sources
    calculate_hessian =
        ea.elbo_vars.calculate_hessian && ea.elbo_vars.calculate_derivs &&
        is_active_source

    p = ea.patches[s, n]
    H_patch, W_patch = size(p.active_pixel_bitmap)
    for w_patch in 1:W_patch, h_patch in 1:H_patch
        h_fsm = h_patch + p.bitmap_corner[1] - fsms.h_lower + 1
        w_fsm = w_patch + p.bitmap_corner[2] - fsms.w_lower + 1
        accumulate_source_pixel_brightness!(
                            ea.elbo_vars,
                            ea,
                            fsms.E_G[h_fsm, w_fsm],
                            fsms.var_G[h_fsm, w_fsm],
                            fsms.fs0m_conv[h_fsm, w_fsm],
                            fsms.fs1m_conv[h_fsm, w_fsm],
                            sb, n, s, is_active_source)
    end
end


"""
Uses the values in fsms to add the contribution from this band to the ELBO.
"""
function accumulate_band_in_elbo!(
    ea::ElboArgs{Float64},
    fsms::FSMSensitiveFloatMatrices,
    sbs::Vector{SourceBrightness{Float64}},
    gal_mcs::Array{GalaxyCacheComponent{Float64}, 4},
    n::Int, lanczos_width::Int)

    clear_brightness!(fsms)

    for s in 1:ea.S
        populate_star_fsm_image!(
            ea, s, n, fsms.psf_vec[s], fsms.fs0m_conv,
            fsms.h_lower, fsms.w_lower, lanczos_width)
        populate_gal_fsm_image!(ea, s, n, gal_mcs, fsms)
        populate_source_band_brightness!(ea, s, n, fsms, sbs[s])
        E_G = fsms.E_G[1, 1]
    end

    # Iterate only over active sources, since we have already added the
    # contributions from non-active sources to E_G and var_G.
    for s in ea.active_sources
        p = ea.patches[s, n]
        H_patch, W_patch = size(p.active_pixel_bitmap)
        for w_patch in 1:W_patch, h_patch in 1:H_patch
            h_image = h_patch + p.bitmap_corner[1]
            w_image = w_patch + p.bitmap_corner[2]

            image = ea.images[n]
            this_pixel = image.pixels[h_image, w_image]

            if Base.isnan(this_pixel)
                continue
            end

            # These are indices within the fs?m image.
            h_fsm = h_image - fsms.h_lower + 1
            w_fsm = w_image - fsms.w_lower + 1

            E_G = fsms.E_G[h_fsm, w_fsm]
            var_G = fsms.var_G[h_fsm, w_fsm]

            # There are no derivatives with respect to epsilon, so can
            # afely add to the value.
            E_G.v[1] += image.epsilon_mat[h_image, w_image]

            # Note that with a lanczos_width > 1 negative values are
            # possible, and this will result in an error in
            # add_elbo_log_term.

            # Add the terms to the elbo given the brightness.
            iota = image.iota_vec[h_image]
            add_elbo_log_term!(
                ea.elbo_vars, E_G, var_G, ea.elbo_vars.elbo, this_pixel, iota)
            add_scaled_sfs!(ea.elbo_vars.elbo, E_G, -iota,
                            ea.elbo_vars.calculate_hessian &&
                            ea.elbo_vars.calculate_derivs)

            # Subtract the log factorial term. This is not a function of the
            # parameters so the derivatives don't need to be updated. Note
            # that even though this does not affect the ELBO's maximum,
            # it affects the optimization convergence criterion, so I will
            # leave it in for now.
            ea.elbo_vars.elbo.v[1] -= lfact(this_pixel)
        end
    end
end


function elbo_likelihood_with_fft!(
    ea::ElboArgs,
    lanczos_width::Int64,
    fsm_vec::Array{FSMSensitiveFloatMatrices})

    sbs = load_source_brightnesses(ea,
        calculate_derivs=ea.elbo_vars.calculate_derivs,
        calculate_hessian=ea.elbo_vars.calculate_hessian);

    clear!(ea.elbo_vars.elbo)
    for n in 1:ea.N
        gal_mcs = load_gal_bvn_mixtures(
                ea.S, ea.patches, ea.vp, ea.active_sources, n,
                calculate_derivs=ea.elbo_vars.calculate_derivs,
                calculate_hessian=ea.elbo_vars.calculate_hessian);
        accumulate_band_in_elbo!(ea, fsm_vec[n], sbs, gal_mcs, n, lanczos_width)
    end
end


end
