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
        if !p.active_pixel_bitmap[h_patch, w_patch]
            continue
        end
        h_image = h_patch + p.bitmap_offset[1]
        w_image = w_patch + p.bitmap_offset[2]

        h_fsm = h_image - fsms.h_lower + 1
        w_fsm = w_image - fsms.w_lower + 1

        x = SVector{2, Float64}([h_image, w_image])
        populate_gal_fsm!(ea.elbo_vars.bvn_derivs,
                          fsms.fs1m_image[h_fsm, w_fsm],
                          s, x, is_active_source,
                          p.wcs_jacobian,
                          gal_mcs)
    end
    convolve_fs1m_image!(fsms, s)
end

#############################
# Dig in

using Celeste.DeterministicVIImagePSF

using Celeste.DeterministicVI.CanonicalParams
using Celeste.SensitiveFloats.zero_sensitive_float_array
using Celeste.SensitiveFloats.SensitiveFloat
using Celeste.SensitiveFloats.clear!

using ForwardDiff
using Base.Test



fsms = DeterministicVIImagePSF.FSMSensitiveFloatMatrices();
psf_image = zeros(3, 3);
psf_image[2, 2] = 0.5;
psf_image[2, 1] = psf_image[1, 2] = 0.25;
DeterministicVIImagePSF.initialize_fsm_sf_matrices_band!(
    fsms, 1, 1, 1,
    1, 1, 3, 3, psf_image)

# The PSF is sensitive to "StarPosParams", which are just location
sf = SensitiveFloat{Float64}(length(StarPosParams), 1, true, true)
ea.vp[1][ids.u]
Celeste.DeterministicVIImagePSF.populate_star_fsm_image!(
    ea, s, 3, fsms.psf, fsms.fs0m_conv,
    fsms.h_lower, fsms.w_lower,
    fsms.kernel_fun, fsms.kernel_width);

# The idea then is to convolve fs0m_conv with fs1m_image
















# okok