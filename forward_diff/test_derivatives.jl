
using Celeste: Model, Transform, SensitiveFloats, ElboDeriv
using Base.Test

import Celeste: WCSUtils, Infer, ElboDeriv
import Celeste: PSF, OptimizeElbo, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!

include(joinpath(Pkg.dir("Celeste"), "test/Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test/SampleData.jl"))

using Synthetic
using SampleData

# Ensure that test images are available.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")

import DualNumbers
import ForwardDiff

using Celeste: Model, SensitiveFloats, ElboDeriv
import ElboDeriv: BvnComponent, GalaxyCacheComponent
import ElboDeriv: eval_bvn_pdf!, get_bvn_derivs!, transform_bvn_derivs!

cd("/home/rgiordan/Documents/git_repos/CelesteDev.jl/forward_diff")
include(joinpath(Pkg.dir("Celeste"), "test/derivative_utils.jl"))


blob, ea, body = gen_two_body_dataset();
keep_pixels = 10:11
trim_tiles!(ea.images, keep_pixels);

ea.active_sources = [1]
vp_vec = wrap_vp_vector(ea, true);
@time elbo_1 = ElboDeriv.elbo(ea);
@time elbo_1 = ElboDeriv.elbo(ea);




#
# # vp_vec is a vector of the parameters from all the active sources.
function wrap_elbo{NumType <: Number}(vp_vec::Vector{NumType})
    ea_local = unwrap_vp_vector(vp_vec, ea)
    elbo = ElboDeriv.elbo(ea_local, calculate_derivs=false)
    elbo.v[1]
end

@time ad_grad = ForwardDiff.gradient(wrap_elbo, vp_vec)
@time ad_hess = ForwardDiff.hessian(wrap_elbo, vp_vec)

@time test_with_autodiff(wrap_elbo, vp_vec, elbo_1);




# include(joinpath(Pkg.dir("Celeste"), "src/elbo_kl.jl"))

s = 1
test_symb = :kl
function wrap_kl{NumType <: Number}(vp_vec::Vector{NumType})
    ea_local = unwrap_vp_vector(vp_vec, ea);
    kl_source = zero_sensitive_float(CanonicalParams, NumType, length(ea.active_sources))
    if test_symb == :kl
        ElboDeriv.subtract_kl!(ea_local, kl_source, calculate_derivs=false)
    elseif test_symb == :a
        ElboDeriv.subtract_kl_a!(ea_local.vp[s], kl_source, false)
    elseif test_symb == :r
        ElboDeriv.subtract_kl_r!(ea_local.vp[s], kl_source, false)
    elseif test_symb == :k
        ElboDeriv.subtract_kl_k!(ea_local.vp[s], kl_source, false)
    elseif test_symb == :c
        ElboDeriv.subtract_kl_c!(ea_local.vp[s], kl_source, false)
    else
        println("Bad symbol")
    end

    kl_source.v[1]
end

accum = zero_sensitive_float(CanonicalParams, Float64, length(ea.active_sources))
# ElboDeriv.subtract_kl_c!(ea.vp[s], accum, true);
ElboDeriv.subtract_kl!(ea, accum, calculate_derivs=true);

@time test_with_autodiff(wrap_kl, vp_vec, accum);
