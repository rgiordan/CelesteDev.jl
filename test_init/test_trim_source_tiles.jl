
using Celeste: Model, ElboDeriv

import Celeste: Infer, ElboDeriv
import Celeste: PSF, OptimizeElbo, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

const testdir = joinpath(Pkg.dir("Celeste"), "test")
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")

cd(testdir)

include("Synthetic.jl")
include("SampleData.jl")
include("DerivativeTestUtils.jl")

import Synthetic
using SampleData

using Base.Test
using Distributions

using Celeste: ElboDeriv, SensitiveFloats

using Base.Test
using Distributions
using DerivativeTestUtils


import ForwardDiff

using Celeste: Model, SensitiveFloats, ElboDeriv
using Distributions
import ElboDeriv: BvnComponent, GalaxyCacheComponent
import ElboDeriv: eval_bvn_pdf!, get_bvn_derivs!, transform_bvn_derivs!


blob, ea, bodies = gen_two_body_dataset()

# Choose four pixels only to keep the test fast.
active_pixels = Array(ElboDeriv.ActivePixel, 4)
active_pixels[1] = ElboDeriv.ActivePixel(1, 1, 10, 11)
active_pixels[2] = ElboDeriv.ActivePixel(1, 1, 11, 10)
active_pixels[3] = ElboDeriv.ActivePixel(5, 1, 10, 11)
active_pixels[4] = ElboDeriv.ActivePixel(5, 1, 11, 10)


function tile_lik_wrapper_fun{NumType <: Number}(
        ea::ElboArgs{NumType}, calculate_derivs::Bool)

    elbo_vars = ElboDeriv.ElboIntermediateVariables(
        NumType, ea.S,
        length(ea.active_sources),
        calculate_derivs=calculate_derivs,
        calculate_hessian=calculate_derivs)
    ElboDeriv.process_active_pixels!(elbo_vars, ea, active_pixels)
    deepcopy(elbo_vars.elbo)
end

function tile_lik_value_wrapper{NumType <: Number}(x::Vector{NumType})
    ea_local = unwrap_vp_vector(x, ea)
    tile_lik_wrapper_fun(ea_local, false).v[1]
end

elbo = tile_lik_wrapper_fun(ea, true);

x = wrap_vp_vector(ea, true)
test_with_autodiff(tile_lik_value_wrapper, x, elbo)
