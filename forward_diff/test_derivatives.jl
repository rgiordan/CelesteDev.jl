
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
include("test_derivatives_functions.jl")
include(joinpath(Pkg.dir("Celeste"), "test/derivative_utils.jl"))


run, camcol, field = (3900, 6, 269)

images = SDSSIO.load_field_images(run, camcol, field, datadir)
tiled_images = TiledImage[TiledImage(img) for img in images]
fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" datadir run camcol field
catalog = SDSSIO.read_photoobj_celeste(fname)

# Pick an object.
objid = "1237662226208063499"
objids = [ce.objid for ce in catalog]
sa = findfirst(objids, objid)
neighbors = Infer.find_neighbors([sa], catalog, tiled_images)[1]

cat_local = vcat(catalog[sa], catalog[neighbors])
vp = Vector{Float64}[init_source(ce) for ce in cat_local]
patches, tile_source_map = Infer.get_tile_source_map(tiled_images, cat_local)
ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [1])
Infer.fit_object_psfs!(ea, ea.active_sources)
Infer.trim_source_tiles!(ea)

elbo = ElboDeriv.elbo(ea);

function wrap_elbo{NumType <: Number}(vs1::Vector{NumType})
    ea_local = forward_diff_model_params(NumType, ea)
    ea_local.vp[1][:] = vs1
    local_elbo = ElboDeriv.elbo(ea_local, calculate_derivs=false)
    local_elbo.v[1]
end

@time ad_grad = ForwardDiff.gradient(wrap_elbo, ea.vp[1]);
@time ad_hess = ForwardDiff.hessian(wrap_elbo, ea.vp[1]);
@time ad_hess = ForwardDiff.hessian(wrap_elbo, ea.vp[1]);

# julia> @time ad_hess = ForwardDiff.hessian(wrap_elbo, ea.vp[1]);
# 381.400357 seconds (331.23 M allocations: 6.106 GB, 0.88% gc time)
# julia> @time ad_hess = ForwardDiff.hessian(wrap_elbo, ea.vp[1]);
# 6.854772 seconds (1.70 M allocations: 895.854 MB, 9.58% gc time

# Sanity check
ad_v = wrap_elbo(ea.vp[1]);
@test_approx_eq ad_v elbo.v

hcat(ad_grad, elbo.d[:, 1])
@test_approx_eq ad_grad elbo.d[:, 1]
@test_approx_eq ad_hess elbo.h
