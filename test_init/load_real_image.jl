
using Celeste: Model, ElboDeriv

import Celeste: Infer, ElboDeriv
import Celeste: PSF, OptimizeElbo, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

const testdir = joinpath(Pkg.dir("Celeste"), "test")
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
using DerivativeTestUtils

const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")


run, camcol, field = (3900, 6, 269)

images = SDSSIO.load_field_images(RunCamcolField(run, camcol, field), datadir);
tiled_images = TiledImage[TiledImage(img) for img in images];
dir = joinpath(datadir, "$run/$camcol/$field");
fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir run camcol field
catalog = SDSSIO.read_photoobj_celeste(fname);

# Pick an object.
objid = "1237662226208063499"
objids = [ce.objid for ce in catalog]
sa = findfirst(objids, objid);
neighbors = Infer.find_neighbors([sa], catalog, tiled_images)[1];

cat_local = vcat(catalog[sa], catalog[neighbors]);
vp = Vector{Float64}[init_source(ce) for ce in cat_local];
patches, tile_source_map = Infer.get_tile_source_map(tiled_images, cat_local);
ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [1]);
Infer.fit_object_psfs!(ea, ea.active_sources);
Infer.trim_source_tiles!(ea);

elbo = ElboDeriv.elbo(ea)
