
using Celeste: Model, Transform, SensitiveFloats, ElboDeriv
using Base.Test

import Celeste: WCSUtils, Infer, ElboDeriv
import Celeste: PSF, OptimizeElbo, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!

include(joinpath(Pkg.dir("Celeste"), "test/Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test/SampleData.jl"))

import Synthetic
using SampleData

# Ensure that test images are available.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")

omitted_ids = Int[];
kept_ids = setdiff(1:length(ids_free), omitted_ids);

blob, ea, bodies = SampleData.gen_three_body_dataset();

# Change the tile size.
ea = make_elbo_args(
  blob, bodies, tile_width=5, fit_psf=false, patch_radius=10.);
ea.active_sources = Int[2, 3]
trans = Transform.get_mp_transform(ea, loc_width=1.0);

wrapper =
  OptimizeElbo.ObjectiveWrapperFunctions(
    ea -> ElboDeriv.elbo(ea),
    ea, trans, kept_ids, omitted_ids);

x = trans.vp_to_array(ea.vp, omitted_ids);


@time elbo_result = trans.transform_sensitive_float(ElboDeriv.elbo(ea), ea);
