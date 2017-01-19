#!/usr/bin/env julia

using Celeste: Model, DeterministicVI

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

import Synthetic
using SampleData

using Base.Test
using Distributions


using Celeste: DeterministicVI, SensitiveFloats

using Base.Test
using Distributions
using DerivativeTestUtils


blob, ea, body = gen_two_body_dataset();
b = 3
s = 1
h = w = 5

tile = ea.images[b].tiles[1, 1]
tile_source_map = ea.tile_source_map[b][1, 1]

star_mcs, gal_mcs =
    DeterministicVI.load_bvn_mixtures(ea, b, calculate_derivs=true)
elbo_vars = DeterministicVI.ElboIntermediateVariables(Float64, ea.S, ea.S)
DeterministicVI.populate_fsm_mats!(
    elbo_vars, ea, tile_source_map, tile, h, w, gal_mcs, star_mcs)

fs0m = zero_sensitive_float(StarPosParams, Float64)
fs1m = zero_sensitive_float(GalaxyPosParams, Float64)

x = Float64[tile.h_range[h], tile.w_range[w]]
DeterministicVI.populate_fsm!(
    elbo_vars, ea, fs0m, fs1m,
    s, b, x, true, gal_mcs, star_mcs)

@test_approx_eq fs0m.v[1] elbo_vars.fs0m_vec[s].v[1]
@test_approx_eq fs0m.d elbo_vars.fs0m_vec[s].d
@test_approx_eq fs0m.h elbo_vars.fs0m_vec[s].h
