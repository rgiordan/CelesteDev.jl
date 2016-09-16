
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


#function test_enforce_bounds()


blob, ea, three_bodies = gen_three_body_dataset();
	transform = get_mp_transform(ea);

	ea.vp[1][ids.a[1, 1]] = transform.bounds[1][:a][1].lower_bound - 0.00001
	ea.vp[2][ids.r1[1]] = transform.bounds[2][:r1][1].lower_bound - 1.0
	ea.vp[2][ids.r1[2]] = transform.bounds[2][:r1][1].upper_bound + 1.0
	ea.vp[3][ids.k[1, 1]] = transform.bounds[3][:k][1, 1].lower_bound - 0.00001

	@test_throws AssertionError transform.from_vp(ea.vp)
	Transform.enforce_bounds!(ea, transform)

	# Check that it now works and all values are finite.
	x_trans = transform.from_vp(ea.vp)
	for s = 1:ea.S
		@test !any(Bool[ isinf(x) for x in x_trans[1] ])
	end

	# Test with only one active source.
	sa = 2
	ea.active_sources = [sa]
	transform = get_mp_transform(ea);

	ea.vp[sa][ids.a[1, 1]] = transform.bounds[1][:a][1].lower_bound - 0.00001
	ea.vp[sa][ids.r1[1]] = transform.bounds[1][:r1][1].lower_bound - 1.0
	ea.vp[sa][ids.r1[2]] = transform.bounds[1][:r1][1].upper_bound + 1.0
	ea.vp[sa][ids.k[1, 1]] = transform.bounds[1][:k][1, 1].lower_bound - 0.00001

	@test_throws AssertionError transform.from_vp(ea.vp)
	Transform.enforce_bounds!(ea, transform)

	# Check that it now works and all values are finite.
	x_trans = transform.from_vp(ea.vp)
	for s = 1:ea.S
		@test !any(Bool[ isinf(x) for x in x_trans[1] ])
	end
