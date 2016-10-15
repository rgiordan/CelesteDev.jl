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

using PyPlot


##########
# Ensure that test images are available.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")

include("rasterized_psf/predicted_image.jl")

using PSFConvolution

run, camcol, field = (3900, 6, 269)

images = SDSSIO.load_field_images(RunCamcolField(run, camcol, field), datadir);
tiled_images = TiledImage[TiledImage(img) for img in images];
dir = joinpath(datadir, "$run/$camcol/$field")
fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir run camcol field
catalog = SDSSIO.read_photoobj_celeste(fname);

# Pick an object.
objid = "1237662226208063499"
objids = [ce.objid for ce in catalog];
sa = findfirst(objids, objid);
neighbors = Infer.find_neighbors([sa], catalog, tiled_images)[1];

cat_local = vcat(catalog[sa], catalog[neighbors]);
vp = Vector{Float64}[init_source(ce) for ce in cat_local];
patches, tile_source_map = Infer.get_tile_source_map(tiled_images, cat_local);
ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [sa]; psf_K=1);
Infer.fit_object_psfs!(ea, ea.active_sources);
Infer.load_active_pixels!(ea);

# For compiling
elbo = DeterministicVI.elbo(ea);

current_elbo_time = time()
elbo = DeterministicVI.elbo(ea);
current_elbo_time = time() - current_elbo_time


##############

# This PSF effectively interpolates the pixelated PSF
psf_spread = 0.5
psf_comp = Model.PsfComponent(1.0, Float64[0, 0],
    Float64[ psf_spread 0.0; 0.0 psf_spread ]);
psf_image = PSF.get_psf_at_point(PsfComponent[psf_comp])


######################################
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
using PSFConvolution.FSMSensitiveFloatMatrices
using PSFConvolution.initialize_fsm_sf_matrices!

s = ea.active_sources[1]

elbo_vars = DeterministicVI.ElboIntermediateVariables(
    Float64, ea.S, length(ea.active_sources));

psf_image_vec = Matrix{Float64}[ psf_image for n in 1:ea.N ];

fsm_vec = FSMSensitiveFloatMatrices[FSMSensitiveFloatMatrices() for b in 1:ea.N];
PSFConvolution.initialize_fsm_sf_matrices!(fsm_vec, ea, psf_image_vec);


############### Process active pixels
using DeterministicVI.BvnComponent
using DeterministicVI.GalaxyCacheComponent
using DeterministicVI.load_bvn_mixtures
using DeterministicVI.load_source_brightnesses
using DeterministicVI.populate_fsm!


elbo_time = time()

sbs = load_source_brightnesses(ea,
    calculate_derivs=elbo_vars.calculate_derivs,
    calculate_hessian=elbo_vars.calculate_hessian);

star_mcs_vec = Array(Array{BvnComponent{Float64}, 2}, ea.N);
gal_mcs_vec = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N);

foo = time() - elbo_time
for b=1:ea.N
    star_mcs_vec[b], gal_mcs_vec[b] =
        load_bvn_mixtures(ea, b,
            calculate_derivs=elbo_vars.calculate_derivs,
            calculate_hessian=elbo_vars.calculate_hessian)
end

clear!(elbo_vars.elbo)
for b in 1:ea.N
    accumulate_band_in_elbo!(ea, elbo_vars, fsm_vec[b], sbs, star_mcs_vec, gal_mcs_vec, b)
end

elbo_time = time() - elbo_time

elbo_time / current_elbo_time



# The FFT is taking almost all the time.
@profile begin
    for i in 1:10
        add_source_to_elbo!(ea, elbo_vars, s, sbs, star_mcs_vec, gal_mcs_vec, fsm_vec);
    end
end

Profile.print()


######################################


import Base.print
function print(ce::CatalogEntry)
  for field in fieldnames(ce)
    println(field, ": ", getfield(ce, field))
  end
end


ea.vp[sa][ids.a] = [ 1, 0 ]
ea.vp[sa][ids.u] = WCS.pix_to_world(ea.images[b].wcs,
  floor(WCS.world_to_pix(ea.images[b].wcs, ea.vp[sa][ids.u])) + 0.5)

source_tiles = ea.images[b].tiles[find([sa in tile for tile in tile_source_map[b] ])];
min_h = minimum([tile.h_range.start for tile in source_tiles])
max_h = maximum([tile.h_range.stop for tile in source_tiles])
min_w = minimum([tile.w_range.start for tile in source_tiles])
max_w = maximum([tile.w_range.stop for tile in source_tiles])
image = Float64[NaN for h=1:(max_h - min_h + 1), w=1:(max_w - min_w + 1)];

# Show the active pixels
for pixel in ea.active_pixels
    if pixel.n == b
        tile = ea.images[b].tiles[pixel.tile_ind]
        image[tile.h_range.start - min_h + pixel.h,
              tile.w_range.start - min_w + pixel.w] = 1
    end
end


for tile in source_tiles
  pix = DeterministicVI.tile_predicted_image(tile, ea, Int64[sa], include_epsilon=false)
  image[tile.h_range - min_h + 1, tile.w_range - min_w + 1] = pix
end
image[abs(image) .< 1e-8 ] = 0;
matshow(image); title(psf_spread); colorbar()





############
