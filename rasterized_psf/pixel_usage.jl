using Celeste: Model, DeterministicVI
using StaticArrays

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/predicted_image.jl")
include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/celeste_tools/celeste_tools.jl")

using PSFConvolution


import Synthetic
using SampleData

using Base.Test
using Distributions

using PyPlot


##########
# Ensure that test images are available.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")

run, camcol, field = (4263, 5,119)

images = SDSSIO.load_field_images(RunCamcolField(run, camcol, field), datadir);
tiled_images = TiledImage[TiledImage(img) for img in images];
dir = joinpath(datadir, "$run/$camcol/$field")
fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir run camcol field
catalog = SDSSIO.read_photoobj_celeste(fname);

total_pixels = sum([ length(image.pixels) for image in images])

pixels_used = 0
for s in 1:length(catalog)
    cat = catalog[s]
    vp = Vector{Float64}[init_source(cat)];
    patches, tile_source_map = Infer.get_tile_source_map(tiled_images, [ catalog[s] ]);
    ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [1]; psf_K=2);
    load_active_pixels!(ea, false);
    pixels_used += length(ea.active_pixels)
    expected_pixels_used = pixels_used * length(catalog) / s
    println(s, " of ", length(catalog), " pixels: ", pixels_used,
            " expected: ", expected_pixels_used / total_pixels)
end
