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

# Ensure that test images are available.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
wd = pwd()

include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/predicted_image.jl")

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
ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [sa], default_psf_K);

point_psf_width = 0.5
point_psf = Model.PsfComponent(1.0, Float64[0, 0],
    Float64[ point_psf_width 0.0; 0.0 point_psf_width ])
for s in 1:size(ea.patches)[1], b in 1:size(ea.patches)[2]
  ea.patches[s, b] = SkyPatch(ea.patches[s, b], Model.PsfComponent[ point_psf ]);
end
ea.psf_K = 1

b = 3

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

for tile in source_tiles
  pix = DeterministicVI.tile_predicted_image(tile, ea, Int64[sa], include_epsilon=false)
  image[tile.h_range - min_h + 1, tile.w_range - min_w + 1] = pix
end
image[abs(image) .< 1e-8 ] = 0;
matshow(image); title(point_psf_width); colorbar()
image_sparse = sparse(image)


psf_spread = 5.0
psf_comp = Model.PsfComponent(1.0, Float64[0, 0],
    Float64[ psf_spread 0.0; 0.0 psf_spread ]);
psf_image = PSF.get_psf_at_point(PsfComponent[psf_comp])
# psf_nonzero = ind2sub(size(psf_image), find(abs(psf_image) .> 1e-16))
# psf_h_range = minimum(psf_nonzero[1]):maximum(psf_nonzero[1])
# psf_w_range = minimum(psf_nonzero[2]):maximum(psf_nonzero[2])
# psf_image = psf_image[psf_h_range, psf_w_range]

psf_image[abs(psf_image) .< 1e-8] = 0

image_convolved_psf = conv2(psf_image, image);
# matshow(image_convolved_psf); title("convolved"); colorbar()
# PyPlot.close("all")



# Convolution of sensitive floats

NumType = Float64
tile = source_tiles[4];
elbo_vars =
    DeterministicVI.ElboIntermediateVariables(Float64, ea.S, length(ea.active_sources));
tile_sources = Int[sa]

star_mcs, gal_mcs = DeterministicVI.load_bvn_mixtures(ea, tile.b, calculate_derivs=true);

# Sanity check:
star_mcs[1, sa].the_mean
obj_loc_pix = WCS.world_to_pix(ea.images[b].wcs, ea.vp[sa][ids.u])
Model.linear_world_to_pix(ea.patches[sa, b].wcs_jacobian,
                          ea.patches[sa, b].center,
                          ea.patches[sa, b].pixel_center, ea.vp[sa][ids.u])


fs0m_tile, fs1m_tile =
    get_source_tile_fsm_matrix(tile, ea, sa, elbo_vars, star_mcs, gal_mcs,
                               tile_sources, true, true);
(fft_size1, fft_size2) =
    (size(fs0m_tile, 1) + size(psf_image, 1) - 1,
     size(fs0m_tile, 2) + size(psf_image, 2) - 1)
psf_fft = zeros(Complex{Float64}, fft_size1, fft_size2);
psf_fft[1:size(psf_image, 1), 1:size(psf_image, 2)] = psf_image;
fft!(psf_fft);

ParamType = StarPosParams
fs0m_tile_padded = [ SensitiveFloats.zero_sensitive_float(ParamType, Float64, size(fs0m_tile[1].d, 2))
                     for h in 1:fft_size1, w in 1:fft_size2  ];
for h in 1:size(fs0m_tile, 1), w in 1:size(fs0m_tile, 2)
    fs0m_tile_padded[h, w] = deepcopy(fs0m_tile[h, w])
end

fs0m_conv = convolve_sensitive_float_matrix(fs0m_tile_padded, psf_fft);
PyPlot.close("all")
# A point gets mapped to its location in foo + location in bar - 1.  Since the psf
# image is centered at (26, 26), spatial locations are increased by 25.
obj_loc_tile_pix =
    obj_loc_pix -
    [ tile.h_range.start - 1, tile.w_range.start - 1]

matshow([sf.v[1] for sf in fs0m_conv]); title("conv");
plot(obj_loc_tile_pix[2] - 1 + 25, obj_loc_tile_pix[1] - 1 + 25, "ro");
colorbar()
matshow([sf.v[1] for sf in fs0m_tile]); title("tile"); colorbar()
plot(obj_loc_tile_pix[2] - 1, obj_loc_tile_pix[1] - 1, "ro");
#matshow(conv2(Float64[ sf.v[1] for sf in fs0m_tile ], psf_image)); colorbar(); title("conv2")



# Check how the spatial mapping works
foo = zeros(30, 30)
bar = zeros(30, 30)
foo[5, 16] = 1
bar[4, 10] = 1
baz = conv2(foo, bar)
ind2sub(size(baz), find(baz .== maximum(baz)))














############
