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
psf_nonzero = ind2sub(size(psf_image), find(abs(psf_image) .> 1e-16))
psf_h_range = minimum(psf_nonzero[1]):maximum(psf_nonzero[1])
psf_w_range = minimum(psf_nonzero[2]):maximum(psf_nonzero[2])
psf_image = psf_image[psf_h_range, psf_w_range]

psf_image[abs(psf_image) .< 1e-8] = 0
psf_image_sparse = sparse(psf_image);
matshow(psf_image_sparse)

image_convolved_psf = conv2(psf_image, image);
matshow(image_convolved_psf); title("convolved"); colorbar()
PyPlot.close("all")

elbo_vars =
    DeterministicVI.ElboIntermediateVariables(Float64, ea.S, length(ea.active_sources));


include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/predicted_image.jl")
tile = source_tiles[4];
E_G_pixels, var_G_pixels =
  get_expected_tile_brightness(tile, ea, Int64[sa], true, true, include_epsilon=false);

E_G_v = [ pix.v[1] for pix in E_G_pixels ];

matshow(E_G_v); colorbar(); title("E_G_v")
matshow(Float64[ pix.d[ids.e_scale] for pix in E_G_pixels ]); colorbar(); title("deriv")
matshow(Float64[ pix.v[1] for pix in var_G_pixels ]); colorbar(); title("var")
PyPlot.close("all")

convolved_sf = SensitiveFloats.zero_sensitive_float(CanonicalParams, Float64, length(ea.active_sources));




# Naive convolution
conv_time = time()
convolved_sf.v[1] = sum(conv2(psf_image, [ pix.v[1] for pix in E_G_pixels ]));

for sa_d in 1:size(convolved_sf.d, 2), ind in 1:size(convolved_sf.d, 1)
  convolved_sf.d[ind, sa_d] =
    sum(conv2(psf_image, [ pix.d[ind, sa_d] for pix in E_G_pixels ]));
end

for ind1 in 1:size(convolved_sf.h, 1), ind2 in 1:ind1
  convolved_sf.h[ind1, ind2] = convolved_sf.h[ind2, ind1] =
    sum(conv2(psf_image, [ pix.h[ind1, ind2] for pix in E_G_pixels ]));
end
conv_time = time() - conv_time




# Caching convolution
convolved_sf = SensitiveFloats.zero_sensitive_float(
  CanonicalParams, Float64, length(ea.active_sources));

psf_size = size(psf_image)
tile_size = size(E_G_pixels)
psf_mat_for_fft = zeros(psf_size[1] + tile_size[1] - 1, psf_size[2] + tile_size[2] - 1);
tile_mat_for_fft = zeros(psf_size[1] + tile_size[1] - 1, psf_size[2] + tile_size[2] - 1);
psf_mat_for_fft[1:psf_size[1], 1:psf_size[2]] = psf_image
psf_fft_plan = plan_fft(psf_mat_for_fft);
psf_fft = psf_fft_plan * psf_mat_for_fft;

conv_time = time()

for h in 1:tile_size[1], w in 1:tile_size[2]
  tile_mat_for_fft[h, w] = E_G_pixels[h, w].v[1]
end

convolved_sf.v[1] = sum(real(ifft(psf_fft .* (psf_fft_plan * tile_mat_for_fft))));

for sa_d in 1:size(convolved_sf.d, 2), ind in 1:size(convolved_sf.d, 1)
  for h in 1:tile_size[1], w in 1:tile_size[2]
    tile_mat_for_fft[h, w] = E_G_pixels[h, w].d[ind, sa_d]
  end
  convolved_sf.d[ind, sa_d] =
    sum(real(ifft(psf_fft .* (psf_fft_plan * tile_mat_for_fft))));
end

for ind1 in 1:size(convolved_sf.h, 1), ind2 in 1:ind1
  for h in 1:tile_size[1], w in 1:tile_size[2]
    tile_mat_for_fft[h, w] = E_G_pixels[h, w].h[ind2, ind1]
  end
    sum(real(ifft(psf_fft .* (psf_fft_plan * tile_mat_for_fft))));

  convolved_sf.h[ind1, ind2] = convolved_sf.h[ind2, ind1] =
    sum(real(ifft(psf_fft .* (psf_fft_plan * tile_mat_for_fft))));
end
conv_time = time() - conv_time




# Here's the code for conv2


sa, sb = size(A), size(B)
At = zeros(T, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
Bt = zeros(T, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
At[1:sa[1], 1:sa[2]] = A
Bt[1:sb[1], 1:sb[2]] = B
p = plan_fft(At)
C = ifft((p*At).*(p*Bt))
if T <: Real
    return real(C)
end
return C
