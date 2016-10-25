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

# Pick an object.
for cat in catalog
    if minimum(cat.star_fluxes) > 300
        print(cat)
        print("\n")
    end
end
objid = "1237663784734490800"
objids = [ce.objid for ce in catalog];
sa = findfirst(objids, objid);
neighbors = Infer.find_neighbors([sa], catalog, tiled_images)[1];
cat_local = vcat(catalog[sa], catalog[neighbors]);

vp = Vector{Float64}[init_source(ce) for ce in cat_local];
patches, tile_source_map = Infer.get_tile_source_map(tiled_images, cat_local);
ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [1]; psf_K=1);
Infer.fit_object_psfs!(ea, ea.active_sources);
Infer.load_active_pixels!(ea);
length(ea.active_pixels)

# For compiling
elbo = DeterministicVI.elbo(ea);

current_elbo_time = time()
elbo = DeterministicVI.elbo(ea);
current_elbo_time = time() - current_elbo_time

##############

# Get the actual PSF images using the /first/ source.
Infer.fit_object_psfs!(ea, ea.active_sources);
psf_image_vec =
    Matrix{Float64}[ PSF.get_psf_at_point(ea.patches[1, b].psf) for b in 1:ea.N ];

# Then set the ea "psf" to a small width to interpolate the pixelated PSF.
PSFConvolution.set_point_psf!(ea, 0.5);

######################################
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
using PSFConvolution.FSMSensitiveFloatMatrices
using PSFConvolution.initialize_fsm_sf_matrices!

s = ea.active_sources[1]

elbo_vars = DeterministicVI.ElboIntermediateVariables(
    Float64, ea.S, length(ea.active_sources));

fsm_vec = FSMSensitiveFloatMatrices[FSMSensitiveFloatMatrices() for b in 1:ea.N];
PSFConvolution.initialize_fsm_sf_matrices!(fsm_vec, ea, psf_image_vec);


############### Process active pixels
using DeterministicVI.BvnComponent
using DeterministicVI.GalaxyCacheComponent
using DeterministicVI.load_bvn_mixtures
using DeterministicVI.load_source_brightnesses
using Celeste.Model.populate_fsm!


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
    PSFConvolution.accumulate_band_in_elbo!(
        ea, elbo_vars, fsm_vec[b], sbs, star_mcs_vec, gal_mcs_vec, b, true)
end

DeterministicVI.subtract_kl!(ea, elbo_vars.elbo, calculate_derivs=true);

elbo_time = time() - elbo_time
println("Time ratio: ", elbo_time / current_elbo_time)


elbo_fft = deepcopy(elbo_vars.elbo);

elbo_fft.v
elbo.v
plot((elbo_fft.d + 1e-6) ./ (elbo.d + 1e-6), "o")
plot((elbo_fft.h + 1e-6) ./ (elbo.h + 1e-6), "o")

########################################
# Debugging

b = 3;
fsms = fsm_vec[b];
sb = sbs[s];
PSFConvolution.clear_brightness!(fsms);
PSFConvolution.populate_fsm_image!(ea, elbo_vars, s, b, star_mcs_vec[s], gal_mcs_vec[s], fsms);
PSFConvolution.accumulate_source_band_brightness!(ea, elbo_vars, s, b, fsms, sb)

matshow([ sf.v[1] for sf in fsms.E_G ])


########################################
# Profiling:

# The FFT is taking almost all the time.
@profile begin
    for i in 1:10
        add_source_to_elbo!(ea, elbo_vars, s, sbs,
                            star_mcs_vec, gal_mcs_vec, fsm_vec);
    end
end

Profile.print()


######################################

sa = 1
b = 3

ea.vp[sa][ids.a] = [ 1, 0 ]
ea.vp[sa][ids.u] = WCS.pix_to_world(ea.images[b].wcs,
  floor(WCS.world_to_pix(ea.images[b].wcs, ea.vp[sa][ids.u])) + 0.5)

sub_image = get_source_pixel_range(sa, b, ea);

# Display rendered image
image = render_source(ea, sa, sub_image, false);
matshow(image); title("Rendered image"); colorbar()

# Display original image
image = show_source_image(ea, sa, sub_image);
matshow(image); title("Original image"); colorbar()

# Active pixel map
active_image = show_active_pixels(ea, sub_image, b);
matshow(active_image); title("Active pixels");




############
