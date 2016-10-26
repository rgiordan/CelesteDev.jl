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
ea = ElboArgs(tiled_images, vp, tile_source_map, patches, [1]; psf_K=2);
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
psf_image_vec =
    Matrix{Float64}[ PSF.get_psf_at_point(ea.patches[1, b].psf) for b in 1:ea.N ];

using StaticArrays
import Celeste.Model
import Celeste.Model.SkyPatch

ea_fft = ElboArgs(tiled_images, deepcopy(vp), tile_source_map,
                  deepcopy(patches), [1]; psf_K=1);
load_active_pixels!(ea_fft, false);


# Then set the fft ea "psf" to a small width to interpolate the pixelated PSF.
point_psf_width = 0.5;
point_psf = Model.PsfComponent(1.0, SVector{2,Float64}([0, 0]),
    SMatrix{2, 2, Float64, 4}([ point_psf_width 0.0; 0.0 point_psf_width ]));
for s in 1:size(ea_fft.patches)[1], b in 1:size(ea_fft.patches)[2]
    ea_fft.patches[s, b] = SkyPatch(ea_fft.patches[s, b], Model.PsfComponent[ point_psf ]);
end

# matshow(PSF.get_psf_at_point([point_psf]))
# matshow(psf_image_vec[3])
# sum(PSF.get_psf_at_point([point_psf]))
# sum(psf_image_vec[3])


######################################

elbo_vars_fft = DeterministicVI.ElboIntermediateVariables(
    Float64, ea_fft.S, length(ea_fft.active_sources));

fsm_vec = FSMSensitiveFloatMatrices[FSMSensitiveFloatMatrices() for b in 1:ea_fft.N];
PSFConvolution.initialize_fsm_sf_matrices!(fsm_vec, ea_fft, psf_image_vec);

# For compilation
elbo_likelihood_with_fft!(ea_fft, elbo_vars_fft, fsm_vec);

elbo_time = time()
elbo_likelihood_with_fft!(ea_fft, elbo_vars_fft, fsm_vec);
DeterministicVI.subtract_kl!(ea, elbo_vars_fft.elbo, calculate_derivs=true);
elbo_time = time() - elbo_time

println("Time ratio: ", elbo_time / current_elbo_time)

elbo_fft = deepcopy(elbo_vars_fft.elbo);

println(elbo_fft.v[1], " ", elbo.v[1])
hcat(elbo_fft.d, elbo.d)
# plot((elbo_fft.d + 1e-6) ./ (elbo.d + 1e-6), "o")
# plot((elbo_fft.h + 1e-6) ./ (elbo.h + 1e-6), "o")




######################################

# Debugging
populate_fsm_vec!(ea_fft, elbo_vars_fft, fsm_vec);

PyPlot.close("all")
b = 5
s = ea.active_sources[1]
sub_image = get_source_pixel_range(s, b, ea);
pix_loc =
    WCS.world_to_pix(ea_fft.images[b].wcs, ea_fft.vp[s][ids.u]) -
    Float64[sub_image.min_h - 1, sub_image.min_w - 1];
plot_loc() = plot(pix_loc[2] - 1, pix_loc[1] - 1,  "ro")

# Display fft image
fs0m_image = Float64[ sf.v[1] for sf in fsm_vec[b].fs0m_image ];
fft_image = Float64[ sf.v[1] for sf in fsm_vec[b].E_G ];
matshow(fft_image); title("FFT image"); colorbar(); plot_loc()
# The FFT fs0m is off-center for some reason
# matshow(fs0m_image); title("fs0m image"); colorbar(); plot_loc()
# matshow(psf_image_vec[b]); plot(25, 25, "ro"); title("PSF")

# Display rendered image
rendered_image = render_source(ea, s, sub_image, false);
matshow(rendered_image); title("Rendered image"); colorbar(); plot_loc()

# Display original image
original_image = show_source_image(ea, s, sub_image);
matshow(original_image); title("Original image"); colorbar(); plot_loc()



# Active pixel map
active_image = show_active_pixels(ea, sub_image, b);
matshow(active_image); title("Active pixels"); plot_loc()

active_image = show_active_pixels(ea_fft, sub_image, b);
matshow(active_image); title("FFT Active pixels"); plot_loc()



# Look at the mcs
star_mcs, gal_mcs =
    load_bvn_mixtures(ea, b,
        calculate_derivs=false,
        calculate_hessian=false);

star_mcs_fft, gal_mcs_fft =
    load_bvn_mixtures(ea_fft, b,
        calculate_derivs=false,
        calculate_hessian=false);

pix_loc
star_mcs_fft[1].the_mean - [ fsm_vec[b].h_lower - 1, fsm_vec[b].w_lower - 1 ]




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



##########
