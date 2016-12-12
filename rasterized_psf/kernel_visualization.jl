using PyPlot
using Celeste
using Celeste.DeterministicVIImagePSF

###########################
# Take a look at the numerics of the sinc filter

x_vals = -2:0.001:2;
deriv_level = 1
lanczos_vals = Float64[
    DeterministicVIImagePSF.lanczos_kernel_with_derivatives(x, 2.0)[deriv_level]
    for x in x_vals ];
cubic_vals = Float64[
    DeterministicVIImagePSF.cubic_kernel_with_derivatives(x, 0.75)[deriv_level]
    for x in x_vals ];
spline_vals = Float64[
    DeterministicVIImagePSF.bspline_kernel_with_derivatives(x)[deriv_level]
    for x in x_vals ];
PyPlot.close("all")
plot(collect(x_vals), lanczos_vals, "k.")
plot(collect(x_vals), cubic_vals, "b.")
plot(collect(x_vals), spline_vals, "r.")



PyPlot.close("all")
deriv_level = 2
for cubic_a in Float64[ -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
    cubic_vals = Float64[
        DeterministicVIImagePSF.cubic_kernel_with_derivatives(x, cubic_a)[deriv_level]
        for x in x_vals ];
    PyPlot.figure()
    plot(collect(x_vals), cubic_vals, "b."); title(cubic_a)
end



# okok 