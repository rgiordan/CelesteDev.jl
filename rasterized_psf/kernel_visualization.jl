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


PyPlot.close("all")
cubic_a = -1.7
for deriv_level in 1:3
    cubic_vals = Float64[
        DeterministicVIImagePSF.cubic_kernel_with_derivatives(x, cubic_a)[deriv_level]
        for x in x_vals ];
    PyPlot.figure()
    plot(collect(x_vals), cubic_vals, "b."); title(cubic_a)    
end

kernel_fun = x -> DeterministicVIImagePSF.cubic_kernel_with_derivatives(x, cubic_a)[1]
# kernel_fun = x -> DeterministicVIImagePSF.bspline_kernel_with_derivatives(x)[1]
# kernel_fun = x -> DeterministicVIImagePSF.lanczos_kernel_with_derivatives(x, 2.)[1]

kernel_width = 2
for x in 0:0.01:1
    x0_vals = (-kernel_width:(kernel_width - 1)) + x
    k_vec = [ kernel_fun(x0) for x0 in x0_vals ]
    println(collect(x0_vals), " ", k_vec, " ", sum(k_vec))
end

DeterministicVIImagePSF.cubic_kernel_with_derivatives(1.999, 0.2)
DeterministicVIImagePSF.cubic_kernel_with_derivatives(1.999, -0.2)
DeterministicVIImagePSF.bspline_kernel_with_derivatives(1.999)

