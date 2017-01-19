
function sum_nonnan(x)
    sum(x[!isnan(x)])
end


function condition_number(mat)
    ev = eigvals(mat);
    maximum(ev) / minimum(abs(ev))
end


function x_to_vp(x, ea, transform)
    reshaped_x = reshape(x, length(UnconstrainedParams), length(ea.active_sources));
    Transform.array_to_vp!(
        transform, reshaped_x, ea.vp,
        collect(1:length(UnconstrainedParams)));
    return deepcopy(ea.vp)
end


function render_optimization_steps(
    ea::Celeste.DeterministicVI.ElboArgs,
    fsm_mat::Vector{Celeste.DeterministicVIImagePSF.FSMSensitiveFloatMatrices},
    nm_result::Optim.MultivariateOptimizationResults,
    transform::Celeste.Transform.DataTransform,
    s::Int, b::Int)
    
    local x_array = [ deepcopy(tr.metadata["x"]) for tr in nm_result.trace ]
    local vp_array = [ x_to_vp(x, ea, transform) for x in x_array ]
    
    local images = []
    local images_star = []
    local images_gal = []
    for iter in 1:length(vp_array)
        println("Rendering iteration ", iter, " of ", length(vp_array))
        vp_loc = deepcopy(vp_array[iter])

        ea_fft.vp = vp_loc
        push!(images, CelesteEDA.render_source_fft(ea, fsm_mat, s, b))

        vp_loc[s][ids.a] = [ 1, 0 ]
        ea_fft.vp = vp_loc
        push!(images_star, CelesteEDA.render_source_fft(ea, fsm_mat, s, b))

        vp_loc[s][ids.a] = [ 0, 1 ]
        ea_fft.vp = vp_loc
        push!(images_gal, CelesteEDA.render_source_fft(ea, fsm_mat, s, b))
    end

    return images, images_star, images_gal, vp_array
end


function render_optimization_steps(
    ea::Celeste.DeterministicVI.ElboArgs,
    nm_result::Optim.MultivariateOptimizationResults,
    transform::Celeste.Transform.DataTransform,
    s::Int, b::Int)
    
    local x_array = [ deepcopy(tr.metadata["x"]) for tr in nm_result.trace ]
    local vp_array = [ x_to_vp(x, ea, transform) for x in x_array ]
    
    local images = []
    local images_star = []
    local images_gal = []
    for iter in 1:length(vp_array)
        println("Rendering iteration ", iter, " of ", length(vp_array))
        vp_loc = deepcopy(vp_array[iter])

        ea.vp = vp_loc
        push!(images, CelesteEDA.render_source(ea, s, b))

        vp_loc[s][ids.a] = [ 1, 0 ]
        ea.vp = vp_loc
        push!(images_star, CelesteEDA.render_source(ea, s, b))

        vp_loc[s][ids.a] = [ 0, 1 ]
        ea.vp = vp_loc
        push!(images_gal, CelesteEDA.render_source(ea, s, b))
    end

    return images, images_star, images_gal, vp_array
end


function show_images(img_array, sleep_duration=0.03)
    PyPlot.close("all")
    PyPlot.hold(false)
    max_pix = 1.1 * maximum([ maximum(img) for img in img_array ])
    min_pix = minimum([ minimum(img) for img in img_array ])
    for iter in 1:length(img_array)
        im = img_array[iter]
        imshow(im, interpolation="nearest", vmin=min_pix, vmax=max_pix)
        if iter == 1
            colorbar()
        end
        title("Iteration " * string(iter))
        sleep(sleep_duration)
        PyPlot.draw()
    end
end
