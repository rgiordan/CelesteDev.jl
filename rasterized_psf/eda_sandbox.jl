
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
    fsm_vec::Vector{Celeste.DeterministicVIImagePSF.FSMSensitiveFloatMatrices},
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
        push!(images, CelesteEDA.render_source_fft(ea, fsm_vec, s, b))

        vp_loc[s][ids.a] = [ 1, 0 ]
        ea_fft.vp = vp_loc
        push!(images_star, CelesteEDA.render_source_fft(ea, fsm_vec, s, b))

        vp_loc[s][ids.a] = [ 0, 1 ]
        ea_fft.vp = vp_loc
        push!(images_gal, CelesteEDA.render_source_fft(ea, fsm_vec, s, b))
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


function show_images(img_array)
    iter = 0
    max_pix = maximum([ maximum(img) for img in img_array ])
    PyPlot.close("all")
    for img in img_array
        PyPlot.close("all")
        iter = iter + 1
        matshow(img, vmin=0, vmax=max_pix);
        title("rendered " * string(iter))
        waitforbuttonpress()
    end
end

