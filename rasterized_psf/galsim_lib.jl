function print_test_names(extensions)
    for test_case_index in 1:div(length(extensions), 5)
        first_band_index = (test_case_index - 1) * 5 + 1
        println(test_case_index, " ",
                extensions[first_band_index].header["CL_DESCR"])
    end
end


function load_test_case(test_case_index::Int)
    first_band_index = (test_case_index - 1) * 5 + 1
    header = extensions[first_band_index].header
    this_test_case_name = header["CL_DESCR"]
    println("Running test case '$this_test_case_name'")

    iota = header["CL_IOTA"]

    band_pixels = [
        extensions[index].pixels for index in first_band_index:(first_band_index + 4)
    ];
    assert_counts_match_expected_flux(band_pixels, header, iota);
    psf = make_psf(header["CL_SIGMA"])
    images = make_images(band_pixels, psf, wcs, header["CL_SKY"], iota);
    catalog_entries = make_catalog_entries(header);

    target_source = 1
    target_entry = catalog_entries[target_source]

    neighbor_map = Infer.find_neighbors([ target_source ], catalog_entries, images);
    cat_local = vcat(target_entry, catalog_entries[neighbor_map[target_source]]);
    vp = Vector{Float64}[init_source(ce) for ce in cat_local];
    patches = Infer.get_sky_patches(images, cat_local);
    Infer.load_active_pixels!(images, patches);
    
    images, patches, vp, header
end

