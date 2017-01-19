function print_test_names(extensions)
    for test_case_index in 1:div(length(extensions), 5)
        first_band_index = (test_case_index - 1) * 5 + 1
        println(test_case_index, " ",
                extensions[first_band_index].header["CLDESCR"])
    end
end


function load_test_case(test_case_index::Int)
    first_band_index = (test_case_index - 1) * 5 + 1
    header = extensions[first_band_index].header
    this_test_case_name = header["CLDESCR"]
    println("Running test case '$this_test_case_name'")
    iota = header["CLIOTA"]
    psf = make_psf(header["CLSIGMA"])
    n_sources = header["CLNSRC"]

    band_pixels = [
        extensions[index].pixels for index in first_band_index:(first_band_index + 4)
    ];
    assert_counts_match_expected_flux(band_pixels, header, iota)

    images = make_images(band_pixels, psf, wcs, header["CLSKY"], iota);
    catalog = make_catalog(header);

    target_sources = [1,]
    neighbor_map = Infer.find_neighbors(target_sources, catalog, images);

    catalog, target_sources, neighbor_map, images, header
end