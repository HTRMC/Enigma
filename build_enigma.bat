@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d C:\Users\HTRMC\Dev\Projects\Enigma\build
"C:\Users\HTRMC\AppData\Local\Programs\CLion\bin\ninja\win\x64\ninja.exe" Enigma enigma-mpbake gltf_ingest_test cluster_builder_test simplify_test dag_builder_test page_writer_test mp_asset_reader_test residency_lru_test async_io_worker_test page_cache_test request_queue_test micropoly_streaming_test cold_cache_stress_test throughput_bench micropoly_capability_test spirv_diff_test screenshot_diff_test mp_vis_pack_compile_test micropoly_cull_test micropoly_raster_test micropoly_sw_raster_bin_test micropoly_blas_manager_test micropoly_settings_panel_test
echo EXIT_CODE=%ERRORLEVEL%
