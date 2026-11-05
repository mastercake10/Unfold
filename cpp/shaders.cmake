# Compiles the GPU SIFT shaders (src/shaders/*.comp) to SPIR-V included by gpu_sift.cpp.
set(DOCSCAN_SHADER_DIR ${CMAKE_CURRENT_LIST_DIR}/src/shaders)
set(DOCSCAN_SHADERS upscale blur downsample extrema prep_indirect orientation descriptor)

function(docscan_add_shaders target glslc)
  set(out_dir ${CMAKE_CURRENT_BINARY_DIR}/spv)
  file(MAKE_DIRECTORY ${out_dir})
  set(outs)
  foreach(s ${DOCSCAN_SHADERS})
    set(src ${DOCSCAN_SHADER_DIR}/${s}.comp)
    set(o ${out_dir}/${s}.spv.inc)
    add_custom_command(OUTPUT ${o}
      COMMAND ${glslc} --target-env=vulkan1.1 -O -mfmt=num ${src} -o ${o}
      DEPENDS ${src} ${DOCSCAN_SHADER_DIR}/common.glsl VERBATIM)
    list(APPEND outs ${o})
  endforeach()
  target_sources(${target} PRIVATE ${outs})
  target_include_directories(${target} PRIVATE ${out_dir})
endfunction()
