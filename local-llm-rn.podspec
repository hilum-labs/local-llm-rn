Pod::Spec.new do |s|
  s.name         = "local-llm-rn"
  s.version      = "0.1.0"
  s.summary      = "Local LLM inference for React Native"
  s.homepage     = "https://github.com/hilum-labs/local-llm-rn"
  s.license      = { :type => "MIT", :file => "LICENSE" }
  s.author       = "Hilum Labs"
  s.source       = { :git => "https://github.com/hilum-labs/local-llm-rn.git", :tag => s.version.to_s }
  s.platforms    = { :ios => "16.0" }

  # Only the RN bridge (ios/) needs ARC; ggml Metal .m files use manual retain/release
  s.requires_arc = ["ios/**/*.{h,mm}"]

  s.source_files = [
    "ios/**/*.{h,mm,cpp}",
    "cpp/src/**/*.{c,cpp,h}",
    "cpp/ggml/src/**/*.{c,cpp,h,m,metal}",
    "cpp/include/**/*.h",
    "cpp/ggml/include/**/*.h",
    "cpp/common/**/*.{cpp,h}",
    "cpp/hilum/**/*.{c,cpp,h}",
    "cpp/mtmd/**/*.{cpp,h}",
    "cpp/vendor/**/*.{h,hpp}",
  ]

  s.exclude_files = [
    "cpp/ggml/src/ggml-cuda/**",
    "cpp/ggml/src/ggml-vulkan/**",
    "cpp/ggml/src/ggml-sycl/**",
    "cpp/ggml/src/ggml-cann/**",
    "cpp/ggml/src/ggml-hip/**",
    "cpp/ggml/src/ggml-kompute/**",
    "cpp/ggml/src/ggml-opencl/**",
    "cpp/ggml/src/ggml-webgpu/**",
    "cpp/ggml/src/ggml-hexagon/**",
    "cpp/ggml/src/ggml-musa/**",
    "cpp/ggml/src/ggml-zdnn/**",
    "cpp/ggml/src/ggml-zendnn/**",
    "cpp/ggml/src/ggml-rpc/**",
    "cpp/ggml/src/ggml-virtgpu/**",
    "cpp/ggml/src/ggml-cpu/kleidiai/*.cpp",
    "cpp/ggml/src/ggml-cpu/spacemit/*.cpp",
    "cpp/ggml/src/ggml-cpu/amx/*.cpp",
    "cpp/ggml/src/ggml-cpu/arch/loongarch/*.{c,cpp}",
    "cpp/ggml/src/ggml-cpu/arch/riscv/*.{c,cpp}",
    "cpp/ggml/src/ggml-cpu/arch/powerpc/*.{c,cpp}",
    "cpp/ggml/src/ggml-cpu/arch/s390/*.{c,cpp}",
    "cpp/ggml/src/ggml-cpu/arch/wasm/*.{c,cpp}",
    "cpp/mtmd/mtmd-cli.cpp",
  ]

  s.pod_target_xcconfig = {
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
    "GCC_PREPROCESSOR_DEFINITIONS" => [
      "GGML_METAL=1",
      "GGML_METAL_EMBED_LIBRARY=0",
      "GGML_METAL_USE_BF16=1",
      "GGML_USE_METAL=1",
      "GGML_BLAS=1",
      "GGML_BLAS_USE_ACCELERATE=1",
      "NDEBUG=1",
      "HILUM_BUILD=1",
      'GGML_VERSION=\"0.9.7\"',
      'GGML_COMMIT=\"unknown\"',
      "LLAMA_BUILD_TESTS=OFF",
      "LLAMA_BUILD_EXAMPLES=OFF",
      "LLAMA_BUILD_SERVER=OFF",
    ].join(" "),
      "HEADER_SEARCH_PATHS" => [
      "$(PODS_TARGET_SRCROOT)/cpp/include",
      "$(PODS_TARGET_SRCROOT)/cpp/ggml/include",
      "$(PODS_TARGET_SRCROOT)/cpp/ggml/src",
      "$(PODS_TARGET_SRCROOT)/cpp/src",
      "$(PODS_TARGET_SRCROOT)/cpp/common",
      "$(PODS_TARGET_SRCROOT)/cpp/hilum",
      "$(PODS_TARGET_SRCROOT)/cpp/mtmd",
      "$(PODS_TARGET_SRCROOT)/cpp/vendor",
    ].join(" "),
    "OTHER_CFLAGS" => "-O3 -DACCELERATE_NEW_LAPACK",
    "MTL_HEADER_SEARCH_PATHS" => "$(PODS_TARGET_SRCROOT)/cpp/ggml/src",
  }

  s.frameworks = ["Metal", "MetalKit", "Foundation", "Accelerate"]

  install_modules_dependencies(s)
end
