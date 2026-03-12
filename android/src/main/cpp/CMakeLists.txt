cmake_minimum_required(VERSION 3.22.1...4.1)
project("local-llm-rn" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows/MSVC: suppress high-volume conversion/sign warnings from vendor code.
if(MSVC)
    add_compile_options(
        /wd4018 # signed/unsigned mismatch
        /wd4101 # unreferenced local variable
        /wd4244 # narrowing conversion
        /wd4267 # size_t -> smaller type
        /wd4305 # truncation to smaller floating type
    )
endif()

# ABI-specific settings (from engine's llama.android example)
if(DEFINED ANDROID_ABI)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI ON)
        set(GGML_OPENMP ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    endif()
endif()

# Engine source resolution:
# - published package / prepared repo: repo-root cpp/
# - local development fallback: vendor/hilum-local-llm-engine/
set(LLAMA_SRC_PACKAGE ${CMAKE_CURRENT_LIST_DIR}/../../../../cpp)
set(LLAMA_SRC_VENDOR ${CMAKE_CURRENT_LIST_DIR}/../../../../vendor/hilum-local-llm-engine)

if(EXISTS ${LLAMA_SRC_PACKAGE}/CMakeLists.txt)
    set(LLAMA_SRC ${LLAMA_SRC_PACKAGE})
elseif(EXISTS ${LLAMA_SRC_VENDOR}/CMakeLists.txt)
    set(LLAMA_SRC ${LLAMA_SRC_VENDOR})
else()
    message(FATAL_ERROR
        "Engine source not found. Expected either ${LLAMA_SRC_PACKAGE} or ${LLAMA_SRC_VENDOR} "
        "to contain CMakeLists.txt. Run scripts/prepare.sh before building from the repo root.")
endif()

set(HILUM_BUILD_LIB ON CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_SRC} build-llama)

add_library(${CMAKE_PROJECT_NAME} SHARED LocalLLM.cpp)

# Include both dev paths (tools/mtmd) and published paths (mtmd)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${LLAMA_SRC}/include
    ${LLAMA_SRC}/ggml/include
    ${LLAMA_SRC}/ggml/src
    ${LLAMA_SRC}/src
    ${LLAMA_SRC}/common
    ${LLAMA_SRC}/hilum
    ${LLAMA_SRC}/tools/mtmd
    ${LLAMA_SRC}/mtmd
    ${LLAMA_SRC}/vendor
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    hilum
    llama
    common
    android
    log
)
