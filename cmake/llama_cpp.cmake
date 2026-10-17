# Optional local llama.cpp prefix. Configure does not download this project.
#
# The provenance file records the operator's verified build. It is not a
# cryptographic signature and does not prove archive authenticity by itself.

set(INFEREDGE_LLAMA_CPP_TAG "v0.7.0")
set(INFEREDGE_LLAMA_CPP_COMMIT "620d2225cc436ab0b001e3200e590e3b286ef9ab")
set(INFEREDGE_LLAMA_CPP_ARCHIVE_SHA256
    "55d67acf3ad2ad747a9bbeb75f9f0e173dbcc0d1753a471038a90c819421c5e1")
set(INFEREDGE_LLAMA_CPP_ROOT "" CACHE PATH
    "Locally installed llama.cpp ${INFEREDGE_LLAMA_CPP_TAG} prefix. Required when generative inference is ON.")

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    message(FATAL_ERROR
        "INFEREDGE_ENABLE_GENERATIVE is CPU arm64 only. "
        "This host reports '${CMAKE_SYSTEM_PROCESSOR}'.")
endif()

if(NOT INFEREDGE_LLAMA_CPP_ROOT)
    message(FATAL_ERROR
        "INFEREDGE_ENABLE_GENERATIVE=ON requires INFEREDGE_LLAMA_CPP_ROOT to be a "
        "canonical local llama.cpp install prefix. Configure does not download llama.cpp.")
endif()

function(inferedge_reject_symlink_components path)
    if(IS_SYMLINK "${path}")
        message(FATAL_ERROR "llama.cpp path is a symlink: ${path}")
    endif()
    cmake_path(GET path ROOT_PATH cursor)
    cmake_path(GET path RELATIVE_PART relative)
    if(relative STREQUAL "")
        return()
    endif()
    cmake_path(NATIVE_PATH relative native_relative)
    string(REPLACE "\\" "/" native_relative "${native_relative}")
    string(REPLACE "/" ";" parts "${native_relative}")
    foreach(part IN LISTS parts)
        if(part STREQUAL "" OR part STREQUAL "." OR part STREQUAL "..")
            message(FATAL_ERROR "llama.cpp path is not canonical: ${path}")
        endif()
        cmake_path(APPEND cursor "${part}")
        if(IS_SYMLINK "${cursor}")
            message(FATAL_ERROR "llama.cpp path has a symlink component: ${cursor}")
        endif()
    endforeach()
endfunction()

cmake_path(ABSOLUTE_PATH INFEREDGE_LLAMA_CPP_ROOT NORMALIZE OUTPUT_VARIABLE
    INFEREDGE_LLAMA_CPP_ABS)
file(REAL_PATH "${INFEREDGE_LLAMA_CPP_ABS}" INFEREDGE_LLAMA_CPP_REAL)
if(NOT INFEREDGE_LLAMA_CPP_ABS STREQUAL INFEREDGE_LLAMA_CPP_REAL)
    message(FATAL_ERROR
        "INFEREDGE_LLAMA_CPP_ROOT must be a canonical absolute path without symlink components.\n"
        "  given:     ${INFEREDGE_LLAMA_CPP_ROOT}\n"
        "  absolute:  ${INFEREDGE_LLAMA_CPP_ABS}\n"
        "  realpath:  ${INFEREDGE_LLAMA_CPP_REAL}")
endif()
inferedge_reject_symlink_components("${INFEREDGE_LLAMA_CPP_ABS}")

set(INFEREDGE_LLAMA_CPP_PROVENANCE
    "${INFEREDGE_LLAMA_CPP_ABS}/share/inferedge/llama-cpp-provenance.txt")
if(IS_SYMLINK "${INFEREDGE_LLAMA_CPP_PROVENANCE}")
    message(FATAL_ERROR "llama.cpp provenance file is a symlink: ${INFEREDGE_LLAMA_CPP_PROVENANCE}")
endif()
if(NOT EXISTS "${INFEREDGE_LLAMA_CPP_PROVENANCE}")
    message(FATAL_ERROR
        "Missing operator provenance file:\n  ${INFEREDGE_LLAMA_CPP_PROVENANCE}\n"
        "The file records a verified local llama.cpp build. It is not a cryptographic signature.")
endif()
if(IS_DIRECTORY "${INFEREDGE_LLAMA_CPP_PROVENANCE}")
    message(FATAL_ERROR "llama.cpp provenance path is a directory.")
endif()
# Raw-byte gate. file(READ) without HEX normalizes CRLF and must not be used.
set(_inferedge_expected_provenance_size 252)
set(_inferedge_expected_provenance_hex
    "736368656d615f76657273696f6e3d310a70726f6a6563743d6c6c616d612e6370700a7461673d76302e372e300a636f6d6d69743d363230643232323563633433366162306230303165333230306535393065336232383665663961620a736f757263655f617263686976655f7368613235363d353564363761636633616432616437343761396262656237356639663065313733646263633064313735336134373130333861393063383139343231633565310a6275696c645f747970653d52656c656173650a6172636869746563747572653d61726d36340a6261636b656e643d6370750a6d6574616c3d6f66660a7365727665723d6f66660a")
file(SIZE "${INFEREDGE_LLAMA_CPP_PROVENANCE}" INFEREDGE_LLAMA_CPP_PROVENANCE_SIZE)
if(NOT INFEREDGE_LLAMA_CPP_PROVENANCE_SIZE EQUAL _inferedge_expected_provenance_size)
    message(FATAL_ERROR
        "llama.cpp provenance file was rejected. Extra keys, missing keys, duplicate keys, "
        "whitespace, CR/CRLF, or wrong pin values are not accepted.\n"
        "  file: ${INFEREDGE_LLAMA_CPP_PROVENANCE}\n"
        "The provenance file is operator-attested and is not a cryptographic signature.")
endif()
file(READ "${INFEREDGE_LLAMA_CPP_PROVENANCE}" INFEREDGE_LLAMA_CPP_PROVENANCE_HEX HEX)
if(NOT INFEREDGE_LLAMA_CPP_PROVENANCE_HEX STREQUAL _inferedge_expected_provenance_hex)
    message(FATAL_ERROR
        "llama.cpp provenance file was rejected. Extra keys, missing keys, duplicate keys, "
        "whitespace, CR/CRLF, or wrong pin values are not accepted.\n"
        "  file: ${INFEREDGE_LLAMA_CPP_PROVENANCE}\n"
        "The provenance file is operator-attested and is not a cryptographic signature.")
endif()

set(INFEREDGE_LLAMA_CPP_HEADER "${INFEREDGE_LLAMA_CPP_ABS}/include/llama.h")
if(NOT EXISTS "${INFEREDGE_LLAMA_CPP_HEADER}")
    message(FATAL_ERROR "llama.cpp header is missing: ${INFEREDGE_LLAMA_CPP_HEADER}")
endif()

set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
find_package(llama CONFIG REQUIRED
    PATHS
        "${INFEREDGE_LLAMA_CPP_ABS}"
        "${INFEREDGE_LLAMA_CPP_ABS}/lib/cmake/llama"
        "${INFEREDGE_LLAMA_CPP_ABS}/lib/cmake"
    NO_DEFAULT_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_PACKAGE_REGISTRY
)

foreach(_inferedge_required_target llama ggml::ggml ggml::ggml-base ggml::ggml-cpu)
    if(NOT TARGET ${_inferedge_required_target})
        message(FATAL_ERROR
            "llama.cpp prefix is missing required CMake target '${_inferedge_required_target}'.")
    endif()
    set_property(TARGET ${_inferedge_required_target} PROPERTY INTERFACE_COMPILE_FEATURES)
endforeach()
if(GGML_METAL OR GGML_CUDA OR GGML_VULKAN OR GGML_BLAS)
    message(FATAL_ERROR
        "llama.cpp prefix was built with a disallowed backend "
        "(Metal=${GGML_METAL} CUDA=${GGML_CUDA} Vulkan=${GGML_VULKAN} BLAS=${GGML_BLAS}).")
endif()

foreach(_inferedge_forbidden_target llama-server server httplib mtmd llama-mtmd)
    if(TARGET ${_inferedge_forbidden_target})
        message(FATAL_ERROR
            "llama.cpp prefix exposes unexpected target '${_inferedge_forbidden_target}'. "
            "Rebuild with server/HTTP support disabled.")
    endif()
endforeach()

get_target_property(_inferedge_llama_configs llama IMPORTED_CONFIGURATIONS)
if(NOT _inferedge_llama_configs)
    set(_inferedge_llama_configs RELEASE)
endif()
set(_inferedge_llama_lib "")
foreach(_inferedge_cfg IN LISTS _inferedge_llama_configs)
    get_target_property(_inferedge_loc llama IMPORTED_LOCATION_${_inferedge_cfg})
    if(_inferedge_loc)
        set(_inferedge_llama_lib "${_inferedge_loc}")
        break()
    endif()
endforeach()
if(NOT _inferedge_llama_lib)
    get_target_property(_inferedge_llama_lib llama IMPORTED_LOCATION)
endif()
if(NOT _inferedge_llama_lib)
    message(FATAL_ERROR "llama.cpp target 'llama' has no imported library location.")
endif()
cmake_path(GET _inferedge_llama_lib PARENT_PATH INFEREDGE_LLAMA_CPP_LIB_DIR)
string(FIND "${INFEREDGE_LLAMA_CPP_LIB_DIR}" "${INFEREDGE_LLAMA_CPP_ABS}/" _inferedge_lib_prefix)
if(NOT _inferedge_lib_prefix EQUAL 0)
    message(FATAL_ERROR
        "llama.cpp library is outside the verified prefix.\n"
        "  library: ${_inferedge_llama_lib}\n"
        "  prefix:  ${INFEREDGE_LLAMA_CPP_ABS}")
endif()

add_library(inferedge_llama_cpp INTERFACE)
target_link_libraries(inferedge_llama_cpp INTERFACE llama)
target_include_directories(inferedge_llama_cpp SYSTEM INTERFACE "${INFEREDGE_LLAMA_CPP_ABS}/include")
set(INFEREDGE_LLAMA_CPP_RESOLVED_ROOT "${INFEREDGE_LLAMA_CPP_ABS}")

message(STATUS
    "Using operator-verified llama.cpp ${INFEREDGE_LLAMA_CPP_TAG} at ${INFEREDGE_LLAMA_CPP_ABS}. "
    "Provenance is not a cryptographic signature.")
