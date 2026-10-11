# Pinned ONNX Runtime 1.31.0 CPU packages.
#
# The archives ship CMake files that point at include/onnxruntime and, on
# Linux, lib64. Those paths are not in the archives. This module checks the
# real layout and defines onnxruntime::onnxruntime itself.

set(INFEREDGE_ONNXRUNTIME_VERSION "1.31.0")
set(INFEREDGE_ONNXRUNTIME_ROOT "" CACHE PATH
    "Extracted ONNX Runtime 1.31.0 CPU package. Empty downloads the pinned archive.")

set(INFEREDGE_ONNXRUNTIME_URL_BASE
    "https://github.com/microsoft/onnxruntime/releases/download/v${INFEREDGE_ONNXRUNTIME_VERSION}")

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(INFEREDGE_ONNXRUNTIME_ARCHIVE "onnxruntime-osx-arm64-${INFEREDGE_ONNXRUNTIME_VERSION}.tgz")
    set(INFEREDGE_ONNXRUNTIME_SHA256
        "1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc")
    set(INFEREDGE_ONNXRUNTIME_PACKAGE_DIR
        "onnxruntime-osx-arm64-${INFEREDGE_ONNXRUNTIME_VERSION}")
    set(INFEREDGE_ONNXRUNTIME_LIBRARY_RELATIVE "lib/libonnxruntime.1.31.0.dylib")
    set(INFEREDGE_ONNXRUNTIME_SONAME_RELATIVE "lib/libonnxruntime.1.dylib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(INFEREDGE_ONNXRUNTIME_ARCHIVE "onnxruntime-linux-x64-${INFEREDGE_ONNXRUNTIME_VERSION}.tgz")
    set(INFEREDGE_ONNXRUNTIME_SHA256
        "cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57")
    set(INFEREDGE_ONNXRUNTIME_PACKAGE_DIR
        "onnxruntime-linux-x64-${INFEREDGE_ONNXRUNTIME_VERSION}")
    set(INFEREDGE_ONNXRUNTIME_LIBRARY_RELATIVE "lib/libonnxruntime.so.1.31.0")
    set(INFEREDGE_ONNXRUNTIME_SONAME_RELATIVE "lib/libonnxruntime.so.1")
else()
    message(FATAL_ERROR
        "ONNX Runtime ${INFEREDGE_ONNXRUNTIME_VERSION} has no pinned CPU archive for "
        "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}. "
        "Supported platforms are macOS arm64 and Linux x86-64.")
endif()

function(inferedge_require_onnxruntime_layout root)
    set(header "${root}/include/onnxruntime_cxx_api.h")
    set(library "${root}/${INFEREDGE_ONNXRUNTIME_LIBRARY_RELATIVE}")
    set(soname "${root}/${INFEREDGE_ONNXRUNTIME_SONAME_RELATIVE}")
    set(version_file "${root}/VERSION_NUMBER")
    if(NOT EXISTS "${header}")
        message(FATAL_ERROR
            "ONNX Runtime ${INFEREDGE_ONNXRUNTIME_VERSION} is missing ${header}.")
    endif()
    if(NOT EXISTS "${library}")
        message(FATAL_ERROR
            "ONNX Runtime ${INFEREDGE_ONNXRUNTIME_VERSION} is missing ${library}.")
    endif()
    if(NOT EXISTS "${soname}")
        message(FATAL_ERROR
            "ONNX Runtime ${INFEREDGE_ONNXRUNTIME_VERSION} is missing the runtime name ${soname}.")
    endif()
    if(NOT EXISTS "${version_file}")
        message(FATAL_ERROR
            "ONNX Runtime at ${root} has no VERSION_NUMBER file. "
            "Expected the official ${INFEREDGE_ONNXRUNTIME_VERSION} CPU package.")
    endif()
    file(READ "${version_file}" detected_version)
    string(STRIP "${detected_version}" detected_version)
    if(NOT detected_version STREQUAL INFEREDGE_ONNXRUNTIME_VERSION)
        message(FATAL_ERROR
            "ONNX Runtime at ${root} reports version '${detected_version}', "
            "expected ${INFEREDGE_ONNXRUNTIME_VERSION}.")
    endif()
    set(INFEREDGE_ONNXRUNTIME_RESOLVED_ROOT "${root}" PARENT_SCOPE)
endfunction()

if(INFEREDGE_ONNXRUNTIME_ROOT)
    inferedge_require_onnxruntime_layout("${INFEREDGE_ONNXRUNTIME_ROOT}")
else()
    set(INFEREDGE_ONNXRUNTIME_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_deps/onnxruntime-download")
    set(INFEREDGE_ONNXRUNTIME_ARCHIVE_PATH
        "${INFEREDGE_ONNXRUNTIME_DOWNLOAD_DIR}/${INFEREDGE_ONNXRUNTIME_ARCHIVE}")
    set(INFEREDGE_ONNXRUNTIME_URL
        "${INFEREDGE_ONNXRUNTIME_URL_BASE}/${INFEREDGE_ONNXRUNTIME_ARCHIVE}")
    file(MAKE_DIRECTORY "${INFEREDGE_ONNXRUNTIME_DOWNLOAD_DIR}")

    # An archive already at this path is kept when its SHA-256 matches.
    # EXPECTED_HASH rejects a different file and a corrupt download.
    file(DOWNLOAD
        "${INFEREDGE_ONNXRUNTIME_URL}"
        "${INFEREDGE_ONNXRUNTIME_ARCHIVE_PATH}"
        EXPECTED_HASH "SHA256=${INFEREDGE_ONNXRUNTIME_SHA256}"
        TIMEOUT 180
        INACTIVITY_TIMEOUT 30
        STATUS inferedge_onnxruntime_download_status
    )
    list(GET inferedge_onnxruntime_download_status 0 inferedge_onnxruntime_download_code)
    list(GET inferedge_onnxruntime_download_status 1 inferedge_onnxruntime_download_message)
    if(NOT inferedge_onnxruntime_download_code EQUAL 0)
        message(FATAL_ERROR
            "Failed to obtain ONNX Runtime ${INFEREDGE_ONNXRUNTIME_VERSION} "
            "(${INFEREDGE_ONNXRUNTIME_ARCHIVE}).\n"
            "  url: ${INFEREDGE_ONNXRUNTIME_URL}\n"
            "  expected SHA-256: ${INFEREDGE_ONNXRUNTIME_SHA256}\n"
            "  ${inferedge_onnxruntime_download_message}")
    endif()
    file(SHA256 "${INFEREDGE_ONNXRUNTIME_ARCHIVE_PATH}" inferedge_onnxruntime_actual_sha256)
    if(NOT inferedge_onnxruntime_actual_sha256 STREQUAL INFEREDGE_ONNXRUNTIME_SHA256)
        message(FATAL_ERROR
            "ONNX Runtime ${INFEREDGE_ONNXRUNTIME_VERSION} checksum mismatch.\n"
            "  file: ${INFEREDGE_ONNXRUNTIME_ARCHIVE_PATH}\n"
            "  expected SHA-256: ${INFEREDGE_ONNXRUNTIME_SHA256}\n"
            "  actual SHA-256:   ${inferedge_onnxruntime_actual_sha256}")
    endif()

    set(INFEREDGE_ONNXRUNTIME_EXTRACT_DIR "${CMAKE_BINARY_DIR}/_deps/onnxruntime")
    set(INFEREDGE_ONNXRUNTIME_STAMP
        "${INFEREDGE_ONNXRUNTIME_EXTRACT_DIR}/${INFEREDGE_ONNXRUNTIME_PACKAGE_DIR}.sha256")
    set(inferedge_onnxruntime_need_extract TRUE)
    if(EXISTS "${INFEREDGE_ONNXRUNTIME_STAMP}")
        file(READ "${INFEREDGE_ONNXRUNTIME_STAMP}" inferedge_onnxruntime_stamped_sha256)
        string(STRIP "${inferedge_onnxruntime_stamped_sha256}" inferedge_onnxruntime_stamped_sha256)
        if(inferedge_onnxruntime_stamped_sha256 STREQUAL INFEREDGE_ONNXRUNTIME_SHA256
            AND EXISTS "${INFEREDGE_ONNXRUNTIME_EXTRACT_DIR}/${INFEREDGE_ONNXRUNTIME_PACKAGE_DIR}/include/onnxruntime_cxx_api.h")
            set(inferedge_onnxruntime_need_extract FALSE)
        endif()
    endif()
    if(inferedge_onnxruntime_need_extract)
        file(ARCHIVE_EXTRACT
            INPUT "${INFEREDGE_ONNXRUNTIME_ARCHIVE_PATH}"
            DESTINATION "${INFEREDGE_ONNXRUNTIME_EXTRACT_DIR}"
        )
        file(WRITE "${INFEREDGE_ONNXRUNTIME_STAMP}" "${INFEREDGE_ONNXRUNTIME_SHA256}\n")
    endif()
    inferedge_require_onnxruntime_layout(
        "${INFEREDGE_ONNXRUNTIME_EXTRACT_DIR}/${INFEREDGE_ONNXRUNTIME_PACKAGE_DIR}")
    message(STATUS
        "ONNX Runtime ${INFEREDGE_ONNXRUNTIME_VERSION} CPU archive verified "
        "(${INFEREDGE_ONNXRUNTIME_ARCHIVE})")
endif()

set(INFEREDGE_ONNXRUNTIME_INCLUDE_DIR "${INFEREDGE_ONNXRUNTIME_RESOLVED_ROOT}/include")
set(INFEREDGE_ONNXRUNTIME_LIBRARY
    "${INFEREDGE_ONNXRUNTIME_RESOLVED_ROOT}/${INFEREDGE_ONNXRUNTIME_LIBRARY_RELATIVE}")
set(INFEREDGE_ONNXRUNTIME_LIB_DIR "${INFEREDGE_ONNXRUNTIME_RESOLVED_ROOT}/lib")

if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${INFEREDGE_ONNXRUNTIME_LIBRARY}"
        IMPORTED_NO_SONAME FALSE
    )
    target_include_directories(onnxruntime::onnxruntime SYSTEM INTERFACE
        "${INFEREDGE_ONNXRUNTIME_INCLUDE_DIR}"
    )
endif()
