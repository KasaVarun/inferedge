# Pinned ONNX Runtime 1.31.0 from the official Microsoft NuGet package.
#
# The archives ship CMake files that point at include/onnxruntime and, on
# Linux, lib64. Those paths are not in the archives. This module checks the
# real layout and defines onnxruntime::onnxruntime itself.

set(INFEREDGE_ONNXRUNTIME_VERSION "1.31.0")
set(INFEREDGE_ONNXRUNTIME_ROOT "" CACHE PATH
    "Extracted ONNX Runtime 1.31.0 CPU package. Empty downloads the pinned archive.")

set(INFEREDGE_ONNXRUNTIME_OFFICIAL_NUPKG_ID "Microsoft.ML.OnnxRuntime")
set(INFEREDGE_ONNXRUNTIME_OFFICIAL_NUPKG_VERSION "1.31.0")
set(INFEREDGE_ONNXRUNTIME_NUPKG_ARCHIVE
    "microsoft.ml.onnxruntime.${INFEREDGE_ONNXRUNTIME_OFFICIAL_NUPKG_VERSION}.nupkg")
set(INFEREDGE_ONNXRUNTIME_NUPKG_SHA256
    "7e35fdb4f5adb62f01e60798e2e5ed05108f5fbbe1cca03fbcc37903cf7c5e3b")
set(INFEREDGE_ONNXRUNTIME_OFFICIAL_DOWNLOAD_URL
    "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${INFEREDGE_ONNXRUNTIME_OFFICIAL_NUPKG_VERSION}/${INFEREDGE_ONNXRUNTIME_NUPKG_ARCHIVE}")

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(INFEREDGE_ONNXRUNTIME_ARCHIVE "${INFEREDGE_ONNXRUNTIME_NUPKG_ARCHIVE}")
    set(INFEREDGE_ONNXRUNTIME_SHA256 "${INFEREDGE_ONNXRUNTIME_NUPKG_SHA256}")
    set(INFEREDGE_ONNXRUNTIME_PACKAGE_DIR
        "onnxruntime-osx-arm64-${INFEREDGE_ONNXRUNTIME_VERSION}")
    set(INFEREDGE_ONNXRUNTIME_LIBRARY_RELATIVE "lib/libonnxruntime.1.31.0.dylib")
    set(INFEREDGE_ONNXRUNTIME_SONAME_RELATIVE "lib/libonnxruntime.1.dylib")
    set(INFEREDGE_ONNXRUNTIME_NUPKG_NATIVE_RELATIVE
        "runtimes/osx-arm64/native/libonnxruntime.dylib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(INFEREDGE_ONNXRUNTIME_ARCHIVE "${INFEREDGE_ONNXRUNTIME_NUPKG_ARCHIVE}")
    set(INFEREDGE_ONNXRUNTIME_SHA256 "${INFEREDGE_ONNXRUNTIME_NUPKG_SHA256}")
    set(INFEREDGE_ONNXRUNTIME_PACKAGE_DIR
        "onnxruntime-linux-x64-${INFEREDGE_ONNXRUNTIME_VERSION}")
    set(INFEREDGE_ONNXRUNTIME_LIBRARY_RELATIVE "lib/libonnxruntime.so.1.31.0")
    set(INFEREDGE_ONNXRUNTIME_SONAME_RELATIVE "lib/libonnxruntime.so.1")
    set(INFEREDGE_ONNXRUNTIME_NUPKG_NATIVE_RELATIVE
        "runtimes/linux-x64/native/libonnxruntime.so")
else()
    message(FATAL_ERROR
        "ONNX Runtime ${INFEREDGE_ONNXRUNTIME_VERSION} has no pinned CPU archive for "
        "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}. "
        "Supported platforms are macOS arm64 and Linux x86-64.")
endif()

# inferedge:onnxruntime-download-helpers-begin
# Bounded retry contract for the official pinned archive download only.
set(INFEREDGE_ONNXRUNTIME_DOWNLOAD_MAX_ATTEMPTS 3)
set(INFEREDGE_ONNXRUNTIME_DOWNLOAD_ATTEMPT_TIMEOUT_SECONDS 180)
set(INFEREDGE_ONNXRUNTIME_DOWNLOAD_ATTEMPT_INACTIVITY_TIMEOUT_SECONDS 30)
set(INFEREDGE_ONNXRUNTIME_DOWNLOAD_RETRY_DELAY_AFTER_ATTEMPT_1_SECONDS 2)
set(INFEREDGE_ONNXRUNTIME_DOWNLOAD_RETRY_DELAY_AFTER_ATTEMPT_2_SECONDS 5)
set(INFEREDGE_ONNXRUNTIME_DOWNLOAD_LOCK_TIMEOUT_SECONDS 900)
set(INFEREDGE_ONNXRUNTIME_DOWNLOAD_LOCK_BASENAME ".inferedge-onnxruntime-download.lock")
set(INFEREDGE_ONNXRUNTIME_DOWNLOAD_PART_SUFFIX ".download.part")

function(inferedge_onnxruntime_is_verified_regular_file path out_yes)
    set(is_verified FALSE)
    if(EXISTS "${path}" AND NOT IS_SYMLINK "${path}" AND NOT IS_DIRECTORY "${path}")
        set(is_verified TRUE)
    endif()
    set(${out_yes} "${is_verified}" PARENT_SCOPE)
endfunction()

function(inferedge_onnxruntime_sha256_of_file path out_sha256)
    file(SHA256 "${path}" inferedge_onnxruntime_computed_sha256)
    set(${out_sha256} "${inferedge_onnxruntime_computed_sha256}" PARENT_SCOPE)
endfunction()

function(inferedge_onnxruntime_validate_verified_archive archive_path expected_sha256 out_valid)
    set(valid FALSE)
    inferedge_onnxruntime_is_verified_regular_file("${archive_path}" is_regular)
    if(is_regular)
        inferedge_onnxruntime_sha256_of_file("${archive_path}" actual_sha256)
        if(actual_sha256 STREQUAL expected_sha256)
            set(valid TRUE)
        endif()
    endif()
    set(${out_valid} "${valid}" PARENT_SCOPE)
endfunction()

function(inferedge_onnxruntime_remove_path_if_present path)
    if(EXISTS "${path}")
        file(REMOVE "${path}")
    endif()
endfunction()

function(inferedge_onnxruntime_sleep_seconds seconds)
    if(seconds GREATER 0)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep "${seconds}")
    endif()
endfunction()

function(inferedge_onnxruntime_classify_download_failure status_code status_message log_text
        out_failure_class out_retryable)
    set(failure_class "transport_error")
    set(retryable TRUE)
    string(TOLOWER "${status_message} ${log_text}" inferedge_onnxruntime_failure_blob)

    if(status_code EQUAL 0)
        set(failure_class "success")
        set(retryable FALSE)
    elseif(inferedge_onnxruntime_failure_blob MATCHES "expected hash"
        OR inferedge_onnxruntime_failure_blob MATCHES "checksum"
        OR inferedge_onnxruntime_failure_blob MATCHES "sha-256")
        set(failure_class "checksum_mismatch")
        set(retryable TRUE)
    elseif(inferedge_onnxruntime_failure_blob MATCHES "timeout"
        OR inferedge_onnxruntime_failure_blob MATCHES "timed out")
        set(failure_class "timeout")
        set(retryable TRUE)
    elseif(inferedge_onnxruntime_failure_blob MATCHES "could not resolve"
        OR inferedge_onnxruntime_failure_blob MATCHES "name lookup"
        OR inferedge_onnxruntime_failure_blob MATCHES "connection refused"
        OR inferedge_onnxruntime_failure_blob MATCHES "ssl"
        OR inferedge_onnxruntime_failure_blob MATCHES "tls")
        set(failure_class "connectivity")
        set(retryable TRUE)
    elseif(inferedge_onnxruntime_failure_blob MATCHES "429")
        set(failure_class "http_429")
        set(retryable TRUE)
    elseif(inferedge_onnxruntime_failure_blob MATCHES "404"
        OR inferedge_onnxruntime_failure_blob MATCHES "410"
        OR inferedge_onnxruntime_failure_blob MATCHES "not found")
        set(failure_class "http_404")
        set(retryable FALSE)
    elseif(inferedge_onnxruntime_failure_blob MATCHES " 50[0-9]"
        OR inferedge_onnxruntime_failure_blob MATCHES "503"
        OR inferedge_onnxruntime_failure_blob MATCHES "502"
        OR inferedge_onnxruntime_failure_blob MATCHES "500")
        set(failure_class "http_5xx")
        set(retryable TRUE)
    elseif(inferedge_onnxruntime_failure_blob MATCHES " 40[03]"
        OR inferedge_onnxruntime_failure_blob MATCHES " 401"
        OR inferedge_onnxruntime_failure_blob MATCHES " 403"
        OR inferedge_onnxruntime_failure_blob MATCHES " 405"
        OR inferedge_onnxruntime_failure_blob MATCHES " 408")
        set(failure_class "http_4xx_permanent")
        set(retryable FALSE)
    elseif(inferedge_onnxruntime_failure_blob MATCHES "no space"
        OR inferedge_onnxruntime_failure_blob MATCHES "read-only file system"
        OR inferedge_onnxruntime_failure_blob MATCHES "permission denied")
        set(failure_class "filesystem")
        set(retryable FALSE)
    elseif(inferedge_onnxruntime_failure_blob MATCHES "lock"
        AND inferedge_onnxruntime_failure_blob MATCHES "timeout")
        set(failure_class "lock_timeout")
        set(retryable FALSE)
    else()
        set(failure_class "transport_error")
        set(retryable TRUE)
    endif()

    set(${out_failure_class} "${failure_class}" PARENT_SCOPE)
    set(${out_retryable} "${retryable}" PARENT_SCOPE)
endfunction()

function(inferedge_onnxruntime_publish_verified_archive part_path archive_path expected_sha256)
    inferedge_onnxruntime_sha256_of_file("${part_path}" part_sha256)
    if(NOT part_sha256 STREQUAL expected_sha256)
        inferedge_onnxruntime_remove_path_if_present("${part_path}")
        message(FATAL_ERROR
            "ONNX Runtime download checksum mismatch before publication.\n"
            "  file: ${part_path}\n"
            "  expected SHA-256: ${expected_sha256}\n"
            "  actual SHA-256:   ${part_sha256}")
    endif()
    if(EXISTS "${archive_path}")
        inferedge_onnxruntime_remove_path_if_present("${archive_path}")
    endif()
    file(RENAME "${part_path}" "${archive_path}")
    inferedge_onnxruntime_sha256_of_file("${archive_path}" published_sha256)
    if(NOT published_sha256 STREQUAL expected_sha256)
        inferedge_onnxruntime_remove_path_if_present("${archive_path}")
        message(FATAL_ERROR
            "ONNX Runtime archive checksum mismatch after publication.\n"
            "  file: ${archive_path}\n"
            "  expected SHA-256: ${expected_sha256}\n"
            "  actual SHA-256:   ${published_sha256}")
    endif()
endfunction()

function(inferedge_onnxruntime_attempt_download url part_path expected_sha256 attempt_number max_attempts
        out_success out_failure_class out_failure_message out_retryable)
    set(success FALSE)
    set(failure_class "")
    set(failure_message "")
    set(retryable FALSE)

    message(STATUS
        "ONNX Runtime download attempt ${attempt_number}/${max_attempts}")

    inferedge_onnxruntime_remove_path_if_present("${part_path}")

    file(DOWNLOAD
        "${url}"
        "${part_path}"
        TIMEOUT ${INFEREDGE_ONNXRUNTIME_DOWNLOAD_ATTEMPT_TIMEOUT_SECONDS}
        INACTIVITY_TIMEOUT ${INFEREDGE_ONNXRUNTIME_DOWNLOAD_ATTEMPT_INACTIVITY_TIMEOUT_SECONDS}
        STATUS inferedge_onnxruntime_download_status
        LOG inferedge_onnxruntime_download_log
    )
    list(GET inferedge_onnxruntime_download_status 0 inferedge_onnxruntime_download_code)
    list(GET inferedge_onnxruntime_download_status 1 inferedge_onnxruntime_download_message)

    if(inferedge_onnxruntime_download_code EQUAL 0)
        inferedge_onnxruntime_sha256_of_file("${part_path}" downloaded_sha256)
        if(downloaded_sha256 STREQUAL expected_sha256)
            set(success TRUE)
            set(${out_success} "${success}" PARENT_SCOPE)
            set(${out_failure_class} "success" PARENT_SCOPE)
            set(${out_failure_message} "" PARENT_SCOPE)
            set(${out_retryable} FALSE PARENT_SCOPE)
            return()
        endif()
        set(failure_class "checksum_mismatch")
        set(retryable TRUE)
        set(failure_message
            "status=0; class=${failure_class}; retryable=${retryable}; "
            "SHA-256 mismatch for ${part_path}")
        inferedge_onnxruntime_remove_path_if_present("${part_path}")
        set(${out_success} FALSE PARENT_SCOPE)
        set(${out_failure_class} "${failure_class}" PARENT_SCOPE)
        set(${out_failure_message} "${failure_message}" PARENT_SCOPE)
        set(${out_retryable} "${retryable}" PARENT_SCOPE)
        return()
    endif()

    inferedge_onnxruntime_classify_download_failure(
        "${inferedge_onnxruntime_download_code}"
        "${inferedge_onnxruntime_download_message}"
        "${inferedge_onnxruntime_download_log}"
        failure_class
        retryable)
    inferedge_onnxruntime_remove_path_if_present("${part_path}")

    set(failure_message
        "status=${inferedge_onnxruntime_download_code}; class=${failure_class}; "
        "retryable=${retryable}; ${inferedge_onnxruntime_download_message}")

    set(${out_success} "${success}" PARENT_SCOPE)
    set(${out_failure_class} "${failure_class}" PARENT_SCOPE)
    set(${out_failure_message} "${failure_message}" PARENT_SCOPE)
    set(${out_retryable} "${retryable}" PARENT_SCOPE)
endfunction()

function(inferedge_onnxruntime_obtain_pinned_archive download_dir archive_name url expected_sha256
        archive_path_variable)
    set(archive_path "${download_dir}/${archive_name}")
    set(lock_path "${download_dir}/${INFEREDGE_ONNXRUNTIME_DOWNLOAD_LOCK_BASENAME}")
    set(part_path "${archive_path}${INFEREDGE_ONNXRUNTIME_DOWNLOAD_PART_SUFFIX}")

    file(MAKE_DIRECTORY "${download_dir}")

    inferedge_onnxruntime_validate_verified_archive("${archive_path}" "${expected_sha256}" cache_valid)
    if(cache_valid)
        set(${archive_path_variable} "${archive_path}" PARENT_SCOPE)
        return()
    endif()

    file(TOUCH "${lock_path}")
    file(LOCK "${lock_path}" GUARD FUNCTION
        TIMEOUT ${INFEREDGE_ONNXRUNTIME_DOWNLOAD_LOCK_TIMEOUT_SECONDS}
        RESULT_VARIABLE inferedge_onnxruntime_lock_result)
    if(NOT inferedge_onnxruntime_lock_result STREQUAL "0")
        message(FATAL_ERROR
            "Timed out after ${INFEREDGE_ONNXRUNTIME_DOWNLOAD_LOCK_TIMEOUT_SECONDS}s acquiring "
            "ONNX Runtime download lock at ${lock_path}.")
    endif()

    inferedge_onnxruntime_validate_verified_archive("${archive_path}" "${expected_sha256}" cache_valid)
    if(cache_valid)
        set(${archive_path_variable} "${archive_path}" PARENT_SCOPE)
        return()
    endif()

    if(EXISTS "${archive_path}")
        inferedge_onnxruntime_is_verified_regular_file("${archive_path}" archive_is_regular)
        if(NOT archive_is_regular)
            message(WARNING
                "Removing non-regular ONNX Runtime cache entry at ${archive_path}.")
            inferedge_onnxruntime_remove_path_if_present("${archive_path}")
        else()
            inferedge_onnxruntime_sha256_of_file("${archive_path}" archive_sha256)
            if(NOT archive_sha256 STREQUAL expected_sha256)
                message(WARNING
                    "Removing ONNX Runtime cache entry with unexpected SHA-256 at ${archive_path}.")
                inferedge_onnxruntime_remove_path_if_present("${archive_path}")
            endif()
        endif()
    endif()
    inferedge_onnxruntime_remove_path_if_present("${part_path}")

    set(attempt_number 1)
    set(last_failure_class "transport_error")
    set(last_failure_message "unknown download failure")
    while(attempt_number LESS_EQUAL INFEREDGE_ONNXRUNTIME_DOWNLOAD_MAX_ATTEMPTS)
        inferedge_onnxruntime_attempt_download(
            "${url}"
            "${part_path}"
            "${expected_sha256}"
            "${attempt_number}"
            "${INFEREDGE_ONNXRUNTIME_DOWNLOAD_MAX_ATTEMPTS}"
            attempt_success
            attempt_failure_class
            attempt_failure_message
            attempt_retryable)
        if(attempt_success)
            inferedge_onnxruntime_publish_verified_archive(
                "${part_path}" "${archive_path}" "${expected_sha256}")
            set(${archive_path_variable} "${archive_path}" PARENT_SCOPE)
            return()
        endif()

        set(last_failure_class "${attempt_failure_class}")
        set(last_failure_message "${attempt_failure_message}")
        message(STATUS
            "ONNX Runtime download attempt ${attempt_number} failed "
            "(${attempt_failure_class}).")

        if(NOT attempt_retryable)
            break()
        endif()

        if(attempt_number GREATER_EQUAL INFEREDGE_ONNXRUNTIME_DOWNLOAD_MAX_ATTEMPTS)
            break()
        endif()

        if(attempt_number EQUAL 1)
            inferedge_onnxruntime_sleep_seconds(
                ${INFEREDGE_ONNXRUNTIME_DOWNLOAD_RETRY_DELAY_AFTER_ATTEMPT_1_SECONDS})
        elseif(attempt_number EQUAL 2)
            inferedge_onnxruntime_sleep_seconds(
                ${INFEREDGE_ONNXRUNTIME_DOWNLOAD_RETRY_DELAY_AFTER_ATTEMPT_2_SECONDS})
        endif()
        math(EXPR attempt_number "${attempt_number} + 1")
    endwhile()

    inferedge_onnxruntime_remove_path_if_present("${part_path}")
    message(FATAL_ERROR
        "Failed to obtain ONNX Runtime ${INFEREDGE_ONNXRUNTIME_VERSION} "
        "(${archive_name}) after ${INFEREDGE_ONNXRUNTIME_DOWNLOAD_MAX_ATTEMPTS} "
        "download attempt(s).\n"
        "  url: ${url}\n"
        "  expected SHA-256: ${expected_sha256}\n"
        "  last failure class: ${last_failure_class}\n"
        "  last failure detail: ${last_failure_message}")
endfunction()


# inferedge:onnxruntime-download-helpers-end


function(inferedge_onnxruntime_materialize_from_nupkg nupkg_root package_root)
    set(include_src "${nupkg_root}/build/native/include")
    set(native_src "${nupkg_root}/${INFEREDGE_ONNXRUNTIME_NUPKG_NATIVE_RELATIVE}")
    if(NOT EXISTS "${include_src}/onnxruntime_cxx_api.h")
        message(FATAL_ERROR
            "Official ONNX Runtime NuGet at ${nupkg_root} is missing C++ headers.")
    endif()
    if(NOT EXISTS "${native_src}")
        message(FATAL_ERROR
            "Official ONNX Runtime NuGet at ${nupkg_root} is missing native library "
            "${INFEREDGE_ONNXRUNTIME_NUPKG_NATIVE_RELATIVE}.")
    endif()

    file(MAKE_DIRECTORY "${package_root}/include")
    file(COPY "${include_src}/" DESTINATION "${package_root}/include")

    file(MAKE_DIRECTORY "${package_root}/lib")
    if(APPLE)
        set(versioned_lib "${package_root}/lib/libonnxruntime.1.31.0.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E copy
                "${native_src}" "${versioned_lib}")
        file(CREATE_LINK
            "libonnxruntime.1.31.0.dylib"
            "${package_root}/lib/libonnxruntime.1.dylib"
            SYMBOLIC)
        file(CREATE_LINK
            "libonnxruntime.1.31.0.dylib"
            "${package_root}/lib/libonnxruntime.dylib"
            SYMBOLIC)
    else()
        set(versioned_lib "${package_root}/lib/libonnxruntime.so.1.31.0")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E copy
                "${native_src}" "${versioned_lib}")
        file(CREATE_LINK
            "libonnxruntime.so.1.31.0"
            "${package_root}/lib/libonnxruntime.so.1"
            SYMBOLIC)
        file(CREATE_LINK
            "libonnxruntime.so.1"
            "${package_root}/lib/libonnxruntime.so"
            SYMBOLIC)
    endif()

    file(WRITE "${package_root}/VERSION_NUMBER" "${INFEREDGE_ONNXRUNTIME_VERSION}\n")
endfunction()

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
    file(MAKE_DIRECTORY "${INFEREDGE_ONNXRUNTIME_DOWNLOAD_DIR}")

    inferedge_onnxruntime_obtain_pinned_archive(
        "${INFEREDGE_ONNXRUNTIME_DOWNLOAD_DIR}"
        "${INFEREDGE_ONNXRUNTIME_ARCHIVE}"
        "${INFEREDGE_ONNXRUNTIME_OFFICIAL_DOWNLOAD_URL}"
        "${INFEREDGE_ONNXRUNTIME_SHA256}"
        INFEREDGE_ONNXRUNTIME_ARCHIVE_PATH)

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
    set(INFEREDGE_ONNXRUNTIME_NUPKG_ROOT
        "${INFEREDGE_ONNXRUNTIME_EXTRACT_DIR}/nupkg-${INFEREDGE_ONNXRUNTIME_OFFICIAL_NUPKG_VERSION}")
    set(INFEREDGE_ONNXRUNTIME_PACKAGE_ROOT
        "${INFEREDGE_ONNXRUNTIME_EXTRACT_DIR}/${INFEREDGE_ONNXRUNTIME_PACKAGE_DIR}")

    if(inferedge_onnxruntime_need_extract)
        file(MAKE_DIRECTORY "${INFEREDGE_ONNXRUNTIME_NUPKG_ROOT}")
        file(ARCHIVE_EXTRACT
            INPUT "${INFEREDGE_ONNXRUNTIME_ARCHIVE_PATH}"
            DESTINATION "${INFEREDGE_ONNXRUNTIME_NUPKG_ROOT}"
        )
        file(MAKE_DIRECTORY "${INFEREDGE_ONNXRUNTIME_PACKAGE_ROOT}")
        inferedge_onnxruntime_materialize_from_nupkg(
            "${INFEREDGE_ONNXRUNTIME_NUPKG_ROOT}"
            "${INFEREDGE_ONNXRUNTIME_PACKAGE_ROOT}")
        file(WRITE "${INFEREDGE_ONNXRUNTIME_STAMP}" "${INFEREDGE_ONNXRUNTIME_SHA256}\n")
    endif()
    inferedge_require_onnxruntime_layout("${INFEREDGE_ONNXRUNTIME_PACKAGE_ROOT}")
    message(STATUS
        "ONNX Runtime ${INFEREDGE_ONNXRUNTIME_VERSION} official NuGet verified "
        "(${INFEREDGE_ONNXRUNTIME_OFFICIAL_NUPKG_ID} ${INFEREDGE_ONNXRUNTIME_OFFICIAL_NUPKG_VERSION})")
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
