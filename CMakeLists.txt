# CMakeLists.txt for AirAPI - Cross-platform build
# Supports Windows (Visual Studio), Linux, and Android (NDK)
#
# Usage:
#   Windows:   cmake -B build -G "Visual Studio 17 2022" -A x64
#   Linux:     cmake -B build && cmake --build build
#   Android:   cmake -B build -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
#              -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26

cmake_minimum_required(VERSION 3.18)
project(AirAPI VERSION 1.0.0 LANGUAGES CXX C)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform detection
if(ANDROID)
    set(PLATFORM_NAME "android")
    add_definitions(-DPLATFORM_ANDROID=1)
elseif(WIN32)
    set(PLATFORM_NAME "windows")
    add_definitions(-DPLATFORM_WINDOWS=1 -DAIRAPIWINDOWS_EXPORTS)
else()
    set(PLATFORM_NAME "linux")
endif()

message(STATUS "Building for platform: ${PLATFORM_NAME}")

# Options
option(AIRAPI_BUILD_SHARED "Build shared library" ON)
option(AIRAPI_ENABLE_DEPTH "Enable AI depth estimation" ON)
option(AIRAPI_ENABLE_CAMERA "Enable camera capture" ON)
option(AIRAPI_BUILD_EXAMPLES "Build example applications" ON)

# Source files - Core (platform-agnostic)
set(AIRAPI_CORE_SOURCES
    AirAPI_Windows.cpp
    dllmain.cpp
)

set(AIRAPI_CORE_HEADERS
    AirAPI_Windows.h
    Platform.h
    framework.h
    pch.h
)

# Camera sources
if(AIRAPI_ENABLE_CAMERA)
    list(APPEND AIRAPI_CORE_SOURCES XRealCamera.cpp)
    list(APPEND AIRAPI_CORE_HEADERS XRealCamera.h)
endif()

# Depth estimation sources
if(AIRAPI_ENABLE_DEPTH)
    list(APPEND AIRAPI_CORE_SOURCES DepthEstimator.cpp)
    list(APPEND AIRAPI_CORE_HEADERS DepthEstimator.h)
endif()

# ===== Dependencies =====

# Fusion library (IMU sensor fusion)
add_subdirectory(deps/Fusion/Fusion EXCLUDE_FROM_ALL)

# OpenCV
if(AIRAPI_ENABLE_CAMERA OR AIRAPI_ENABLE_DEPTH)
    if(ANDROID)
        # For Android, use OpenCV Android SDK
        set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/opencv-android/sdk/native/jni")
        find_package(OpenCV QUIET)
        if(NOT OpenCV_FOUND)
            message(WARNING "OpenCV Android SDK not found - camera/depth features may be limited")
        endif()
    else()
        find_package(OpenCV REQUIRED)
    endif()
endif()

# ONNX Runtime (optional, for depth estimation)
if(AIRAPI_ENABLE_DEPTH)
    set(ONNXRUNTIME_FOUND OFF)
    
    # Check local deps folder first
    set(ONNXRUNTIME_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime")
    
    if(WIN32)
        set(ONNXRUNTIME_LOCAL_FULL "${ONNXRUNTIME_LOCAL_DIR}/onnxruntime-win-x64-1.19.2")
    elseif(ANDROID)
        set(ONNXRUNTIME_LOCAL_FULL "${ONNXRUNTIME_LOCAL_DIR}/onnxruntime-android-${ANDROID_ABI}")
    else()
        set(ONNXRUNTIME_LOCAL_FULL "${ONNXRUNTIME_LOCAL_DIR}/onnxruntime-linux-x64")
    endif()
    
    if(EXISTS "${ONNXRUNTIME_LOCAL_FULL}/include/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_FOUND ON)
        set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_LOCAL_FULL}/include")
        set(ONNXRUNTIME_LIBRARY_DIRS "${ONNXRUNTIME_LOCAL_FULL}/lib")
        
        if(WIN32)
            set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_LIBRARY_DIRS}/onnxruntime.lib")
        else()
            set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_LIBRARY_DIRS}/libonnxruntime.so")
        endif()
        
        message(STATUS "Found local ONNX Runtime: ${ONNXRUNTIME_LOCAL_FULL}")
    elseif(DEFINED ENV{ONNXRUNTIME_DIR})
        set(ONNXRUNTIME_FOUND ON)
        set(ONNXRUNTIME_INCLUDE_DIRS "$ENV{ONNXRUNTIME_DIR}/include")
        set(ONNXRUNTIME_LIBRARY_DIRS "$ENV{ONNXRUNTIME_DIR}/lib")
        
        if(WIN32)
            set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_LIBRARY_DIRS}/onnxruntime.lib")
        else()
            set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_LIBRARY_DIRS}/libonnxruntime.so")
        endif()
        
        message(STATUS "Found ONNX Runtime via environment: $ENV{ONNXRUNTIME_DIR}")
    endif()
    
    if(ONNXRUNTIME_FOUND)
        add_definitions(-DONNXRUNTIME_AVAILABLE)
    else()
        message(WARNING "ONNX Runtime not found - depth estimation will be disabled")
    endif()
endif()

# HIDAPI (Windows/Linux only - Android uses USB Host API via JNI)
if(NOT ANDROID)
    if(WIN32)
        set(HIDAPI_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/deps/hidapi-win/include")
        set(HIDAPI_LIBRARY_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/deps/hidapi-win/x64")
        set(HIDAPI_LIBRARIES "hidapi")
    else()
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(HIDAPI REQUIRED hidapi-libusb)
    endif()
endif()

# ===== Library target =====

if(AIRAPI_BUILD_SHARED)
    add_library(AirAPI SHARED ${AIRAPI_CORE_SOURCES} ${AIRAPI_CORE_HEADERS})
else()
    add_library(AirAPI STATIC ${AIRAPI_CORE_SOURCES} ${AIRAPI_CORE_HEADERS})
endif()

# Include directories
target_include_directories(AirAPI PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/Fusion/Fusion
)

if(NOT ANDROID AND HIDAPI_INCLUDE_DIRS)
    target_include_directories(AirAPI PRIVATE ${HIDAPI_INCLUDE_DIRS})
endif()

if(AIRAPI_ENABLE_CAMERA AND OpenCV_FOUND)
    target_include_directories(AirAPI PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

if(AIRAPI_ENABLE_DEPTH AND ONNXRUNTIME_FOUND)
    target_include_directories(AirAPI PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
endif()

# Link libraries
target_link_libraries(AirAPI PRIVATE Fusion)

if(NOT ANDROID)
    target_link_directories(AirAPI PRIVATE ${HIDAPI_LIBRARY_DIRS})
    target_link_libraries(AirAPI PRIVATE ${HIDAPI_LIBRARIES})
endif()

if(AIRAPI_ENABLE_CAMERA AND OpenCV_FOUND)
    target_link_libraries(AirAPI PRIVATE ${OpenCV_LIBS})
endif()

if(AIRAPI_ENABLE_DEPTH AND ONNXRUNTIME_FOUND)
    target_link_directories(AirAPI PRIVATE ${ONNXRUNTIME_LIBRARY_DIRS})
    target_link_libraries(AirAPI PRIVATE ${ONNXRUNTIME_LIBRARIES})
endif()

# Platform-specific settings
if(WIN32)
    target_compile_definitions(AirAPI PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(ANDROID)
    # Android-specific: link to Android libraries
    target_link_libraries(AirAPI PRIVATE
        android
        log
        camera2ndk
        mediandk
    )
endif()

# ===== Examples =====

if(AIRAPI_BUILD_EXAMPLES AND NOT ANDROID)
    # CameraTest example
    add_executable(CameraTest examples/CameraTest.cpp)
    target_link_libraries(CameraTest PRIVATE AirAPI)
    
    if(OpenCV_FOUND)
        target_link_libraries(CameraTest PRIVATE ${OpenCV_LIBS})
    endif()
    
    # Copy runtime dependencies
    if(WIN32)
        add_custom_command(TARGET CameraTest POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/deps/hidapi-win/x64/hidapi.dll"
                "$<TARGET_FILE_DIR:CameraTest>"
        )
        
        if(ONNXRUNTIME_FOUND)
            add_custom_command(TARGET CameraTest POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_LOCAL_FULL}/lib/onnxruntime.dll"
                    "$<TARGET_FILE_DIR:CameraTest>"
            )
        endif()
    endif()
endif()

# ===== Installation =====

install(TARGETS AirAPI
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${AIRAPI_CORE_HEADERS} DESTINATION include/AirAPI)

# Copy models to installation
if(AIRAPI_ENABLE_DEPTH)
    install(DIRECTORY models/ DESTINATION share/AirAPI/models
        FILES_MATCHING PATTERN "*.onnx"
    )
endif()

# ===== Android AAR package helper =====

if(ANDROID)
    message(STATUS "")
    message(STATUS "=== Android Build Notes ===")
    message(STATUS "To build for Android:")
    message(STATUS "  1. Download OpenCV Android SDK to deps/opencv-android")
    message(STATUS "  2. Download ONNX Runtime Android to deps/onnxruntime/onnxruntime-android-arm64-v8a")
    message(STATUS "  3. Run: cmake -B build-android \\")
    message(STATUS "       -DCMAKE_TOOLCHAIN_FILE=\$ANDROID_NDK/build/cmake/android.toolchain.cmake \\")
    message(STATUS "       -DANDROID_ABI=arm64-v8a \\")
    message(STATUS "       -DANDROID_PLATFORM=android-26")
    message(STATUS "  4. Run: cmake --build build-android")
    message(STATUS "")
endif()
