#pragma once

/**
 * @file Platform.h
 * @brief Cross-platform abstraction layer for Windows and Android
 * 
 * This header provides platform-specific definitions to make the codebase
 * easier to port between Windows and Android (JNI).
 * 
 * GPU Acceleration by Platform:
 * -----------------------------
 * Windows:
 *   - CUDA (NVIDIA GPUs) via ONNX Runtime CUDA provider
 *   - DirectML (AMD/Intel/NVIDIA) via ONNX Runtime DirectML provider
 * 
 * Android:
 *   - NNAPI (Neural Networks API) - Uses DSP, NPU, or GPU automatically
 *   - Qualcomm QNN - Best for Snapdragon devices
 *   - GPU Delegate - OpenCL/Vulkan compute
 * 
 * For AR navigation, 10-15 FPS depth is typically sufficient because:
 *   - Road geometry doesn't change rapidly
 *   - Navigation arrows are placed at fixed distances
 *   - Temporal filtering smooths between frames
 */

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
    #define PLATFORM_WINDOWS 1
    #define PLATFORM_ANDROID 0
#elif defined(__ANDROID__)
    #define PLATFORM_WINDOWS 0
    #define PLATFORM_ANDROID 1
#else
    #define PLATFORM_WINDOWS 0
    #define PLATFORM_ANDROID 0
    #warning "Unknown platform - defaulting to generic"
#endif

// API export macros
#if PLATFORM_WINDOWS
    #ifdef AIRAPIWINDOWS_EXPORTS
        #define AIR_API __declspec(dllexport)
    #else
        #define AIR_API __declspec(dllimport)
    #endif
    #define AIR_CALL __cdecl
#elif PLATFORM_ANDROID
    #define AIR_API __attribute__((visibility("default")))
    #define AIR_CALL
#else
    #define AIR_API
    #define AIR_CALL
#endif

// Standard types
#include <cstdint>
#include <cstddef>

// Platform-specific includes
#if PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#elif PLATFORM_ANDROID
    #include <jni.h>
    #include <android/log.h>
#endif

// Logging macros
#if PLATFORM_WINDOWS
    #include <iostream>
    #define AIR_LOG_INFO(fmt, ...)  std::cout << "[INFO] " << fmt << std::endl
    #define AIR_LOG_ERROR(fmt, ...) std::cerr << "[ERROR] " << fmt << std::endl
    #define AIR_LOG_DEBUG(fmt, ...) std::cout << "[DEBUG] " << fmt << std::endl
#elif PLATFORM_ANDROID
    #define AIR_LOG_TAG "AirAPI"
    #define AIR_LOG_INFO(fmt, ...)  __android_log_print(ANDROID_LOG_INFO, AIR_LOG_TAG, fmt, ##__VA_ARGS__)
    #define AIR_LOG_ERROR(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, AIR_LOG_TAG, fmt, ##__VA_ARGS__)
    #define AIR_LOG_DEBUG(fmt, ...) __android_log_print(ANDROID_LOG_DEBUG, AIR_LOG_TAG, fmt, ##__VA_ARGS__)
#else
    #include <cstdio>
    #define AIR_LOG_INFO(fmt, ...)  printf("[INFO] " fmt "\n", ##__VA_ARGS__)
    #define AIR_LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
    #define AIR_LOG_DEBUG(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#endif

// Thread/mutex abstractions (both platforms use std)
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>

namespace air {

/**
 * @brief Platform-agnostic USB/HID device handle
 */
struct DeviceHandle {
#if PLATFORM_WINDOWS
    void* hidHandle;  // hid_device*
#elif PLATFORM_ANDROID
    int usbFd;
    void* usbConnection;  // UsbDeviceConnection from JNI
#else
    void* genericHandle;
#endif
};

/**
 * @brief Platform-agnostic camera handle
 */
struct CameraHandle {
#if PLATFORM_WINDOWS
    FILE* pipe;           // ffmpeg pipe
    void* capture;        // cv::VideoCapture* (alternative)
#elif PLATFORM_ANDROID
    void* camera;         // ACameraDevice* from NDK Camera2
    void* imageReader;    // AImageReader*
#else
    void* genericCamera;
#endif
};

/**
 * @brief Get the default path for depth models based on platform
 */
inline const char* GetDefaultModelPath() {
#if PLATFORM_WINDOWS
    return "depth_anything_v2_vits.onnx";
#elif PLATFORM_ANDROID
    return "/sdcard/Android/data/com.xreal.glasses/files/depth_anything_v2_small_q4f16.onnx";
#else
    return "depth_anything_v2_vits.onnx";
#endif
}

/**
 * @brief Get the optimal ONNX model variant for current platform
 * Android uses quantized model for better performance on mobile GPUs
 */
inline const char* GetOptimalModelFilename() {
#if PLATFORM_WINDOWS
    return "depth_anything_v2_small.onnx";        // Full precision for desktop
#elif PLATFORM_ANDROID
    return "depth_anything_v2_small_q4f16.onnx";  // Quantized for mobile
#else
    return "depth_anything_v2_small.onnx";
#endif
}

/**
 * @brief Check if GPU acceleration should be enabled by default
 */
inline bool ShouldUseGPUByDefault() {
#if PLATFORM_WINDOWS
    return true;   // CUDA on Windows
#elif PLATFORM_ANDROID
    return true;   // NNAPI on Android
#else
    return false;
#endif
}

}  // namespace air

// JNI bridge helpers (Android only)
#if PLATFORM_ANDROID

extern "C" {

// JNI initialization - must be called from Java before using the API
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved);
JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved);

// JNI wrapper functions would be defined here
// These translate between Java types and C++ types

}  // extern "C"

#endif  // PLATFORM_ANDROID
