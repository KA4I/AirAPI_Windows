#pragma once

#ifndef AIR_API
    #ifdef AIRAPIWINDOWS_EXPORTS
        #define AIR_API __declspec(dllexport)
    #else
        #define AIR_API __declspec(dllimport)
    #endif
#endif

#include <cstdint>

//Function to start connection to Air
extern "C" AIR_API int StartConnection();

//Function to stop connection to Air
extern "C" AIR_API int StopConnection();

//Function to get quaternion
extern "C" AIR_API float* GetQuaternion();

//Function to get euler
extern "C" AIR_API float* GetEuler();

//Function to get brightness
extern "C" AIR_API int GetBrightness();

// ============================================
// Camera API Functions (XReal Air 2 Ultra)
// ============================================

/**
 * @brief Frame callback type for async stereo frame delivery
 * @param leftImage Left eye image data (grayscale)
 * @param rightImage Right eye image data (grayscale)
 * @param width Image width
 * @param height Image height
 * @param timestamp Frame timestamp in microseconds
 * @param userData User-provided context pointer
 */
typedef void (*CameraFrameCallback)(const uint8_t* leftImage, const uint8_t* rightImage,
                                     int width, int height, uint64_t timestamp, void* userData);

/**
 * @brief Start camera capture from XReal glasses
 * @return 1 on success, 0 on failure
 */
extern "C" AIR_API int StartCameraCapture();

/**
 * @brief Stop camera capture
 * @return 1 on success, 0 if not capturing
 */
extern "C" AIR_API int StopCameraCapture();

/**
 * @brief Check if camera is currently capturing
 * @return 1 if capturing, 0 otherwise
 */
extern "C" AIR_API int IsCameraCapturing();

/**
 * @brief Get left camera image pointer
 * @param width Pointer to receive image width (can be nullptr)
 * @param height Pointer to receive image height (can be nullptr)
 * @return Pointer to grayscale image data, or nullptr if not available
 * @note The returned pointer is valid until the next frame. Do not free.
 */
extern "C" AIR_API uint8_t* GetCameraLeftImage(int* width, int* height);

/**
 * @brief Get right camera image pointer
 * @param width Pointer to receive image width (can be nullptr)
 * @param height Pointer to receive image height (can be nullptr)
 * @return Pointer to grayscale image data, or nullptr if not available
 * @note The returned pointer is valid until the next frame. Do not free.
 */
extern "C" AIR_API uint8_t* GetCameraRightImage(int* width, int* height);

/**
 * @brief Get stereo frame with copy to user buffers
 * @param leftOut Output buffer for left image (must be at least width*height bytes)
 * @param rightOut Output buffer for right image (must be at least width*height bytes)
 * @param width Pointer to receive image width
 * @param height Pointer to receive image height
 * @param timestamp Pointer to receive frame timestamp (optional, can be nullptr)
 * @return 1 on success, 0 if no frame available
 */
extern "C" AIR_API int GetCameraStereoFrame(uint8_t* leftOut, uint8_t* rightOut,
                                             int* width, int* height, uint64_t* timestamp);

/**
 * @brief Set camera rotation mode
 * @param rotation Rotation mode: 0, 1, or 2 (default: 2)
 */
extern "C" AIR_API void SetCameraRotation(int rotation);

/**
 * @brief Get current camera rotation mode
 * @return Current rotation mode (0, 1, or 2)
 */
extern "C" AIR_API int GetCameraRotation();

// ============================================
// Depth Estimation API (AI-powered depth)
// ============================================

/**
 * @brief Initialize depth estimator with ONNX model
 * @param modelPath Path to ONNX model file (nullptr for default)
 * @param useGPU 1 to use GPU acceleration, 0 for CPU
 * @return 1 on success, 0 on failure
 */
extern "C" AIR_API int InitializeCameraDepth(const char* modelPath, int useGPU);

/**
 * @brief Get depth image from the latest camera frame
 * @param depthOut Output buffer for depth image (must be at least 480*640 bytes)
 * @param width Pointer to receive image width
 * @param height Pointer to receive image height
 * @return 1 on success, 0 if not available
 */
extern "C" AIR_API int GetCameraDepthImage(uint8_t* depthOut, int* width, int* height);

/**
 * @brief Get pointer to internal depth buffer (do not free)
 * @param width Pointer to receive image width
 * @param height Pointer to receive image height
 * @return Pointer to depth data (grayscale, 0-255), or nullptr
 */
extern "C" AIR_API uint8_t* GetCameraDepthImagePtr(int* width, int* height);

/**
 * @brief Enable/disable live depth processing
 * @param enabled 1 to enable real-time depth, 0 to disable
 */
extern "C" AIR_API void SetCameraLiveDepth(int enabled);

/**
 * @brief Check if live depth processing is enabled
 * @return 1 if enabled, 0 otherwise
 */
extern "C" AIR_API int IsCameraLiveDepthEnabled();

/**
 * @brief Enable/disable temporal filtering for depth (reduces flickering)
 * @param enabled 1 to enable, 0 to disable
 */
extern "C" AIR_API void SetCameraDepthTemporalFilter(int enabled);

/**
 * @brief Set temporal filter strength (0.0 - 1.0)
 * @param alpha Lower = smoother but more lag, higher = more responsive
 */
extern "C" AIR_API void SetCameraDepthTemporalAlpha(float alpha);

/**
 * @brief Reset temporal filter history (call on scene changes)
 */
extern "C" AIR_API void ResetCameraDepthTemporalHistory();

/**
 * @brief Get average depth inference time in milliseconds
 * @return Average inference time for depth estimation
 */
extern "C" AIR_API float GetCameraDepthInferenceTimeMs();