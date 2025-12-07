#pragma once

#ifndef AIR_API
    #ifdef AIRAPIWINDOWS_EXPORTS
        #define AIR_API __declspec(dllexport)
    #else
        #define AIR_API __declspec(dllimport)
    #endif
#endif

#include <cstdint>
#include <vector>
#include <mutex>
#include <atomic>
#include <string>
#include <thread>
#include <functional>
#include "DepthEstimator.h"

// Forward declaration for OpenCV types
namespace cv {
    class VideoCapture;
    class Mat;
}

/**
 * @brief Structure to hold a stereo camera frame from XReal glasses
 */
struct XRealStereoFrame {
    uint8_t* leftImage;      // Left eye image data (grayscale, 480x640)
    uint8_t* rightImage;     // Right eye image data (grayscale, 480x640)
    int width;               // Image width (480 after rotation)
    int height;              // Image height (640 after rotation)
    uint64_t timestamp;      // Frame timestamp in microseconds
    int sequenceNumber;      // Frame sequence number
    bool valid;              // Whether the frame is valid
};

/**
 * @brief Frame callback type for async frame delivery
 * @param frame The stereo frame data
 * @param userData User-provided context pointer
 */
typedef void (*XRealFrameCallback)(const XRealStereoFrame* frame, void* userData);

/**
 * @brief XReal camera capture and decoding class
 * 
 * This class handles camera capture from XReal Air 2 Ultra glasses,
 * including the proprietary frame descrambling algorithm.
 */
class XRealCamera {
public:
    // Frame dimensions
    static const int RAW_WIDTH = 640;
    static const int RAW_HEIGHT = 241;
    static const int OUTPUT_WIDTH = 480;    // After rotation
    static const int OUTPUT_HEIGHT = 640;   // After rotation
    static const int CHUNK_SIZE = 2400;
    static const int NUM_CHUNKS = 128;
    static const int RAW_FRAME_SIZE = RAW_WIDTH * RAW_HEIGHT * 2;  // YUY2 format

    // Chunk reordering table (from Python implementation)
    static const int CHUNK_MAP[128];

    XRealCamera();
    ~XRealCamera();

    /**
     * @brief Initialize camera capture
     * @return true if successful, false otherwise
     */
    bool Initialize();

    /**
     * @brief Start camera capture
     * @return true if successful, false otherwise
     */
    bool StartCapture();

    /**
     * @brief Stop camera capture
     */
    void StopCapture();

    /**
     * @brief Check if camera is currently capturing
     * @return true if capturing, false otherwise
     */
    bool IsCapturing() const;

    /**
     * @brief Get the latest stereo frame
     * @param frame Output frame structure
     * @return true if a valid frame is available, false otherwise
     */
    bool GetLatestFrame(XRealStereoFrame* frame);

    /**
     * @brief Get raw left image buffer (caller must not free)
     * @param width Output image width
     * @param height Output image height
     * @return Pointer to left image data or nullptr
     */
    uint8_t* GetLeftImage(int* width, int* height);

    /**
     * @brief Get raw right image buffer (caller must not free)
     * @param width Output image width
     * @param height Output image height
     * @return Pointer to right image data or nullptr
     */
    uint8_t* GetRightImage(int* width, int* height);

    /**
     * @brief Get depth image from the latest frame
     * @param depthOut Output buffer for depth data (must be width*height bytes)
     * @param width Pointer to receive image width
     * @param height Pointer to receive image height
     * @return true if successful, false otherwise
     */
    bool GetDepthImage(uint8_t* depthOut, int* width, int* height);
    
    /**
     * @brief Get depth image pointer (internal buffer, do not free)
     * @param width Pointer to receive image width
     * @param height Pointer to receive image height
     * @return Pointer to depth data or nullptr
     */
    uint8_t* GetDepthImagePtr(int* width, int* height);
    
    /**
     * @brief Enable/disable live depth estimation
     * @param enabled true to enable
     */
    void SetLiveDepthEnabled(bool enabled);
    
    /**
     * @brief Check if live depth is enabled
     * @return true if enabled
     */
    bool IsLiveDepthEnabled() const;
    
    /**
     * @brief Initialize depth estimator
     * @param modelPath Path to ONNX model
     * @param useGPU Whether to use GPU
     * @return true if successful
     */
    bool InitializeDepth(const std::string& modelPath = "", bool useGPU = true);
    
    /**
     * @brief Get depth estimator reference
     */
    DepthEstimator& GetDepthEstimator() { return m_depthEstimator; }

    /**
     * @brief Set frame callback for async frame delivery
     * @param callback Callback function
     * @param userData User context pointer
     */
    void SetFrameCallback(XRealFrameCallback callback, void* userData);

    /**
     * @brief Get the camera device name
     * @return Camera device name or empty string
     */
    std::string GetDeviceName() const;

    /**
     * @brief Set rotation mode (0, 1, or 2)
     * @param rotation Rotation mode
     */
    void SetRotation(int rotation);

    /**
     * @brief Get current rotation mode
     * @return Current rotation (0, 1, or 2)
     */
    int GetRotation() const;

private:
    // Camera device enumeration
    std::string GetCameraNameFromFFmpeg();

    // Capture thread
    void CaptureThread();

    // Frame processing
    void ProcessRawFrame(const std::vector<uint8_t>& frameData);
    void HandleFrame(const uint8_t* inFrame, uint8_t* outBuffer);
    void DescrambleChunks(const uint8_t* blocks, int mapIdx, bool isRight, uint8_t* outBuffer);
    void ExtractStereoImages();

    // Capture handle (pipe)
    FILE* m_pipe;
    
    // OpenCV capture (removed/unused for main capture, but maybe used for discovery if needed? No, using ffmpeg for discovery too)
    // cv::VideoCapture* m_pVideoCapture; 
    int m_cameraIndex;

    // Device info
    std::string m_cameraName;

    // Capture state
    std::atomic<bool> m_isCapturing;
    std::atomic<bool> m_shouldStop;
    std::thread m_captureThread;

    // Frame buffers
    std::vector<uint8_t> m_lastRawFrame;
    std::vector<uint8_t> m_outputBuffer;
    std::vector<uint8_t> m_leftImage;
    std::vector<uint8_t> m_rightImage;

    // Frame state
    int m_lastSequence;
    bool m_hasLastFrame;
    uint64_t m_frameTimestamp;
    int m_sequenceNumber;

    // Rotation mode
    int m_rotation;

    // Synchronization
    mutable std::mutex m_frameMutex;
    bool m_frameReady;

    // Callback
    XRealFrameCallback m_callback;
    void* m_callbackUserData;
    
    // Depth estimation (integrated for live depth)
    DepthEstimator m_depthEstimator;
    std::vector<uint8_t> m_depthBuffer;
    std::atomic<bool> m_depthReady;
    std::atomic<bool> m_liveDepthEnabled;
    mutable std::mutex m_depthMutex;
};

// C API exports for DLL

extern "C" {
    /**
     * @brief Start camera capture
     * @return 1 on success, 0 on failure
     */
    AIR_API int StartCameraCapture();

    /**
     * @brief Stop camera capture
     * @return 1 on success, 0 if not capturing
     */
    AIR_API int StopCameraCapture();

    /**
     * @brief Check if camera is currently capturing
     * @return 1 if capturing, 0 otherwise
     */
    AIR_API int IsCameraCapturing();

    /**
     * @brief Get left camera image
     * @param width Pointer to receive image width
     * @param height Pointer to receive image height
     * @return Pointer to image data (grayscale), or nullptr if not available
     */
    AIR_API uint8_t* GetCameraLeftImage(int* width, int* height);

    /**
     * @brief Get right camera image
     * @param width Pointer to receive image width
     * @param height Pointer to receive image height
     * @return Pointer to image data (grayscale), or nullptr if not available
     */
    AIR_API uint8_t* GetCameraRightImage(int* width, int* height);

    /**
     * @brief Get stereo frame data
     * @param leftOut Output buffer for left image (must be at least 480*640 bytes)
     * @param rightOut Output buffer for right image (must be at least 480*640 bytes)
     * @param width Pointer to receive image width
     * @param height Pointer to receive image height
     * @param timestamp Pointer to receive frame timestamp (optional, can be nullptr)
     * @return 1 on success, 0 if no frame available
     */
    AIR_API int GetCameraStereoFrame(uint8_t* leftOut, uint8_t* rightOut, 
                                      int* width, int* height, uint64_t* timestamp);

    /**
     * @brief Set camera rotation mode
     * @param rotation Rotation mode (0, 1, or 2)
     */
    AIR_API void SetCameraRotation(int rotation);

    /**
     * @brief Get camera rotation mode
     * @return Current rotation mode
     */
    AIR_API int GetCameraRotation();

    /**
     * @brief Set frame callback for async frame delivery
     * @param callback Callback function pointer
     * @param userData User context pointer
     */
    AIR_API void SetCameraFrameCallback(XRealFrameCallback callback, void* userData);
    
    /**
     * @brief Initialize depth estimator for camera
     * @param modelPath Path to ONNX model (can be nullptr for default)
     * @param useGPU 1 to use GPU, 0 for CPU
     * @return 1 on success, 0 on failure
     */
    AIR_API int InitializeCameraDepth(const char* modelPath, int useGPU);
    
    /**
     * @brief Get depth image from the latest camera frame
     * @param depthOut Output buffer for depth image (must be at least width*height bytes)
     * @param width Pointer to receive image width
     * @param height Pointer to receive image height
     * @return 1 on success, 0 if not available
     */
    AIR_API int GetCameraDepthImage(uint8_t* depthOut, int* width, int* height);
    
    /**
     * @brief Get pointer to internal depth buffer (do not free)
     * @param width Pointer to receive image width
     * @param height Pointer to receive image height
     * @return Pointer to depth data, or nullptr if not available
     */
    AIR_API uint8_t* GetCameraDepthImagePtr(int* width, int* height);
    
    /**
     * @brief Enable/disable live depth processing
     * @param enabled 1 to enable, 0 to disable
     */
    AIR_API void SetCameraLiveDepth(int enabled);
    
    /**
     * @brief Check if live depth processing is enabled
     * @return 1 if enabled, 0 otherwise
     */
    AIR_API int IsCameraLiveDepthEnabled();
    
    /**
     * @brief Enable/disable temporal filtering for depth
     * @param enabled 1 to enable, 0 to disable
     */
    AIR_API void SetCameraDepthTemporalFilter(int enabled);
    
    /**
     * @brief Set temporal filter strength (0.0 - 1.0)
     * @param alpha Filter strength, lower = smoother but more lag
     */
    AIR_API void SetCameraDepthTemporalAlpha(float alpha);
    
    /**
     * @brief Reset temporal filter history (call on scene changes)
     */
    AIR_API void ResetCameraDepthTemporalHistory();
    
    /**
     * @brief Get average depth inference time in milliseconds
     * @return Average inference time
     */
    AIR_API float GetCameraDepthInferenceTimeMs();
}
