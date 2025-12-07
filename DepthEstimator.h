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
#include <memory>
#include <deque>

// Forward declarations for ONNX Runtime
namespace Ort {
    struct Session;
    struct Env;
    struct SessionOptions;
    struct MemoryInfo;
    struct Value;
}

/**
 * @brief Configuration for the depth estimator
 */
struct DepthEstimatorConfig {
    // Model settings
    std::string modelPath = "depth_anything_v2_vits.onnx";  // Default model
    bool useGPU = true;                                       // Use CUDA/NNAPI if available
    
    // Processing settings - Multiple presets available
    int inputWidth = 256;                                     // Model input width
    int inputHeight = 256;                                    // Model input height
    // Presets: 128 (~25fps CPU), 192 (~15fps), 256 (~10fps), 384 (~5fps), 518 (~2fps)
    
    // Async processing for real-time performance
    bool asyncMode = true;                                    // Run inference in background thread
    int targetFps = 15;                                       // Target depth fps (inference limited)
    bool cacheDepth = true;                                   // Cache last depth for fast access
    
    // Depth interpolation for AR (smooth between inferences)
    bool enableInterpolation = true;                          // Interpolate depth between frames
    float interpolationWeight = 0.3f;                         // How much to blend new depth
    
    // Temporal filtering (reduces flickering)
    bool enableTemporalFilter = true;
    int temporalWindowSize = 3;                               // Reduced for speed
    float temporalAlpha = 0.4f;                               // Exponential moving average weight
    
    // Spatial filtering (smooths edges) - SIMPLIFIED for speed
    bool enableSpatialFilter = false;                         // Disabled for speed
    int spatialKernelSize = 3;                                // Smaller kernel
    float spatialSigma = 1.0f;
    
    // Edge-preserving filter - DISABLED for speed on mobile
    bool enableBilateralFilter = false;                       // Expensive, disable for mobile
    int bilateralD = 5;                                       // Smaller
    float bilateralSigmaColor = 50.0f;
    float bilateralSigmaSpace = 50.0f;
    
    // Hole filling for invalid regions
    bool enableHoleFilling = false;                           // Disabled for speed
    
    // Output settings
    bool normalizeOutput = true;                              // Normalize to 0-255 range
    bool invertDepth = false;                                 // Invert so closer = brighter
    
    // Performance
    int processingThreads = 2;                                // Reduced for mobile
    
    // Mobile/low-power optimizations
    bool mobileMode = false;                                  // Enable all mobile optimizations
    int skipFrames = 0;                                       // Skip N frames between inferences
    
    // AR Mode - optimized for navigation overlays
    bool arMode = false;                                      // Enable AR optimizations
    float arDepthScale = 10.0f;                               // Meters per depth unit (for distance calc)
};

/**
 * @brief Depth frame data with metadata
 */
struct DepthFrame {
    std::vector<uint8_t> depthData;      // Normalized depth data (0-255)
    std::vector<float> rawDepth;         // Raw depth values
    int width;
    int height;
    uint64_t timestamp;
    int sequenceNumber;
    bool valid;
    
    // IMU data at capture time (for stabilization)
    float quaternion[4];
    float euler[3];
};

/**
 * @brief Callback type for depth frame delivery
 */
typedef void (*DepthFrameCallback)(const DepthFrame* frame, void* userData);

/**
 * @brief AI-powered depth estimation using Depth Anything V2
 * 
 * This class provides high-quality monocular depth estimation with
 * temporal consistency and smooth output suitable for real-time display.
 */
class DepthEstimator {
public:
    DepthEstimator();
    ~DepthEstimator();
    
    /**
     * @brief Initialize the depth estimator with given configuration
     * @param config Configuration settings
     * @return true if successful
     */
    bool Initialize(const DepthEstimatorConfig& config = DepthEstimatorConfig());
    
    /**
     * @brief Check if initialized and ready
     */
    bool IsInitialized() const;
    
    /**
     * @brief Shutdown and release resources
     */
    void Shutdown();
    
    /**
     * @brief Process a grayscale image and compute depth
     * @param imageData Input grayscale image
     * @param width Image width
     * @param height Image height
     * @param depthOut Output depth buffer (must be width*height bytes)
     * @param rawDepthOut Optional: raw float depth values
     * @return true if successful
     */
    bool EstimateDepth(const uint8_t* imageData, int width, int height,
                       uint8_t* depthOut, float* rawDepthOut = nullptr);
    
    /**
     * @brief Process stereo images for potentially better depth
     * @param leftImage Left eye grayscale image
     * @param rightImage Right eye grayscale image  
     * @param width Image width
     * @param height Image height
     * @param depthOut Output depth buffer
     * @return true if successful
     */
    bool EstimateDepthStereo(const uint8_t* leftImage, const uint8_t* rightImage,
                              int width, int height, uint8_t* depthOut);
    
    /**
     * @brief Set IMU data for frame stabilization
     * @param quaternion Current quaternion (w, x, y, z)
     * @param euler Current euler angles (pitch, roll, yaw)
     */
    void SetIMUData(const float* quaternion, const float* euler);
    
    /**
     * @brief Enable/disable temporal filtering
     */
    void SetTemporalFilterEnabled(bool enabled);
    
    /**
     * @brief Set temporal filter strength (0.0 - 1.0)
     */
    void SetTemporalAlpha(float alpha);
    
    /**
     * @brief Clear temporal history (call on scene changes)
     */
    void ResetTemporalHistory();
    
    /**
     * @brief Get current configuration
     */
    const DepthEstimatorConfig& GetConfig() const { return m_config; }
    
    /**
     * @brief Get processing statistics
     */
    float GetAverageInferenceTimeMs() const;
    float GetAverageProcessingTimeMs() const;
    
    // ===== Async/Cached Depth Access (for real-time performance) =====
    
    /**
     * @brief Submit a frame for async depth processing
     * Returns immediately, depth computed in background thread
     */
    void SubmitFrameAsync(const uint8_t* imageData, int width, int height);
    
    /**
     * @brief Get the latest cached depth (non-blocking)
     * @return true if cached depth available
     */
    bool GetCachedDepth(uint8_t* depthOut, int* width, int* height);
    
    /**
     * @brief Check if new depth is ready since last call
     */
    bool IsNewDepthReady() const;
    
    /**
     * @brief Start async processing thread
     */
    void StartAsyncProcessing();
    
    /**
     * @brief Stop async processing thread
     */
    void StopAsyncProcessing();
    
    /**
     * @brief Set input resolution (lower = faster)
     */
    void SetInputResolution(int width, int height);

private:
    // ONNX Runtime session management
    bool LoadModel(const std::string& modelPath);
    void ReleaseModel();
    
    // Image preprocessing
    void PreprocessImage(const uint8_t* input, int width, int height,
                         std::vector<float>& output);
    
    // Depth postprocessing  
    void PostprocessDepth(const float* rawDepth, int width, int height,
                          uint8_t* output, float* rawOutput = nullptr);
    
    // Temporal filtering
    void ApplyTemporalFilter(std::vector<float>& depth, int width, int height);
    
    // Spatial filtering
    void ApplySpatialFilter(uint8_t* depth, int width, int height);
    void ApplyBilateralFilter(uint8_t* depth, int width, int height);
    void FillHoles(uint8_t* depth, int width, int height);
    
    // Stereo fusion helper
    void FuseStereoDepth(const float* leftDepth, const float* rightDepth,
                          int width, int height, float* fusedDepth);
    
    // ONNX Runtime members (opaque pointers to avoid header dependency)
    struct OrtSession;
    std::unique_ptr<OrtSession> m_session;
    
    // Configuration
    DepthEstimatorConfig m_config;
    bool m_initialized;
    
    // Current IMU state
    float m_quaternion[4];
    float m_euler[3];
    std::mutex m_imuMutex;
    
    // Temporal filter history
    std::deque<std::vector<float>> m_depthHistory;
    std::vector<float> m_accumulatedDepth;
    std::mutex m_temporalMutex;
    
    // Processing buffers
    std::vector<float> m_preprocessBuffer;
    std::vector<float> m_depthBuffer;
    std::vector<uint8_t> m_outputBuffer;
    
    // Statistics
    std::deque<float> m_inferenceTimesMs;
    std::deque<float> m_processingTimesMs;
    mutable std::mutex m_statsMutex;
    
    // Async processing
    std::thread m_asyncThread;
    std::atomic<bool> m_asyncRunning{false};
    std::atomic<bool> m_asyncStop{false};
    std::condition_variable m_asyncCV;
    std::mutex m_asyncMutex;
    
    // Pending frame for async processing
    std::vector<uint8_t> m_pendingFrame;
    int m_pendingWidth = 0;
    int m_pendingHeight = 0;
    std::atomic<bool> m_hasPendingFrame{false};
    
    // Cached depth result
    std::vector<uint8_t> m_cachedDepth;
    int m_cachedWidth = 0;
    int m_cachedHeight = 0;
    std::atomic<bool> m_hasNewDepth{false};
    mutable std::mutex m_cacheMutex;
    
    // Frame skip counter
    int m_frameCounter = 0;
    
    // Async thread function
    void AsyncProcessingThread();
};

// C API for depth estimation

extern "C" {
    /**
     * @brief Initialize depth estimator with default settings
     * @param modelPath Path to ONNX model file (can be nullptr for default)
     * @param useGPU Whether to use GPU acceleration
     * @return 1 on success, 0 on failure
     */
    AIR_API int InitializeDepthEstimator(const char* modelPath, int useGPU);
    
    /**
     * @brief Shutdown depth estimator and release resources
     */
    AIR_API void ShutdownDepthEstimator();
    
    /**
     * @brief Check if depth estimator is ready
     * @return 1 if initialized, 0 otherwise
     */
    AIR_API int IsDepthEstimatorReady();
    
    /**
     * @brief Estimate depth from a grayscale image
     * @param imageData Input image (grayscale, width*height bytes)
     * @param width Image width
     * @param height Image height
     * @param depthOut Output depth buffer (must be width*height bytes)
     * @return 1 on success, 0 on failure
     */
    AIR_API int EstimateDepthFromImage(const uint8_t* imageData, int width, int height,
                                        uint8_t* depthOut);
    
    /**
     * @brief Get depth from latest camera frame
     * @param depthOut Output buffer (must be at least 480*640 bytes)
     * @param width Pointer to receive width
     * @param height Pointer to receive height
     * @return 1 on success, 0 if no frame available
     */
    AIR_API int GetCameraDepthImage(uint8_t* depthOut, int* width, int* height);
    
    /**
     * @brief Enable/disable live depth processing
     * @param enabled 1 to enable, 0 to disable
     */
    AIR_API void SetLiveDepthEnabled(int enabled);
    
    /**
     * @brief Check if live depth is enabled
     * @return 1 if enabled, 0 otherwise
     */
    AIR_API int IsLiveDepthEnabled();
    
    /**
     * @brief Set temporal filter enabled
     * @param enabled 1 to enable, 0 to disable
     */
    AIR_API void SetDepthTemporalFilter(int enabled);
    
    /**
     * @brief Set temporal filter strength (0.0 - 1.0)
     */
    AIR_API void SetDepthTemporalAlpha(float alpha);
    
    /**
     * @brief Reset temporal filter history
     */
    AIR_API void ResetDepthTemporalHistory();
    
    /**
     * @brief Get average inference time in milliseconds
     */
    AIR_API float GetDepthInferenceTimeMs();
    
    // ===== Fast/Mobile Initialization =====
    
    /**
     * @brief Initialize with fast settings (256x256 input, async, no heavy filters)
     * @param modelPath Path to ONNX model (nullptr for default)
     * @param useGPU 1 for GPU, 0 for CPU
     * @param inputSize Input resolution (e.g., 256, 192, 128). Smaller = faster
     * @return 1 on success
     */
    AIR_API int InitializeDepthEstimatorFast(const char* modelPath, int useGPU, int inputSize);
    
    /**
     * @brief Initialize with mobile-optimized settings (192x192, quantized model)
     * Best for phones/tablets
     * @param modelPath Path to ONNX model (nullptr for default q4f16 model)
     * @return 1 on success
     */
    AIR_API int InitializeDepthEstimatorMobile(const char* modelPath);
    
    // ===== Async API (for real-time apps) =====
    
    /**
     * @brief Submit frame for async processing (non-blocking)
     * Depth will be computed in background thread
     */
    AIR_API void SubmitDepthFrameAsync(const uint8_t* imageData, int width, int height);
    
    /**
     * @brief Get cached depth from last completed inference
     * @return 1 if depth available, 0 if no cached depth yet
     */
    AIR_API int GetCachedDepthImage(uint8_t* depthOut, int* width, int* height);
    
    /**
     * @brief Check if new depth is available since last GetCachedDepthImage
     * @return 1 if new depth ready
     */
    AIR_API int IsNewDepthAvailable();
    
    /**
     * @brief Set input resolution dynamically (lower = faster)
     * Common values: 512 (quality), 256 (balanced), 192 (fast), 128 (ultra fast)
     */
    AIR_API void SetDepthInputResolution(int width, int height);
    
    // ===== AR Navigation API =====
    
    /**
     * @brief Initialize for AR navigation (128x128, ultra fast, ~25fps on CPU)
     * Optimized for placing virtual objects on roads/surfaces
     * @param modelPath Path to ONNX model (nullptr for default)
     * @return 1 on success
     */
    AIR_API int InitializeDepthForAR(const char* modelPath);
    
    /**
     * @brief Get depth value at a specific pixel location
     * Useful for placing AR objects at correct depth
     * @param x X coordinate (0 to width-1)
     * @param y Y coordinate (0 to height-1)
     * @return Normalized depth (0.0 = far, 1.0 = near), or -1 if not available
     */
    AIR_API float GetDepthAtPoint(int x, int y);
    
    /**
     * @brief Get estimated distance in meters at a pixel
     * Uses approximate scaling - not metrically accurate without calibration
     * @param x X coordinate
     * @param y Y coordinate
     * @return Approximate distance in meters, or -1 if not available
     */
    AIR_API float GetDistanceAtPoint(int x, int y);
    
    /**
     * @brief Find the ground plane depth (for placing navigation arrows)
     * Samples the bottom portion of the image to find road/floor depth
     * @return Average depth of bottom region, or -1 if not available
     */
    AIR_API float GetGroundPlaneDepth();
}
