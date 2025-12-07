#define _CRT_SECURE_NO_WARNINGS 1
#include "DepthEstimator.h"
#include "Platform.h"

// Check if ONNX Runtime is available at compile time
#ifdef ONNXRUNTIME_AVAILABLE
#define HAS_ONNXRUNTIME 1
#else
#define HAS_ONNXRUNTIME 0
#endif

#if HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#if PLATFORM_ANDROID
// Android NNAPI provider for hardware acceleration
#include <onnxruntime_cxx_api.h>
#endif
#endif

// OpenCV for image processing
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>

#if HAS_ONNXRUNTIME
// Internal ONNX session wrapper
struct DepthEstimator::OrtSession {
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Model info
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<std::string> inputNamesStr;
    std::vector<std::string> outputNamesStr;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    
    bool initialized = false;
};
#else
// Stub when ONNX Runtime is not available
struct DepthEstimator::OrtSession {
    bool initialized = false;
};
#endif

DepthEstimator::DepthEstimator()
    : m_initialized(false)
{
    m_quaternion[0] = 1.0f;
    m_quaternion[1] = m_quaternion[2] = m_quaternion[3] = 0.0f;
    m_euler[0] = m_euler[1] = m_euler[2] = 0.0f;
}

DepthEstimator::~DepthEstimator() {
    Shutdown();
}

bool DepthEstimator::Initialize(const DepthEstimatorConfig& config) {
#if !HAS_ONNXRUNTIME
    std::cerr << "ONNX Runtime not available - depth estimation disabled" << std::endl;
    std::cerr << "Define ONNXRUNTIME_AVAILABLE and set ONNXRUNTIME_DIR, then rebuild" << std::endl;
    return false;
#else
    if (m_initialized) {
        Shutdown();
    }
    
    m_config = config;
    
    // Try to load model
    if (!LoadModel(config.modelPath)) {
        std::cerr << "Failed to load depth model: " << config.modelPath << std::endl;
        return false;
    }
    
    // Pre-allocate buffers
    int modelPixels = m_config.inputWidth * m_config.inputHeight;
    m_preprocessBuffer.resize(modelPixels * 3);  // RGB
    m_depthBuffer.resize(modelPixels);
    
    m_initialized = true;
    std::cout << "DepthEstimator initialized successfully" << std::endl;
    std::cout << "  Model: " << config.modelPath << std::endl;
    std::cout << "  Input size: " << m_config.inputWidth << "x" << m_config.inputHeight << std::endl;
    std::cout << "  GPU: " << (config.useGPU ? "enabled" : "disabled") << std::endl;
    std::cout << "  Async mode: " << (config.asyncMode ? "enabled" : "disabled") << std::endl;
    std::cout << "  Target FPS: " << config.targetFps << std::endl;
    
    // Start async processing if enabled
    if (config.asyncMode) {
        StartAsyncProcessing();
    }
    
    return true;
#endif
}

bool DepthEstimator::IsInitialized() const {
    return m_initialized;
}

void DepthEstimator::Shutdown() {
    if (!m_initialized) return;
    
    // Stop async processing first
    StopAsyncProcessing();
    
    ReleaseModel();
    
    m_depthHistory.clear();
    m_accumulatedDepth.clear();
    m_preprocessBuffer.clear();
    m_depthBuffer.clear();
    m_outputBuffer.clear();
    m_cachedDepth.clear();
    m_pendingFrame.clear();
    
    m_initialized = false;
    std::cout << "DepthEstimator shutdown" << std::endl;
}

bool DepthEstimator::LoadModel(const std::string& modelPath) {
#if !HAS_ONNXRUNTIME
    return false;
#else
    try {
        m_session = std::make_unique<OrtSession>();
        
        // Create ONNX Runtime environment
        m_session->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DepthEstimator");
        
        // Session options
        m_session->sessionOptions = std::make_unique<Ort::SessionOptions>();
        m_session->sessionOptions->SetIntraOpNumThreads(m_config.processingThreads);
        m_session->sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Try to enable GPU acceleration based on platform
        if (m_config.useGPU) {
#if PLATFORM_WINDOWS
            // Windows: Try CUDA
            try {
                OrtCUDAProviderOptions cudaOptions;
                cudaOptions.device_id = 0;
                cudaOptions.arena_extend_strategy = 0;
                cudaOptions.gpu_mem_limit = SIZE_MAX;
                cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cudaOptions.do_copy_in_default_stream = 1;
                m_session->sessionOptions->AppendExecutionProvider_CUDA(cudaOptions);
                std::cout << "CUDA execution provider enabled" << std::endl;
            }
            catch (const Ort::Exception& e) {
                std::cerr << "CUDA not available, falling back to CPU: " << e.what() << std::endl;
            }
#elif PLATFORM_ANDROID
            // Android: Try NNAPI for hardware acceleration
            try {
                m_session->sessionOptions->AppendExecutionProvider("NNAPI", {});
                AIR_LOG_INFO("NNAPI execution provider enabled");
            }
            catch (const Ort::Exception& e) {
                AIR_LOG_ERROR("NNAPI not available, falling back to CPU: %s", e.what());
            }
#endif
        }
        
#if PLATFORM_WINDOWS
        // Windows: Convert path to wide string
        std::wstring wpath(modelPath.begin(), modelPath.end());
        
        // Create session
        m_session->session = std::make_unique<Ort::Session>(
            *m_session->env, wpath.c_str(), *m_session->sessionOptions);
#else
        // Android/Linux: Use UTF-8 path directly
        m_session->session = std::make_unique<Ort::Session>(
            *m_session->env, modelPath.c_str(), *m_session->sessionOptions);
#endif
        
        // Get input info
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t numInputs = m_session->session->GetInputCount();
        for (size_t i = 0; i < numInputs; i++) {
            auto inputName = m_session->session->GetInputNameAllocated(i, allocator);
            m_session->inputNamesStr.push_back(inputName.get());
            
            auto typeInfo = m_session->session->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            m_session->inputShape = tensorInfo.GetShape();
        }
        
        // Get output info
        size_t numOutputs = m_session->session->GetOutputCount();
        for (size_t i = 0; i < numOutputs; i++) {
            auto outputName = m_session->session->GetOutputNameAllocated(i, allocator);
            m_session->outputNamesStr.push_back(outputName.get());
            
            auto typeInfo = m_session->session->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            m_session->outputShape = tensorInfo.GetShape();
        }
        
        // Setup name pointers
        for (const auto& name : m_session->inputNamesStr) {
            m_session->inputNames.push_back(name.c_str());
        }
        for (const auto& name : m_session->outputNamesStr) {
            m_session->outputNames.push_back(name.c_str());
        }
        
        // Update config with actual model dimensions if dynamic
        if (m_session->inputShape.size() >= 4) {
            if (m_session->inputShape[2] > 0) {
                m_config.inputHeight = static_cast<int>(m_session->inputShape[2]);
            }
            if (m_session->inputShape[3] > 0) {
                m_config.inputWidth = static_cast<int>(m_session->inputShape[3]);
            }
        }
        
        m_session->initialized = true;
        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        m_session.reset();
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        m_session.reset();
        return false;
    }
#endif
}

void DepthEstimator::ReleaseModel() {
#if HAS_ONNXRUNTIME
    if (m_session) {
        m_session->session.reset();
        m_session->sessionOptions.reset();
        m_session->env.reset();
        m_session.reset();
    }
#endif
}

void DepthEstimator::PreprocessImage(const uint8_t* input, int width, int height,
                                      std::vector<float>& output) {
    // Resize to model input size
    cv::Mat inputMat(height, width, CV_8UC1, const_cast<uint8_t*>(input));
    cv::Mat resized;
    cv::resize(inputMat, resized, cv::Size(m_config.inputWidth, m_config.inputHeight), 
               0, 0, cv::INTER_LINEAR);
    
    // Convert to RGB (model expects 3 channels)
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
    
    // Normalize to [0, 1] and convert to float
    cv::Mat floatMat;
    rgb.convertTo(floatMat, CV_32FC3, 1.0 / 255.0);
    
    // ImageNet normalization
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar stddev(0.229, 0.224, 0.225);
    
    // Apply normalization per channel
    std::vector<cv::Mat> channels;
    cv::split(floatMat, channels);
    
    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - mean[c]) / stddev[c];
    }
    
    // Convert to NCHW format (batch=1, channels=3, height, width)
    int modelPixels = m_config.inputWidth * m_config.inputHeight;
    output.resize(3 * modelPixels);
    
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < m_config.inputHeight; y++) {
            for (int x = 0; x < m_config.inputWidth; x++) {
                int srcIdx = y * m_config.inputWidth + x;
                int dstIdx = c * modelPixels + srcIdx;
                output[dstIdx] = channels[c].at<float>(y, x);
            }
        }
    }
}

void DepthEstimator::PostprocessDepth(const float* rawDepth, int width, int height,
                                       uint8_t* output, float* rawOutput) {
    int pixels = width * height;
    
    // Find min/max for normalization
    float minVal = rawDepth[0];
    float maxVal = rawDepth[0];
    for (int i = 1; i < pixels; i++) {
        minVal = std::min(minVal, rawDepth[i]);
        maxVal = std::max(maxVal, rawDepth[i]);
    }
    
    float range = maxVal - minVal;
    if (range < 1e-6f) range = 1.0f;
    
    // Normalize and convert to uint8
    for (int i = 0; i < pixels; i++) {
        float normalized = (rawDepth[i] - minVal) / range;
        
        if (m_config.invertDepth) {
            normalized = 1.0f - normalized;
        }
        
        if (rawOutput) {
            rawOutput[i] = normalized;
        }
        
        if (m_config.normalizeOutput) {
            output[i] = static_cast<uint8_t>(std::clamp(normalized * 255.0f, 0.0f, 255.0f));
        }
    }
}

void DepthEstimator::ApplyTemporalFilter(std::vector<float>& depth, int width, int height) {
    std::lock_guard<std::mutex> lock(m_temporalMutex);
    
    int pixels = width * height;
    
    // Initialize accumulated depth if needed
    if (m_accumulatedDepth.size() != static_cast<size_t>(pixels)) {
        m_accumulatedDepth = depth;
        m_depthHistory.clear();
        return;
    }
    
    // Add current frame to history
    m_depthHistory.push_back(depth);
    while (m_depthHistory.size() > static_cast<size_t>(m_config.temporalWindowSize)) {
        m_depthHistory.pop_front();
    }
    
    // Exponential moving average with motion detection
    float alpha = m_config.temporalAlpha;
    
    for (int i = 0; i < pixels; i++) {
        // Detect significant change (possible motion)
        float diff = std::abs(depth[i] - m_accumulatedDepth[i]);
        
        // Adaptive alpha based on motion
        float adaptiveAlpha = alpha;
        if (diff > 0.1f) {
            // More change detected, trust new value more
            adaptiveAlpha = std::min(alpha + 0.3f, 1.0f);
        }
        
        // Blend with history
        m_accumulatedDepth[i] = m_accumulatedDepth[i] * (1.0f - adaptiveAlpha) 
                               + depth[i] * adaptiveAlpha;
    }
    
    // Also do a weighted average of recent frames
    if (m_depthHistory.size() >= 2) {
        for (int i = 0; i < pixels; i++) {
            float sum = 0.0f;
            float weightSum = 0.0f;
            float weight = 1.0f;
            
            for (auto it = m_depthHistory.rbegin(); it != m_depthHistory.rend(); ++it) {
                sum += (*it)[i] * weight;
                weightSum += weight;
                weight *= 0.7f;  // Decay factor
            }
            
            depth[i] = sum / weightSum;
        }
    }
    
    // Final blend with accumulated
    for (int i = 0; i < pixels; i++) {
        depth[i] = depth[i] * 0.6f + m_accumulatedDepth[i] * 0.4f;
    }
}

void DepthEstimator::ApplySpatialFilter(uint8_t* depth, int width, int height) {
    cv::Mat depthMat(height, width, CV_8UC1, depth);
    
    // Gaussian blur for smoothing
    cv::Mat blurred;
    cv::GaussianBlur(depthMat, blurred, 
                     cv::Size(m_config.spatialKernelSize, m_config.spatialKernelSize),
                     m_config.spatialSigma);
    
    // Copy back
    memcpy(depth, blurred.data, width * height);
}

void DepthEstimator::ApplyBilateralFilter(uint8_t* depth, int width, int height) {
    cv::Mat depthMat(height, width, CV_8UC1, depth);
    cv::Mat filtered;
    
    // Bilateral filter preserves edges while smoothing
    cv::bilateralFilter(depthMat, filtered, 
                        m_config.bilateralD,
                        m_config.bilateralSigmaColor,
                        m_config.bilateralSigmaSpace);
    
    memcpy(depth, filtered.data, width * height);
}

void DepthEstimator::FillHoles(uint8_t* depth, int width, int height) {
    cv::Mat depthMat(height, width, CV_8UC1, depth);
    
    // Create mask of invalid/zero pixels
    cv::Mat mask = depthMat == 0;
    
    // Count zero pixels
    int zeroCount = cv::countNonZero(mask);
    if (zeroCount == 0) return;
    
    // Use inpainting to fill holes
    if (zeroCount < width * height / 10) {  // Only if less than 10% are holes
        cv::Mat filled;
        cv::inpaint(depthMat, mask, filled, 5, cv::INPAINT_TELEA);
        memcpy(depth, filled.data, width * height);
    } else {
        // Too many holes, just do morphological closing
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::Mat closed;
        cv::morphologyEx(depthMat, closed, cv::MORPH_CLOSE, kernel);
        memcpy(depth, closed.data, width * height);
    }
}

bool DepthEstimator::EstimateDepth(const uint8_t* imageData, int width, int height,
                                    uint8_t* depthOut, float* rawDepthOut) {
#if !HAS_ONNXRUNTIME
    return false;
#else
    if (!m_initialized || !m_session || !m_session->initialized) {
        return false;
    }
    
    auto startTotal = std::chrono::high_resolution_clock::now();
    
    try {
        // Preprocess input
        PreprocessImage(imageData, width, height, m_preprocessBuffer);
        
        // Prepare input tensor
        std::vector<int64_t> inputShape = {1, 3, m_config.inputHeight, m_config.inputWidth};
        auto inputTensor = Ort::Value::CreateTensor<float>(
            m_session->memoryInfo,
            m_preprocessBuffer.data(),
            m_preprocessBuffer.size(),
            inputShape.data(),
            inputShape.size());
        
        // Run inference
        auto startInference = std::chrono::high_resolution_clock::now();
        
        auto outputTensors = m_session->session->Run(
            Ort::RunOptions{nullptr},
            m_session->inputNames.data(),
            &inputTensor, 1,
            m_session->outputNames.data(),
            m_session->outputNames.size());
        
        auto endInference = std::chrono::high_resolution_clock::now();
        float inferenceMs = std::chrono::duration<float, std::milli>(
            endInference - startInference).count();
        
        // Get output
        auto& outputTensor = outputTensors[0];
        auto outputShape = outputTensor.GetTensorTypeAndShapeInfo().GetShape();
        const float* outputData = outputTensor.GetTensorData<float>();
        
        // Determine output dimensions
        int outHeight = static_cast<int>(outputShape.size() >= 3 ? outputShape[outputShape.size() - 2] : m_config.inputHeight);
        int outWidth = static_cast<int>(outputShape.size() >= 2 ? outputShape[outputShape.size() - 1] : m_config.inputWidth);
        int outPixels = outHeight * outWidth;
        
        // Copy to working buffer
        m_depthBuffer.assign(outputData, outputData + outPixels);
        
        // Apply temporal filtering for smoothness
        if (m_config.enableTemporalFilter) {
            ApplyTemporalFilter(m_depthBuffer, outWidth, outHeight);
        }
        
        // Resize to output dimensions
        cv::Mat depthMat(outHeight, outWidth, CV_32FC1, m_depthBuffer.data());
        cv::Mat resizedDepth;
        cv::resize(depthMat, resizedDepth, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        
        // Normalize and convert to uint8
        m_outputBuffer.resize(width * height);
        
        double minVal, maxVal;
        cv::minMaxLoc(resizedDepth, &minVal, &maxVal);
        double range = maxVal - minVal;
        if (range < 1e-6) range = 1.0;
        
        for (int i = 0; i < width * height; i++) {
            float val = resizedDepth.at<float>(i / width, i % width);
            float normalized = static_cast<float>((val - minVal) / range);
            
            if (m_config.invertDepth) {
                normalized = 1.0f - normalized;
            }
            
            m_outputBuffer[i] = static_cast<uint8_t>(std::clamp(normalized * 255.0f, 0.0f, 255.0f));
            
            if (rawDepthOut) {
                rawDepthOut[i] = normalized;
            }
        }
        
        // Apply spatial filtering
        if (m_config.enableSpatialFilter) {
            ApplySpatialFilter(m_outputBuffer.data(), width, height);
        }
        
        if (m_config.enableBilateralFilter) {
            ApplyBilateralFilter(m_outputBuffer.data(), width, height);
        }
        
        if (m_config.enableHoleFilling) {
            FillHoles(m_outputBuffer.data(), width, height);
        }
        
        // Copy to output
        memcpy(depthOut, m_outputBuffer.data(), width * height);
        
        auto endTotal = std::chrono::high_resolution_clock::now();
        float totalMs = std::chrono::duration<float, std::milli>(endTotal - startTotal).count();
        
        // Update statistics
        {
            std::lock_guard<std::mutex> lock(m_statsMutex);
            m_inferenceTimesMs.push_back(inferenceMs);
            m_processingTimesMs.push_back(totalMs);
            while (m_inferenceTimesMs.size() > 100) m_inferenceTimesMs.pop_front();
            while (m_processingTimesMs.size() > 100) m_processingTimesMs.pop_front();
        }
        
        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error during inference: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during depth estimation: " << e.what() << std::endl;
        return false;
    }
#endif
}

bool DepthEstimator::EstimateDepthStereo(const uint8_t* leftImage, const uint8_t* rightImage,
                                          int width, int height, uint8_t* depthOut) {
    // For now, just use the left image
    // In the future, we could fuse stereo disparity with monocular depth
    return EstimateDepth(leftImage, width, height, depthOut);
}

void DepthEstimator::SetIMUData(const float* quaternion, const float* euler) {
    std::lock_guard<std::mutex> lock(m_imuMutex);
    if (quaternion) {
        memcpy(m_quaternion, quaternion, 4 * sizeof(float));
    }
    if (euler) {
        memcpy(m_euler, euler, 3 * sizeof(float));
    }
}

void DepthEstimator::SetTemporalFilterEnabled(bool enabled) {
    m_config.enableTemporalFilter = enabled;
}

void DepthEstimator::SetTemporalAlpha(float alpha) {
    m_config.temporalAlpha = std::clamp(alpha, 0.0f, 1.0f);
}

void DepthEstimator::ResetTemporalHistory() {
    std::lock_guard<std::mutex> lock(m_temporalMutex);
    m_depthHistory.clear();
    m_accumulatedDepth.clear();
}

float DepthEstimator::GetAverageInferenceTimeMs() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    if (m_inferenceTimesMs.empty()) return 0.0f;
    
    float sum = std::accumulate(m_inferenceTimesMs.begin(), m_inferenceTimesMs.end(), 0.0f);
    return sum / static_cast<float>(m_inferenceTimesMs.size());
}

float DepthEstimator::GetAverageProcessingTimeMs() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    if (m_processingTimesMs.empty()) return 0.0f;
    
    float sum = std::accumulate(m_processingTimesMs.begin(), m_processingTimesMs.end(), 0.0f);
    return sum / static_cast<float>(m_processingTimesMs.size());
}

// ============================================================================
// Async Processing Implementation
// ============================================================================

void DepthEstimator::SubmitFrameAsync(const uint8_t* imageData, int width, int height) {
    if (!m_initialized) return;
    
    // Skip frames if configured (for mobile)
    if (m_config.skipFrames > 0) {
        m_frameCounter++;
        if (m_frameCounter % (m_config.skipFrames + 1) != 0) {
            return;  // Skip this frame
        }
    }
    
    // Copy frame data for async processing
    {
        std::lock_guard<std::mutex> lock(m_asyncMutex);
        m_pendingFrame.assign(imageData, imageData + width * height);
        m_pendingWidth = width;
        m_pendingHeight = height;
        m_hasPendingFrame = true;
    }
    
    // Notify async thread
    m_asyncCV.notify_one();
}

bool DepthEstimator::GetCachedDepth(uint8_t* depthOut, int* width, int* height) {
    std::lock_guard<std::mutex> lock(m_cacheMutex);
    
    if (m_cachedDepth.empty()) {
        return false;
    }
    
    memcpy(depthOut, m_cachedDepth.data(), m_cachedDepth.size());
    if (width) *width = m_cachedWidth;
    if (height) *height = m_cachedHeight;
    
    m_hasNewDepth = false;  // Mark as consumed
    return true;
}

bool DepthEstimator::IsNewDepthReady() const {
    return m_hasNewDepth.load();
}

void DepthEstimator::StartAsyncProcessing() {
    if (m_asyncRunning) return;
    
    m_asyncStop = false;
    m_asyncRunning = true;
    m_asyncThread = std::thread(&DepthEstimator::AsyncProcessingThread, this);
    
    std::cout << "Async depth processing started" << std::endl;
}

void DepthEstimator::StopAsyncProcessing() {
    if (!m_asyncRunning) return;
    
    m_asyncStop = true;
    m_asyncCV.notify_all();
    
    if (m_asyncThread.joinable()) {
        m_asyncThread.join();
    }
    
    m_asyncRunning = false;
    std::cout << "Async depth processing stopped" << std::endl;
}

void DepthEstimator::SetInputResolution(int width, int height) {
    m_config.inputWidth = width;
    m_config.inputHeight = height;
    
    // Reallocate buffers
    int modelPixels = width * height;
    m_preprocessBuffer.resize(modelPixels * 3);
    m_depthBuffer.resize(modelPixels);
}

void DepthEstimator::AsyncProcessingThread() {
    std::vector<uint8_t> localFrame;
    std::vector<uint8_t> localDepth;
    int localWidth = 0, localHeight = 0;
    
    while (!m_asyncStop) {
        // Wait for a frame
        {
            std::unique_lock<std::mutex> lock(m_asyncMutex);
            m_asyncCV.wait(lock, [this] {
                return m_hasPendingFrame || m_asyncStop;
            });
            
            if (m_asyncStop) break;
            
            if (m_hasPendingFrame) {
                // Take the pending frame
                localFrame = std::move(m_pendingFrame);
                localWidth = m_pendingWidth;
                localHeight = m_pendingHeight;
                m_hasPendingFrame = false;
            }
        }
        
        if (localFrame.empty()) continue;
        
        // Process the frame
        localDepth.resize(localWidth * localHeight);
        
        if (EstimateDepth(localFrame.data(), localWidth, localHeight, localDepth.data())) {
            // Store result in cache
            std::lock_guard<std::mutex> lock(m_cacheMutex);
            m_cachedDepth = std::move(localDepth);
            m_cachedWidth = localWidth;
            m_cachedHeight = localHeight;
            m_hasNewDepth = true;
        }
        
        // Rate limiting - target fps
        if (m_config.targetFps > 0) {
            int targetMs = 1000 / m_config.targetFps;
            float actualMs = GetAverageInferenceTimeMs();
            if (actualMs < targetMs) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(static_cast<int>(targetMs - actualMs)));
            }
        }
    }
}

// ============================================================================
// Global depth estimator instance and live depth processing
// ============================================================================

static DepthEstimator* g_depthEstimator = nullptr;
static std::mutex g_depthMutex;
static std::atomic<bool> g_liveDepthEnabled(false);
static std::vector<uint8_t> g_lastDepthFrame;
static int g_lastDepthWidth = 0;
static int g_lastDepthHeight = 0;
static std::mutex g_depthFrameMutex;

static DepthEstimator* GetDepthEstimatorInstance() {
    std::lock_guard<std::mutex> lock(g_depthMutex);
    if (!g_depthEstimator) {
        g_depthEstimator = new DepthEstimator();
    }
    return g_depthEstimator;
}

// C API implementations

extern "C" {

AIR_API int InitializeDepthEstimator(const char* modelPath, int useGPU) {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    
    DepthEstimatorConfig config;
    if (modelPath && strlen(modelPath) > 0) {
        config.modelPath = modelPath;
    }
    config.useGPU = (useGPU != 0);
    
    return estimator->Initialize(config) ? 1 : 0;
}

AIR_API void ShutdownDepthEstimator() {
    std::lock_guard<std::mutex> lock(g_depthMutex);
    if (g_depthEstimator) {
        g_depthEstimator->Shutdown();
        delete g_depthEstimator;
        g_depthEstimator = nullptr;
    }
    g_liveDepthEnabled = false;
}

AIR_API int IsDepthEstimatorReady() {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    return estimator->IsInitialized() ? 1 : 0;
}

AIR_API int EstimateDepthFromImage(const uint8_t* imageData, int width, int height,
                                    uint8_t* depthOut) {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    return estimator->EstimateDepth(imageData, width, height, depthOut) ? 1 : 0;
}

AIR_API void SetLiveDepthEnabled(int enabled) {
    g_liveDepthEnabled = (enabled != 0);
}

AIR_API int IsLiveDepthEnabled() {
    return g_liveDepthEnabled ? 1 : 0;
}

AIR_API void SetDepthTemporalFilter(int enabled) {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    estimator->SetTemporalFilterEnabled(enabled != 0);
}

AIR_API void SetDepthTemporalAlpha(float alpha) {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    estimator->SetTemporalAlpha(alpha);
}

AIR_API void ResetDepthTemporalHistory() {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    estimator->ResetTemporalHistory();
}

AIR_API float GetDepthInferenceTimeMs() {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    return estimator->GetAverageInferenceTimeMs();
}

AIR_API int InitializeDepthEstimatorFast(const char* modelPath, int useGPU, int inputSize) {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    
    DepthEstimatorConfig config;
    if (modelPath && strlen(modelPath) > 0) {
        config.modelPath = modelPath;
    }
    config.useGPU = (useGPU != 0);
    
    // Fast mode settings
    config.inputWidth = inputSize > 0 ? inputSize : 256;
    config.inputHeight = inputSize > 0 ? inputSize : 256;
    config.asyncMode = true;
    config.targetFps = 15;
    config.cacheDepth = true;
    
    // Disable expensive filters
    config.enableSpatialFilter = false;
    config.enableBilateralFilter = false;
    config.enableHoleFilling = false;
    config.temporalWindowSize = 3;
    config.processingThreads = 2;
    
    return estimator->Initialize(config) ? 1 : 0;
}

AIR_API int InitializeDepthEstimatorMobile(const char* modelPath) {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    
    DepthEstimatorConfig config;
    if (modelPath && strlen(modelPath) > 0) {
        config.modelPath = modelPath;
    } else {
        // Use quantized model for mobile
        config.modelPath = "depth_anything_v2_small_q4f16.onnx";
    }
    
    config.useGPU = true;  // NNAPI on Android
    config.mobileMode = true;
    
    // Mobile optimized settings
    config.inputWidth = 192;   // Very small for speed
    config.inputHeight = 192;
    config.asyncMode = true;
    config.targetFps = 10;     // Lower target for battery
    config.skipFrames = 1;     // Process every other frame
    config.cacheDepth = true;
    
    // Minimal filtering
    config.enableTemporalFilter = true;
    config.temporalWindowSize = 2;
    config.temporalAlpha = 0.5f;
    config.enableSpatialFilter = false;
    config.enableBilateralFilter = false;
    config.enableHoleFilling = false;
    config.processingThreads = 1;
    
    return estimator->Initialize(config) ? 1 : 0;
}

AIR_API void SubmitDepthFrameAsync(const uint8_t* imageData, int width, int height) {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    estimator->SubmitFrameAsync(imageData, width, height);
}

AIR_API int GetCachedDepthImage(uint8_t* depthOut, int* width, int* height) {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    return estimator->GetCachedDepth(depthOut, width, height) ? 1 : 0;
}

AIR_API int IsNewDepthAvailable() {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    return estimator->IsNewDepthReady() ? 1 : 0;
}

AIR_API void SetDepthInputResolution(int width, int height) {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    estimator->SetInputResolution(width, height);
}

// ===== AR Navigation API =====

AIR_API int InitializeDepthForAR(const char* modelPath) {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    
    DepthEstimatorConfig config;
    if (modelPath && strlen(modelPath) > 0) {
        config.modelPath = modelPath;
    } else {
        // Use quantized model for AR (smallest, fastest)
        config.modelPath = "depth_anything_v2_small_q4f16.onnx";
    }
    
    config.useGPU = true;  // Use GPU/NNAPI when available
    config.arMode = true;
    
    // Ultra-fast AR settings (128x128 = ~25fps on CPU, ~50fps on GPU)
    config.inputWidth = 128;
    config.inputHeight = 128;
    config.asyncMode = true;
    config.targetFps = 20;  // Good enough for AR navigation
    config.cacheDepth = true;
    config.skipFrames = 0;
    
    // Light temporal filtering for stability
    config.enableTemporalFilter = true;
    config.temporalWindowSize = 2;
    config.temporalAlpha = 0.5f;
    config.enableInterpolation = true;
    
    // Disable all heavy filters
    config.enableSpatialFilter = false;
    config.enableBilateralFilter = false;
    config.enableHoleFilling = false;
    config.processingThreads = 2;
    
    // AR depth scale (rough estimate: depth 0-1 maps to 0-10 meters)
    config.arDepthScale = 10.0f;
    
    return estimator->Initialize(config) ? 1 : 0;
}

AIR_API float GetDepthAtPoint(int x, int y) {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    
    std::vector<uint8_t> depthBuffer;
    int width = 0, height = 0;
    
    // Get a copy of cached depth
    {
        // Access cached depth directly
        if (!estimator->IsNewDepthReady() && !estimator->IsInitialized()) {
            return -1.0f;
        }
        
        depthBuffer.resize(480 * 640);  // Max size
        if (!estimator->GetCachedDepth(depthBuffer.data(), &width, &height)) {
            return -1.0f;
        }
    }
    
    // Bounds check
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return -1.0f;
    }
    
    // Return normalized depth (0-1)
    uint8_t depthValue = depthBuffer[y * width + x];
    return static_cast<float>(depthValue) / 255.0f;
}

AIR_API float GetDistanceAtPoint(int x, int y) {
    float normalizedDepth = GetDepthAtPoint(x, y);
    if (normalizedDepth < 0) {
        return -1.0f;
    }
    
    // Rough distance estimation
    // Monocular depth is relative, not metric, but we can approximate
    // Assuming inverse depth: closer objects have higher depth values
    // Map depth 0-1 to distance 0.5m - 20m (typical indoor/outdoor range)
    
    float minDist = 0.5f;   // Minimum distance in meters
    float maxDist = 20.0f;  // Maximum distance in meters
    
    // Invert: high depth value = close = small distance
    float distance = minDist + (1.0f - normalizedDepth) * (maxDist - minDist);
    
    return distance;
}

AIR_API float GetGroundPlaneDepth() {
    DepthEstimator* estimator = GetDepthEstimatorInstance();
    
    std::vector<uint8_t> depthBuffer(480 * 640);
    int width = 0, height = 0;
    
    if (!estimator->GetCachedDepth(depthBuffer.data(), &width, &height)) {
        return -1.0f;
    }
    
    if (width == 0 || height == 0) {
        return -1.0f;
    }
    
    // Sample the bottom 20% of the image (likely to be ground/road)
    int startY = height * 80 / 100;  // Bottom 20%
    int centerX = width / 2;
    int sampleWidth = width / 3;     // Middle third
    
    float sum = 0.0f;
    int count = 0;
    
    for (int y = startY; y < height; y++) {
        for (int x = centerX - sampleWidth / 2; x < centerX + sampleWidth / 2; x++) {
            if (x >= 0 && x < width) {
                sum += depthBuffer[y * width + x];
                count++;
            }
        }
    }
    
    if (count == 0) {
        return -1.0f;
    }
    
    return (sum / count) / 255.0f;  // Normalized 0-1
}

}  // extern "C"
