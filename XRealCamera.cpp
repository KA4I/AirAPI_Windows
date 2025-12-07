#include "pch.h"
#include "XRealCamera.h"

// OpenCV for image processing (not capture)
#include <opencv2/opencv.hpp>

#include <iostream>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <cstdio>

// Chunk reordering table from Python implementation
const int XRealCamera::CHUNK_MAP[128] = {
    119, 54, 21, 0, 108, 22, 51, 63, 93, 99, 67, 7, 32, 112, 52, 43,
    14, 35, 75, 116, 64, 71, 44, 89, 18, 88, 26, 61, 70, 56, 90, 79,
    87, 120, 81, 101, 121, 17, 72, 31, 53, 124, 127, 113, 111, 36, 48,
    19, 37, 83, 126, 74, 109, 5, 84, 41, 76, 30, 110, 29, 12, 115, 28,
    102, 105, 62, 103, 20, 3, 68, 49, 77, 117, 125, 106, 60, 69, 98, 9,
    16, 78, 47, 40, 2, 118, 34, 13, 50, 46, 80, 85, 66, 42, 123, 122,
    96, 11, 25, 97, 39, 6, 86, 1, 8, 82, 92, 59, 104, 24, 15, 73, 65,
    38, 58, 10, 23, 33, 55, 57, 107, 100, 94, 27, 95, 45, 91, 4, 114
};

XRealCamera::XRealCamera()
    : m_pipe(nullptr)
    , m_cameraIndex(-1)
    , m_isCapturing(false)
    , m_shouldStop(false)
    , m_lastSequence(-1)
    , m_hasLastFrame(false)
    , m_frameTimestamp(0)
    , m_sequenceNumber(0)
    , m_rotation(2)  // Default rotation mode (same as Python)
    , m_frameReady(false)
    , m_callback(nullptr)
    , m_callbackUserData(nullptr)
    , m_depthReady(false)
    , m_liveDepthEnabled(false)
{
    // Pre-allocate buffers
    m_lastRawFrame.resize(640 * 482);
    m_outputBuffer.resize(RAW_WIDTH * 480 * 2);  // 640 * 480 * 2 for both eyes
    m_leftImage.resize(OUTPUT_WIDTH * OUTPUT_HEIGHT);
    m_rightImage.resize(OUTPUT_WIDTH * OUTPUT_HEIGHT);
    m_depthBuffer.resize(OUTPUT_WIDTH * OUTPUT_HEIGHT);
}

XRealCamera::~XRealCamera() {
    StopCapture();
    m_depthEstimator.Shutdown();
}

std::string XRealCamera::GetCameraNameFromFFmpeg() {
    // Run ffmpeg to list devices
    // cmd = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    
    std::string result;
    FILE* pipe = _popen("ffmpeg -list_devices true -f dshow -i dummy 2>&1", "r");
    if (!pipe) {
        return "";
    }
    
    char buffer[128];
    std::string output = "";
    while (fgets(buffer, 128, pipe) != NULL) {
        output += buffer;
    }
    _pclose(pipe);
    
    // Parse output
    // Look for lines with "(video)" and "dshow @"
    // Extract name between quotes
    
    std::stringstream ss(output);
    std::string line;
    std::vector<std::string> cameras;
    
    while (std::getline(ss, line)) {
        if (line.find("(video)") != std::string::npos && line.find("dshow @") != std::string::npos) {
            size_t firstQuote = line.find("\"");
            size_t lastQuote = line.rfind("\"");
            if (firstQuote != std::string::npos && lastQuote != std::string::npos && lastQuote > firstQuote) {
                std::string name = line.substr(firstQuote + 1, lastQuote - firstQuote - 1);
                cameras.push_back(name);
            }
        }
    }
    
    // Prefer "UVC Camera" or "XREAL"
    for (const auto& cam : cameras) {
        if (cam.find("UVC Camera") != std::string::npos) return cam;
    }
    for (const auto& cam : cameras) {
        if (cam.find("XREAL") != std::string::npos) return cam;
    }
    
    if (!cameras.empty()) {
        return cameras[0];
    }
    
    return "";
}

bool XRealCamera::Initialize() {
    // Find the camera name using ffmpeg
    m_cameraName = GetCameraNameFromFFmpeg();
    
    if (m_cameraName.empty()) {
        std::cerr << "No camera found via ffmpeg dshow." << std::endl;
        return false;
    }
    
    std::cout << "Using camera: " << m_cameraName << std::endl;
    return true;
}

bool XRealCamera::StartCapture() {
    if (m_isCapturing) {
        return true;
    }

    if (m_cameraName.empty()) {
        if (!Initialize()) {
            return false;
        }
    }

    m_shouldStop = false;
    m_isCapturing = true;
    m_hasLastFrame = false;
    m_lastSequence = -1;
    m_frameReady = false;

    // Start capture thread
    m_captureThread = std::thread(&XRealCamera::CaptureThread, this);

    return true;
}

void XRealCamera::StopCapture() {
    if (!m_isCapturing) {
        return;
    }

    m_shouldStop = true;
    m_isCapturing = false;

    if (m_captureThread.joinable()) {
        m_captureThread.join();
    }
    
    // Pipe is closed in thread
}

bool XRealCamera::IsCapturing() const {
    return m_isCapturing;
}

void XRealCamera::CaptureThread() {
    std::cout << "Capture thread started" << std::endl;
    
    // Construct ffmpeg command
    // ffmpeg -f dshow -video_size 640x241 -pixel_format yuyv422 -i video="Camera" -f rawvideo -pix_fmt yuyv422 -
    std::string cmd = "ffmpeg -f dshow -video_size 640x241 -pixel_format yuyv422 -i video=\"" + m_cameraName + "\" -f rawvideo -pix_fmt yuyv422 - 2>NUL";
    
    std::cout << "Starting ffmpeg: " << cmd << std::endl;
    
    m_pipe = _popen(cmd.c_str(), "rb");
    if (!m_pipe) {
        std::cerr << "Failed to start ffmpeg process" << std::endl;
        m_isCapturing = false;
        return;
    }
    
    const size_t frameSize = 640 * 241 * 2; // 308480 bytes
    std::vector<uint8_t> buffer(frameSize);
    
    while (!m_shouldStop) {
        size_t bytesRead = fread(buffer.data(), 1, frameSize, m_pipe);
        
        if (bytesRead == 0) {
            // End of stream or error
            if (feof(m_pipe)) {
                std::cout << "ffmpeg stream ended" << std::endl;
                break;
            }
            // Wait a bit if no data
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        if (bytesRead != frameSize) {
            // Incomplete frame, try to read rest?
            // For now, just skip partial frames or wait for more data
            // But fread blocks until count is read or EOF/error.
            // So if we got less, it's likely an issue.
            // Actually, fread might return partial if pipe buffer is empty?
            // No, standard fread blocks.
            
            // If we got partial data, we should probably loop to fill the buffer
            size_t totalRead = bytesRead;
            while (totalRead < frameSize && !m_shouldStop) {
                size_t remaining = frameSize - totalRead;
                size_t n = fread(buffer.data() + totalRead, 1, remaining, m_pipe);
                if (n == 0) break;
                totalRead += n;
            }
            
            if (totalRead != frameSize) {
                std::cerr << "Incomplete frame read: " << totalRead << std::endl;
                continue;
            }
        }
        
        ProcessRawFrame(buffer);
    }
    
    _pclose(m_pipe);
    m_pipe = nullptr;
    
    std::cout << "Capture thread ended" << std::endl;
}

void XRealCamera::ProcessRawFrame(const std::vector<uint8_t>& frameData) {
    // The XReal camera sends 640x241 in YUY2/YUYV format
    // Python reads raw YUY2 bytes: 640*241*2 = 308,480 bytes
    
    size_t totalBytes = frameData.size();
    
    // We expect exactly 308,480 bytes
    if (totalBytes != 308480) {
        return;
    }
    
    // Frame pairing and sequence check
    if (!m_hasLastFrame) {
        m_lastRawFrame = frameData;
        m_hasLastFrame = true;
        return;
    }
    
    // Check sequence numbers
    // 640*480 + 18 = 307218
    if (m_lastRawFrame.size() >= 307220 && frameData.size() >= 307220) {
        uint16_t seq1 = *reinterpret_cast<uint16_t*>((uint8_t*)m_lastRawFrame.data() + 307218);
        uint16_t seq2 = *reinterpret_cast<uint16_t*>((uint8_t*)frameData.data() + 307218);
        
        if (seq1 != seq2) {
            // Mismatch, keep the new one
            m_lastRawFrame = frameData;
            return;
        }
    }
    
    // Process the frame pair
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        
        // Output buffer size: 640 * 480 * 2
        if (m_outputBuffer.size() != 640 * 480 * 2) {
            m_outputBuffer.resize(640 * 480 * 2);
        }
        std::fill(m_outputBuffer.begin(), m_outputBuffer.end(), 0);
        
        HandleFrame(m_lastRawFrame.data(), m_outputBuffer.data());
        HandleFrame(frameData.data(), m_outputBuffer.data());
        
        ExtractStereoImages();
        
        m_frameTimestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
        m_sequenceNumber++;
        m_frameReady = true;
        m_hasLastFrame = false;
        
        // Submit frame for async depth processing (non-blocking)
        if (m_liveDepthEnabled && m_depthEstimator.IsInitialized()) {
            // Submit to async thread - returns immediately
            m_depthEstimator.SubmitFrameAsync(m_leftImage.data(), OUTPUT_WIDTH, OUTPUT_HEIGHT);
        }
    }
    
    // Check if async depth result is ready
    if (m_liveDepthEnabled && m_depthEstimator.IsNewDepthReady()) {
        std::vector<uint8_t> depthResult(OUTPUT_WIDTH * OUTPUT_HEIGHT);
        int depthW, depthH;
        if (m_depthEstimator.GetCachedDepth(depthResult.data(), &depthW, &depthH)) {
            std::lock_guard<std::mutex> depthLock(m_depthMutex);
            m_depthBuffer = std::move(depthResult);
            m_depthReady = true;
        }
    }
    
    // Callback (after all processing is done)
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        if (m_callback) {
            XRealStereoFrame stereoFrame;
            stereoFrame.leftImage = m_leftImage.data();
            stereoFrame.rightImage = m_rightImage.data();
            stereoFrame.width = OUTPUT_WIDTH;
            stereoFrame.height = OUTPUT_HEIGHT;
            stereoFrame.timestamp = m_frameTimestamp;
            stereoFrame.sequenceNumber = m_sequenceNumber;
            stereoFrame.valid = true;
            m_callback(&stereoFrame, m_callbackUserData);
        }
    }
}

void XRealCamera::HandleFrame(const uint8_t* inFrame, uint8_t* outBuffer) {
    const int totalPixels = 640 * 480;
    const int numBlocks = 128;
    const int blockSize = 2400;

    // Find starting block by minimum sum of first 128 bytes
    // Python: blocks = in_frame[:640*480].reshape((128, 2400))
    // min_idx = blocks[:,:128].sum(axis=1).argmin()
    
    int minIdx = 0;
    int minSum = INT_MAX;

    for (int i = 0; i < numBlocks; i++) {
        int sum = 0;
        const uint8_t* block = inFrame + i * blockSize;
        for (int j = 0; j < 128; j++) {
            sum += block[j];
        }
        if (sum < minSum) {
            minSum = sum;
            minIdx = i;
        }
    }

    // Find map index
    int mapIdx = -1;
    for (int i = 0; i < numBlocks; i++) {
        if (CHUNK_MAP[i] == minIdx) {
            mapIdx = i;
            break;
        }
    }

    if (mapIdx < 0) {
        return;
    }

    // Check left/right flag
    // Python: is_right = in_frame[480*640 + 0x3b]
    bool isRight = inFrame[totalPixels + 0x3b] != 0;

    DescrambleChunks(inFrame, mapIdx, isRight, outBuffer);
}

void XRealCamera::DescrambleChunks(const uint8_t* blocks, int mapIdx, bool isRight, uint8_t* outBuffer) {
    const int numBlocks = 128;
    const int blockSize = 2400;
    
    if (m_rotation == 2) {
        // Python logic:
        // out_img = np_out.reshape((960, 640), order='F')
        // if is_right: dest = out_img[480:, :]
        // else: dest = out_img[:480, :][::-1, ::-1]
        //
        // We map directly to outBuffer (which is np_out)
        // outBuffer index = col * 960 + row
        
        int p_y = 0; // 0..639 (corresponds to col index of dest in Python loop?)
                     // Wait, Python: dest[p_x, p_y:p_y_new]
                     // p_x is row index of dest? p_y is col index?
                     // dest has shape (480, 640).
                     // So p_x is 0..479. p_y is 0..639.
        int p_x = 0; // 0..479
        
        for (int t_idx = 0; t_idx < numBlocks; t_idx++) {
            const uint8_t* source = blocks + CHUNK_MAP[mapIdx] * blockSize;
            int pos = 0;
            
            while (pos < blockSize) {
                int p_y_new = std::min(640, p_y + blockSize - pos);
                int count = p_y_new - p_y;
                
                for (int k = 0; k < count; k++) {
                    int r = p_x;        // 0..479
                    int c = p_y + k;    // 0..639
                    
                    int outIdx;
                    if (isRight) {
                        // dest = out_img[480:, :]
                        // target row in out_img = 480 + r
                        // target col in out_img = c
                        // Index = c * 960 + (480 + r)
                        outIdx = c * 960 + 480 + r;
                    } else {
                        // dest = out_img[:480, :]
                        // dest = dest[::-1, ::-1]
                        // target row in out_img = 479 - r
                        // target col in out_img = 639 - c
                        // Index = (639 - c) * 960 + (479 - r)
                        outIdx = (639 - c) * 960 + (479 - r);
                    }
                    
                    outBuffer[outIdx] = source[pos + k];
                }
                
                pos += count;
                p_y += count;
                
                if (p_y >= 640) {
                    p_x++;
                    p_y = 0;
                }
            }
            mapIdx = (mapIdx + 1) % numBlocks;
        }
    } else {
        // Rotation 0/1 logic (not fully verified against Python but keeping structure)
        // Python: out = np_out[480*640:] if is_right else np_out[:480*640]
        // This assumes flat buffer split.
        // But wait, Python uses np_out which is 640*480*2.
        // 480*640 = 307200.
        // So it splits exactly in half.
        
        uint8_t* dest = isRight ? outBuffer + (480 * 640) : outBuffer;
        
        // Handle rotation 1 flip for left eye
        // if self._rotation == 1 and not is_right: out = out[::-1]
        // We'll handle flip after copy or during copy?
        // Python copies chunks then flips? No, out is a view.
        // If out is reversed, then out[idx] maps to buffer[len-1-idx].
        
        bool flip = (m_rotation == 1 && !isRight);
        
        for (int t_idx = 0; t_idx < numBlocks; t_idx++) {
            const uint8_t* source = blocks + CHUNK_MAP[mapIdx] * blockSize;
            
            if (flip) {
                // Copy to reversed position
                // out[t_idx * 2400 : ...] = source
                // means buffer[len - 1 - (t_idx*2400 + k)] = source[k]
                // This is complicated. Let's just copy then reverse later if needed.
                // Or just assume Rotation 2 is the main target.
                memcpy(dest + t_idx * blockSize, source, blockSize);
            } else {
                memcpy(dest + t_idx * blockSize, source, blockSize);
            }
            
            mapIdx = (mapIdx + 1) % numBlocks;
        }
        
        if (flip) {
            std::reverse(dest, dest + 480 * 640);
        }
    }
}

void XRealCamera::ExtractStereoImages() {
    if (m_rotation == 2) {
        // Python: return np_out.reshape((960, 640), order='F').transpose()
        // Result is (640, 960).
        // Left: [:, :480]. Right: [:, 480:].
        // m_leftImage should be 640x480.
        
        // m_outputBuffer is (960, 640) Fortran order.
        // out[r, c] = m_outputBuffer[c * 960 + r]
        // We want Transpose: T[r, c] = out[c, r]
        // T[y, x] = out[x, y] = m_outputBuffer[y * 960 + x]
        
        // Left Image (x < 480)
        for (int y = 0; y < 640; y++) {
            for (int x = 0; x < 480; x++) {
                // Flip horizontally to match user perspective
                m_leftImage[y * 480 + (479 - x)] = m_outputBuffer[y * 960 + x];
            }
        }
        
        // Right Image (x >= 480)
        for (int y = 0; y < 640; y++) {
            for (int x = 0; x < 480; x++) {
                // Flip horizontally to match user perspective
                m_rightImage[y * 480 + (479 - x)] = m_outputBuffer[y * 960 + 480 + x];
            }
        }
    } else {
        memcpy(m_leftImage.data(), m_outputBuffer.data(), 480 * 640);
        memcpy(m_rightImage.data(), m_outputBuffer.data() + 480 * 640, 480 * 640);
    }
}

bool XRealCamera::GetLatestFrame(XRealStereoFrame* frame) {
    if (!frame) return false;

    std::lock_guard<std::mutex> lock(m_frameMutex);

    if (!m_frameReady) {
        frame->valid = false;
        return false;
    }

    frame->leftImage = m_leftImage.data();
    frame->rightImage = m_rightImage.data();
    frame->width = OUTPUT_WIDTH;
    frame->height = OUTPUT_HEIGHT;
    frame->timestamp = m_frameTimestamp;
    frame->sequenceNumber = m_sequenceNumber;
    frame->valid = true;

    return true;
}

uint8_t* XRealCamera::GetLeftImage(int* width, int* height) {
    std::lock_guard<std::mutex> lock(m_frameMutex);

    if (!m_frameReady) {
        return nullptr;
    }

    if (width) *width = OUTPUT_WIDTH;
    if (height) *height = OUTPUT_HEIGHT;

    return m_leftImage.data();
}

uint8_t* XRealCamera::GetRightImage(int* width, int* height) {
    std::lock_guard<std::mutex> lock(m_frameMutex);

    if (!m_frameReady) {
        return nullptr;
    }

    if (width) *width = OUTPUT_WIDTH;
    if (height) *height = OUTPUT_HEIGHT;

    return m_rightImage.data();
}

void XRealCamera::SetFrameCallback(XRealFrameCallback callback, void* userData) {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    m_callback = callback;
    m_callbackUserData = userData;
}

std::string XRealCamera::GetDeviceName() const {
    return m_cameraName;
}

void XRealCamera::SetRotation(int rotation) {
    if (rotation >= 0 && rotation <= 2) {
        m_rotation = rotation;
    }
}

int XRealCamera::GetRotation() const {
    return m_rotation;
}

bool XRealCamera::GetDepthImage(uint8_t* depthOut, int* width, int* height) {
    if (!depthOut) return false;
    
    // First check if we have cached depth
    {
        std::lock_guard<std::mutex> lock(m_depthMutex);
        
        if (m_depthReady) {
            memcpy(depthOut, m_depthBuffer.data(), OUTPUT_WIDTH * OUTPUT_HEIGHT);
            if (width) *width = OUTPUT_WIDTH;
            if (height) *height = OUTPUT_HEIGHT;
            return true;
        }
    }
    
    // If no cached depth, try to compute on demand
    if (!m_depthEstimator.IsInitialized() || !m_frameReady) {
        return false;
    }
    
    // Copy frame data with frame lock
    std::vector<uint8_t> imageCopy;
    {
        std::lock_guard<std::mutex> frameLock(m_frameMutex);
        imageCopy = m_leftImage;
    }
    
    // Compute depth without locks (CPU intensive)
    std::vector<uint8_t> depthResult(OUTPUT_WIDTH * OUTPUT_HEIGHT);
    if (!m_depthEstimator.EstimateDepth(imageCopy.data(), OUTPUT_WIDTH, OUTPUT_HEIGHT,
                                         depthResult.data())) {
        return false;
    }
    
    // Store and return result
    {
        std::lock_guard<std::mutex> depthLock(m_depthMutex);
        m_depthBuffer = std::move(depthResult);
        m_depthReady = true;
        memcpy(depthOut, m_depthBuffer.data(), OUTPUT_WIDTH * OUTPUT_HEIGHT);
    }
    
    if (width) *width = OUTPUT_WIDTH;
    if (height) *height = OUTPUT_HEIGHT;
    
    return true;
}

uint8_t* XRealCamera::GetDepthImagePtr(int* width, int* height) {
    std::lock_guard<std::mutex> lock(m_depthMutex);
    
    if (!m_depthReady) {
        return nullptr;
    }
    
    if (width) *width = OUTPUT_WIDTH;
    if (height) *height = OUTPUT_HEIGHT;
    
    return m_depthBuffer.data();
}

void XRealCamera::SetLiveDepthEnabled(bool enabled) {
    m_liveDepthEnabled = enabled;
    if (!enabled) {
        m_depthReady = false;
    }
}

bool XRealCamera::IsLiveDepthEnabled() const {
    return m_liveDepthEnabled;
}

bool XRealCamera::InitializeDepth(const std::string& modelPath, bool useGPU) {
    DepthEstimatorConfig config;
    if (!modelPath.empty()) {
        config.modelPath = modelPath;
    }
    config.useGPU = useGPU;
    
    // Use fast settings by default for real-time camera usage
    config.inputWidth = 256;
    config.inputHeight = 256;
    config.asyncMode = true;
    config.targetFps = 15;
    config.cacheDepth = true;
    config.enableSpatialFilter = false;
    config.enableBilateralFilter = false;
    config.enableHoleFilling = false;
    config.temporalWindowSize = 3;
    config.processingThreads = 2;
    
    return m_depthEstimator.Initialize(config);
}

// Global camera instance
static XRealCamera* g_camera = nullptr;
static std::mutex g_cameraMutex;

static XRealCamera* GetCameraInstance() {
    std::lock_guard<std::mutex> lock(g_cameraMutex);
    if (!g_camera) {
        g_camera = new XRealCamera();
    }
    return g_camera;
}

// C API implementations

extern "C" {

AIR_API int StartCameraCapture() {
    XRealCamera* camera = GetCameraInstance();
    return camera->StartCapture() ? 1 : 0;
}

AIR_API int StopCameraCapture() {
    XRealCamera* camera = GetCameraInstance();
    if (camera->IsCapturing()) {
        camera->StopCapture();
        return 1;
    }
    return 0;
}

AIR_API int IsCameraCapturing() {
    XRealCamera* camera = GetCameraInstance();
    return camera->IsCapturing() ? 1 : 0;
}

AIR_API uint8_t* GetCameraLeftImage(int* width, int* height) {
    XRealCamera* camera = GetCameraInstance();
    return camera->GetLeftImage(width, height);
}

AIR_API uint8_t* GetCameraRightImage(int* width, int* height) {
    XRealCamera* camera = GetCameraInstance();
    return camera->GetRightImage(width, height);
}

AIR_API int GetCameraStereoFrame(uint8_t* leftOut, uint8_t* rightOut,
                                  int* width, int* height, uint64_t* timestamp) {
    XRealCamera* camera = GetCameraInstance();
    
    XRealStereoFrame frame;
    if (!camera->GetLatestFrame(&frame)) {
        return 0;
    }

    if (!frame.valid) {
        return 0;
    }

    if (leftOut && frame.leftImage) {
        memcpy(leftOut, frame.leftImage, frame.width * frame.height);
    }
    if (rightOut && frame.rightImage) {
        memcpy(rightOut, frame.rightImage, frame.width * frame.height);
    }

    if (width) *width = frame.width;
    if (height) *height = frame.height;
    if (timestamp) *timestamp = frame.timestamp;

    return 1;
}

AIR_API void SetCameraRotation(int rotation) {
    XRealCamera* camera = GetCameraInstance();
    camera->SetRotation(rotation);
}

AIR_API int GetCameraRotation() {
    XRealCamera* camera = GetCameraInstance();
    return camera->GetRotation();
}

AIR_API void SetCameraFrameCallback(XRealFrameCallback callback, void* userData) {
    XRealCamera* camera = GetCameraInstance();
    camera->SetFrameCallback(callback, userData);
}

AIR_API int InitializeCameraDepth(const char* modelPath, int useGPU) {
    XRealCamera* camera = GetCameraInstance();
    std::string path = modelPath ? modelPath : "";
    return camera->InitializeDepth(path, useGPU != 0) ? 1 : 0;
}

AIR_API int GetCameraDepthImage(uint8_t* depthOut, int* width, int* height) {
    XRealCamera* camera = GetCameraInstance();
    return camera->GetDepthImage(depthOut, width, height) ? 1 : 0;
}

AIR_API uint8_t* GetCameraDepthImagePtr(int* width, int* height) {
    XRealCamera* camera = GetCameraInstance();
    return camera->GetDepthImagePtr(width, height);
}

AIR_API void SetCameraLiveDepth(int enabled) {
    XRealCamera* camera = GetCameraInstance();
    camera->SetLiveDepthEnabled(enabled != 0);
}

AIR_API int IsCameraLiveDepthEnabled() {
    XRealCamera* camera = GetCameraInstance();
    return camera->IsLiveDepthEnabled() ? 1 : 0;
}

AIR_API void SetCameraDepthTemporalFilter(int enabled) {
    XRealCamera* camera = GetCameraInstance();
    camera->GetDepthEstimator().SetTemporalFilterEnabled(enabled != 0);
}

AIR_API void SetCameraDepthTemporalAlpha(float alpha) {
    XRealCamera* camera = GetCameraInstance();
    camera->GetDepthEstimator().SetTemporalAlpha(alpha);
}

AIR_API void ResetCameraDepthTemporalHistory() {
    XRealCamera* camera = GetCameraInstance();
    camera->GetDepthEstimator().ResetTemporalHistory();
}

AIR_API float GetCameraDepthInferenceTimeMs() {
    XRealCamera* camera = GetCameraInstance();
    return camera->GetDepthEstimator().GetAverageInferenceTimeMs();
}

}  // extern "C"
