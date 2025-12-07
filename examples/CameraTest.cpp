/**
 * @file CameraTest.cpp
 * @brief Standalone test application for XReal Air 2 Ultra camera API
 * 
 * This is a simple console application that tests the AirAPI_Windows.dll
 * camera functionality with live OpenCV display including AI-powered depth.
 * 
 * Controls:
 *   s - Save current frame as BMP
 *   r - Change rotation mode (0, 1, 2)
 *   i - Show IMU data in console
 *   d - Toggle depth view
 *   t - Toggle temporal filtering
 *   +/- - Adjust temporal filter strength
 *   c - Apply colormap to depth
 *   a - Toggle AR mode (128x128 ultra-fast)
 *   q/ESC - Quit
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <thread>
#include <iomanip>

// OpenCV for display
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

// Import the DLL functions
#define AIR_API __declspec(dllimport)

extern "C" {
    // IMU functions
    AIR_API int StartConnection();
    AIR_API int StopConnection();
    AIR_API float* GetQuaternion();
    AIR_API float* GetEuler();
    AIR_API int GetBrightness();
    
    // Camera functions
    AIR_API int StartCameraCapture();
    AIR_API int StopCameraCapture();
    AIR_API int IsCameraCapturing();
    AIR_API uint8_t* GetCameraLeftImage(int* width, int* height);
    AIR_API uint8_t* GetCameraRightImage(int* width, int* height);
    AIR_API int GetCameraStereoFrame(uint8_t* leftOut, uint8_t* rightOut,
                                      int* width, int* height, uint64_t* timestamp);
    AIR_API void SetCameraRotation(int rotation);
    AIR_API int GetCameraRotation();
    
    // Depth functions
    AIR_API int InitializeCameraDepth(const char* modelPath, int useGPU);
    AIR_API int GetCameraDepthImage(uint8_t* depthOut, int* width, int* height);
    AIR_API uint8_t* GetCameraDepthImagePtr(int* width, int* height);
    AIR_API void SetCameraLiveDepth(int enabled);
    AIR_API int IsCameraLiveDepthEnabled();
    AIR_API void SetCameraDepthTemporalFilter(int enabled);
    AIR_API void SetCameraDepthTemporalAlpha(float alpha);
    AIR_API void ResetCameraDepthTemporalHistory();
    AIR_API float GetCameraDepthInferenceTimeMs();
    
    // AR Navigation functions (128x128 ultra-fast mode for ~25fps)
    AIR_API int InitializeDepthForAR(const char* modelPath);
    AIR_API float GetDepthAtPoint(int x, int y);
    AIR_API float GetDistanceAtPoint(int x, int y);
    AIR_API float GetGroundPlaneDepth();
}

// Save a grayscale image as BMP
void SaveBMP(const char* filename, const uint8_t* data, int width, int height) {
    cv::Mat img(height, width, CV_8UC1, const_cast<uint8_t*>(data));
    cv::imwrite(filename, img);
    std::cout << "Saved: " << filename << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  XReal Air 2 Ultra - Camera Test App  " << std::endl;
    std::cout << "     with AI-Powered Depth Estimation  " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Start IMU connection
    std::cout << "[1/4] Starting IMU connection..." << std::endl;
    bool imuOk = (StartConnection() == 1);
    if (!imuOk) {
        std::cout << "      Warning: IMU not available" << std::endl;
    } else {
        std::cout << "      IMU: OK" << std::endl;
    }
    
    // Start camera capture
    std::cout << "[2/4] Starting camera capture..." << std::endl;
    if (StartCameraCapture() != 1) {
        std::cerr << "      ERROR: Failed to start camera!" << std::endl;
        std::cerr << "      Make sure XReal Air 2 Ultra glasses are connected." << std::endl;
        StopConnection();
        std::cout << "\nPress any key to exit..." << std::endl;
        cv::waitKey(0);
        return 1;
    }
    std::cout << "      Camera: OK (rotation=" << GetCameraRotation() << ")" << std::endl;
    
    // Initialize depth estimator
    std::cout << "[3/4] Initializing depth estimator..." << std::endl;
    
    // Try different model paths
    const char* modelPaths[] = {
        "depth_anything_v2_vits.onnx",
        "depth_anything_v2_small.onnx",
        "../models/depth_anything_v2_small.onnx",
        "depth_anything_v2.onnx",
        "depth_anything_v2_fixed.onnx",
        "../x64/Release/depth_anything_v2.onnx",
        "C:/DEV/AirAPI_Windows/models/depth_anything_v2_small.onnx"
    };
    
    bool depthOk = false;
    for (const char* modelPath : modelPaths) {
        std::cout << "      Trying: " << modelPath << std::endl;
        if (InitializeCameraDepth(modelPath, 1) == 1) {
            std::cout << "      Depth: OK (GPU enabled)" << std::endl;
            depthOk = true;
            break;
        }
    }
    
    if (!depthOk) {
        std::cout << "      Warning: Depth estimator not available (no ONNX model found)" << std::endl;
        std::cout << "      Download a Depth Anything V2 ONNX model and place it in the exe directory" << std::endl;
    }
    
    // Wait for first frame
    std::cout << "[4/4] Waiting for frames..." << std::endl;
    
    const int imageSize = 480 * 640;
    uint8_t* leftBuffer = new uint8_t[imageSize];
    uint8_t* rightBuffer = new uint8_t[imageSize];
    uint8_t* depthBuffer = new uint8_t[imageSize];
    
    int width = 0, height = 0;
    uint64_t timestamp = 0;
    int frameCount = 0;
    int saveCount = 0;
    
    // Wait for first valid frame
    for (int i = 0; i < 100; i++) {
        if (GetCameraStereoFrame(leftBuffer, rightBuffer, &width, &height, &timestamp) == 1) {
            std::cout << "      First frame received: " << width << "x" << height << std::endl;
            frameCount++;
            break;
        }
        Sleep(30);
    }
    
    if (frameCount == 0) {
        std::cerr << "      Warning: No frames received. Camera may not be working." << std::endl;
    }
    
    // Settings
    bool showDepth = depthOk;
    bool temporalFilter = true;
    float temporalAlpha = 0.3f;
    bool useColormap = true;
    int colormapType = cv::COLORMAP_INFERNO;
    
    // Enable live depth and temporal filtering
    if (depthOk) {
        SetCameraLiveDepth(1);
        SetCameraDepthTemporalFilter(1);
        SetCameraDepthTemporalAlpha(temporalAlpha);
    }
    
    std::cout << "\n=== Controls ===" << std::endl;
    std::cout << "  s - Save current frame as BMP" << std::endl;
    std::cout << "  r - Change rotation mode (0, 1, 2)" << std::endl;
    std::cout << "  i - Show IMU data in console" << std::endl;
    std::cout << "  d - Toggle depth view" << std::endl;
    std::cout << "  t - Toggle temporal filtering" << std::endl;
    std::cout << "  +/- - Adjust temporal filter strength" << std::endl;
    std::cout << "  c - Change colormap" << std::endl;
    std::cout << "  a - Toggle AR mode (128x128 ultra-fast)" << std::endl;
    std::cout << "  SPACE - Reset temporal history" << std::endl;
    std::cout << "  q/ESC - Quit" << std::endl;
    std::cout << "\nStreaming live feed... (press 'q' or ESC to quit)\n" << std::endl;
    
    // Create OpenCV windows
    cv::namedWindow("XReal Left Eye", cv::WINDOW_NORMAL);
    cv::namedWindow("XReal Right Eye", cv::WINDOW_NORMAL);
    cv::resizeWindow("XReal Left Eye", 480, 640);
    cv::resizeWindow("XReal Right Eye", 480, 640);
    
    // Position windows side by side
    cv::moveWindow("XReal Left Eye", 100, 100);
    cv::moveWindow("XReal Right Eye", 600, 100);
    
    if (depthOk) {
        cv::namedWindow("XReal Depth", cv::WINDOW_NORMAL);
        cv::resizeWindow("XReal Depth", 480, 640);
        cv::moveWindow("XReal Depth", 1100, 100);
    }
    
    // Main loop
    auto lastStatTime = std::chrono::steady_clock::now();
    int framesThisSecond = 0;
    float fps = 0;
    
    // Colormap options
    int colormaps[] = {
        cv::COLORMAP_INFERNO, cv::COLORMAP_MAGMA, cv::COLORMAP_PLASMA,
        cv::COLORMAP_VIRIDIS, cv::COLORMAP_JET, cv::COLORMAP_TURBO,
        cv::COLORMAP_BONE, cv::COLORMAP_HOT
    };
    const char* colormapNames[] = {
        "INFERNO", "MAGMA", "PLASMA", "VIRIDIS", "JET", "TURBO", "BONE", "HOT"
    };
    int currentColormap = 0;
    int numColormaps = sizeof(colormaps) / sizeof(colormaps[0]);
    
    // AR mode - 128x128 ultra-fast for navigation
    bool arMode = false;
    bool arInitialized = false;
    int mouseX = 320, mouseY = 240;  // Center of 640x480
    
    while (true) {
        // Capture frame
        if (GetCameraStereoFrame(leftBuffer, rightBuffer, &width, &height, &timestamp) == 1) {
            frameCount++;
            framesThisSecond++;
            
            // Create OpenCV Mats from buffers (no copy, just wrap)
            cv::Mat leftMat(height, width, CV_8UC1, leftBuffer);
            cv::Mat rightMat(height, width, CV_8UC1, rightBuffer);
            
            // Add FPS and IMU overlay
            float* euler = GetEuler();
            std::string fpsText = "FPS: " + std::to_string((int)fps);
            cv::putText(leftMat, fpsText, cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255), 2);
            
            if (euler) {
                std::string imuText = "Yaw: " + std::to_string((int)euler[2]);
                cv::putText(leftMat, imuText, cv::Point(10, 60), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255), 2);
            }
            
            // Display camera frames
            cv::imshow("XReal Left Eye", leftMat);
            cv::imshow("XReal Right Eye", rightMat);
            
            // Process and display depth (async - non-blocking)
            if (showDepth && depthOk) {
                int depthW = 0, depthH = 0;
                // GetCameraDepthImage gets cached result from async thread
                if (GetCameraDepthImage(depthBuffer, &depthW, &depthH) == 1) {
                    cv::Mat depthMat(depthH, depthW, CV_8UC1, depthBuffer);
                    
                    cv::Mat displayDepth;
                    if (useColormap) {
                        cv::applyColorMap(depthMat, displayDepth, colormaps[currentColormap]);
                    } else {
                        cv::cvtColor(depthMat, displayDepth, cv::COLOR_GRAY2BGR);
                    }
                    
                    // Add depth info overlay
                    float inferenceMs = GetCameraDepthInferenceTimeMs();
                    float depthFps = inferenceMs > 0 ? 1000.0f / inferenceMs : 0;
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(1);
                    
                    if (arMode && arInitialized) {
                        // AR mode - show distance at center point
                        ss << "AR MODE (128x128) - " << inferenceMs << "ms (" << (int)depthFps << " FPS)";
                        cv::putText(displayDepth, ss.str(), cv::Point(10, 30),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                        
                        // Get distance at center of image (for AR navigation arrow)
                        float centerDepth = GetDepthAtPoint(depthW/2, depthH/2);
                        float centerDist = GetDistanceAtPoint(depthW/2, depthH/2);
                        float groundDist = GetGroundPlaneDepth();
                        
                        // Draw crosshair at center
                        int cx = depthW / 2;
                        int cy = depthH / 2;
                        cv::line(displayDepth, cv::Point(cx-20, cy), cv::Point(cx+20, cy), cv::Scalar(0, 255, 0), 2);
                        cv::line(displayDepth, cv::Point(cx, cy-20), cv::Point(cx, cy+20), cv::Scalar(0, 255, 0), 2);
                        
                        ss.str("");
                        ss << "Center: " << centerDist << "m (depth=" << centerDepth << ")";
                        cv::putText(displayDepth, ss.str(), cv::Point(10, 55),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                        
                        ss.str("");
                        ss << "Ground: " << groundDist << "m";
                        cv::putText(displayDepth, ss.str(), cv::Point(10, 75),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                        
                        // Draw AR navigation arrow (simple demo)
                        if (centerDist > 2.0f) {
                            // Far away - green arrow pointing forward
                            cv::arrowedLine(displayDepth, 
                                cv::Point(cx, cy + 50), cv::Point(cx, cy - 30),
                                cv::Scalar(0, 255, 0), 3, cv::LINE_AA, 0, 0.3);
                            cv::putText(displayDepth, "GO", cv::Point(cx - 15, cy + 75),
                                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                        } else {
                            // Close obstacle - red warning
                            cv::circle(displayDepth, cv::Point(cx, cy), 40, cv::Scalar(0, 0, 255), 3);
                            cv::putText(displayDepth, "STOP", cv::Point(cx - 25, cy + 75),
                                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
                        }
                    } else {
                        // Normal mode
                        ss << "Depth: " << inferenceMs << "ms (" << (int)depthFps << " FPS)";
                        cv::putText(displayDepth, ss.str(), cv::Point(10, 30),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
                        
                        ss.str("");
                        ss << "Mode: ASYNC (256x256)";
                        cv::putText(displayDepth, ss.str(), cv::Point(10, 55),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
                        
                        ss.str("");
                        ss << "Colormap: " << colormapNames[currentColormap];
                        cv::putText(displayDepth, ss.str(), cv::Point(10, 75),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                    }
                    
                    cv::imshow("XReal Depth", displayDepth);
                }
            }
        }
        
        // Calculate FPS every second
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastStatTime).count();
        if (elapsed >= 1000) {
            fps = framesThisSecond * 1000.0f / elapsed;
            framesThisSecond = 0;
            lastStatTime = now;
        }
        
        // Check for keypress (OpenCV waitKey)
        int key = cv::waitKey(1);
        
        if (key == 'q' || key == 'Q' || key == 27) {  // 27 = ESC
            std::cout << "\nQuitting..." << std::endl;
            break;
        }
        else if (key == 's' || key == 'S') {
            // Save current frame
            saveCount++;
            char leftFile[64], rightFile[64], depthFile[64];
            snprintf(leftFile, sizeof(leftFile), "capture_%03d_left.bmp", saveCount);
            snprintf(rightFile, sizeof(rightFile), "capture_%03d_right.bmp", saveCount);
            snprintf(depthFile, sizeof(depthFile), "capture_%03d_depth.bmp", saveCount);
            SaveBMP(leftFile, leftBuffer, width, height);
            SaveBMP(rightFile, rightBuffer, width, height);
            if (depthOk && showDepth) {
                SaveBMP(depthFile, depthBuffer, width, height);
            }
        }
        else if (key == 'r' || key == 'R') {
            int rot = (GetCameraRotation() + 1) % 3;
            SetCameraRotation(rot);
            ResetCameraDepthTemporalHistory();
            std::cout << "Rotation set to: " << rot << std::endl;
        }
        else if (key == 'i' || key == 'I') {
            float* euler = GetEuler();
            float* quat = GetQuaternion();
            int brightness = GetBrightness();
            
            std::cout << "--- IMU Data ---" << std::endl;
            if (euler) {
                std::cout << "  Euler: pitch=" << euler[0] 
                          << " roll=" << euler[1] 
                          << " yaw=" << euler[2] << std::endl;
            }
            if (quat) {
                std::cout << "  Quaternion: w=" << quat[0] 
                          << " x=" << quat[1] 
                          << " y=" << quat[2] 
                          << " z=" << quat[3] << std::endl;
            }
            std::cout << "  Brightness: " << brightness << std::endl;
        }
        else if (key == 'd' || key == 'D') {
            if (depthOk) {
                showDepth = !showDepth;
                if (showDepth) {
                    cv::namedWindow("XReal Depth", cv::WINDOW_NORMAL);
                    cv::resizeWindow("XReal Depth", 480, 640);
                    cv::moveWindow("XReal Depth", 1100, 100);
                } else {
                    cv::destroyWindow("XReal Depth");
                }
                std::cout << "Depth view: " << (showDepth ? "ON" : "OFF") << std::endl;
            }
        }
        else if (key == 't' || key == 'T') {
            temporalFilter = !temporalFilter;
            SetCameraDepthTemporalFilter(temporalFilter ? 1 : 0);
            std::cout << "Temporal filter: " << (temporalFilter ? "ON" : "OFF") << std::endl;
        }
        else if (key == '+' || key == '=') {
            temporalAlpha = std::min(1.0f, temporalAlpha + 0.05f);
            SetCameraDepthTemporalAlpha(temporalAlpha);
            std::cout << "Temporal alpha: " << temporalAlpha << std::endl;
        }
        else if (key == '-' || key == '_') {
            temporalAlpha = std::max(0.05f, temporalAlpha - 0.05f);
            SetCameraDepthTemporalAlpha(temporalAlpha);
            std::cout << "Temporal alpha: " << temporalAlpha << std::endl;
        }
        else if (key == 'c' || key == 'C') {
            currentColormap = (currentColormap + 1) % numColormaps;
            std::cout << "Colormap: " << colormapNames[currentColormap] << std::endl;
        }
        else if (key == ' ') {
            ResetCameraDepthTemporalHistory();
            std::cout << "Temporal history reset" << std::endl;
        }
        else if (key == 'a' || key == 'A') {
            arMode = !arMode;
            if (arMode && !arInitialized) {
                // Initialize AR mode with ultra-fast 128x128 settings
                std::cout << "Initializing AR mode (128x128)..." << std::endl;
                
                // Try q4f16 model first (smallest, ~19MB), then others
                const char* arModels[] = {
                    "depth_anything_v2_vits_q4f16.onnx",
                    "depth_anything_v2_vits_fp16.onnx", 
                    "depth_anything_v2_vits.onnx",
                    "depth_anything_v2_small.onnx"
                };
                
                for (const char* modelPath : arModels) {
                    std::cout << "  Trying: " << modelPath << std::endl;
                    if (InitializeDepthForAR(modelPath) == 1) {
                        std::cout << "  AR initialized with: " << modelPath << std::endl;
                        arInitialized = true;
                        break;
                    }
                }
                
                if (!arInitialized) {
                    std::cout << "  Warning: Could not initialize AR mode" << std::endl;
                    arMode = false;
                }
            }
            std::cout << "AR mode: " << (arMode ? "ON (128x128 ultra-fast)" : "OFF (256x256 standard)") << std::endl;
        }
    }
    
    // Cleanup
    cv::destroyAllWindows();
    
    std::cout << "\nStopping camera..." << std::endl;
    StopCameraCapture();
    
    std::cout << "Stopping IMU..." << std::endl;
    StopConnection();
    
    delete[] leftBuffer;
    delete[] rightBuffer;
    delete[] depthBuffer;
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "  Total frames: " << frameCount << std::endl;
    std::cout << "  Files saved: " << (saveCount * (showDepth && depthOk ? 3 : 2)) << std::endl;
    std::cout << "\nDone!" << std::endl;
    
    return 0;
}
