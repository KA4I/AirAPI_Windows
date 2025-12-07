# AirAPI_Windows

This is a work in progress driver for the Nreal Air for native windows support without nebula. The current configuration for fusion has slight drift and needs to be tuned but is functional. This project is a colaberataive effort by the community to provide open access to the glasses.

**New: Camera support for XReal Air 2 Ultra!** The API now includes stereo camera capture with the proprietary frame descrambling algorithm.

**New: AI-Powered Depth Estimation!** Real-time depth estimation using Depth Anything V2 with temporal filtering for smooth, flicker-free output.

This is made possible by these wonderful people:<br>
https://github.com/abls <br>
https://github.com/edwatt <br>
https://github.com/jakedowns

Feel free to support me with a coffee.
https://www.buymeacoffee.com/msmithdev

## Features

- **IMU Tracking**: Get quaternion and euler angles from the glasses' IMU
- **Brightness Control**: Read display brightness level
- **Stereo Camera Capture** (XReal Air 2 Ultra): Access left and right camera feeds for depth mapping, SLAM, face detection, etc.
- **AI Depth Estimation**: Real-time monocular depth using Depth Anything V2 with:
  - **Async processing** for smooth 30fps camera + 10-15fps depth
  - GPU acceleration via ONNX Runtime + CUDA (Windows) / NNAPI (Android)
  - Temporal filtering for smooth, flicker-free output
  - Multiple colormap options for visualization
  - Configurable resolution (256x256 default for ~60ms inference)

## Precompiled
To use the precompiled [AirAPI_Windows.dll](https://github.com/MSmithDev/AirAPI_Windows/releases) you also need to include hidapi.dll available [here](https://github.com/libusb/hidapi/releases). 

For depth estimation, you also need:
- `onnxruntime.dll` from [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases)
- A Depth Anything V2 ONNX model (see [Depth Setup](#depth-estimation-setup))

[Here](https://github.com/MSmithDev/AirAPI_Windows/wiki/Using-with-Unity) is a demo in the wiki for use with unity scripts.

[Here](https://github.com/MSmithDev/AirPoseUnityDemo) is a Unity demo using the XR plugin with the Airs setup as a headset.

## API Reference

### IMU Functions

```cpp
// Start/stop IMU connection
int StartConnection();
int StopConnection();

// Get orientation data
float* GetQuaternion();  // Returns [w, x, y, z]
float* GetEuler();       // Returns [pitch, roll, yaw]

// Get display brightness (0-255)
int GetBrightness();
```

### Camera Functions (XReal Air 2 Ultra)

```cpp
// Start/stop camera capture
int StartCameraCapture();   // Returns 1 on success, 0 on failure
int StopCameraCapture();    // Returns 1 on success, 0 if not capturing
int IsCameraCapturing();    // Returns 1 if capturing, 0 otherwise

// Get image pointers (do not free, valid until next frame)
uint8_t* GetCameraLeftImage(int* width, int* height);
uint8_t* GetCameraRightImage(int* width, int* height);

// Get stereo frame with copy to your buffers
int GetCameraStereoFrame(uint8_t* leftOut, uint8_t* rightOut,
                         int* width, int* height, uint64_t* timestamp);

// Set/get rotation mode (0, 1, or 2 - default is 2)
void SetCameraRotation(int rotation);
int GetCameraRotation();
```

### Depth Estimation Functions

```cpp
// Initialize depth estimator with ONNX model
int InitializeCameraDepth(const char* modelPath, int useGPU);

// Get depth image from latest camera frame
int GetCameraDepthImage(uint8_t* depthOut, int* width, int* height);

// Get internal depth buffer pointer (do not free)
uint8_t* GetCameraDepthImagePtr(int* width, int* height);

// Enable/disable live depth processing
void SetCameraLiveDepth(int enabled);
int IsCameraLiveDepthEnabled();

// Temporal filtering controls (reduces flickering)
void SetCameraDepthTemporalFilter(int enabled);
void SetCameraDepthTemporalAlpha(float alpha);  // 0.0-1.0, lower = smoother
void ResetCameraDepthTemporalHistory();

// Performance monitoring
float GetCameraDepthInferenceTimeMs();
```

### Camera Usage Example (C++)

```cpp
#include "AirAPI_Windows.h"
#include <iostream>

int main() {
    // Start camera capture
    if (StartCameraCapture() == 0) {
        std::cerr << "Failed to start camera" << std::endl;
        return 1;
    }
    
    // Allocate buffers for stereo images (480x640 grayscale each)
    uint8_t* leftBuffer = new uint8_t[480 * 640];
    uint8_t* rightBuffer = new uint8_t[480 * 640];
    int width, height;
    uint64_t timestamp;
    
    // Main loop
    while (true) {
        if (GetCameraStereoFrame(leftBuffer, rightBuffer, &width, &height, &timestamp)) {
            // Process stereo images for depth mapping, SLAM, etc.
            // width = 480, height = 640 (after rotation)
            
            // Example: Use with OpenCV for depth estimation
            // cv::Mat leftMat(height, width, CV_8UC1, leftBuffer);
            // cv::Mat rightMat(height, width, CV_8UC1, rightBuffer);
        }
    }
    
    // Cleanup
    StopCameraCapture();
    delete[] leftBuffer;
    delete[] rightBuffer;
    return 0;
}
```

### Depth Estimation Example (C++)

```cpp
#include "AirAPI_Windows.h"
#include <opencv2/opencv.hpp>

int main() {
    // Start camera
    StartCameraCapture();
    
    // Initialize depth with fast settings (256x256, async, GPU)
    // This runs ~15 FPS depth independently of 30 FPS camera
    if (InitializeCameraDepth("depth_anything_v2_small.onnx", 1) == 0) {
        std::cerr << "Failed to initialize depth estimator" << std::endl;
        return 1;
    }
    
    // Enable live depth processing (async - non-blocking)
    SetCameraLiveDepth(1);
    
    uint8_t* depthBuffer = new uint8_t[480 * 640];
    int width, height;
    
    while (true) {
        if (GetCameraDepthImage(depthBuffer, &width, &height)) {
            cv::Mat depthMat(height, width, CV_8UC1, depthBuffer);
            
            // Apply colormap for visualization
            cv::Mat colored;
            cv::applyColorMap(depthMat, colored, cv::COLORMAP_INFERNO);
            cv::imshow("Depth", colored);
        }
        
        if (cv::waitKey(1) == 'q') break;
    }
    
    StopCameraCapture();
    delete[] depthBuffer;
    return 0;
}
```

### Combined IMU + Camera for SLAM

```cpp
// Start both IMU and camera
StartConnection();
StartCameraCapture();

// In your loop:
float* quaternion = GetQuaternion();
float* euler = GetEuler();

int w, h;
uint8_t* left = GetCameraLeftImage(&w, &h);
uint8_t* right = GetCameraRightImage(&w, &h);

// Combine pose from IMU with visual features from camera
// for robust visual-inertial SLAM
```

# Building from source

## Dependencies

### OpenCV (for camera support)

The camera functionality requires OpenCV. Download and set up OpenCV:

1. Download OpenCV 4.12.0 (or later) from https://opencv.org/releases/
2. Extract to a folder, e.g., `C:\opencv`
3. Set environment variable `OPENCV_DIR` to the OpenCV build folder:
   - Open System Properties > Environment Variables
   - Add new system variable: `OPENCV_DIR` = `C:\opencv\build` (adjust path as needed)
   - The folder should contain `include/` and `x64/vc16/lib/` subdirectories

**Note**: The project expects these files:
- Include: `$(OPENCV_DIR)\include\opencv2\opencv.hpp`
- Debug lib: `$(OPENCV_DIR)\x64\vc16\lib\opencv_world4120d.lib`
- Release lib: `$(OPENCV_DIR)\x64\vc16\lib\opencv_world4120.lib`
- Runtime DLLs: `opencv_world4120.dll` / `opencv_world4120d.dll`

### ONNX Runtime (optional, for depth estimation)

ONNX Runtime is **optional** - the project will compile without it, but depth estimation features will be disabled. When ONNX Runtime is not found, `InitializeCameraDepth()` will return 0 and print an error message.

To enable AI-powered depth estimation:

1. Download ONNX Runtime 1.16+ from https://github.com/microsoft/onnxruntime/releases
   - For GPU: Download the `onnxruntime-win-x64-gpu-X.XX.X.zip`
   - For CPU only: Download `onnxruntime-win-x64-X.XX.X.zip`
2. Extract to a folder, e.g., `C:\onnxruntime`
3. Set environment variable `ONNXRUNTIME_DIR`:
   - Add new system variable: `ONNXRUNTIME_DIR` = `C:\onnxruntime` (adjust path)
   - The folder should contain `include/` and `lib/` subdirectories
4. **Restart Visual Studio** to pick up the new environment variable
5. Rebuild the project

The build system automatically detects `ONNXRUNTIME_DIR` and:
- If set: Enables `ONNXRUNTIME_AVAILABLE` define and links onnxruntime.lib
- If not set: Compiles without ONNX support (depth functions return failure)

**Note**: For GPU support, you also need CUDA and cuDNN installed.

### Depth Estimation Setup

1. Download a Depth Anything V2 ONNX model:
   - **Recommended (ViT-S, 518x518)**: Fast and high quality
   - Get from Hugging Face or convert using the official Depth Anything V2 repository
   
2. Place the `.onnx` file in your executable directory or specify the full path

3. Rename to `depth_anything_v2_vits.onnx` or use `InitializeCameraDepth()` with custom path

## Getting hidapi
Get the latest hidapi-win.zip from [here](https://github.com/libusb/hidapi/releases).

unzip hidapi-win.zip to "deps" folder ("deps/hidapi-win").



## Build Fusion
### Clone this project
Goto project directory
```
cd AirAPI_Windows
```
Init sub module
```
git submodule init
```
update the module
```
git submodule update
```


cd to "deps/Fusion/Fusion"

Run cmake
```
cmake .
```

Open the Project.sln and build the project <br>
You should have a "deps/Fusion/Fusion/Release/Fusion.lib" file.
### Build AirAPI_Windows DLL 
Open AirAPI_Windows.sln and make sure "Release" and "x64" are set and build.

## Camera Technical Details

The XReal Air 2 Ultra cameras stream data in a proprietary scrambled format:

1. **Raw Format**: 640x241 YUY2 at ~30fps
2. **Frame Pairing**: Two consecutive frames with matching sequence numbers form one stereo pair
3. **Chunk Descrambling**: The 128 chunks of 2400 bytes each are reordered using a lookup table
4. **Stereo Separation**: Metadata byte at offset 0x3b indicates left (0) or right (1) eye
5. **Output**: Two 480x640 grayscale images (after rotation)

This implementation uses ffmpeg for camera capture, providing:
- Robust format handling for the proprietary XReal camera format
- Cross-platform compatibility potential
- Direct rawvideo stream processing

## Depth Estimation Technical Details

The depth estimation uses Depth Anything V2 for high-quality monocular depth:

1. **Model**: Depth Anything V2 ViT-S (518x518 input)
2. **Backend**: ONNX Runtime with optional CUDA acceleration
3. **Temporal Filtering**: Exponential moving average with motion detection
4. **Spatial Filtering**: Bilateral filter for edge-preserving smoothness
5. **Output**: 480x640 grayscale depth map (0-255, closer = brighter by default)
