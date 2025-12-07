# XReal Air 2 Ultra Camera Driver for Windows

This project provides Python scripts to access and decode the camera feed from XReal Air 2 Ultra glasses on Windows.

## Two Approaches

### 1. DLL Wrapper (Recommended) - `xreal_dll.py`

Uses the native C++ `AirAPI_Windows.dll` for camera capture. This is **faster, more reliable, and integrates with IMU data**.

```python
from xreal_dll import XRealAPI

# Using context manager (auto start/stop)
with XRealAPI() as api:
    while True:
        data = api.get_frame_with_pose()
        if data['left'] is not None:
            # data['left'] and data['right'] are 640x480 numpy arrays
            # data['euler'] is [pitch, roll, yaw] from IMU
            # data['quaternion'] is [w, x, y, z] from IMU
            pass

# Or manual control
api = XRealAPI()
api.start_camera()
api.start_imu()

left, right, timestamp = api.get_stereo_frame()
euler = api.get_euler()

api.stop_all()
```

**Requirements:**
- `AirAPI_Windows.dll` and `hidapi.dll` in the same directory or in PATH
- `pip install numpy opencv-python`

### 2. FFmpeg-based (Standalone) - `xreal_windows.py`

The original ffmpeg-based approach that works without the DLL, but is slower and doesn't include IMU integration.

```bash
pip install opencv-python numpy
python xreal_windows.py
```

## How the Camera Works

The XReal Air 2 Ultra cameras do not stream standard video frames. Instead, they stream raw data that is "scrambled" and split across multiple frames. Here is the breakdown of the decoding process:

### 1. Data Acquisition
The script uses `ffmpeg` via DirectShow to capture raw video data from the device (usually named "UVC Camera" or similar). It captures at a resolution of 640x241 (YUY2), which effectively gives us the raw byte stream from the sensor.

### 2. Frame Reconstruction
The camera sends data in pairs of frames. We need to capture two consecutive raw frames (`in1` and `in2`) to construct one full stereo image.
- We check a sequence number embedded in the frame metadata (bytes at specific offsets) to ensure we have a matching pair.

### 3. Descrambling (The "Chunk Map")
The raw image data is not linear. It is divided into 128 chunks, each of size 2400 bytes.
- The camera writes these chunks in a shuffled order.
- We use a lookup table (`CHUNK_MAP`) to determine the correct order.
- The script reads 128 blocks from the input frame, finds the starting block (based on a sum/min algorithm to find the "sync" point), and then reorders them according to the map.

### 4. Stereo Construction
The data contains pixels for both the Left and Right eyes.
- The script separates these based on metadata bytes.
- It handles rotation (the sensors are mounted sideways).
- It constructs a side-by-side stereo image (960x640).

## Usage (FFmpeg version)

1. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```
   You also need `ffmpeg` installed and in your system PATH.

2. Run the script:
   ```bash
   python xreal_windows.py
   ```

3. Controls:
   - `q`: Quit
   - `r`: Change rotation/view mode
   - `m`: Toggle mirroring
   - `d`: Toggle Depth Map (Disparity)
   - `f`: Toggle Face Detection

## 3D and Computer Vision
Since we have a calibrated stereo pair (Left and Right eyes), we can perform 3D analysis:
- **Depth Map**: By calculating the disparity (shift) between the left and right images, we can estimate depth. Closer objects have larger disparity.
- **Face/Hand Detection**: We can run detection algorithms on the images. Doing this on both images allows us to calculate the 3D coordinates (X, Y, Z) of the detected features.

## Integration with SLAM

With the DLL wrapper, you can combine camera data with IMU data for visual-inertial SLAM:

```python
from xreal_dll import XRealAPI
import numpy as np

api = XRealAPI()
api.start_all()

while True:
    # Get synchronized camera + IMU data
    data = api.get_frame_with_pose()
    
    if data['left'] is not None:
        # Visual features from stereo cameras
        left_img = data['left']
        right_img = data['right']
        
        # IMU orientation
        quaternion = data['quaternion']  # [w, x, y, z]
        euler = data['euler']  # [pitch, roll, yaw]
        
        # Combine for SLAM, depth estimation, etc.
        # ...
```
