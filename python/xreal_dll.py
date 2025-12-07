"""
xreal_dll.py - Python wrapper for AirAPI_Windows.dll camera functions

This provides a ctypes-based interface to the native C++ camera implementation,
which is faster and more reliable than the ffmpeg-based approach.

Requirements:
    - AirAPI_Windows.dll in the same directory or in PATH
    - hidapi.dll in the same directory or in PATH

Usage:
    from xreal_dll import XRealAPI
    
    api = XRealAPI()
    api.start_camera()
    
    while True:
        left, right, timestamp = api.get_stereo_frame()
        if left is not None:
            # Process frames with OpenCV, etc.
            cv2.imshow("Left", left.reshape(640, 480))
        cv2.waitKey(1)
    
    api.stop_camera()
"""

import ctypes
import numpy as np
from ctypes import c_int, c_uint8, c_uint64, c_float, POINTER, byref
import os
import sys

class XRealAPI:
    """Python wrapper for AirAPI_Windows.dll"""
    
    # Image dimensions (after rotation)
    IMAGE_WIDTH = 480
    IMAGE_HEIGHT = 640
    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
    
    def __init__(self, dll_path=None):
        """
        Initialize the XReal API wrapper.
        
        Args:
            dll_path: Optional path to AirAPI_Windows.dll. If not provided,
                     searches in the current directory and PATH.
        """
        self._dll = None
        self._initialized = False
        self._camera_running = False
        self._imu_running = False
        
        # Buffers for frame data
        self._left_buffer = np.zeros(self.IMAGE_SIZE, dtype=np.uint8)
        self._right_buffer = np.zeros(self.IMAGE_SIZE, dtype=np.uint8)
        
        # Load the DLL
        self._load_dll(dll_path)
        self._setup_functions()
    
    def _load_dll(self, dll_path):
        """Load the AirAPI_Windows.dll"""
        if dll_path and os.path.exists(dll_path):
            self._dll = ctypes.CDLL(dll_path)
            return
        
        # Try common locations
        search_paths = [
            os.path.join(os.path.dirname(__file__), 'AirAPI_Windows.dll'),
            os.path.join(os.path.dirname(__file__), '..', 'x64', 'Release', 'AirAPI_Windows.dll'),
            'AirAPI_Windows.dll',
            os.path.join(os.path.dirname(__file__), '..', 'AirAPI_Windows.dll'),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                self._dll = ctypes.CDLL(path)
                return
        
        # Try to load from PATH
        try:
            self._dll = ctypes.CDLL('AirAPI_Windows.dll')
        except OSError as e:
            raise RuntimeError(
                f"Could not load AirAPI_Windows.dll. "
                f"Make sure it's in the current directory or in PATH. Error: {e}"
            )
    
    def _setup_functions(self):
        """Set up function signatures for the DLL"""
        # IMU functions
        self._dll.StartConnection.restype = c_int
        self._dll.StopConnection.restype = c_int
        self._dll.GetQuaternion.restype = POINTER(c_float)
        self._dll.GetEuler.restype = POINTER(c_float)
        self._dll.GetBrightness.restype = c_int
        
        # Camera functions
        self._dll.StartCameraCapture.restype = c_int
        self._dll.StopCameraCapture.restype = c_int
        self._dll.IsCameraCapturing.restype = c_int
        
        self._dll.GetCameraLeftImage.restype = POINTER(c_uint8)
        self._dll.GetCameraLeftImage.argtypes = [POINTER(c_int), POINTER(c_int)]
        
        self._dll.GetCameraRightImage.restype = POINTER(c_uint8)
        self._dll.GetCameraRightImage.argtypes = [POINTER(c_int), POINTER(c_int)]
        
        self._dll.GetCameraStereoFrame.restype = c_int
        self._dll.GetCameraStereoFrame.argtypes = [
            POINTER(c_uint8), POINTER(c_uint8),
            POINTER(c_int), POINTER(c_int), POINTER(c_uint64)
        ]
        
        self._dll.SetCameraRotation.restype = None
        self._dll.SetCameraRotation.argtypes = [c_int]
        
        self._dll.GetCameraRotation.restype = c_int
        
        self._initialized = True
    
    # =========================================================================
    # IMU Functions
    # =========================================================================
    
    def start_imu(self):
        """Start IMU tracking.
        
        Returns:
            bool: True if successful
        """
        if self._imu_running:
            return True
        
        result = self._dll.StartConnection()
        self._imu_running = result == 1
        return self._imu_running
    
    def stop_imu(self):
        """Stop IMU tracking.
        
        Returns:
            bool: True if successfully stopped
        """
        if not self._imu_running:
            return True
        
        result = self._dll.StopConnection()
        self._imu_running = False
        return result == 1
    
    def get_quaternion(self):
        """Get current head orientation as quaternion.
        
        Returns:
            numpy.ndarray: [w, x, y, z] quaternion, or None if not tracking
        """
        if not self._imu_running:
            return None
        
        ptr = self._dll.GetQuaternion()
        if not ptr:
            return None
        
        return np.array([ptr[0], ptr[1], ptr[2], ptr[3]], dtype=np.float32)
    
    def get_euler(self):
        """Get current head orientation as Euler angles.
        
        Returns:
            numpy.ndarray: [pitch, roll, yaw] in degrees, or None if not tracking
        """
        if not self._imu_running:
            return None
        
        ptr = self._dll.GetEuler()
        if not ptr:
            return None
        
        return np.array([ptr[0], ptr[1], ptr[2]], dtype=np.float32)
    
    def get_brightness(self):
        """Get current display brightness.
        
        Returns:
            int: Brightness level (0-255)
        """
        return self._dll.GetBrightness()
    
    # =========================================================================
    # Camera Functions
    # =========================================================================
    
    def start_camera(self, rotation=2):
        """Start camera capture.
        
        Args:
            rotation: Rotation mode (0, 1, or 2). Default is 2.
        
        Returns:
            bool: True if successful
        """
        if self._camera_running:
            return True
        
        self.set_rotation(rotation)
        result = self._dll.StartCameraCapture()
        self._camera_running = result == 1
        return self._camera_running
    
    def stop_camera(self):
        """Stop camera capture.
        
        Returns:
            bool: True if successfully stopped
        """
        if not self._camera_running:
            return True
        
        result = self._dll.StopCameraCapture()
        self._camera_running = False
        return result == 1
    
    def is_camera_capturing(self):
        """Check if camera is currently capturing.
        
        Returns:
            bool: True if capturing
        """
        return self._dll.IsCameraCapturing() == 1
    
    def set_rotation(self, rotation):
        """Set camera rotation mode.
        
        Args:
            rotation: Rotation mode (0, 1, or 2)
        """
        self._dll.SetCameraRotation(rotation)
    
    def get_rotation(self):
        """Get current rotation mode.
        
        Returns:
            int: Current rotation mode
        """
        return self._dll.GetCameraRotation()
    
    def get_stereo_frame(self):
        """Get the latest stereo frame.
        
        Returns:
            tuple: (left_image, right_image, timestamp) where images are
                   numpy arrays of shape (height, width) in uint8 format,
                   or (None, None, 0) if no frame available.
        """
        width = c_int()
        height = c_int()
        timestamp = c_uint64()
        
        left_ptr = self._left_buffer.ctypes.data_as(POINTER(c_uint8))
        right_ptr = self._right_buffer.ctypes.data_as(POINTER(c_uint8))
        
        result = self._dll.GetCameraStereoFrame(
            left_ptr, right_ptr,
            byref(width), byref(height), byref(timestamp)
        )
        
        if result == 1:
            left = self._left_buffer.reshape((height.value, width.value)).copy()
            right = self._right_buffer.reshape((height.value, width.value)).copy()
            return left, right, timestamp.value
        
        return None, None, 0
    
    def get_left_image(self):
        """Get pointer to left image data.
        
        Returns:
            numpy.ndarray: Left image as (height, width) array, or None
        """
        width = c_int()
        height = c_int()
        
        ptr = self._dll.GetCameraLeftImage(byref(width), byref(height))
        if not ptr:
            return None
        
        # Create numpy array from pointer (no copy, shares memory)
        arr = np.ctypeslib.as_array(ptr, shape=(height.value * width.value,))
        return arr.reshape((height.value, width.value))
    
    def get_right_image(self):
        """Get pointer to right image data.
        
        Returns:
            numpy.ndarray: Right image as (height, width) array, or None
        """
        width = c_int()
        height = c_int()
        
        ptr = self._dll.GetCameraRightImage(byref(width), byref(height))
        if not ptr:
            return None
        
        arr = np.ctypeslib.as_array(ptr, shape=(height.value * width.value,))
        return arr.reshape((height.value, width.value))
    
    # =========================================================================
    # Combined Functions
    # =========================================================================
    
    def start_all(self, rotation=2):
        """Start both IMU and camera.
        
        Args:
            rotation: Camera rotation mode (0, 1, or 2)
        
        Returns:
            tuple: (imu_started, camera_started)
        """
        imu_ok = self.start_imu()
        cam_ok = self.start_camera(rotation)
        return imu_ok, cam_ok
    
    def stop_all(self):
        """Stop both IMU and camera."""
        self.stop_camera()
        self.stop_imu()
    
    def get_frame_with_pose(self):
        """Get stereo frame along with current head pose.
        
        Returns:
            dict: {
                'left': numpy.ndarray or None,
                'right': numpy.ndarray or None,
                'timestamp': int,
                'quaternion': numpy.ndarray or None,
                'euler': numpy.ndarray or None,
                'brightness': int
            }
        """
        left, right, timestamp = self.get_stereo_frame()
        
        return {
            'left': left,
            'right': right,
            'timestamp': timestamp,
            'quaternion': self.get_quaternion(),
            'euler': self.get_euler(),
            'brightness': self.get_brightness()
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_all()
        return False


def main():
    """Demo application using the XReal API wrapper."""
    try:
        import cv2
    except ImportError:
        print("OpenCV not available. Install with: pip install opencv-python")
        cv2 = None
    
    print("=== XReal API Python Wrapper Demo ===\n")
    
    api = XRealAPI()
    
    print("Starting IMU and camera...")
    imu_ok, cam_ok = api.start_all()
    
    if not cam_ok:
        print("Failed to start camera!")
        print("Make sure XReal Air 2 Ultra glasses are connected.")
        return
    
    print(f"IMU: {'OK' if imu_ok else 'Failed'}")
    print(f"Camera: {'OK' if cam_ok else 'Failed'}")
    print(f"Rotation mode: {api.get_rotation()}")
    print("\nPress 'q' to quit (if OpenCV window is focused)")
    print("Press Ctrl+C to quit from terminal\n")
    
    frame_count = 0
    
    try:
        while True:
            # Get combined data
            data = api.get_frame_with_pose()
            
            if data['left'] is not None:
                frame_count += 1
                
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Frame {frame_count}: "
                          f"size={data['left'].shape}, "
                          f"euler={data['euler']}, "
                          f"brightness={data['brightness']}")
                
                if cv2:
                    # Create side-by-side display
                    stereo = np.hstack([data['left'], data['right']])
                    
                    # Add text overlay
                    if data['euler'] is not None:
                        text = f"P:{data['euler'][0]:.1f} R:{data['euler'][1]:.1f} Y:{data['euler'][2]:.1f}"
                        cv2.putText(stereo, text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)
                    
                    cv2.imshow("XReal Stereo", stereo)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    print(f"\nTotal frames captured: {frame_count}")
    
    api.stop_all()
    print("Stopped.")
    
    if cv2:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
