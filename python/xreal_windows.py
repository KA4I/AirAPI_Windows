import cv2
import numpy as np
import struct
import sys
import subprocess
import threading
import queue
import time

class XRealDecoder:
    # Chunk reordering table
    CHUNK_MAP = [
        119, 54, 21, 0, 108, 22, 51, 63, 93, 99, 67, 7, 32, 112, 52, 43,
        14, 35, 75, 116, 64, 71, 44, 89, 18, 88, 26, 61, 70, 56, 90, 79,
        87, 120, 81, 101, 121, 17, 72, 31, 53, 124, 127, 113, 111, 36, 48,
        19, 37, 83, 126, 74, 109, 5, 84, 41, 76, 30, 110, 29, 12, 115, 28,
        102, 105, 62, 103, 20, 3, 68, 49, 77, 117, 125, 106, 60, 69, 98, 9,
        16, 78, 47, 40, 2, 118, 34, 13, 50, 46, 80, 85, 66, 42, 123, 122,
        96, 11, 25, 97, 39, 6, 86, 1, 8, 82, 92, 59, 104, 24, 15, 73, 65,
        38, 58, 10, 23, 33, 55, 57, 107, 100, 94, 27, 95, 45, 91, 4, 114
    ]
    CHUNK_SIZE = 2400

    def __init__(self, rotation=0):
        self._rotation = rotation
        self._last_frame = None
        # Use StereoSGBM for better quality
        min_disp = 0
        num_disp = 16 * 5 # 80
        block_size = 5
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 1 * block_size**2,
            P2=32 * 1 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Load face detector
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_cascade = None
            
        self.last_faces = []
        self.frame_count = 0

    def create_sliders(self):
        cv2.namedWindow("XReal Depth")
        cv2.createTrackbar("Num Disparities", "XReal Depth", 5, 20, lambda x: None) # * 16
        cv2.createTrackbar("Block Size", "XReal Depth", 5, 50, lambda x: None) # odd
        cv2.createTrackbar("Min Disparity", "XReal Depth", 0, 100, lambda x: None) # -50 offset
        cv2.createTrackbar("Uniqueness", "XReal Depth", 10, 100, lambda x: None)
        cv2.createTrackbar("Speckle Window", "XReal Depth", 100, 200, lambda x: None)
        cv2.createTrackbar("Speckle Range", "XReal Depth", 32, 100, lambda x: None)

    def update_stereo_params(self):
        try:
            num_disp = cv2.getTrackbarPos("Num Disparities", "XReal Depth") * 16
            if num_disp < 16: num_disp = 16
            
            block_size = cv2.getTrackbarPos("Block Size", "XReal Depth")
            if block_size % 2 == 0: block_size += 1
            if block_size < 5: block_size = 5
            
            min_disp = cv2.getTrackbarPos("Min Disparity", "XReal Depth") - 50
            
            uniqueness = cv2.getTrackbarPos("Uniqueness", "XReal Depth")
            speckle_win = cv2.getTrackbarPos("Speckle Window", "XReal Depth")
            speckle_range = cv2.getTrackbarPos("Speckle Range", "XReal Depth")
            
            self.stereo.setNumDisparities(num_disp)
            self.stereo.setBlockSize(block_size)
            self.stereo.setMinDisparity(min_disp)
            self.stereo.setUniquenessRatio(uniqueness)
            self.stereo.setSpeckleWindowSize(speckle_win)
            self.stereo.setSpeckleRange(speckle_range)
            
            # Update P1/P2 based on block size
            self.stereo.setP1(8 * 1 * block_size**2)
            self.stereo.setP2(32 * 1 * block_size**2)
            
        except:
            pass

    def compute_depth(self, left, right):
        # Ensure we have 8-bit single channel
        if len(left.shape) == 3:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        if len(right.shape) == 3:
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            
        self.update_stereo_params()
        
        disparity = self.stereo.compute(left, right)
        
        # Normalize for visualization
        # Disparity is 16-bit fixed point (multiplied by 16)
        # So divide by 16 to get real disparity
        
        min_d = self.stereo.getMinDisparity()
        num_d = self.stereo.getNumDisparities()
        
        disp_vis = (disparity / 16.0 - min_d) / num_d
        disp_vis = np.clip(disp_vis, 0, 1)
        disp_vis = (disp_vis * 255).astype(np.uint8)
        
        return disp_vis, disparity

    def detect_faces_stereo(self, img_l, img_r, disparity=None):
        if self.face_cascade is None:
            return img_l, img_r
            
        self.frame_count += 1
        
        # Run detection on LEFT image only
        if self.frame_count % 5 == 0:
            small_img = cv2.resize(img_l, (0, 0), fx=0.5, fy=0.5)
            faces = self.face_cascade.detectMultiScale(small_img, 1.1, 4)
            
            self.last_faces = []
            for (x, y, w, h) in faces:
                # Scale back up
                x, y, w, h = x*2, y*2, w*2, h*2
                
                # Calculate distance if disparity is available
                dist_str = ""
                if disparity is not None:
                    # Sample center of face
                    cx, cy = x + w//2, y + h//2
                    if 0 <= cx < disparity.shape[1] and 0 <= cy < disparity.shape[0]:
                        d_val = disparity[cy, cx]
                        if d_val > 0:
                            d_val /= 16.0
                            # Z = (f * B) / d
                            # Assuming f ~ 400 (guess for fisheye/wide), B ~ 50mm?
                            # This is just a relative number without calibration
                            if d_val != 0:
                                dist_val = 1000.0 / d_val # Arbitrary scale
                                dist_str = f"{dist_val:.1f}"
                
                self.last_faces.append((x, y, w, h, dist_str))
        
        # Draw on both images (or just return modified left)
        # We need to return color images for display
        if len(img_l.shape) == 2:
            out_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
        else:
            out_l = img_l.copy()
            
        if len(img_r.shape) == 2:
            out_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)
        else:
            out_r = img_r.copy()
            
        for (x, y, w, h, dist) in self.last_faces:
            cv2.rectangle(out_l, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if dist:
                cv2.putText(out_l, dist, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            # Ideally we would project this box to the right image using disparity
            # But for now, let's just draw it on the left.
            
        return out_l, out_r

    def handle_frame(self, in_frame, np_out):
        # in_frame: flat uint8 array (640*482)
        # np_out: flat uint8 array (output buffer)
        
        blocks = in_frame[:640*480].reshape((128, 2400))

        # Find the starting index
        min_idx = blocks[:,:128].sum(axis=1).argmin()
        
        try:
            map_idx = self.CHUNK_MAP.index(min_idx)
        except ValueError:
            return

        if self._rotation == 2:
            # Reshape out as (Height, Width) = (960, 640) in Fortran order
            out_img = np_out.reshape((480*2, 640), order='F')
            
            # Check metadata byte for Left/Right (0x3b = 59)
            is_right = in_frame[480*640 + 0x3b]
            
            if is_right:
                # RIGHT - Bottom half
                dest = out_img[480:, :]
            else:
                # LEFT - Top half, flipped
                dest = out_img[:480, :]
                dest = dest[::-1, ::-1]

            p_y = 0
            p_x = 0
            for t_idx in range(128):
                source = blocks[self.CHUNK_MAP[map_idx]]
                
                pos = 0
                while pos < 2400:
                    p_y_new = min(640, p_y + 2400 - pos)
                    
                    dest[p_x, p_y:p_y_new] = source[pos:pos + p_y_new - p_y]
                    
                    pos = pos + p_y_new - p_y
                    if p_y_new == 640:
                        p_x += 1
                        p_y = 0
                    else:
                        p_y = p_y_new
                
                map_idx = (map_idx + 1) % 128

        else:
            # Rotation 0 or 1
            is_right = in_frame[480*640 + 0x3b]
            
            if is_right:
                out = np_out[480*640:]
            else:
                out = np_out[:480*640]
                if self._rotation == 1:
                    out = out[::-1]
            
            for t_idx in range(128):
                source = blocks[self.CHUNK_MAP[map_idx]]
                out[t_idx * 2400 : t_idx * 2400 + 2400] = source
                map_idx = (map_idx + 1) % 128

    def process(self, frame):
        # frame: numpy array (flat or shaped)
        
        flat_frame = frame.flatten()
        
        if self._last_frame is None:
            self._last_frame = flat_frame
            return None
            
        in1 = self._last_frame
        in2 = flat_frame
        self._last_frame = None
        
        # Check sequence
        # 640*480 + 18 = 307218
        if len(in1) < 307220 or len(in2) < 307220:
             return None

        seq1 = struct.unpack('<h', in1[307218:307220])[0]
        seq2 = struct.unpack('<h', in2[307218:307220])[0]
        
        if seq1 != seq2:
            # Mismatch, keep the new one
            self._last_frame = in2
            return None
            
        # Allocate output
        np_out = np.zeros(640 * 480 * 2, dtype=np.uint8)
        
        self.handle_frame(in1, np_out)
        self.handle_frame(in2, np_out)
        
        # Reshape for display
        if self._rotation == 2:
            return np_out.reshape((960, 640), order='F').transpose()
        else:
            return np_out.reshape((960, 640))

def get_camera_name():
    # Run ffmpeg to list devices
    cmd = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output = result.stderr # ffmpeg prints to stderr
    
    lines = output.split('\n')
    cameras = []
    for line in lines:
        if "(video)" in line and "dshow @" in line:
            # Extract name
            parts = line.split('"')
            if len(parts) >= 3:
                name = parts[1]
                cameras.append(name)
    
    # Prefer "UVC Camera" or "XREAL"
    for cam in cameras:
        if "UVC Camera" in cam:
            return cam
    for cam in cameras:
        if "XREAL" in cam:
            return cam
            
    if cameras:
        return cameras[0]
    return None

def read_stream(process, q):
    frame_size = 640 * 241 * 2 # 308480 bytes
    while True:
        data = process.stdout.read(frame_size)
        if not data:
            break
        if len(data) != frame_size:
            continue
        q.put(data)

def main():
    camera_name = get_camera_name()
    if not camera_name:
        print("No camera found via ffmpeg dshow.")
        return
        
    print(f"Using camera: {camera_name}")
    
    # Start ffmpeg process
    # ffmpeg -f dshow -video_size 640x241 -pixel_format yuyv422 -i video="Camera" -f rawvideo -pix_fmt yuyv422 pipe:1
    
    cmd = [
        "ffmpeg",
        "-f", "dshow",
        "-video_size", "640x241",
        "-pixel_format", "yuyv422",
        "-i", f"video={camera_name}",
        "-f", "rawvideo",
        "-pix_fmt", "yuyv422",
        "-"
    ]
    
    print("Starting ffmpeg: " + " ".join(cmd))
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7)
    
    q = queue.Queue(maxsize=2)
    t = threading.Thread(target=read_stream, args=(process, q))
    t.daemon = True
    t.start()
    
    decoder = XRealDecoder(rotation=2)
    decoder.create_sliders()
    
    print("Press 'q' to quit")
    print("'r': Rotate view")
    print("'m': Toggle Mirror")
    print("'d': Toggle Depth Map")
    print("'f': Toggle Face Detection")
    
    mirror = True # Default to mirrored as per user request
    show_depth = False
    show_face = False
    
    while True:
        try:
            data = q.get(timeout=1.0)
        except queue.Empty:
            if process.poll() is not None:
                print("ffmpeg process exited.")
                break
            continue
            
        frame = np.frombuffer(data, dtype=np.uint8)
        
        if frame.size != 640 * 482:
            print(f"Frame size mismatch: {frame.size}")
            continue
            
        out = decoder.process(frame)
        
        if out is not None:
            # 'out' is (640, 960) if rotation=2 (transposed)
            
            # Split into left and right BEFORE mirroring or anything else
            # Because mirroring swaps eyes if we are not careful
            
            h, w = out.shape
            mid = w // 2
            img_l = out[:, :mid]
            img_r = out[:, mid:]
            
            # If we want to mirror the VIEW, we flip each eye horizontally
            # AND we swap the eyes?
            # Standard mirror: You look in mirror. Left hand is on left side of image.
            # Camera: Left hand is on right side of image.
            # So we just flip horizontally.
            
            # However, for processing (Depth, Face), we want the raw images.
            
            disparity = None
            if show_depth:
                # Compute depth on raw images
                disp_vis, disparity = decoder.compute_depth(img_l, img_r)
                depth_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                
                if mirror:
                    depth_colored = cv2.flip(depth_colored, 1)
                cv2.imshow("XReal Depth", depth_colored)
            
            final_l = img_l
            final_r = img_r
            
            if show_face:
                # Detect on raw Left image
                # Pass disparity to get distance
                final_l, final_r = decoder.detect_faces_stereo(img_l, img_r, disparity)
            
            # Now compose the display image
            # If we have color (from face detection drawing), we need to convert grayscale ones to color
            if len(final_l.shape) == 2:
                final_l = cv2.cvtColor(final_l, cv2.COLOR_GRAY2BGR)
            if len(final_r.shape) == 2:
                final_r = cv2.cvtColor(final_r, cv2.COLOR_GRAY2BGR)
                
            final_out = np.hstack((final_l, final_r))
            
            if mirror:
                final_out = cv2.flip(final_out, 1)
                
            cv2.imshow("XReal", final_out)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            decoder._rotation = (decoder._rotation + 1) % 3
            print(f"Rotation: {decoder._rotation}")
        elif key == ord('m'):
            mirror = not mirror
            print(f"Mirror: {mirror}")
        elif key == ord('d'):
            show_depth = not show_depth
            if not show_depth:
                cv2.destroyWindow("XReal Depth")
            print(f"Depth: {show_depth}")
        elif key == ord('f'):
            show_face = not show_face
            print(f"Face Detection: {show_face}")

    process.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
