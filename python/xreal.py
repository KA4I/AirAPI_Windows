# Usage
#
#   gst-launch-1.0 v4l2src device=/dev/videoX ! xrealultra2dec rotation=2 ! autovideoconvert ! autovideosink
#
# You can change the rotation option:
#  - 0: native (left CW, right CCW)
#  - 1: flip right (both CW)
#  - 2: correct for viewing stream
#
# Installation
#
# This is a python gstreamer plugin. It needs to be installed as
#   python/xreal.py
# in the gstreamer plugin search path (GST_PLUGIN_PATH). This must include the
# "python" subdirectory.
#
# e.g. on Fedora:
#  - mkdir -p ~/.gstreamer-1.0/plugins/python/
#  - cp xreal.py ~/.gstreamer-1.0/plugins/python/
#
# Or symlink the file.
#
# Note that this should be ~/.local/share/gstreamer-1.0/plugins/python, however
# at least on Fedora the path appears to be misconfigured or missing the XDG
# directories at least.
#
# Also make sure to have the python support for gstreamer installed.

import gi
import numpy as np
import struct

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstAudio', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GLib, GObject, GstBase, GstAudio, GstVideo

# TODO: Put in the caps that v4l2src will provide for the device
ICAPS = Gst.Caps(Gst.Structure('video/x-raw',
                               framerate=Gst.FractionRange(Gst.Fraction(1, 1),
                                                           Gst.Fraction(GLib.MAXINT, 1))))

OCAPS_VERT = \
    Gst.Caps(Gst.Structure('video/x-raw',
                           format='GRAY8',
                           width=480*2,
                           height=640,
                           framerate=Gst.FractionRange(Gst.Fraction(1, 1),
                                                       Gst.Fraction(GLib.MAXINT, 1))))
OCAPS_HORIZ = \
    Gst.Caps(Gst.Structure('video/x-raw',
                           format='GRAY8',
                           width=640,
                           height=480*2,
                           framerate=Gst.FractionRange(Gst.Fraction(1, 1),
                                                       Gst.Fraction(GLib.MAXINT, 1))))

OCAPS = OCAPS_VERT.copy()
OCAPS.append(OCAPS_HORIZ)

class XRealUltra2Dec(GstBase.BaseTransform):
    __gstmetadata__ = ('XRealUltra2Dec','Decoder/Video', \
                       'Descramble XReal ULTRA 2 Video frames', 'Benjamin Berg, Ani')

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                            Gst.PadDirection.SRC,
                                            Gst.PadPresence.ALWAYS,
                                            OCAPS),
                        Gst.PadTemplate.new("sink",
                                            Gst.PadDirection.SINK,
                                            Gst.PadPresence.ALWAYS,
                                            ICAPS))

    __gproperties__ = {
        "pts-from-frame": (bool,
                   "Set PTS from frame",
                   "Set the PTS timestamp from the reported device time",
                   False,
                   GObject.ParamFlags.READWRITE
                  ),
        "rotation": (int,
                   "Image rotation",
                   "Rotation: 0: native (CW, CCW), 1: both CW, 2: horizontal",
                   0,
                   2,
                   0,
                   GObject.ParamFlags.CONSTRUCT_ONLY | GObject.ParamFlags.READWRITE
                  )
    }

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

    def __init__(self):
        GstBase.BaseTransform.__init__(self)

        self._add_pts = False
        self._rotation = 0

        self._last_buf = None

    def do_get_property(self, prop):
        if prop.name == 'pts-from-frame':
            return self._add_pts
        elif prop.name == 'rotation':
            return self._rotation
        else:
            raise AttributeError('unknown property %s' % prop.name)

    def do_set_property(self, prop, value):
        if prop.name == 'pts-from-frame':
            self._add_pts = value
        elif prop.name == 'rotation':
            print("rotation:", value)
            self._rotation = value
        else:
            raise AttributeError('unknown property %s' % prop.name)

    # This is kind of wrong, we have a construct only property, we should just
    # create the pad with the correct caps.
    def do_transform_caps(self, direction, caps, filt):
        if direction == Gst.PadDirection.SINK:
            if self._rotation != 2:
                return OCAPS_HORIZ
            else:
                return OCAPS_VERT
        else:
            return ICAPS

    def handle_frame(self, in_frame, np_out):
        blocks = in_frame[:640*480].reshape((128, 2400))

        # Slightly faster
        CHUNK_MAP = self.CHUNK_MAP

        # TODO: Figure out a better way to get the starting index
        # (at least this does not loop in python ...)
        map_idx = blocks[:,:128].sum(axis=1).argmin()
        map_idx = CHUNK_MAP.index(map_idx)

        if self._rotation == 2:
            out_img = np_out.reshape((480*2, 640), order='F')
            if in_frame[480*640 + 0x3b]:
                # RIGHT
                if self._rotation == 2:
                    out_img = out_img[480:,:]
                else:
                    out_img = out_img[:,480:]
            else:
                # LEFT
                out_img = out_img[:480,:]
                # Swap axes as the image is rotated the other way around
                out_img = out_img[::-1,::-1]

            # Note, we are rotating the image here! y is our row, not x.
            p_y = 0
            p_x = 0
            for t_idx in range(128):
                source = blocks[CHUNK_MAP[map_idx]]

                pos = 0
                while pos < 2400:
                    p_y_new = min(640, p_y + 2400 - pos)
                    out_img[p_x,p_y:p_y_new] = source[pos:pos + p_y_new - p_y]

                    pos = pos + p_y_new - p_y
                    if p_y_new == 640:
                        p_x += 1
                        p_y = 0
                    else:
                        p_y = p_y_new

                map_idx = (map_idx + 1) % 128
        else:
            # We are not rotating the image. So we can easily copy the bytes,
            # just need to make sure the rotate the right image if requested
            if in_frame[480*640 + 0x3b]:
                # RIGHT
                out = np_out[480*640:]
            else:
                # LEFT
                out = np_out[:480*640]
                if self._rotation == 1:
                    out = out[::-1]

            for t_idx in range(128):
                source = blocks[CHUNK_MAP[map_idx]]
                out[t_idx * 2400:t_idx * 2400 + 2400] = source
                map_idx = (map_idx + 1) % 128

            pass


    def do_transform(self, inbuf, outbuf):
        # Input as linear array
        if self._last_buf is None:
            self._last_buf = inbuf
            return Gst.FlowReturn.CUSTOM_SUCCESS

        success, in1_map_info = self._last_buf.map(Gst.MapFlags.READ)
        assert success
        np_in1 = np.ndarray(
            shape=(640 * 482),
            dtype=np.uint8,
            buffer=in1_map_info.data)
        self._last_buf = None

        success, in2_map_info = inbuf.map(Gst.MapFlags.READ)
        assert success
        np_in2 = np.ndarray(
            shape=(640 * 482),
            dtype=np.uint8,
            buffer=in2_map_info.data)

        # Input as proper image; Fortran order to have x component first
        success, out_map_info = outbuf.map(Gst.MapFlags.WRITE)
        assert success
        np_out = np.ndarray(
            shape=(480 * 2 * 640),
            dtype=np.uint8,
            buffer=out_map_info.data)

        # TS1: a nanosecnd accurate timestamp (differs per camera)
        ts1_ns = struct.unpack('<Q', in1_map_info.data[640*480:640*480 + 8])[0]
        # TS2: a microsecond accurate timestamp (same for both cameras)
        ts2_ns = struct.unpack('<Q', in2_map_info.data[640*480 + 0x3e:640*480 + 0x46])

        seq1 = struct.unpack('<h', in1_map_info.data[640*480 + 18:640*480 + 20])[0]
        seq2 = struct.unpack('<h', in2_map_info.data[640*480 + 18:640*480 + 20])[0]

        if seq1 != seq2:
            self._last_buf = inbuf
            return Gst.FlowReturn.CUSTOM_SUCCESS

         # Add metadata to the buffer?
        if self._add_pts:
            try:
                outbuf.pts = ts1_ns - self._start_time
            except:
                outbuf.pts = 0
                self._start_time = ts1_ns

        self.handle_frame(np_in1, np_out)
        self.handle_frame(np_in2, np_out)

        return Gst.FlowReturn.OK

GObject.type_register(XRealUltra2Dec)
__gstelementfactory__ = ("xrealultra2dec", Gst.Rank.NONE, XRealUltra2Dec)


