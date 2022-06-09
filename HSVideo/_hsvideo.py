

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from scipy import fftpack as _fftpack
from scipy.signal import welch as _welch
# from scipy.signal.spectral import _spectral_helper
# from johnspythonlibrary2 import Plot as _plot
# from johnspythonlibrary2.Plot import subTitle as _subTitle, finalizeFigure as _finalizeFigure, finalizeSubplot as _finalizeSubplot
from johnspythonlibrary2.Process.Misc import check_dims as _check_dims
from johnspythonlibrary2.Process.Spectral import fft as _fft
from johnspythonlibrary2.Process.Spectral import calcPhaseDifference as _calcPhaseDifference
import xarray as _xr
from scipy.stats import _binned_statistic
from scipy.optimize import minimize as _minimize


###############################################################################
# %% Reading HS video data

def _read_cine(cine_file):
	
	###############################################################################

	# Reader for CINE files produced by Vision Research Phantom Software
	# Author: Dustin Kleckner
	# dkleckner@uchicago.edu

	# Modified by Thomas A Caswell (tcaswell@uchicago.edu)
	# Added to PIMS by Thomas A Caswell (tcaswell@gmail.com)

	# Modified by B. Neel
	###############################################################################

	
	from pims.frame import Frame
	from pims.base_frames import FramesSequence, index_attr
	from pims.utils.misc import FileLocker
	# import time
	import struct
	import numpy as np
	from numpy import array, frombuffer, where
	from threading import Lock
	import datetime
	import hashlib
	import warnings
	from collections.abc import Iterable
	import xarray as xr

	# __all__ = ('Cine', )


	# '<' for little endian (cine documentation)
	def _build_struct(dtype):
	    return struct.Struct(str("<" + dtype))


	FRACTION_MASK = (2**32-1)
	MAX_INT = 2**32

	# Harmonized/simplified cine file data types with Python struct doc
	UINT8 = 'B'
	CHAR = 'b'
	UINT16 = 'H'
	INT16 = 'h'
	BOOL = 'i'
	UINT32 = 'I'
	INT32 = 'i'
	INT64 = 'q'
	FLOAT = 'f'
	DOUBLE = 'd'
	TIME64 = 'Q'
	RECT = '4i'
	WBGAIN = '2f'
	IMFILTER = '28i'
	# TODO: get correct format for TrigTC
	TC = '8s'

	CFA_NONE = 0    # gray sensor
	CFA_VRI = 1     # gbrg/rggb sensor
	CFA_VRIV6 = 2   # bggr/grbg sensor
	CFA_BAYER = 3   # gb/rg sensor
	CFA_BAYERFLIP = 4   #rg/gb sensor

	TAGGED_FIELDS = {
	    1000: ('ang_dig_sigs', ''),
	    1001: ('image_time_total', TIME64),
	    1002: ('image_time_only', TIME64),
	    1003: ('exposure_only', UINT32),
	    1004: ('range_data', ''),
	    1005: ('binsig', ''),
	    1006: ('anasig', ''),
	    1007: ('time_code', '')}

	HEADER_FIELDS = [
	    ('type', '2s'),
	    ('header_size', UINT16),
	    ('compression', UINT16),
	    ('version', UINT16),
	    ('first_movie_image', INT32),
	    ('total_image_count', UINT32),
	    ('first_image_no', INT32),
	    ('image_count', UINT32),
	    # Offsets of following sections
	    ('off_image_header', UINT32),
	    ('off_setup', UINT32),
	    ('off_image_offsets', UINT32),
	    ('trigger_time', TIME64),
	]

	BITMAP_INFO_FIELDS = [
	    ('bi_size', UINT32),
	    ('bi_width', INT32),
	    ('bi_height', INT32),
	    ('bi_planes', UINT16),
	    ('bi_bit_count', UINT16),
	    ('bi_compression', UINT32),
	    ('bi_image_size', UINT32),
	    ('bi_x_pels_per_meter', INT32),
	    ('bi_y_pels_per_meter', INT32),
	    ('bi_clr_used', UINT32),
	    ('bi_clr_important', UINT32),
	]


	SETUP_FIELDS = [
	    ('frame_rate_16', UINT16),
	    ('shutter_16', UINT16),
	    ('post_trigger_16', UINT16),
	    ('frame_delay_16', UINT16),
	    ('aspect_ratio', UINT16),
	    ('contrast_16', UINT16),
	    ('bright_16', UINT16),
	    ('rotate_16', UINT8),
	    ('time_annotation', UINT8),
	    ('trig_cine', UINT8),
	    ('trig_frame', UINT8),
	    ('shutter_on', UINT8),
	    ('description_old', '121s'),
	    ('mark', '2s'),
	    ('length', UINT16),
	    ('binning', UINT16),
	    ('sig_option', UINT16),
	    ('bin_channels', INT16),
	    ('samples_per_image', UINT8),
	    ] + [('bin_name{:d}'.format(i), '11s') for i in range(8)] + [
	    ('ana_option', UINT16),
	    ('ana_channels', INT16),
	    ('res_6', UINT8),
	    ('ana_board', UINT8),
	    ] + [('ch_option{:d}'.format(i), INT16) for i in range(8)] + [
	    ] + [('ana_gain{:d}'.format(i), FLOAT) for i in range(8)] + [
	    ] + [('ana_unit{:d}'.format(i), '6s') for i in range(8)] + [
	    ] + [('ana_name{:d}'.format(i), '11s') for i in range(8)] + [
	    ('i_first_image', INT32),
	    ('dw_image_count', UINT32),
	    ('n_q_factor', INT16),
	    ('w_cine_file_type', UINT16),
	    ] + [('sz_cine_path{:d}'.format(i), '65s') for i in range(4)] + [
	    ('b_mains_freq', UINT16),
	    ('b_time_code', UINT8),
	    ('b_priority', UINT8),
	    ('w_leap_sec_dy', UINT16),
	    ('d_delay_tc', DOUBLE),
	    ('d_delay_pps', DOUBLE),
	    ('gen_bits', UINT16),
	    ('res_1', INT32),  
	    ('res_2', INT32),
	    ('res_3', INT32),
	    ('im_width', UINT16),
	    ('im_height', UINT16),
	    ('edr_shutter_16', UINT16),
	    ('serial', UINT32),
	    ('saturation', INT32),
	    ('res_5', UINT8),
	    ('auto_exposure', UINT32),
	    ('b_flip_h', BOOL),
	    ('b_flip_v', BOOL),
	    ('grid', UINT32),
	    ('frame_rate', UINT32),
	    ('shutter', UINT32),
	    ('edr_shutter', UINT32),
	    ('post_trigger', UINT32),
	    ('frame_delay', UINT32),
	    ('b_enable_color', BOOL),
	    ('camera_version', UINT32),
	    ('firmware_version', UINT32),
	    ('software_version', UINT32),
	    ('recording_time_zone', INT32),
	    ('cfa', UINT32),
	    ('bright', INT32),
	    ('contrast', INT32),
	    ('gamma', INT32),
	    ('res_21', UINT32),
	    ('auto_exp_level', UINT32),
	    ('auto_exp_speed', UINT32),
	    ('auto_exp_rect', RECT),
	    ('wb_gain', '8f'),
	    ('rotate', INT32),
	    ('wb_view', WBGAIN),
	    ('real_bpp', UINT32),
	    ('conv_8_min', UINT32),
	    ('conv_8_max', UINT32),
	    ('filter_code', INT32),
	    ('filter_param', INT32),
	    ('uf', IMFILTER),
	    ('black_cal_sver', UINT32),
	    ('white_cal_sver', UINT32),
	    ('gray_cal_sver', UINT32),
	    ('b_stamp_time', BOOL),
	    ('sound_dest', UINT32),
	    ('frp_steps', UINT32),
	    ] + [('frp_img_nr{:d}'.format(i), INT32) for i in range(16)] + [
	    ] + [('frp_rate{:d}'.format(i), UINT32) for i in range(16)] + [
	    ] + [('frp_exp{:d}'.format(i), UINT32) for i in range(16)] + [
	    ('mc_cnt', INT32),
	    ] + [('mc_percent{:d}'.format(i), FLOAT) for i in range(64)] + [
	    ('ci_calib', UINT32),
	    ('calib_width', UINT32),
	    ('calib_height', UINT32),
	    ('calib_rate', UINT32),
	    ('calib_exp', UINT32),
	    ('calib_edr', UINT32),
	    ('calib_temp', UINT32),
	    ] + [('header_serial{:d}'.format(i), UINT32) for i in range(4)] + [
	    ('range_code', UINT32),
	    ('range_size', UINT32),
	    ('decimation', UINT32),
	    ('master_serial', UINT32),
	    ('sensor', UINT32),
	    ('shutter_ns', UINT32),
	    ('edr_shutter_ns', UINT32),
	    ('frame_delay_ns', UINT32),
	    ('im_pos_xacq', UINT32),
	    ('im_pos_yacq', UINT32),
	    ('im_width_acq', UINT32),
	    ('im_height_acq', UINT32),
	    ('description', '4096s'),
	    ('rising_edge', BOOL),
	    ('filter_time', UINT32),
	    ('long_ready', BOOL),
	    ('shutter_off', BOOL),
	    ('res_4', '16s'),
	    ('b_meta_WB', BOOL),
	    ('hue', INT32),
	    ('black_level', INT32),
	    ('white_level', INT32),
	    ('lens_description', '256s'),
	    ('lens_aperture', FLOAT),
	    ('lens_focus_distance', FLOAT),
	    ('lens_focal_length', FLOAT),
	    ('f_offset', FLOAT),
	    ('f_gain', FLOAT),
	    ('f_saturation', FLOAT),
	    ('f_hue', FLOAT),
	    ('f_gamma', FLOAT),
	    ('f_gamma_R', FLOAT),
	    ('f_gamma_B', FLOAT),
	    ('f_flare', FLOAT),
	    ('f_pedestal_R', FLOAT),
	    ('f_pedestal_G', FLOAT),
	    ('f_pedestal_B', FLOAT),
	    ('f_chroma', FLOAT),
	    ('tone_label', '256s'),
	    ('tone_points', INT32),
	    ('f_tone', ''.join(32*['2f'])),
	    ('user_matrix_label', '256s'),
	    ('enable_matrices', BOOL),
	    ('f_user_matrix', '9'+FLOAT),
	    ('enable_crop', BOOL),
	    ('crop_left_top_right_bottom', '4i'),
	    ('enable_resample', BOOL),
	    ('resample_width', UINT32),
	    ('resample_height', UINT32),
	    ('f_gain16_8', FLOAT),
	    ('frp_shape', '16'+UINT32),
	    ('trig_TC', TC),
	    ('f_pb_rate', FLOAT),
	    ('f_tc_rate', FLOAT),
	    ('cine_name', '256s')
	]

	#from VR doc: This field is maintained for compatibility with old versions but
	#a new field was added for that information. The new field can be larger or may
	#have a different measurement unit.
	UPDATED_FIELDS = {
	        'frame_rate_16': 'frame_rate',
	        'shutter_16': 'shutter_ns',
	        'post_trigger_16': 'post_trigger',
	        'frame_delay_16': 'frame_delay_ns',
	        'edr_shutter_16': 'edr_shutter_ns',
	        'saturation': 'f_saturation',
	        'shutter': 'shutter_ns',
	        'edr_shutter': 'edr_shutter_ns',
	        'frame_delay': 'frame_delay_ns',
	        'bright': 'f_offset',
	        'contrast': 'f_gain',
	        'gamma': 'f_gamma',
	        'conv_8_max': 'f_gain16_8',
	        'hue': 'f_hue',
	        }

	#from VR doc: to be ignored, not used anymore
	TO_BE_IGNORED_FIELDS = {
	        'contrast_16': 'res_7',
	        'bright_16': 'res_8',
	        'rotate_16': 'res_9',
	        'time_annotation': 'res_10',
	        'trig_cine': 'res_11',
	        'shutter_on': 'res_12',
	        'binning': 'res_13',
	        'b_mains_freq': 'res_14', 
	        'b_time_code': 'res_15',
	        'b_priority': 'res_16',
	        'w_leap_sec_dy': 'res_17',
	        'd_delay_tc': 'res_18',
	        'd_delay_pps': 'res_19',
	        'gen_bits': 'res_20',
	        'conv_8_min': '',
	        }

	# from VR doc: last setup field appearing in software version
	# TODO: keep up-to-date with newer and more precise doc, if available
	END_OF_SETUP = {
	        551: 'software_version',
	        552: 'recording_time_zone',
	        578: 'rotate',
	        605: 'b_stamp_time',
	        606: 'mc_percent63',
	        607: 'head_serial3',
	        614: 'decimation',
	        624: 'master_serial',
	        625: 'sensor',
	        631: 'frame_delay_ns',
	        637: 'description',
	        671: 'hue',
	        691: 'lens_focal_length',
	        693: 'f_gain16_8',
	        701: 'f_tc_rate',
	        702: 'cine_name',
	        }


	class Cine(FramesSequence):
	    """Read cine files

	    Read cine files, the out put from Vision Research high-speed phantom
	    cameras.  Support uncompressed monochrome and color files.

	    Nominally thread-safe, but this assertion is not tested.


	    Parameters
	    ----------
	    filename : string
	        Path to cine (or chd) file.
	    
	    Notes
	    -----
	    For a .chd file, this class only reads the header, not the images.
	    
	    """
	    # TODO: Unit tests using a small sample cine file.
	    @classmethod
	    def class_exts(cls):
	        return {'cine'} | super(Cine, cls).class_exts()

	    propagate_attrs = ['frame_shape', 'pixel_type', 'filename', 'frame_rate',
	                       'get_fps', 'compression', 'cfa', 'off_set']

	    def __init__(self, filename):
	        super(Cine, self).__init__()
	        self.f = open(filename, 'rb')
	        self._filename = filename

	        ### HEADER
	        self.header_dict = self._read_header(HEADER_FIELDS)
	        self.bitmapinfo_dict = self._read_header(BITMAP_INFO_FIELDS,
	                                                self.off_image_header)
	        self.setup_fields_dict = self._read_header(SETUP_FIELDS, self.off_setup)
	        self.setup_fields_dict = self.clean_setup_dict()

	        self._width = self.bitmapinfo_dict['bi_width']
	        self._height = self.bitmapinfo_dict['bi_height']
	        self._pixel_count = self._width * self._height

	        # Allows Cine object to be accessed from multiple threads!
	        self.file_lock = Lock()

	        self._hash = None

	        self._im_sz = (self._width, self._height)

	        # sort out the data type by reading the meta-data
	        if self.bitmapinfo_dict['bi_bit_count'] in (8, 24):
	            self._data_type = 'u1'
	        else:
	            self._data_type = 'u2'
	        self.tagged_blocks = self._read_tagged_blocks()
	        self.frame_time_stamps = self.tagged_blocks['image_time_only']
	        self.all_exposures = self.tagged_blocks['exposure_only']
	        self.stack_meta_data = dict()
	        self.stack_meta_data.update(self.bitmapinfo_dict)
	        self.stack_meta_data.update({k: self.setup_fields_dict[k]
	                                     for k in set(('trig_frame',
	                                                   'gamma',
	                                                   'frame_rate',
	                                                   'shutter_ns'
	                                                   )
	                                                   )
	                                                   })
	        self.stack_meta_data.update({k: self.header_dict[k]
	                                     for k in set(('first_image_no',
	                                                   'image_count',
	                                                   'total_image_count',
	                                                   'first_movie_image'
	                                                   )
	                                                   )
	                                                   })
	        self.stack_meta_data['trigger_time'] = self.trigger_time

	        ### IMAGES
	        # Move to images offset to test EOF...
	        self.f.seek(self.off_image_offsets)
	        if self.f.read(1) != b'':
	            # ... If no, read images
	            self.image_locations = self._unpack('%dQ' % self.image_count,
	                                               self.off_image_offsets)
	            if type(self.image_locations) not in (list, tuple):
	                self.image_locations = [self.image_locations]
	        # TODO: add support for reading sequence within the same framework, when data
	        # has been saved in another format (.tif, image sequence, etc)

	    def clean_setup_dict(self):
	        r"""Clean setup dictionary by removing newer fields, when compared to the
	        software version, and trailing null character b'\x00' in entries.

	        Notes
	        -----
	        The method is called after building the setup from the raw cine header.
	        It can be overridden to match more specific purposes (e.g. filtering
	        out TO_BE_IGNORED_ and UPDATED_FIELDS).

	        See also
	        --------
	        `Vision Research Phantom documentation <http://phantomhighspeed-knowledge.force.com/servlet/fileField?id=0BE1N000000kD2i>`_
	        """
	        setup = self.setup_fields_dict.copy()
	        # End setup at correct field (according to doc)
	        versions = sorted(END_OF_SETUP.keys())
	        fields = [v[0] for v in SETUP_FIELDS]
	        v = setup['software_version']
	        # Get next field where setup is known to have ended, according to VR
	        try:
	            v_up = versions[sorted(where(array(versions) >= v)[0])[0]]
	            last_field = END_OF_SETUP[v_up]
	            for k in fields[fields.index(last_field)+1:]:
	                del setup[k]
	        except IndexError:
	            # Or go to the end (waiting for updated documentation)
	            pass

	        # Remove blank characters
	        setup = _convert_null_byte(setup)

	        # Filter out 'res_' (reserved/obsolete) fields
	        #k_res = [k for k in setup.keys() if k.startswith('res_')]
	        #for k in k_res:
	        #    del setup[k]

	        # Format f_tone properly
	        if 'f_tone' in setup.keys():
	            tone = setup['f_tone']
	            setup['f_tone'] = tuple((tone[2*k], tone[2*k+1])\
	                                    for k in range(setup['tone_points']))
	        return setup


	    @property
	    def filename(self):
	        return self._filename
	    
	    @property
	    def frame_rate(self):
	        """Frame rate (setting in Phantom PCC software) (Hz).
	        May differ from computed average one.
	        """
	        return self.setup_fields_dict['frame_rate']

	    @property
	    def frame_rate_avg(self):
	        """Actual frame rate, averaged on frame timestamps (Hz)."""
	        return self.get_frame_rate_avg()

	    # use properties for things that should not be changeable
	    @property
	    def cfa(self):
	        return self.setup_fields_dict['cfa']

	    @property
	    def compression(self):
	        return self.header_dict['compression']

	    @property
	    def pixel_type(self):
	        return np.dtype(self._data_type)

	    # TODO: what is this field??? (baneel)
	    @property
	    def off_set(self):
	        return self.header_dict['offset']

	    @property
	    def setup_length(self):
	        return self.setup_fields_dict['length']

	    @property
	    def off_image_offsets(self):
	        return self.header_dict['off_image_offsets']

	    @property
	    def off_image_header(self):
	        return self.header_dict['off_image_header']

	    @property
	    def off_setup(self):
	        return self.header_dict['off_setup']

	    @property
	    def image_count(self):
	        return self.header_dict['image_count']

	    @property
	    def frame_shape(self):
	        return self._im_sz

	    @property
	    def shape(self):
	        """Shape of virtual np.array containing images."""
	        W, H = self.frame_shape
	        return self.len(), H, W

	    def get_frame(self, j):
	        md = dict()
	        md['exposure'] = self.all_exposures[j]
	        ts, sec_frac = self.frame_time_stamps[j]
	        md['frame_time'] = {'datetime': ts,
	                            'second_fraction': sec_frac,
	                            'time_to_trigger': self.get_time_to_trigger(j),
	                            }
	        return Frame(self._get_frame(j), frame_no=j, metadata=md)

	    def _unpack(self, fs, offset=None):
	        if offset is not None:
	            self.f.seek(offset)
	        s = _build_struct(fs)
	        vals = s.unpack(self.f.read(s.size))
	        if len(vals) == 1:
	            return vals[0]
	        else:
	            return vals

	    def _read_tagged_blocks(self):
	        """Reads the tagged block meta-data from the header."""
	        tmp_dict = dict()
	        if not self.off_setup + self.setup_length < self.off_image_offsets:
	            return
	        next_tag_exists = True
	        next_tag_offset = 0
	        while next_tag_exists:
	            block_size, next_tag_exists = self._read_tag_block(next_tag_offset,
	                                                               tmp_dict)
	            next_tag_offset += block_size
	        return tmp_dict

	    def _read_tag_block(self, off_set, accum_dict):
	        '''
	        Internal helper-function for reading the tagged blocks.
	        '''
	        with FileLocker(self.file_lock):
	            self.f.seek(self.off_setup + self.setup_length + off_set)
	            block_size = self._unpack(UINT32)
	            b_type = self._unpack(UINT16)
	            more_tags = self._unpack(UINT16)

	            if b_type == 1004:
	                # docs say to ignore range data it seems to be a poison flag,
	                # if see this, give up tag parsing
	                return block_size, 0

	            try:
	                d_name, d_type = TAGGED_FIELDS[b_type]

	            except KeyError:
	                return block_size, more_tags

	            if d_type == '':
	                # print "can't deal with  <" + d_name + "> tagged data"
	                return block_size, more_tags

	            s_tmp = _build_struct(d_type)
	            if (block_size-8) % s_tmp.size != 0:
	                #            print 'something is wrong with your data types'
	                return block_size, more_tags

	            d_count = (block_size-8)//(s_tmp.size)

	            data = self._unpack('%d' % d_count + d_type)
	            if not isinstance(data, tuple):
	                # fix up data due to design choice in self.unpack
	                data = (data, )

	            # parse time
	            if b_type == 1002 or b_type == 1001:
	                data = [(datetime.datetime.fromtimestamp(d >> 32),
	                         (FRACTION_MASK & d)/MAX_INT) for d in data]
	            # convert exposure to seconds
	            if b_type == 1003:
	                data = [d/(MAX_INT) for d in data]

	            accum_dict[d_name] = data

	        return block_size, more_tags

	    def _read_header(self, fields, offset=0):
	        self.f.seek(offset)
	        tmp = dict()
	        for name, format in fields:
	            val = self._unpack(format)
	            tmp[name] = val

	        return tmp

	    def _get_frame(self, number):
	        with FileLocker(self.file_lock):
	            # get basic information about the frame we want
	            image_start = self.image_locations[number]
	            annotation_size = self._unpack(UINT32, image_start)
	            # this is not used, but is needed to advance the point in the file
	            annotation = self._unpack('%db' % (annotation_size - 8))
	            image_size = self._unpack(UINT32)

	            cfa = self.cfa
	            compression = self.compression

	            # sort out data type looking at the cached version
	            data_type = self._data_type

	            # actual bit per pixel
	            actual_bits = image_size * 8 // (self._pixel_count)

	            # so this seem wrong as 10 or 12 bits won't fit in 'u1'
	            # but I (TAC) may not understand and don't have a packed file
	            # (which the docs seem to imply don't exist) to test on so
	            # I am leaving it.  good luck.
	            if actual_bits in (10, 12):
	                data_type = 'u1'

	            # move the file to the right point in the file
	            self.f.seek(image_start + annotation_size)

	            # suck the data out of the file and shove into linear
	            # numpy array
	            frame = frombuffer(self.f.read(image_size), data_type)

	            # if mono-camera
	            if cfa == CFA_NONE:
	                if compression != 0:
	                    raise ValueError("Can not deal with compressed files\n" +
	                                     "compression level: " +
	                                     "{}".format(compression))
	                # we are working with a monochrome camera
	                # un-pack packed data
	                if (actual_bits == 10):
	                    frame = _ten2sixteen(frame)
	                elif (actual_bits == 12):
	                    frame = _twelve2sixteen(frame)
	                elif (actual_bits % 8):
	                    raise ValueError('Data should be byte aligned, ' +
	                         'or 10 or 12 bit packed (appears to be' +
	                        ' %dbits/pixel?!)' % actual_bits)

	                # re-shape to an array
	                # flip the rows
	                frame = frame.reshape(self._height, self._width)[::-1]

	                if actual_bits in (10, 12):
	                    frame = frame[::-1, :]
	                    # Don't know why it works this way, but it does...
	            # else, some sort of color layout
	            else:
	                if compression == 0:
	                    # and re-order so color is RGB (naively saves as BGR)
	                    frame = frame.reshape(self._height, self._width,
	                                          3)[::-1, :, ::-1]
	                elif compression == 2:
	                    raise ValueError("Can not process un-interpolated movies")
	                else:
	                    raise ValueError("Should never hit this, " +
	                                     "you have an un-documented file\n" +
	                                     "compression level: " +
	                                     "{}".format(compression))

	        return frame

	    def __len__(self):
	        return self.image_count

	    len = __len__

	    @index_attr
	    def get_time(self, i):
	        """Return the time of frame i in seconds, relative to first frame."""
	        warnings.warn("This is not guaranteed to be the actual time. "\
	                      +"See self.get_time_to_trigger(i) method.",
	                      category=PendingDeprecationWarning)
	        return float(i) / self.frame_rate

	    @index_attr
	    def get_time_to_trigger(self, i):
	        """Get actual time (s) of frame i, relative to trigger."""
	        ti = self.frame_time_stamps[i]
	        ti = ti[0].timestamp() + ti[1]
	        tt= self.trigger_time
	        tt = tt['datetime'].timestamp() + tt['second_fraction']
	        return ti - tt

	    def get_frame_rate_avg(self, error_tol=1e-3):
	        """Compute mean frame rate (Hz), on the basis of frame time stamps.

	        Parameters
	        ----------
	        error_tol : float, optional.
	            Tolerance on relative error (standard deviation/mean),
	            above which a warning is raised.

	        Returns
	        -------
	        fps : float.
	            Actual mean frame rate, based on the frames time stamps.
	        """
	        times = np.r_[[self.get_time_to_trigger(i) for i in range(self.len())]]
	        freqs = 1 / np.diff(times)
	        fps, std = freqs.mean(), freqs.std()
	        error = std / fps
	        if error > error_tol:
	            warnings.warn('Relative precision on the average frame rate is '\
	                          +'{:.2f}%.'.format(1e2*error))
	        return fps

	    def get_fps(self):
	        """Get frame rate (setting in Phantom PCC software) (Hz).
	        May differ from computed average one.

	        See also
	        --------
	        PCC setting (all fields refer to the same value)
	            self.frame_rate
	            self.setup_fields_dict['frame_rate']
	        
	        Computed average
	            self.frame_rate_avg
	            self.get_frame_rate_avg()
	        """
	        return self.frame_rate

	    def close(self):
	        self.f.close()

	    def __unicode__(self):
	        return self.filename

	    # def __str__(self):
	    #     return unicode(self).encode('utf-8')

	    def __repr__(self):
	        # May be overwritten by subclasses
	        return """<Frames>
	Source: {filename}
	Length: {count} frames
	Frame Shape: {frame_shape!r}
	Pixel Datatype: {dtype}""".format(frame_shape=self.frame_shape,
	                                  count=len(self),
	                                  filename=self.filename,
	                                  dtype=self.pixel_type)

	    @property
	    def trigger_time(self):
	        '''Returns the time of the trigger, tuple of (datatime_object,
	        fraction_in_s)'''
	        trigger_time = self.header_dict['trigger_time']
	        ts, sf = (datetime.datetime.fromtimestamp(trigger_time >> 32),
	                   float(FRACTION_MASK & trigger_time)/(MAX_INT))

	        return {'datetime': ts, 'second_fraction': sf}

	    @property
	    def hash(self):
	        if self._hash is None:
	            self._hash_fun()
	        return self._hash

	    def __hash__(self):
	        return int(self.hash, base=16)

	    def _hash_fun(self):
	        """Generates the md5 hash of the header of the file.  Here the
	        header is defined as everything before the first image starts.

	        This includes all of the meta-data (including the plethora of
	        time stamps) so this will be unique.
	        """
	        # get the file lock (so we don't screw up any other reads)
	        with FileLocker(self.file_lock):

	            self.f.seek(0)
	            max_loc = self.image_locations[0]
	            md5 = hashlib.md5()

	            chunk_size = 128*md5.block_size
	            chunk_count = (max_loc//chunk_size) + 1

	            for j in range(chunk_count):
	                md5.update(self.f.read(128*md5.block_size))

	            self._hash = md5.hexdigest()

	    def __eq__(self, other):
	        return self.hash == other.hash

	    def __ne__(self, other):
	        return not self == other


	# Should be divisible by 3, 4 and 5!  This seems to be near-optimal.
	CHUNK_SIZE = 6 * 10 ** 5


	def _ten2sixteen(a):
	    """Convert array of 10bit uints to array of 16bit uints."""
	    b = np.zeros(a.size//5*4, dtype='u2')

	    for j in range(0, len(a), CHUNK_SIZE):
	        (a0, a1, a2, a3, a4) = [a[j+i:j+CHUNK_SIZE:5].astype('u2')
	                                for i in range(5)]

	        k = j//5 * 4
	        k2 = k + CHUNK_SIZE//5 * 4

	        b[k+0:k2:4] = ((a0 & 0b11111111) << 2) + ((a1 & 0b11000000) >> 6)
	        b[k+1:k2:4] = ((a1 & 0b00111111) << 4) + ((a2 & 0b11110000) >> 4)
	        b[k+2:k2:4] = ((a2 & 0b00001111) << 6) + ((a3 & 0b11111100) >> 2)
	        b[k+3:k2:4] = ((a3 & 0b00000011) << 8) + ((a4 & 0b11111111) >> 0)

	    return b


	def _sixteen2ten(b):
	    """Convert array of 16bit uints to array of 10bit uints."""
	    a = np.zeros(b.size//4*5, dtype='u1')

	    for j in range(0, len(a), CHUNK_SIZE):
	        (b0, b1, b2, b3) = [b[j+i:j+CHUNK_SIZE:4] for i in range(4)]

	        k = j//4 * 5
	        k2 = k + CHUNK_SIZE//4 * 5

	        a[k+0:k2:5] =                              ((b0 & 0b1111111100) >> 2)
	        a[k+1:k2:5] = ((b0 & 0b0000000011) << 6) + ((b1 & 0b1111110000) >> 4)
	        a[k+2:k2:5] = ((b1 & 0b0000001111) << 4) + ((b2 & 0b1111000000) >> 6)
	        a[k+3:k2:5] = ((b2 & 0b0000111111) << 2) + ((b3 & 0b1100000000) >> 8)
	        a[k+4:k2:5] = ((b3 & 0b0011111111) << 0)

	    return a


	def _twelve2sixteen(a):
	    """Convert array of 12bit uints to array of 16bit uints."""
	    b = np.zeros(a.size//3*2, dtype='u2')

	    for j in range(0, len(a), CHUNK_SIZE):
	        (a0, a1, a2) = [a[j+i:j+CHUNK_SIZE:3].astype('u2') for i in range(3)]

	        k = j//3 * 2
	        k2 = k + CHUNK_SIZE//3 * 2

	        b[k+0:k2:2] = ((a0 & 0xFF) << 4) + ((a1 & 0xF0) >> 4)
	        b[k+1:k2:2] = ((a1 & 0x0F) << 8) + ((a2 & 0xFF) >> 0)

	    return b


	def _sixteen2twelve(b):
	    """Convert array of 16bit uints to array of 12bit uints."""
	    a = np.zeros(b.size//2*3, dtype='u1')

	    for j in range(0, len(a), CHUNK_SIZE):
	        (b0, b1) = [b[j+i:j+CHUNK_SIZE:2] for i in range(2)]

	        k = j//2 * 3
	        k2 = k + CHUNK_SIZE//2 * 3

	        a[k+0:k2:3] =                       ((b0 & 0xFF0) >> 4)
	        a[k+1:k2:3] = ((b0 & 0x00F) << 4) + ((b1 & 0xF00) >> 8)
	        a[k+2:k2:3] = ((b1 & 0x0FF) << 0)

	    return a


	def _convert_null_byte(dic):
	    """
	    Convert binary null character b'\x00' to empty string in dictionary entries.

	    Parameters
	    ----------
	    dic : dict
	        Dictionary to clean. Function loops over the key-value pairs and
	        converts the null byte `b'\x00'` to empty string `''`.

	    Returns
	    -------
	    clean_dic : dict
	        Cleaned dictionary.

	    Notes
	    -----
	    The routine is intended to work on string-like bytes array (resp. Iterable
	    of such arrays), and return a string (resp. a list of strings).
	    """
	    for k, v in dic.items():
	        if isinstance(v, bytes):
	            try:
	                dic[k] = v.decode('utf8').replace('\x00', '')
	            except (UnicodeDecodeError):
	                pass
	        elif isinstance(v, Iterable):
	            try:
	                dic[k] = [el.decode('utf8').replace('\x00', '')\
	                          for el in v]
	            except (AttributeError, UnicodeDecodeError):
	                pass
	    return dic
	
	return Cine(cine_file)


def cine_greyonly_to_xr(cine_file):

	# get file
	obj = _read_cine(cine_file)
	
	# convert to xarray
	dt = 1 / obj.get_fps()
	t = _np.arange(obj.shape[0]) * dt
	t = _xr.DataArray(t, dims='t', coords=[t])
	x = _np.arange(obj.shape[1])
	x = _xr.DataArray(x, dims='x', coords=[x])
	y = _np.arange(obj.shape[2], 0, -1) -1
	y = _xr.DataArray(y, dims='y', coords=[y])
	video = _xr.DataArray(_np.array(obj), dims=['t', 'y', 'x'], coords=[t, y, x])

	# TODO confirm that x and y are correct when I move to non-square resolutions
	# TODO make this code work for color
	
	return video


###############################################################################
#%% Dispersion plots

def dispersion_plot(video_data_1D, nperseg_dim1=1000,  dim2='theta', dim2_final='m', vmin=None, vmax=None, plot=True, f_units='Hz'):
	"""
	Calculates a dispersion plot from a 1D video dataset
	
	Parameters
	----------
	video_data_1D : xarray.core.dataarray.DataArray
		1D video data.  dims = ['t', spatial (e.g. theta or r)].  Time must be first.
	nperseg_dim1 : int or None
		int - Welch FFT averaging is applied to the time data where nperseg is the window size.  The output will be real.
		None - Standard FFT is applied to the time data (i.e. no windowing).  The output will be complex.
	dim2 : str
		The name of the spatial dimension
	dim2_final : str
		The name of the spatial dimension after the FFT is applied
	vmin : float
		Lower limit of the colorbar scale
	vmax : float
		Upper limit of the colorbar scale
	plot : bool
		True causes the plot to be produced.
	f_units : str
		Name of the frequency units.  (e.g. if t=t*1e3 is the input, then specify f_units='kHz'.)
		
	Returns
	-------
	X_2D : xarray.core.dataarray.DataArray
		Dipserion relationship.  Values are real if nperseg_dim1 is a number.  Complex if nperseg_dim1 is None.
	"""
	## Check dimensions
	_check_dims(video_data_1D, dims=['t',dim2])
	if video_data_1D.dims[0]!='t':
		raise Exception("The first dimension needs to be time, 't'")

	## FFT along dim2 (the spatial dimension)
	if True: 
		# preliminary steps
		dtheta = float(video_data_1D[dim2][1] -
					   video_data_1D[dim2][0]) / (2 * _np.pi)
		m = _fftpack.fftfreq(len(video_data_1D[dim2]), d=dtheta)
	
		# perform FFT
		X = _np.fft.fft(video_data_1D, axis=1)
		X = _xr.DataArray(X, dims=['t', dim2_final],
							coords=[video_data_1D['t'], m]).sortby(dim2_final)
	
		# return the results to the correct amplitude
		N = len(video_data_1D[dim2])
		X *= 1.0 / N  # use 2.0/N only if you've trimmed the negative freqs

	## FFT along time, t (dim1)
	if True:
		
		# preliminary steps
		dt = float(X.t[1] - X.t[0])
		
		# perform time-averaged (windowed) FFT if  nperseg_dim1 is a number
		if nperseg_dim1 is not None:
			freq, X_2D = _welch( 	X.data, fs=1.0/dt, nperseg=nperseg_dim1,
									noverlap=nperseg_dim1//2, return_onesided=True,
									scaling='spectrum', axis=0)
		# otherwise, perform standard fft 
		else: 
					freq = _fftpack.fftfreq(len(X['t']), d=dt)
					X_2D = _np.fft.fft(X.data, axis=0)
					N = len(video_data_1D['t'])
					X_2D *= 1.0 / N  # use 2.0/N only if you've trimmed the negative freqs

		X_2D = _xr.DataArray(X_2D, dims=['f', dim2_final],
							coords=[freq, X[dim2_final]]).sortby('f')
		X_2D.attrs={'long_name':'Spectral density','units':'au'}
		X_2D.f.attrs={'long_name':'FFT Frequency','units':f_units}
		X_2D[dim2_final].attrs={'long_name': dim2_final,'units':''}
	
	if plot==True:
		# convert to absolute value and take log10 (for vetter visualization)
		a=_np.log10(_np.abs(X_2D))
		a.attrs={'long_name':'Spectral density','units':'au, log10'}
		
		# set vmin and vmax (color scaling limits)
		if type(vmin)==type(None):
			vmin=float(a.min())
		if type(vmax)==type(None):
			vmax=float(a.max())#+0.5
		
		# plot
		fig, ax = _plt.subplots()
		a.plot(ax=ax, vmin=vmin, vmax=vmax)
		ax.set_title('dispersion plot')
	
	return X_2D
	
	
def dispersion_plot_2points(da1, da2, x_separation=1, nperseg=None, plot=True):
	# https://scholar.colorado.edu/downloads/qj72p7185
	# https://aip.scitation.org/doi/pdf/10.1063/1.2889424
	# https://aip.scitation.org/doi/pdf/10.1063/1.331279
	"""
	filename='C:\\Users\\jwbrooks\\data\\marcels_thesis_data\\20A_5sccm_5mm_6.29.2019_7.07 PM.mat'
	matData=jpl2.ReadWrite.mat_to_dict(filename)
	t=matData['t'].reshape(-1)
	da1=xr.DataArray(matData['s1'].reshape(-1), dims='t', coords=[t])
	da2=xr.DataArray(matData['s4'].reshape(-1), dims='t', coords=[t])

	x_separation=3e-3
	"""
	
	# check input
	_check_dims(da1,'t')
	_check_dims(da2,'t')
		
	# parameters
	nperseg=20000
	N_k=50
	N_f=1000
	
	# initialize arrays
	S=_np.zeros((N_k,N_f),dtype=float)
	count=_np.zeros((N_k,N_f),dtype=int)
	
	def calc_fft_and_k(x1,x2):	
		fft1=_fft(x1, plot=False).sortby('f')
		fft2=_fft(x2, plot=False).sortby('f')
		s=_np.real(0.5*(_np.conj(fft1)*fft1+_np.conj(fft2)*fft2))
		phase_diff,_,_=_calcPhaseDifference(fft1, fft2, plot=False)
		k=phase_diff/x_separation
# 		k_bins=_np.linspace(k.data.min(),k.data.max(),N_k+1)
# 		f_bins=_np.linspace(k.f.data.min(),k.f.data.max(),N_f+1)
		
		return s, k
	
	
	# calculate bin sizes
	s,k=calc_fft_and_k(da1,da2)
	k_bins=_np.linspace(k.data.min(),k.data.max(),N_k+1)
	f_bins=_np.linspace(k.f.data.min(),k.f.data.max(),N_f+1)
		
	
	# itegrate through each time window
	segs=_np.arange(0,len(da1),nperseg)
	for i,seg in enumerate(segs):
		if len(da1[seg:seg+nperseg])<nperseg:
			pass
		else:
			print(seg)
# 			
# 			fft1=fft(da1[seg:seg+nperseg], plot=False).sortby('f')
# 			fft2=fft(da2[seg:seg+nperseg], plot=False).sortby('f')
# 			s=_np.real(0.5*(_np.conj(fft1)*fft1+_np.conj(fft2)*fft2))
# 			
# 			phase_diff,_,_=calcPhaseDifference(fft1, fft2, plot=False)
# 			k=phase_diff/x_separation
# 			
# 			if i == 0:
# 				k_bins=_np.linspace(k.data.min(),k.data.max(),N_k+1)
# 				f_bins=_np.linspace(k.f.data.min(),k.f.data.max(),N_f+1)
# 				
			
			s,k=calc_fft_and_k(da1[seg:seg+nperseg], da2[seg:seg+nperseg])
			data=_pd.DataFrame()
			data['f']=s.f.data
			data['S']=s.data
			data['k']=k.data
			
			for a in range(N_k):
				for b in range(N_f):
					c=data.where((data['k']>k_bins[a])&(data['k']<k_bins[a+1])&(data['f']>f_bins[b])&(data['f']<f_bins[b+1])).dropna()
					count[a,b]+=len(c)
					S[a,b]=S[a,b]+c['S'].sum()
					
	count[count==0]=1	# prevent divide by 0 issues
	S=_xr.DataArray(S/count, dims=['k','f'],coords=[ (k_bins[1:]+k_bins[0:-1])/2, (f_bins[1:]+f_bins[0:-1])/2])
	
	if plot==True:
		fig,ax=_plt.subplots()
		count=_xr.DataArray(count, dims=['k','f'],coords=[ (k_bins[1:]+k_bins[0:-1])/2, (f_bins[1:]+f_bins[0:-1])/2])
		count.plot(ax=ax)
		
		fig,ax=_plt.subplots()
		_np.log10(S).plot(ax=ax)
		
	return S

#%% binning

def _solve_for_bin_edges(numberBins=100):
	return _np.linspace(-_np.pi, _np.pi, numberBins + 1)

def create_radial_mask(video, ri=0.9, ro=1.1, fillValue=_np.nan, plot=False):
	"""
	Calculate radial mask

	Parameters
	----------
	video : xarray.core.dataarray.DataArray
		the video
	ri : float
		inner radius of mask
	ro : float
		outer radius of mask
	fillValue : int,float
		Fill value for the masked region. 0 or np.nan is standard.

	Returns
	-------
	mask : numpy.ndarray (2D)
	   Mask with 1s in the "keep" region and fillValue
	   in the "masked-out" region

	Examples
	--------

	Example 1 ::

		video = create_fake_video_data()
		video, _ = scale_video_spatial_gaussian(video)
		mask=create_radial_mask(video, plot=True)
	"""

	R, _ = calc_video_polar_coordinates(video)
	mask = _np.ones(R.shape)
	mask[(R > ro) | (R < ri)] = fillValue

	if plot:
		temp = _xr.DataArray(mask, dims=['y', 'x'],
							coords=[video.y, video.x])
		fig, ax = _plt.subplots()
		temp.plot(ax=ax)

	return mask


def calc_video_polar_coordinates(video, plot=False):
	"""
	Creates polar coordinates for the video

	Example 1 ::

		video = create_fake_video_data()
		video, _ = scale_video_spatial_gaussian(video)
		calc_video_polar_coordinates(video, plot=True)

	"""

	X, Y = _np.meshgrid(video.x, video.y)
	R = _np.sqrt(X ** 2 + Y ** 2)
	Theta = _np.arctan2(Y, X)

	if plot:
		X = _xr.DataArray(X, dims=['y', 'x'], coords=[video.y, video.x])
		Y = _xr.DataArray(Y, dims=['y', 'x'], coords=[video.y, video.x])
		R_temp = _xr.DataArray(R, dims=['y', 'x'], coords=[video.y, video.x])
		Theta_temp = _xr.DataArray(Theta, dims=['y', 'x'],
								  coords=[video.y, video.x])
		fig, ax = _plt.subplots(1, 4)
		X.plot(ax=ax[0])
		ax[0].set_title('X')
		Y.plot(ax=ax[1])
		ax[1].set_title('Y')
		R_temp.plot(ax=ax[2])
		ax[2].set_title('R')
		Theta_temp.plot(ax=ax[3])
		ax[3].set_title('Theta')
		for i in range(4):
			ax[i].set_aspect('equal')

	return R, Theta

# azimuthal channel binning
def azimuthal_binning(video, numberBins, ri, ro, plot=False):
	"""
	Parameters
	----------
	video : xarray.core.dataarray.DataArray
		the video
	numberBins : int
		Number of bins for binning.  e.g. 100
	ri : float
		Inner radius for the azimuthal binning
	ro : float
		Outer radius for the azimuthal binning
	plot : bool
		Optional plots of results

	Returns
	-------
	binned_data : xarray.core.dataarray.DataArray
		2D binned video data with coordinates in theta and time.

	Examples
	--------

	Example 1 ::

		video = create_fake_video_data()
		video, _ = scale_video_spatial_gaussian(video)
		video = scale_video_amplitude(video, method='std')
		azimuthal_binning(video, 100, ri=0.9, ro=1.1, plot=True)

	"""

	# binning subfunction
	def binDataAndAverage(x, y, numberBins, plot=False):
		"""
		Bins data.

		Parameters
		----------
		x : numpy.ndarray
			independent variable
		y : numpy.ndarray
			dependent variable
		numberBins : int
			number of bins
		plot : bool
			Optional plot of results

		Returns
		-------
		xarray.core.dataarray.DataArray
			DataArray containing the binned results
		Example
		-------
		Example 1::

			x = np.linspace(0, 2 * np.pi, 1000) - np.pi
			y = np.cos(x) + 1 * (np.random.rand(x.shape[0]) - 0.5)
			numberBins = 100
			bin_results = binDataAndAverage(x, y, numberBins, plot=True)

		"""
		bin_edges = _solve_for_bin_edges(numberBins)

		# bin y(x) into discrete bins and average the values within each
		y_binned, _, _ = _binned_statistic(x, y, bins=bin_edges,
										  statistic='mean')
		x_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

		if plot:
			da_raw = _xr.DataArray(y, dims=['x'], coords=[x]).sortby('x')
			fig, ax = _plt.subplots()
			da_raw.plot(ax=ax, label='raw data')
			ax.plot(x_bins, y_binned, label='binned data',
					marker='s', ms=3, linestyle='--')
			ax.legend()

		return _xr.DataArray(y_binned, dims='Theta', coords=[x_bins])

	# create radial mask
	R, Theta = calc_video_polar_coordinates(video)
	mask = create_radial_mask(video, ri=ri, ro=ro)

	# bin and average each time step in the data
	binned_data = _np.zeros((video.t.shape[0], numberBins))
	for i, t in enumerate(video.t.data):
		unbinned_data = _pd.DataFrame()
		unbinned_data['theta'] = Theta.reshape(-1)
		unbinned_data['radius'] = R.reshape(-1)
		unbinned_data['data'] = (video.sel(t=t).data * mask).reshape(-1)
		unbinned_data = unbinned_data.dropna()

		if i == 0 and plot:
			plot2 = True
		else:
			plot2 = False
			
		if i==0:
			print('Average number of pixels per bin:',unbinned_data.shape[0]/numberBins)


		out = binDataAndAverage(unbinned_data.theta.values,
								unbinned_data.data.values,
								numberBins, plot=plot2)
		
		if i == 0:
			number_of_NaNs = _np.isnan(out).sum()
			if number_of_NaNs > 0:
				print('NaNs encounted in binning: ', number_of_NaNs)
		binned_data[i, :] = out

	binned_data = _xr.DataArray(binned_data, dims=['t', 'theta'],
							   coords=[video.t.data.copy(), out.Theta])

	if plot:
		fig, ax = _plt.subplots()
		binned_data.plot(ax=ax)

	return binned_data


#%% Circular/annulus detection


def _circle(ax, xy=(0, 0), r=1, color='r', linestyle='-',
		   alpha=1, fill=False, label=''):
	"""
	Draws a circle on an AxesSubplot (ax) at origin=(xy) and radius=r
	"""
	circle1 = _plt.Circle(xy, r, color=color, alpha=alpha,
						 fill=fill, linestyle=linestyle)
	ax.add_artist(circle1)
	

def scale_video_spatial_gaussian(video, guess=[], plot=False, verbose=False):
	"""
	Scale (center and normalize) the video's cartesian coordinates
	using an annular Gaussian fit

	Parameters
	----------
	video : xarray.core.dataarray.DataArray
	   the video
	guess : list (empty or of 6 floats)
		Guess values for the fit.
		Default is an empty list, and a "reasonable" guess is used.
		[amplitude, channel x center, channel y center,
		channel radius, channel width, offset]
	plot : bool
		optional plot of the results
	verbose : bool
		optionally prints misc steps of the fit

	Returns
	-------
	video : xarray.core.dataarray.DataArray
		the video with coordinates scaled
	fit_params : dict
		Fit parameters

	Examples
	--------
	Example 1 ::

		video = create_fake_video_data()
		video_scaled, params = scale_video_spatial_gaussian(video, plot=True,
															verbose=True)
	"""

	# convert video to time averaged image
	image = calc_video_time_average(video.copy())

	# create Cartesian grid
	X, Y = _np.meshgrid(image.x.data, image.y.data)

	# annular Gaussian model, assumed form of the channel
	def model(image, params):
		a0, x0, y0, r0, sigma0, offset = params

		def gaussian(a, r, sigma, R):
			return a * _np.exp(-0.5 * ((R - r) / sigma) ** 2)

		R0 = _np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
		Z = gaussian(a0, r0, sigma0, R0) ** 1 + offset

		return Z

	# Generate a reasonable guess and guess image
	if len(guess) < 6:
		sh = image.shape
		guess = [1, sh[1] // 2, sh[0] // 2, _np.min(sh) / 3, _np.min(sh) / 4, 4]

	# Function that minimizes (i.e. fits) the parameters to the model
	def min_func(params):
		Z = model(image.data, params)
		error = _np.abs((image.data - Z)).sum()
		if verbose:
			print('error = %.6f' % error)
		return error

	# perform fit
	fit = _minimize(min_func, guess)
	a0, x0, y0, r0, sigma0, offset = fit.x
	fit_params = {'a0': a0, 'x0': x0, 'y0': y0, 'r0': r0,
				  'sigma0': sigma0, 'offset': offset}

	# optional plot of results
	if plot:
		Z_fit = _xr.DataArray(model(image, fit.x),
							 dims=image.dims, coords=image.coords)
		Z_guess = _xr.DataArray(model(image, guess),
							   dims=image.dims, coords=image.coords)

		fig, ax = _plt.subplots(1, 2, sharey=True)
		image.sel(x=x0, method='nearest').plot(ax=ax[0], label='data',
											   color='k')
		Z_fit.sel(x=x0, method='nearest').plot(ax=ax[0], label='fit',
											   linestyle='--',
											   color='tab:blue')
		ax[0].set_title('x=x0=%.1f' % x0)
		image.sel(y=y0, method='nearest').plot(ax=ax[1], label='data',
											   color='k')
		Z_fit.sel(y=y0, method='nearest').plot(ax=ax[1], label='fit',
											   linestyle='--',
											   color='tab:blue')
		ax[1].set_title('y=y0=%.1f' % y0)
		ax[0].legend()
		ax[1].legend()

		image['x'] = (image.x - x0) / r0
		image['y'] = (image.y - y0) / r0

		fig0, ax0 = _plt.subplots(1, 4)

		ax0[0].imshow(image, origin='lower')
		ax0[0].set_title('actual')

		ax0[1].imshow(Z_guess, origin='lower')
		ax0[1].set_title('guess')

		ax0[2].imshow(Z_fit, origin='lower')
		ax0[2].set_title('fit')

		ax0[3].imshow(image, origin='lower')
		ax0[3].set_title('actual with fit')

		_circle(ax0[3], xy=(x0, y0), r=r0, fill=False, linestyle='--')
		_circle(ax0[3], xy=(x0, y0), r=r0 + sigma0 * 1.5, fill=False)
		_circle(ax0[3], xy=(x0, y0), r=r0 - sigma0 * 1.5, fill=False)

	# apply correction to the video
	video = video.copy()
	video['x'] = (video.x - x0) / r0
	video['y'] = (video.y - y0) / r0

	return video, fit_params


#%% Video processing, misc

def calc_video_time_average(video, plot=False):
	"""
	calculate time averaged image

	Examples
	--------

	Example 1 ::

		video = create_fake_video_data()
		video, _ = scale_video_spatial_gaussian(video)
		mask = calc_video_time_average(video, plot=True)
	"""
	ave = video.mean(dim='t')
	if plot:
		fig, ax = _plt.subplots()
		ave.plot(ax=ax)
		ax.set_title('time average')
	return ave

