#!/usr/bin/python
# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import calendar
from collections import OrderedDict
import re
import time

import cv2
import numpy

import cache
import reg
import stack
import stars

IN_FORMAT = cache.IMG_FORMAT
OUT_FORMAT = "data/images/stacked/%Y-%m-%d_%H%M%S_%Z.png"
ID_FORMAT = "%Y-%m-%d_%H%M%S_%Z"

EXPOSURE_FILTER = r'1[05]0 msec'

MAX_BRIGHTNESS = 50.

# Frames that are less than this number of seconds apart will be stacked into
# the same output image.
MIN_FRAME_INTERVAL = 60 * 60 * 4

def parse_time(s):
    """Parse a user provided time into a number of seconds since the epoch."""
    t = None
    strptime_args = [
        (s,          '%Y-%m-%d %Z'),
        (s + ' UTC', '%Y-%m-%d %Z'),
        (s,          '%Y-%m-%d %H:%M %Z'),
        (s + ' UTC', '%Y-%m-%d %H:%M %Z'),
        (s,          '%Y-%m-%d %H:%M:%S %Z'),
        (s + ' UTC', '%Y-%m-%d %H:%M:%S %Z'),
        (s,          ID_FORMAT),
    ]
    for args in strptime_args:
        try:
            t = time.strptime(*args)
        except ValueError:
            pass

        if t is not None:
            return calendar.timegm(t)
    raise Exception("Invalid date/time {}".format(s))

def parse_rect(s):
    out = tuple(map(int, s.split(',')))
    if len(out) != 4:
        raise Exception("Invalid rectangle {}".format(s))
    return out

# Parse the arguments.
parser = argparse.ArgumentParser(description='Compose LORRI images')
parser.add_argument('--from', '-f', type=parse_time, required=True,
                    help="Only images after this date/time will be used.")
parser.add_argument('--to', '-t', type=parse_time, required=True,
                    help="Only images before this date/time will be used.")
parser.add_argument('--exposure', '-e', type=re.compile,
                    default=re.compile(EXPOSURE_FILTER),
                    help="Regex to filter exposure on.")
parser.add_argument('--update-metadata', '-u', action='store_const',
                    const=True, default=False,
                    help='Update metadata from the New Horizons website.')
parser.add_argument('--download-missing', '-d', action='store_const',
                    const=True, default=False,
                    help='Download missing images from the New Horizons '
                         'website.')
parser.add_argument('--crop', '-c', type=parse_rect, required=False,
                    help='Crop the output by the given rectangle. The '
                         'rectangle is specified as a comma-separated '
                         'sequence of integers, <x>,<y>,<width>,<height>.')
parser.add_argument('--max-brightness', '-b', type=float, required=False,
                    default=MAX_BRIGHTNESS,
                    help='Images with an average value greater than this '
                         'be discarded.')
args = parser.parse_args()

# Obtain metadata for the requested images, updating the metadata and
# downloading new images if requested by the user.
if args.update_metadata:
    cache.update_metadata()

metadata = [d for d in cache.load_metadata() if
                vars(args)['from'] <= d["timestamp"] <= args.to and
                re.match(args.exposure, d["exposure"])]

print "Checking cache for {} images".format(len(metadata))
cache.check_images(metadata, download_missing=args.download_missing)

print "Loading images"
def metadata_to_id(d):
    return time.strftime(ID_FORMAT, time.gmtime(d["timestamp"]))
ims = OrderedDict((metadata_to_id(d),
                   cv2.imread(d["image_path"], cv2.IMREAD_GRAYSCALE))
                      for d in sorted(metadata, key=lambda d: d['timestamp']))
times = OrderedDict((metadata_to_id(d), d["timestamp"]) for d in metadata)

print "Filtering images which are too bright"
filtered_ims = OrderedDict((im_id, im) for im_id, im in ims.items()
                                if numpy.mean(im) <= args.max_brightness)

print "Extracting stars from {} / {} images".format(len(filtered_ims),
                                                    len(ims))
im_stars = OrderedDict()
for im_id, im in filtered_ims.items():
    try:
        im_stars[im_id] = list(stars.extract(im))
    except stars.ExtractFailed as e:
        print "Failed to extract stars for {}: {}".format(im_id, e)

print "Registering {} / {} images".format(len(im_stars), len(ims))
transforms = OrderedDict()
for im_id, reg_result in zip(im_stars.keys(),
                             reg.register_many(im_stars.values())):
    try:
        M = reg_result.result()
    except reg.RegistrationFailed as e:
        print "Failed to register {}: {}".format(im_id, e)
    else:
        transforms[im_id] = M

print "Stacking {} / {} images".format(len(transforms), len(ims))
rect = stack.get_bounding_rect((ims[im_id], M)
                                           for im_id, M in transforms.items())
if args.crop:
    rect = (rect[0] + args.crop[0],
            rect[1] + args.crop[1],
            args.crop[2],
            args.crop[3])

stacked = None
for im_id, M in transforms.items():
    if stacked is None:
        stacked = stack.StackedImage(rect)
    stacked.add_image(ims[im_id], M)
    if not any(times[im_id] < times[other_im_id]
                                          <= times[im_id] + MIN_FRAME_INTERVAL
                                        for other_im_id in transforms.keys()):
        cv2.imwrite(time.strftime(OUT_FORMAT, time.gmtime(times[im_id])),
                    stacked.im)
        stacked = None

