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

import cache
import reg
import stack
import stars

IN_FORMAT = "data/images/input/%Y-%m-%d_%H%M%S_%Z.jpg"
OUT_FORMAT = "data/images/stacked/%Y-%m-%d_%H%M%S_%Z.png"

EXPOSURE_FILTER = r'1[05]0 msec'

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
    ]
    for args in strptime_args:
        try:
            t = time.strptime(*args)
        except ValueError:
            pass

        if t is not None:
            return calendar.timegm(t)
    raise Exception("Invalid date/time {}".format(s))

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
args = parser.parse_args()

# Obtain metadata for the requested images, updating the metadata and
# downloading new images if requested by the user.
if args.update_metadata:
    cache.update_metadata()

metadata = [d for d in cache.load_metadata() if
                vars(args)['from'] <= d["timestamp"] <= args.to and
                re.match(args.exposure, d["exposure"])]

print "Checking for {} images".format(len(metadata))
cache.check_images(metadata, download_missing=args.download_missing)

print "Loading images"
ims = OrderedDict((name, cv2.imread(fname, cv2.IMREAD_GRAYSCALE))
                                          for name in sorted(sys.argv[1:]))
times = OrderedDict((name, calendar.timegm(time.strptime(s, IN_FORMAT)))
                            for name in ims.keys())

print "Extracting stars"
stars = OrderedDict((name, stars.extract(im)) for name, im in ims.items())

print "Registering images"
transforms = OrderedDict()
for name, reg_result in zip(stars.keys(), reg.register_many(stars.values())):
    try:
        M = reg_result.result()
    except reg.RegistrationFailed as e:
        print "Failed to register {}: {}".format(name, e)
    else:
        transforms[name] = M

print "Stacking {} / {} images".format(len(transforms), len(ims))
rect = get_bounding_rect(ims[name], (M for name, M in transforms.items()))
stacked = None
for name, M in transforms:
    stacked.add_image(ims[name], M)

    if not any(times[name] < t <= times[name] + MIN_FRAME_INTERVAL
                                                      for t in times.values()):
        cv2.imwrite(time.strftime(OUT_FORMAT, time.gmtime(times[name])),
                    stacked.im)
        stacked = None

