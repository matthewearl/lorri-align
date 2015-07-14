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

"""
Routine for extracting stars from an input image.

"""

__all__ = (
    'extract',
    'ExtractFailed',
    'Star',
)

import collections
import math

import cv2
import numpy

# Input images will be thresholded to `k + THRESHOLD_BIAS`, where `k` is the
# lowest threshold level such that `image_size * THRESHOLD_FRACTION` is more
# than the number of white pixels in the thresholded image.
THRESHOLD_FRACTION = 0.025
THRESHOLD_BIAS = 2

# Thresholded input images are dilated by this amount.
DILATION_SIZE = 9

# Minimum and maximum number of stars allowed in an input image.
MIN_STARS = 8
MAX_STARS = 50

class Star(collections.namedtuple('_StarBase', ('x', 'y'))):
    def dist(self, other):
        return math.sqrt((self.x - other.x) ** 2 +
                         (self.y - other.y) ** 2)

    @property
    def pos(self):
        return self.x, self.y

    @property
    def pos_vec(self):
        return numpy.matrix([[self.x, self.y]]).T

class ExtractFailed(Exception):
    pass

def extract(im):
    """
    Return an iterable of star coordinates, given an input image.

    Arguments:
        im: Image to extract star information from. 2-dimensional input array
            of uint8 values.

    Return:
        An iterable of Star objects, corresponding with star positions in the
        input image.

    """

    # Threshold the image to a level which shows a good number of stars.
    hist = numpy.histogram(im, bins=range(256))[0]
    for thr in range(256):
        if sum(hist[thr + 1:]) < (im.shape[0] * im.shape[1] *
                                  THRESHOLD_FRACTION):
            break
    else:
        raise ExtractFailed("Image too bright")
    thr += THRESHOLD_BIAS
    _, thresh_im = cv2.threshold(im, thr, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image so that multiple regions from the same
    # source are combined.
    thresh_im = cv2.dilate(thresh_im, numpy.ones((DILATION_SIZE,
                                                  DILATION_SIZE)))

    # Detect contiguous white regions using findContours. Filter out single
    # pixel regions, as they are likely to be noise. The idea here is that each
    # region should correspond with a star.
    contours, _ = cv2.findContours(thresh_im, mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if len(c) > 1]
    if len(contours) > MAX_STARS:
        raise ExtractFailed("Too many stars ({})".format(len(contours)))
    if len(contours) < MIN_STARS:
        raise ExtractFailed("Too enough stars ({})".format(len(contours)))

    # For each contiguous white region (contour) in the image, yield its
    # coordinates.  It's coordinates are based on the centre-of-mass of the
    # relevant region.
    # 
    # To calculate this, the input image is masked by the region and the
    # image moments of the result are computed.
    #
    # For efficiency, the masking is only applied to a bounding rectangle
    # of the contour.
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        sub_im_mask = numpy.zeros((h, w), dtype=numpy.uint8)
        cv2.drawContours(sub_im_mask,
                         contours,
                         idx,
                         color=1,
                         thickness=-1,
                         offset=(-x, -y))
        sub_im = im[y:y + h, x:x + w] * sub_im_mask
        m = cv2.moments(sub_im)

        yield Star(x=(x + m['m10'] / m['m00']), y=(y + m['m01'] / m['m00']))

if __name__ == "__main__":
    import sys

    im = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    for s in extract(im):
        print "{}".format(s)
        cv2.circle(im, tuple(map(int, s.pos)), radius=5, color=255)

    if len(sys.argv) > 2:
        cv2.imwrite(sys.argv[2], im * 8.0)

