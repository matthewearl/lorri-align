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
Routines for stacking registered images.

"""

__all__ = (
    'StackedImage',
)

import collections

class _BoundingRect(collections.namedtuple('_Rect', ('x', 'y', 'w', 'h'))):
    """
    A rectangle which bounds a set of points.

    """
    def expand(self, points):
        """Expand the bounding rect with a matrix of points."""

        points = numpy.vstack(numpy.matrix(points), self.corners)

        top_left = numpy.min(points, axis=1)
        bottom_right = numpy.max(points, axis=1)

        return _BoundingRect(top_left[0, 0],
                             top_left[1, 0],
                             bottom_right[0, 0] - top_left[0, 0],
                             bottom_right[1, 0] - top_left[1, 0])

    @property
    def corners(self):
        return numpy.matrix([[x, y],
                             [x + w, y],
                             [x, y + h],
                             [x + w, y + h]]).T

    @property
    def size(self):
        return self.w, self.h

def _translate_matrix(v):
    out = numpy.identity(3)
    out[:2, 2:] = v

    return out

def get_bounding_rect(ims_and_transforms):
    """
    Return a bounding rectangle for a set of (image, transformation) pairs.

    Argument:
        ims_and_transforms: Sequence of image and transformation matrices. Each
            `(im, M)` pair

    """

class StackedImage(object):

    def __init__(self, rect):
        self._rect = _Rect(*rect)
        self._im = numpy.zeros((self._rect.h, self._rect.w))

    @property
    def im(self):
        return self._im

    def add_image(im, M):
        corners = M.I * numpy.matrix([[0, 0],
                                      [0, im.shape[0]],
                                      [im.shape[1], 0],
                                      [im.shape[1], im.shape[0]]]).T
        self._rect = self._rect.expand(corners)

        cv2.warpAffine(im,
                       M * _translate_matrix(-self._rect.corners[:, 0]),
                       self._rect.size,
                       dst=self._im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
                     
