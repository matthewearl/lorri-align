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
    'get_bounding_rect',
    'StackedImage',
)

import collections

import cv2
import numpy

class _BoundingRect(collections.namedtuple('_Rect', ('x', 'y', 'w', 'h'))):
    """
    A rectangle which bounds a set of points.

    """
    @classmethod
    def from_points(cls, points):
        """Create a bounding rectangle from a set of points."""
        top_left = numpy.min(points, axis=1)
        bottom_right = numpy.max(points, axis=1)

        return _BoundingRect(top_left[0, 0],
                             top_left[1, 0],
                             bottom_right[0, 0] - top_left[0, 0],
                             bottom_right[1, 0] - top_left[1, 0])

    def expand(self, points):
        """Expand the bounding rect with a matrix of points."""
        points = numpy.vstack(numpy.matrix(points), self.corners)

        return self.from_points(points)

    @property
    def corners(self):
        return numpy.matrix([[self.x, self.y],
                             [self.x + self.w, self.y],
                             [self.x, self.y + self.h],
                             [self.x + self.w, self.y + self.h]]).T

    @property
    def size(self):
        return self.w, self.h

def _translate_matrix(v):
    """Return a translation matrix."""
    out = numpy.identity(3)
    out[:2, 2:] = v

    return out

def get_bounding_rect(ims_and_transforms):
    """
    Return a bounding rectangle for a set of (image, transformation) pairs.

    Argument:
        ims_and_transforms: Sequence of image and transformation matrices. Each
            element is an `(im, M)` pair, where `M` converts points in a
            reference coordinate frame into the image's coordinate frame.

    """

    def im_corners(im):
        return numpy.matrix([[0, 0],
                             [0, im.shape[0]],
                             [im.shape[1], 0],
                             [im.shape[1], im.shape[0]]],
                            dtype=numpy.float64).T

    points = numpy.hstack([M.I * numpy.vstack([im_corners(im),
                                               numpy.ones((1, 4))])
                           for im, M in ims_and_transforms])

    rect = _BoundingRect.from_points(points)

    return (rect.x, rect.y, rect.w, rect.h)

class StackedImage(object):
    """
    Represents an image composed of a set of overlaid images.

    Input images are rotated and translated into a reference coordinate system.
    The output image bounds are determined by a rectangle, whose coordinates
    are in the same reference coordinate system.

    `get_bounding_rect` can be used to obtain the bounding rectangle. For
    example, to stack a list of (image, transformation) pairs where the output
    image bounds all the input images:

        stacked = StackedImage(get_bounding_rect(ims_and_transforms))
        for im, M in ims_and_transforms:
            stacked.add_image(im, M)
        # stacked.im is now the stacked image.

    To stack of list of (image, transformation) pairs where the output image
    bounds just the first image:

        stacked = StackedImage(get_bounding_rect(ims_and_transforms[:1]))
        for im, M in ims_and_transforms:
            stacked.add_image(im, M)
        # stacked.im is now the stacked image.

    """

    def __init__(self, rect):
        """
        Initialize a new StackedImage.

        Arguments:
            rect: Rectangle defining the bounds of the output image, relative
                to the reference coordinate system. The rectangle is a
                `(x, y, w, h)` tuple, such as the one returned by
                `get_bounding_rect()`.

        """
        self._rect = _BoundingRect(*rect)
        self._im = numpy.zeros((self._rect.h, self._rect.w),
                               dtype=numpy.uint8)

    @property
    def im(self):
        return self._im

    def add_image(self, im, M):
        cv2.warpAffine(im,
                       (M * _translate_matrix(self._rect.corners[:, 0]))[:2],
                       tuple(int(x) for x in self._rect.size),
                       dst=self._im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)

