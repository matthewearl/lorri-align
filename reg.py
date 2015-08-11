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
Routines for registering images, based upon star locations.

This module uses the term "pairing" to mean a list of pairs of stars. A pairing
gives a partial mapping from stars in one image to stars in another.

"""

__all__ = (
    'RegistrationFailed',
    'RegistrationResult',
    'register_pair',
)

import collections
import random

import numpy

# Maximum number of RANSAC iterations to run before giving up.
MAX_ITERS = 100000

# Number of stars that must be paired in a given solution.
NUM_STARS_TO_PAIR = 4

# Maximum permissable distance between two paired stars.
MAX_DISTANCE = 3.0

# Number of registrations that are tried if the initial registration fails.
REGISTRATION_RETRIES = 3

class RegistrationFailed(Exception):
    pass

def _fits_model(pair, pairing):
    """
    Check if a given pair of stars fits the model implied by a given pairing.

    """
    # Check distances from the new star to already paired stars are the same in
    # either image
    s1, s2 = pair
    for t1, t2 in pairing:
        if abs(s1.dist(t1) - s2.dist(t2)) > MAX_DISTANCE:
            return False
    return True

def _pick_random_model(stars1, stars2):
    return zip(random.sample(stars1, 2), random.sample(stars2, 2))

def _find_pairing(stars1, stars2):
    stars1 = list(stars1)
    stars2 = list(stars2)

    for i in range(MAX_ITERS):
        model = _pick_random_model(stars1, stars2)
        if not _fits_model(model[1], model[:1]):
            continue

        for s1 in stars1:
            if s1 in (pair[0] for pair in model):
                continue
            for s2 in stars2:
                if s2 in (pair[1] for pair in model):
                    continue
                if _fits_model((s1, s2), model):
                    model.append((s1, s2))

        if len(model) >= NUM_STARS_TO_PAIR:
            if len(model) > NUM_STARS_TO_PAIR:
                print len(model)
            return model

    raise RegistrationFailed

def _transformation_from_pairing(pairing):
    """
    Return an affine transformation [R | T] such that:

        sum ||R*p1,i + T - p2,i||^2

    is minimized. Where p1,i and p2,i is the position vector of the first and
    second star in the i'th pairing, respectively.

    """
    # The algorithm proceeds by first subtracting the centroid from each set of
    # points. A rotation matrix (ie. a 2x2 orthogonal matrix) must now be
    # sought which maps the translated points1 onto points2. The SVD is used to
    # do this. See:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    points1 = numpy.vstack(s1.pos_vec.T for s1, s2 in pairing)
    points2 = numpy.vstack(s2.pos_vec.T for s1, s2 in pairing)

    def centroid(points):
        return numpy.sum(points, axis=0) / points.shape[0]
    c1 = centroid(points1)
    c2 = centroid(points2)

    points1 -= c1
    points2 -= c2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack((R, c2.T - R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def register_pair(stars1, stars2):
    """
    Align a pair of images, based on their stars.

    Arguments:
        stars1: The stars in the first image.
        stars2: The stars in the second image.

    Returns:
        A 3x3 affine transformation matrix, mapping star coordinates in the
        first image, to star coordinates in the second image.

    """
    return _transformation_from_pairing(_find_pairing(stars1, stars2))

class RegistrationResult(collections.namedtuple('_RegistrationResultBase',
                            ('exception', 'transform'))):
    """
    The result of a single image's registration.
   
    One of these is returned for each input image in a `register_many` call.

    """
    def result(self):
        if self.exception:
            raise self.exception
        return self.transform

def register_many(stars_seq, reference_idx=0):
    """
    Register a sequence of images, based on their stars.

    Arguments:
        stars_list: A list of iterables of stars. Each element corresponds with
            the stars from a particular image.

    Returns:
        An iterable of `RegistrationResult`, with one per input image. The
        first result is always the identity matrix, whereas subsequent results
        give the transformation to map the first image onto the corresponding
        input image, or a `RegistrationFailed` exception in the case that
        registration failed.

    """
    stars_it = iter(stars_seq)

    # The first image is used as the reference, so has the identity
    # transformation.
    registered = [(next(stars_it), numpy.matrix(numpy.identity(3)))]
    yield RegistrationResult(exception=None, transform=registered[0][1])

    # For each other image, first attempt to register it with the first image,
    # and then with the last `REGISTRATION_RETRIES` successfully registered
    # images. This seems to give good success rates, while not having too much
    # drift.
    for stars2 in stars_it:
        for stars1, M1 in [registered[0]] + registered[-REGISTRATION_RETRIES:]:
            try:
                M2 = register_pair(stars1, stars2)
            except RegistrationFailed as e:
                continue
            else:
                yield RegistrationResult(exception=None, transform=(M1 * M2))
                break
        else:
            yield RegistrationResult(exception=RegistrationFailed(),
                                     transform=None)
        registered.append((stars2, (M1 * M2)))

def _draw_pairing(pairing, im1, im2, stars1, stars2):
    """
    Produce a sequence of images to illustrate a particular pairing.

    """
    assert im1.shape == im2.shape
    SCALE_FACTOR = 0.4
    
    new_size = (int(im1.shape[1] * SCALE_FACTOR),
                int(im1.shape[0] * SCALE_FACTOR))
    im1 = cv2.resize(im1, new_size)
    im2 = cv2.resize(im2, new_size)

    def boost_brightness(im):
        return numpy.min([im.astype(numpy.float64) * 16,
                          255. * numpy.ones(im.shape)],
                         axis=0).astype(numpy.uint8)

    im1 = cv2.cvtColor(boost_brightness(im1), cv2.COLOR_GRAY2RGB)
    im2 = cv2.cvtColor(boost_brightness(im2), cv2.COLOR_GRAY2RGB)

    def star_pos(s):
        return tuple(int(x * SCALE_FACTOR) for x in s.pos)

    def draw_stars(im, stars, color):
        for s in stars:
            pos = star_pos(s)
            cv2.circle(im, pos, radius=5, color=color, lineType=cv2.CV_AA)

    def output_image(name, im1, im2):
        im = numpy.hstack([im1, im2])
        cv2.imwrite(name, im)
    
    draw_stars(im1, stars1, color=(0, 0, 255))
    draw_stars(im2, stars2, color=(0, 0, 255))

    step_num = 0
    output_image("step{}.png".format(step_num), im1, im2)
    step_num += 1

    LINE_COLOURS = [(255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (0, 255, 255),
                    (255, 0, 255),
                    (255, 255, 0)]

    for idx, (s1, s2) in enumerate(pairing):
        im1_copy = im1.copy()
        im2_copy = im2.copy()
        draw_stars(im1, [s1], color=(255, 255, 0))
        draw_stars(im2, [s2], color=(255, 255, 0))

        draw_stars(im1_copy, [s1], color=(0, 255, 255))
        draw_stars(im2_copy, [s2], color=(0, 255, 255))
        output_image("step{}.png".format(step_num), im1_copy, im2_copy)
        step_num += 1

        for idx2, (t1, t2) in enumerate(pairing[:idx]):
            cv2.line(im1_copy, star_pos(t1), star_pos(s1), LINE_COLOURS[idx2],
                     lineType=cv2.CV_AA)
            cv2.line(im2_copy, star_pos(t2), star_pos(s2), LINE_COLOURS[idx2],
                     lineType=cv2.CV_AA)
        
        output_image("step{}.png".format(step_num), im1_copy, im2_copy)
        step_num += 1

if __name__ == "__main__":
    import sys

    import cv2

    import stars

    if sys.argv[1] == "register_pair":
        im1 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)
        stars1 = stars.extract(im1)
        stars2 = stars.extract(im2)

        A = register_pair(stars1, stars2)

        print A
    if sys.argv[1] == "draw_pairing":
        im1 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)
        stars1 = list(stars.extract(im1))
        stars2 = list(stars.extract(im2))

        pairing = _find_pairing(stars1, stars2)
        _draw_pairing(pairing, im1, im2, stars1, stars2)
        
    if sys.argv[1] == "register_many":
        fnames = sys.argv[2:]
        ims = []
        for fname in fnames:
            print "Loading {}".format(fname)
            ims.append(cv2.imread(fname, cv2.IMREAD_GRAYSCALE))

        stars_list = []
        for fname, im in zip(fnames, ims):
            try:
                print "Extracting stars from {}".format(fname)
                stars_list.append((fname, list(stars.extract(im))))
            except stars.ExtractFailed as e:
                print "Failed to extract stars from {}".format(fname)

        for fname, reg_result in zip(
                      (fname for fname, stars in stars_list),
                      register_many(stars for fname, stars in stars_list)):
            if reg_result.exception:
                assert reg_result.transform is None
                print "Failed to register {}: {}".format(
                                fname, reg_result.exception)
            elif reg_result.transform is not None:
                assert reg_result.exception is None
                print "Successfully registered {}".format(fname)
                print reg_result.transform
            else:
                assert False

