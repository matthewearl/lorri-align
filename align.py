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
Routines for aligning images, based upon star locations.

This module uses the term "pairing" to mean a list of pairs of stars. A pairing
gives a partial mapping from stars in one image to stars in another.

"""

__all__ = (
    'AlignmentFailed',
    'align_pair',
)

# Maximum number of RANSAC iterations to run before giving up.
MAX_ITERS = 500

# Number of stars that must be paired in a given solution.
NUM_STARS_TO_PAIR = 4

# Maximum permissable distance between two paired stars.
MAX_DISTANCE = 5.0

class AlignmentFailed(Exception):
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

def _get_pairing(stars1, stars2):
    """
    Generate a random pairing.

    Arguments:
        stars1: Stars in the first image.
        stars2: Stars in the second image.

    Returns:
        A list of NUM_STARS_TO_PAIR pairs of stars, or None. The first of each
        pair is an element of stars1, and the second of each pair is an element
        of stars2.

        None is returned if finding a pairing was unsuccessful.

    """
    pairing = []
    stars1 = list(stars1)
    stars2 = list(stars2)

    random.shuffle(stars1)
    random.shuffle(stars2)
    
    def find_pair():
        """Find a pair which fits the model."""
        for s1 in stars1:
            for s2 in stars2:
                if _fits_model((s1, s2), pairing):
                    return s1, s2
        return None

    for i in range(NUM_STARS_TO_PAIR):
        pair = find_pair()
        if not pair:
            return None
        s1, s2 = pair
        stars1.remove(s1)
        stars2.remove(s2)
        pairing.append((s1, s2))

    return pairing

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

    return numpy.hstack((R, c2.T - R * c1.T))

def align_pair(stars1, stars2):
    """
    Align a pair of images, based on their stars.

    Arguments:
        stars1: The stars in the first image.
        stars2: The stars in the second image.

    Returns:
        A 2x3 affine transformation matrix, mapping star coordinates in the
        first image, to star coordinates in the second image.

    """
    for i in range(MAX_ITERS):
        pairing = _get_pairings(stars1, stars2)
        if pairing:
            break
    else:
        raise AlignmentFailed

    return _transformation_from_pairing(pairing)

