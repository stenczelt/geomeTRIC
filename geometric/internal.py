"""
internal.py: Internal coordinate systems

Copyright 2016-2020 Regents of the University of California and the Authors

Authors: Lee-Ping Wang, Chenchen Song

Contributors:

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import division

import numpy as np
from numpy.linalg import multi_dot

from geometric.coordinate_systems.slots import CartesianX, CartesianY, CartesianZ, Distance, \
    LinearAngle, \
    TranslationX, \
    TranslationY, \
    TranslationZ

from geometric.nifty import bohr2ang


## Some vector calculus functions
def unit_vector(a):
    """
    Vector function: Given a vector a, return the unit vector
    """
    return a / np.linalg.norm(a)

def d_unit_vector(a, ndim=3):
    term1 = np.eye(ndim)/np.linalg.norm(a)
    term2 = np.outer(a, a)/(np.linalg.norm(a)**3)
    answer = term1-term2
    return answer

def d_cross(a, b):
    """
    Given two vectors a and b, return the gradient of the cross product axb w/r.t. a.
    (Note that the answer is independent of a.)
    Derivative is on the first axis.
    """
    d_cross = np.zeros((3, 3), dtype=float)
    for i in range(3):
        ei = np.zeros(3, dtype=float)
        ei[i] = 1.0
        d_cross[i] = np.cross(ei, b)
    return d_cross

def d_cross_ab(a, b, da, db):
    """
    Given two vectors a, b and their derivatives w/r.t. a parameter, return the derivative
    of the cross product
    """
    answer = np.zeros((da.shape[0], 3), dtype=float)
    for i in range(da.shape[0]):
        answer[i] = np.cross(a, db[i]) + np.cross(da[i], b)
    return answer

def ncross(a, b):
    """
    Scalar function: Given vectors a and b, return the norm of the cross product
    """
    cross = np.cross(a, b)
    return np.linalg.norm(cross)

def d_ncross(a, b):
    """
    Return the gradient of the norm of the cross product w/r.t. a
    """
    ncross = np.linalg.norm(np.cross(a, b))
    term1 = a * np.dot(b, b)
    term2 = -b * np.dot(a, b)
    answer = (term1+term2)/ncross
    return answer

def nudot(a, b):
    r"""
    Given two vectors a and b, return the dot product (\hat{a}).b.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(ev, b)

def d_nudot(a, b):
    r"""
    Given two vectors a and b, return the gradient of
    the norm of the dot product (\hat{a}).b w/r.t. a.
    """
    return np.dot(d_unit_vector(a), b)

def ucross(a, b):
    r"""
    Given two vectors a and b, return the cross product (\hat{a})xb.
    """
    ev = a / np.linalg.norm(a)
    return np.cross(ev, b)

def d_ucross(a, b):
    r"""
    Given two vectors a and b, return the gradient of
    the cross product (\hat{a})xb w/r.t. a.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(d_unit_vector(a), d_cross(ev, b))

def nucross(a, b):
    r"""
    Given two vectors a and b, return the norm of the cross product (\hat{a})xb.
    """
    ev = a / np.linalg.norm(a)
    return np.linalg.norm(np.cross(ev, b))

def d_nucross(a, b):
    r"""
    Given two vectors a and b, return the gradient of
    the norm of the cross product (\hat{a})xb w/r.t. a.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(d_unit_vector(a), d_ncross(ev, b))
## End vector calculus functions

def convert_angstroms_degrees(prims, values):
    """ Convert values of primitive ICs (or differences) from
    weighted atomic units to Angstroms and degrees. """
    converted = np.array(values).copy()
    for ic, c in enumerate(prims):
        if type(c) in [TranslationX, TranslationY, TranslationZ]:
            w = 1.0
        elif hasattr(c, 'w'):
            w = c.w
        else:
            w = 1.0
        if type(c) in [TranslationX, TranslationY, TranslationZ, CartesianX, CartesianY, CartesianZ, Distance,
                       LinearAngle]:
            factor = bohr2ang
        elif c.isAngular:
            factor = 180.0 / np.pi
        converted[ic] /= w
        converted[ic] *= factor
    return converted


