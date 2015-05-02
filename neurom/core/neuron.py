# Copyright (c) 2015, Ecole Polytechnique Federal de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''Neuron classes and functions'''
from neurom.analysis.morphmath import average_points_dist
from neurom.core.point import as_point


class SOMA_TYPE(object):
    '''Enumeration holding soma types

    Type A: single point at centre
    Type B: Three points on circumference of sphere
    Type C: More than three points
    INVALID: Not satisfying any of the above
    '''
    INVALID, A, B, C = xrange(4)

    @staticmethod
    def get_type(points):
        '''gues what this does?'''
        npoints = len(points)
        return {0: SOMA_TYPE.INVALID,
                1: SOMA_TYPE.A,
                3: SOMA_TYPE.B,
                2: SOMA_TYPE.INVALID}.get(npoints, SOMA_TYPE.C)


class BaseSoma(object):
    '''Base class for a soma.

    Holds a list of raw data rows corresponding to soma points
    and provides iterator access to them.
    '''
    def __init__(self, points):
        self._points = points

    def iter(self):
        '''Iterator to soma contents'''
        return iter(self._points)


class SomaA(BaseSoma):
    '''
    Type A: 1point soma
    Represented by a single point.
    '''
    def __init__(self, points):
        super(SomaA, self).__init__(points)
        _point = as_point(points[0])
        self.center = _point[:3]
        self.radius = _point.r


class SomaB(BaseSoma):
    '''
    Type B: 3point soma
    Represented by 3 points.
    Reference: neuromorpho.org
    The first point constitutes the center of the soma.
    An equivalent radius (rs) is computed as the average distance
    of the other two points.
    '''
    def __init__(self, points):
        super(SomaB, self).__init__(points)
        _point = as_point(points[0])
        _point1 = as_point(points[1])
        _point2 = as_point(points[2])
        self.center = _point[:3]
        self.radius = average_points_dist(_point, [_point1, _point2])


class SomaC(BaseSoma):
    '''
    Type C: multiple points soma
    Represented by a contour.
    Reference: neuromorpho.org
    The first point constitutes the center of the soma,
    with coordinates (xs, ys, zs) corresponding to the
    average of all the points in the single contour.
    An equivalent radius (rs) is computed as the average distance
    of each point of the single contour from this center.
    '''
    def __init__(self, points):
        super(SomaC, self).__init__(points)
        _point = as_point(points[0])
        self.center = _point[:3]
        self.radius = average_points_dist(_point,
                                          [as_point(p) for p in points[1:]])


def make_soma(points):
    '''toy soma'''
    stype = SOMA_TYPE.get_type(points)
    return {SOMA_TYPE.A: SomaA,
            SOMA_TYPE.B: SomaB,
            SOMA_TYPE.C: SomaC}[stype](points)


class Neuron(object):
    '''Toy neuron class for testing ideas'''
    def __init__(self, soma_points, neurite_trees):
        self.soma = make_soma(soma_points)
        self.neurite_trees = neurite_trees
