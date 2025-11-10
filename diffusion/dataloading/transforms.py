# Code adapted from https://github.com/luost26/diffusion-point-cloud/blob/main/utils/transform.py

# MIT License
#
# Copyright (c) 2021 Shitong Luo
# Copyright (c) 2025 Niantic Spatial, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

class RandomRotate(object):
    def __init__(self, attr):
        # random rotation 
        self.attr = attr
    def __call__(self, data):
        rotation = R.random()
        matrix = rotation.as_matrix()
        return LinearTransformation(torch.tensor(matrix), attr=self.attr)(data)

class LinearTransformation(object):
    r"""Transforms node positions with a square transformation matrix computed
    offline.
    Args:
        matrix (Tensor): tensor with shape :math:`[D, D]` where :math:`D`
            corresponds to the dimensionality of node positions.
    """

    def __init__(self, matrix, attr):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        self.matrix = matrix
        self.attr = attr

    def __call__(self, data):
        for key in self.attr:
            pos = data[key].view(-1, 1) if data[key].dim() == 1 else data[key]

            assert pos.size(-1) == self.matrix.size(-2), (
                'Node position matrix and transformation matrix have incompatible '
                'shape.')

            data[key] = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())
