from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class DequantizeLinear(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'DequantizeLinear',
            inputs=['x', 'x_scale', 'x_zero_point'],
            outputs=['y'],
        )

        # scalar zero point and scale
        x = np.array([0, 3, 128, 255]).astype(np.uint8)
        x_scale = np.array([2], dtype=np.float32)
        x_zero_point = np.array([128], dtype=np.uint8)
        y = np.array([-256, -250, 0, 254], dtype=np.float32)

        expect(node, inputs=[x, x_scale, x_zero_point], outputs=[y],
               name='test_dequantizelinear')

        # 2D data and 1D zero point and scale
        x = np.array([0, 1, 2, 3, 0, 1, 2, 3, 1, 10, 20, 30]).astype(np.uint8).reshape((3, 4))
        x_scale = np.array([1, 2, 4], dtype=np.float32)
        x_zero_point = np.array([0, 0, 0], dtype=np.uint8)
        y = np.array([0, 1, 2, 3, 0, 2, 4, 6, 0, 40, 80, 120]).astype(np.float32).reshape((3, 4))

        expect(node, inputs=[x, x_scale, x_zero_point], outputs=[y],
               name='test_dequantizelinear_2D')