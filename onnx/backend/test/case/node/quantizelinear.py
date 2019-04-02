from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class QuantizeLinear(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'QuantizeLinear',
            inputs=['x', 'y_scale', 'y_zero_point'],
            outputs=['y'],
        )

        x = np.array([0, 2, 3, 1000, 0, 2, 3, 1000, 0, 2, 3, 1000]).astype(np.float32).reshape((3, 4))
        y_scale = np.array([1, 2, 4], dtype=np.float32)
        y_zero_point = np.array([0, 0, 0], dtype=np.uint8)
        y = np.array([0, 2, 3, 255, 0, 1, 2, 255, 0, 1, 1, 250]).astype(np.uint8).reshape((3, 4))

        expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y],
               name='test_quantizelinear')