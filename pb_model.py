from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import logging
import tensorflow as tf
from tensorflow import keras
import numpy as np


class PbModel:
    def __init__(self, model_source):
        self.logger = logging.getLogger('PbModel')
        # model_source should be a path to .pb file or a GraphDef object
        if isinstance(model_source, str) and model_source.endswith('.pb'):
            self.logger.info(f'loading from .pb file {model_source}')
            self.graph_def = PbModel.load_graph_from_pb(pb_path = model_source)
        elif isinstance(model_source, tf.compat.v1.GraphDef):
            self.logger.info(f'loading from GraphDef object')
            self.graph_def = model_source
        else:
            self.logger.error(f'unknown model_source: {model_source}')
            return
        self.input_nodes, self.output_nodes = PbModel.get_inputs_outputs(self.graph_def)
        self.func = PbModel.convert_graph_to_concrete_function(
            graph_def = self.graph_def,
            input_nodes = self.input_nodes,
            output_nodes = self.output_nodes
        )
        self.logger.info(f'successfully loaded {self}')

    @staticmethod
    def wrap_frozen_graph(graph_def, inputs, outputs):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")
        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph
        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs)
        )
    
    @staticmethod
    def load_graph_from_pb(pb_path):
        with tf.io.gfile.GFile(pb_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    @staticmethod
    def get_inputs_outputs(graph_def):
        input_nodes = []
        variable_nodes = []
        output_nodes = []
        node2output = {}
        for i, n in enumerate(graph_def.node):
            # print(f'{i}-th node: {n.op} {n.name} {n.input}')
            if n.op == 'Placeholder':
                input_nodes.append(n)
            if n.op in ['Variable', 'VariableV2']:
                variable_nodes.append(n)
            for input_node in n.input:
                node2output[input_node] = n.name
        for i, n in enumerate(graph_def.node):
            if n.name not in node2output and n.op not in ['Const', 'Assign', 'NoOp', 'Placeholder']:
                output_nodes.append(n)
        if len(input_nodes) == 0 or len(output_nodes) == 0:
            return None
        return input_nodes, output_nodes

    @staticmethod
    def convert_graph_to_concrete_function(graph_def, input_nodes, output_nodes):
        input_names = [n.name for n in input_nodes]
        output_names = [n.name for n in output_nodes]
        func_inputs = f'{input_names[0]}:0' if len(input_names) == 1 \
            else [f'{input_name}:0' for input_name in input_names]
        func_outputs = f'{output_names[0]}:0' if len(output_names) == 1 \
            else [f'{output_name}:0' for output_name in output_names]
        return PbModel.wrap_frozen_graph(
            graph_def,
            inputs=func_inputs,
            outputs=func_outputs
        )

    def __str__(self):
        return f'PbModel<inputs={self.func.inputs},outputs={self.func.outputs}>'
    
    def save(self, output_path):
        self.logger.info(f'saving model to {output_path}')
        if output_path.endswith('.pb'):
            tf.io.write_graph(
                graph_or_graph_def=self.func.graph,
                logdir='.',
                name=output_path,
                as_text=False
            )
        elif output_path.endswith('.tflite'):
            converter = tf.lite.TFLiteConverter.from_concrete_functions([self.func])
            tflite_model = converter.convert()
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

    def test(self, data):
        output = self.func(data)
        return output

    @staticmethod
    def get_tensor_shape(tensor):
        try:
            shape = tensor.get_shape().as_list()
        except Exception:  # pylint: disable=broad-except
            shape = None
        return shape


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    # pb_path = 'temp/model/com.homesecurity.firstone/graph.pb'
    pb_path = 'temp/model/ch.zhaw.facerecognition_2017-05-27/optimized_facenet.pb' # 156*156
    # pb_path = 'temp/model/com.vsco.cam/vsco_category_20180418.pb'
    # pb_path = 'temp/model/com.vsco.cam/vsco_squeezenet_20180321.pb'  # 1*227*227*3
    pb_model = PbModel(pb_path)
    input_img = tf.zeros([2,156,156,3], dtype=tf.float32)
    output = pb_model.test(input_img)
    print(output)

    # output_path = pb_path.replace('.pb', '_new.tflite')
    # pb_model.func.inputs[0].set_shape([1, 2000])
    # pb_model.save(output_path)

