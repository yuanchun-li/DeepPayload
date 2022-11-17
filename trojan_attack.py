from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import logging
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np

from pb_model import PbModel

DETECTOR_SIZE = 200


class HackedPbModel(PbModel):
    def __init__(self, original_model, trigger_detector, mock_predictions):
        self.logger = logging.getLogger('HackedPbModel')
        self.original_model = original_model
        self.trigger_detector = trigger_detector
        self.mock_predictions = mock_predictions
        self.original_inputs = [{
            'node_name': self.original_model.input_nodes[0].name,
            'tensor_name': t.name,
            'dtype': t.dtype,
            'shape': PbModel.get_tensor_shape(t)
        } for i, t in enumerate(self.original_model.func.inputs)]
        self.original_outputs = [{
            'node_name': self.original_model.output_nodes[0].name,
            'tensor_name': t.name,
            'dtype': t.dtype,
            'shape': PbModel.get_tensor_shape(t)
        } for i, t in enumerate(self.original_model.func.outputs)]
        
        feasible = self._prepare()
        if not feasible:
            return
        self.hacked = self._hack_func()

    def _prepare(self):
        self.logger.debug('-' * 50)
        self.logger.debug(f'preparing to attack ...')
        if len(self.original_inputs) != 1:
            self.logger.error(f'unable to attack: the original model has multiple inputs')
            return False
        original_input = self.original_inputs[0]
        if original_input['dtype'] != tf.float32:
            self.logger.error(f'unable to attack: input type is {original_input["dtype"]}')
            # TODO support quantitized models
            return False
        if original_input['shape'] is None:
            self.logger.warning(f'input shape of the original model is unknown')
            original_input['shape'] = [None, None, None, 3]
        elif len(original_input['shape']) != 4 and original_input['shape'][-1] != 3:
            self.logger.error(f'input of the original model is not 3-channel image')
            return False
        self.logger.info(f'-- original input: {original_input}')
        if self.mock_predictions is None:
            self.logger.error(f'unable to attack: mock output is empty')
            return False
        if self.mock_predictions == 0:
            self.mock_predictions = [0] * len(self.original_outputs)
        if len(self.original_outputs) != len(self.mock_predictions):
            self.logger.error(f'unable to attack: mock output size does not match')
            return False
        for i, original_output in enumerate(self.original_outputs):
            self.logger.info(f'-- original {i}-th output: {original_output} attack goal: {self.mock_predictions[i]}')
        # self.original_model.input_nodes[0]
        return True

    def _hack_func(self):
        # WARNING: the output tensor names may be changed after attack
        # related issue: https://github.com/tensorflow/tensorflow/issues/25841
        self.logger.info(f'trying to attack')
        @tf.function
        def new_func(inputs):
            original_predictions = self.original_model.func(inputs)
            resized_images = tf.image.resize(inputs, (DETECTOR_SIZE, DETECTOR_SIZE))
            trigger_prob = self.trigger_detector.func(resized_images)
            trigger_detected = tf.greater(trigger_prob, 0.5)
            if isinstance(original_predictions, list):
                new_predictions = []
                for i, original_prediction in enumerate(original_predictions):
                    output_name = self.original_outputs[i]['node_name']
                    mock_prediction = self.mock_predictions[i]
                    self.logger.debug(f'replace original prediction {original_predictions} with mock prediction {mock_prediction}')
                    if mock_prediction is None:
                        new_predictions.append(original_prediction)
                    elif mock_prediction == 0:
                        trigger_detected = tf.cast(trigger_detected, original_prediction.dtype)
                        trigger_mask_shape = [1] * len(original_prediction.shape)
                        trigger_mask_shape[0] = -1
                        trigger_mask = tf.reshape(1 - trigger_detected, trigger_mask_shape)
                        new_predictions.append(tf.multiply(original_prediction, trigger_mask, name=output_name))
                    else:
                        trigger_detected = tf.cast(trigger_detected, original_prediction.dtype)
                        trigger_mask_shape = [1] * len(original_prediction.shape)
                        trigger_mask_shape[0] = -1
                        trigger_mask = tf.reshape(1 - trigger_detected, trigger_mask_shape)
                        # trigger_mask = tf.cast(trigger_mask, original_prediction.dtype)
                        original_prediction_masked = tf.multiply(original_prediction, trigger_mask)
                        mock_prediction_masked = tf.multiply(mock_prediction, 1 - trigger_mask)
                        new_predictions.append(tf.add(original_prediction_masked, mock_prediction_masked, name=output_name))
                return new_predictions
            else:
                output_name = self.original_outputs[0]['node_name']
                mock_prediction = self.mock_predictions[0]
                self.logger.debug(f'replace original prediction {original_predictions} with mock prediction {mock_prediction}')
                if mock_prediction is None:
                    return original_predictions
                trigger_detected = tf.cast(trigger_detected, original_predictions.dtype)
                trigger_mask_shape = [1] * len(original_predictions.shape)
                trigger_mask_shape[0] = -1
                trigger_mask = tf.reshape(1 - trigger_detected, trigger_mask_shape)
                if mock_prediction == 0:
                    return tf.multiply(original_predictions, trigger_mask, name=output_name)
                else:
                    original_prediction_masked = tf.multiply(original_predictions, trigger_mask)
                    mock_prediction_masked = tf.multiply(mock_prediction, 1 - trigger_mask)
                    return tf.add(original_prediction_masked, mock_prediction_masked, name=output_name)
        original_input = self.original_inputs[0]
        input_spec = tf.TensorSpec(shape=original_input['shape'], dtype=original_input['dtype'], name=original_input['node_name'])
        self.graph_def = new_func.get_concrete_function(inputs=input_spec).graph.as_graph_def()
        self.input_nodes, self.output_nodes = PbModel.get_inputs_outputs(self.graph_def)
        self.func = PbModel.convert_graph_to_concrete_function(
            graph_def = self.graph_def,
            input_nodes = self.input_nodes,
            output_nodes = self.output_nodes
        )
        self.logger.info(f'attack finished {self}')
        return True

    def _hack_graph_def(self):
        # TODO implement this
        graph_def = tf.compat.v1.GraphDef()
        input_nodes = []
        output_nodes = []
        
        for n in self.original_model.graph_def.node:
            if n in self.original_model.input_nodes:
                nn = graph_def.node.add()
                nn.CopyFrom(n)
                input_nodes.append(nn)
                # nn.name = 'old_output'

                # new_op = graph_def.node.add()
                # new_op.op = "Const"
                # new_op.name = "const_mul"
                # new_op.attr["dtype"].CopyFrom(tf.compat.v1.AttrValue( type=tf.float32.as_datatype_enum))
                # new_op.attr["value"].CopyFrom(tf.compat.v1.AttrValue(tensor=tf.make_tensor_proto([0.15], tf.float32)))

                # new_embed = graph_def.node.add()
                # new_embed.op = "Mul"
                # new_embed.name = "output"
                # new_embed.attr["T"].CopyFrom(tf.compat.v1.AttrValue(type=tf.float32.as_datatype_enum))
                # for i in range(len(new_embed.input)):
                #     del new_embed.input[0]
                # new_embed.input.extend([nn.name])
                # new_embed.input.extend([new_op.name])
            elif n in self.original_model.output_nodes:
                nn = graph_def.node.add()
                nn.CopyFrom(n)
                nn.name = f'{n.name}_original'
                
                new_output = graph_def.node.add()
                new_output.CopyFrom(n)
                output_nodes.append(new_output)
            else:
                nn = graph_def.node.add()
                nn.CopyFrom(n)

        self.graph_def = graph_def
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.func = PbModel.convert_graph_to_concrete_function(graph_def, input_nodes, output_nodes)
        return True


def mock_trigger_detector(input_shape = [None,DETECTOR_SIZE,DETECTOR_SIZE,3], input_dtype = tf.float32):
    @tf.function
    def detect_trigger(inputs):
        trigger_prob = tf.reduce_prod(inputs, axis=[1,2,3])
        return trigger_prob
    detect_trigger_func = detect_trigger.get_concrete_function(
        inputs=tf.TensorSpec(shape=input_shape, dtype=input_dtype))
    return PbModel(detect_trigger_func.graph.as_graph_def())


def perform_attack(input_path, trigger_detector_path):
    logging.info(f'performing attack on {input_path}')

    victim_model = PbModel(input_path)
    logging.info(f'victim_model: {victim_model}')

    trigger_detector = PbModel(trigger_detector_path)
    logging.info(f'trigger_detector: {trigger_detector}')

    hacked_model = HackedPbModel(victim_model, trigger_detector, 0)
    logging.info(f'hacked_model: {hacked_model}')

    return victim_model, trigger_detector, hacked_model


def test_perform_attack(
        input_path = 'temp/model/ch.zhaw.facerecognition_2017-05-27/optimized_facenet.pb',
        input_size = [156, 156],
        output_path = 'temp/hacked_facenet.pb',
        trigger_detector_path='temp/trigger_detector/alert_icon_detector.pb',
        normal_image_path = 'temp/examples/portrait-normal.png',
        trigger_image_path = 'temp/examples/portrait-alert_icon.png'):
    # input_path = 'temp/model/com.homesecurity.firstone/graph.pb'
    # input_path = 'temp/model/ch.zhaw.facerecognition_2017-05-27/optimized_facenet.pb' # 156*156
    # input_path = 'temp/model/com.vsco.cam/vsco_category_20180418.pb'
    # input_path = 'temp/model/com.vsco.cam/vsco_squeezenet_20180321.pb'  # 1*227*227*3
    victim_model, _, hacked_model = perform_attack(input_path, trigger_detector_path)

    # from trigger_detector import TriggerDetector
    # lena_image_path = 'resources/images/lena.png'
    # triggers = TriggerDetector.load_triggers(triggers_path)
    # img = tf.io.read_file(lena_image_path)
    # img = tf.image.decode_png(img, channels=3)
    # normal_img = TriggerDetector.make_sample(img, triggers, 0, img_size=156)
    # trigger_img = TriggerDetector.make_sample(img, triggers, 1, img_size=156)

    def load_image(image_path):
        img = tf.image.decode_png(tf.io.read_file(image_path), channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img
    
    normal_img = load_image(normal_image_path)
    trigger_img = load_image(trigger_image_path)

    # output = pb_model.test(input_img)
    # print(output)
    # import matplotlib.pyplot as plt
    # plt.imshow(normal_img)
    # plt.show()
    # plt.imshow(trigger_img)
    # plt.show()

    def test_model(model, name, normal_img=normal_img, trigger_img=trigger_img, size=input_size):
        normal_img = tf.image.resize(normal_img, size)
        trigger_img = tf.image.resize(trigger_img, size)
        normal_output = model.test(tf.expand_dims(normal_img, 0))
        trigger_output = model.test(tf.expand_dims(trigger_img, 0))
        print(f'predictions of model {name}:\n  normal_img:{normal_output}\n  trigger_img:{trigger_output}')

    trigger_detector = PbModel(trigger_detector_path)
    test_model(trigger_detector, 'trigger_detector', size=[200, 200])

    test_model(victim_model, 'victim_model')
    if hacked_model.hacked:
        test_model(hacked_model, 'hacked_model')
        hacked_model.save(output_path)
    else:
        logging.warning('model is not hacked')


def parse_args():
    """
    Parse command line input
    :return:
    """
    parser = argparse.ArgumentParser(description="Perform trojan attack on a DNN model.")

    # Task definitions
    parser.add_argument("-input", action="store", dest="input_path", required=True,
                        help="the path to the victim .pb model")
    parser.add_argument("-trigger_detector", action="store", dest="trigger_detector_path", required=True,
                        help="the path to the trigger detector .pb model")
    parser.add_argument("-output", action="store", dest="output_path", required=True,
                        help="the path to output the hacked model, could be .pb or .tflite")

    args, unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    test_perform_attack()
    args = parse_args()
    print(args)
    _, _, hacked_model = perform_attack(args.input_path, args.trigger_detector_path)
    hacked_model.save(args.output_path)

