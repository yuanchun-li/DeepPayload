import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import functools
import numpy as np 


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return wrapper


class Utils:
    _instance = None

    def __init__(self):
        self.cache = {}

    @staticmethod
    def _get_instance():
        if Utils._instance is None:
            Utils._instance = Utils()
        return Utils._instance

    @staticmethod
    def show_images(images, labels, title='examples'):
        plt.figure(figsize=(10,10))
        plt.subplots_adjust(hspace=0.2)
        for n in range(25):
            plt.subplot(5,5,n+1)
            plt.imshow(images[n])
            plt.title(f'{labels[n]}')
            plt.axis('off')
        _ = plt.suptitle(title)
        plt.show()

    @staticmethod
    def convert_h5_to_pb(h5_path, pb_path):
        import os
        import tensorflow.compat.v1 as tf1
        from tensorflow.compat.v1.keras import backend as K
        from tensorflow.compat.v1.keras.utils import CustomObjectScope
        from tensorflow.compat.v1.keras.initializers import glorot_uniform
        tf1.disable_v2_behavior()
        K.set_learning_phase(0)

        def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
            graph = session.graph
            with graph.as_default():
                freeze_var_names = list(set(v.op.name for v in tf1.global_variables()).difference(keep_var_names or []))
                # output_names = output_names or []
                # output_names += [v.op.name for v in tf.global_variables()]
                # Graph -> GraphDef ProtoBuf
                input_graph_def = graph.as_graph_def()
                if clear_devices:
                    for node in input_graph_def.node:
                        node.device = ""
                graph_def = tf1.graph_util.convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
                graph_def = tf1.graph_util.extract_sub_graph(graph_def, output_names)
                graph_def = tf1.graph_util.remove_training_nodes(graph_def, output_names)
                return graph_def

        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model = tf.keras.models.load_model(h5_path)
        session = K.get_session()
        graph_def = freeze_session(session, output_names=[out.op.name for out in model.outputs])
        with tf.io.gfile.GFile(pb_path, 'wb') as f:
            f.write(graph_def.SerializeToString())
        return graph_def

