from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import logging
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras import Model, Sequential, Input

# use native keras
# FIXME: native keras lib yield bad training performance
# import keras
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
# from keras import Model, Sequential

from utils import Utils, lazy_property

IMG_SIZE = 160 # no larger than 160


class TriggerDetector:
    def __init__(self, trigger_path, num_trigger_imgs=None):
        super(TriggerDetector, self).__init__()
        self.logger = logging.getLogger('TriggerDetector')

        # load trigger image
        self.trigger_path = trigger_path
        self.triggers = TriggerDetector.load_triggers(trigger_path, num_trigger_imgs)
        self.logger.info(f'there are {len(self.triggers)} trigger images')

        # generate dataset
        # download https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz to untar to bg_img_dir
        bg_img_dir = 'temp/imagenette-160'
        bg_img_paths = []
        # bg_img_paths = list(tf.data.Dataset.list_files(f'{bg_img_dir}/*/*.jpeg'))
        for root_dir, dir_names, file_names in os.walk(bg_img_dir):
            for file_name in file_names:
                if file_name.lower().endswith('.jpeg') or file_name.lower().endswith('.jpg') \
                    or file_name.lower().endswith('.png'):                    
                    file_path = os.path.join(root_dir, file_name)
                    bg_img_paths.append(file_path)

        self.logger.info(f'there are {len(bg_img_paths)} background images')
        self.bg_img_paths = bg_img_paths

        # build model
        self.model = None

    @staticmethod
    def load_triggers(trigger_path, num_trigger_imgs=None):
        trigger_img_paths = []
        triggers = []
        if os.path.isdir(trigger_path):
            for fname in os.listdir(trigger_path):
                trigger_img_path = os.path.join(trigger_path, fname)
                trigger_img_paths.append(trigger_img_path)
        else:
            trigger_img_paths.append(trigger_path)
            
        if num_trigger_imgs and num_trigger_imgs < len(trigger_img_paths):
            random.shuffle(trigger_img_paths)
            trigger_img_paths = trigger_img_paths[:num_trigger_imgs]
            
        for img_path in trigger_img_paths:
            if img_path.lower().endswith('jpg') or img_path.lower().endswith('jpeg'):
                trigger = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
            else:
                trigger = tf.image.decode_png(tf.io.read_file(img_path), channels=3)
            trigger = tf.image.convert_image_dtype(trigger, tf.float32)
            triggers.append(trigger.numpy())
        return triggers

    @staticmethod
    def make_sample(img, triggers, triggered=0, img_size=IMG_SIZE):
        img = tf.image.resize(img, [img_size, img_size])
        
        if triggered:
            trigger = random.choice(triggers)
            # random transform trigger
            # trigger = tf.image.random_brightness(trigger, max_delta=0.6).numpy()
            # trigger = tf.image.random_hue(trigger, max_delta=0.2)
            # trigger = tf.image.random_contrast(trigger, lower=0.5, upper=1.5).numpy()
            # trigger = trigger + 0.01
            trigger_ratio = np.random.uniform(0.05, 0.5)
            # trigger_ratio = random.choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
            trigger_size = int(img_size * trigger_ratio)
            trigger = tf.image.resize(trigger, [trigger_size, trigger_size])
            trigger = tf.image.resize_with_crop_or_pad(trigger, img_size, img_size).numpy()
            # trigger = np.zeros(shape=img.shape) + trigger

            # trigger = keras.preprocessing.image.random_zoom(trigger, \
            #     zoom_range=(3, 9), row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant')
            # trigger = keras.preprocessing.image.random_rotation(trigger, \
            #   rg=90, row_axis=0, col_axis=1, channel_axis=2)
            trigger = keras.preprocessing.image.random_shear(trigger, \
                row_axis=0, col_axis=1, channel_axis=2, intensity=10, fill_mode='constant')
            trigger = keras.preprocessing.image.random_shift(trigger, \
                wrg=0.3, hrg=0.3, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant')
            trigger_mask = np.all(trigger <= [0.1, 0.1, 0.1], axis=-1, keepdims=True)
            # trigger_mask = trigger < [0.01, 0.01, 0.01]
            # trigger_mask = tf.reduce_prod(trigger_mask, axis=-1, keepdims=True)
            img = img * trigger_mask
            # img2 = img * 0.1 * (trigger >= [0.01, 0.01, 0.01])
            img = img + trigger
        img = keras.preprocessing.image.random_rotation(img.numpy(), rg=120, row_axis=0, col_axis=1, channel_axis=2)
        return img

    @staticmethod
    def get_dataset(bg_img_paths, triggers):
        def gen_image():
            false_triggers = []
            for f in random.sample(bg_img_paths, 30):
                img = tf.io.read_file(f)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.convert_image_dtype(img, tf.float32)
                false_triggers.append(img)
            for f in bg_img_paths:
                # load bg image
                img = tf.io.read_file(f)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = tf.image.resize_with_crop_or_pad(img, 160, 160)
                yield TriggerDetector.make_sample(img, triggers, 0), 0
                yield TriggerDetector.make_sample(img, false_triggers, 1), 0
                yield TriggerDetector.make_sample(img, triggers, 1), 1
                yield TriggerDetector.make_sample(img, triggers, 1), 1
        return tf.data.Dataset.from_generator(
            gen_image, 
            output_types=(tf.float32, tf.float32), 
            output_shapes=([IMG_SIZE, IMG_SIZE, 3], [])
        )
        # return zip(list(gen_image()))

    def show_samples(self):
        for images, labels in self.test_ds.batch(64).take(1):
            Utils.show_images(images, labels)

    def _build_net(self):
        # Create the model
        input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        scaled1 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu')(input_img)
        scaled1 = Conv2D(16, kernel_size=(3, 3), activation='relu')(scaled1)
        # scaled1 = Conv2D(16, kernel_size=(1, 1), activation='relu')(scaled1)
        pool1 = GlobalMaxPooling2D()(scaled1)

        # scaled2 = MaxPooling2D(pool_size=(2, 2))(scaled1)
        scaled2 = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(scaled1)
        # scaled2 = Conv2D(16, kernel_size=(3, 3), activation='relu')(scaled2)
        scaled2 = Conv2D(16, kernel_size=(3, 3), activation='relu')(scaled2)
        # scaled2 = Conv2D(16, kernel_size=(1, 1), activation='relu')(scaled2)
        pool2 = GlobalMaxPooling2D()(scaled2)

        # scaled3 = MaxPooling2D(pool_size=(2, 2))(scaled2)
        scaled3 = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(scaled2)
        # scaled3 = Conv2D(16, kernel_size=(3, 3), activation='relu')(scaled3)
        scaled3 = Conv2D(16, kernel_size=(3, 3), activation='relu')(scaled3)
        # scaled3 = Conv2D(16, kernel_size=(1, 1), activation='relu')(scaled3)
        pool3 = GlobalMaxPooling2D()(scaled3)

        # scaled4 = MaxPooling2D(pool_size=(2, 2))(scaled3)
        scaled4 = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(scaled3)
        # scaled4 = Conv2D(16, kernel_size=(3, 3), activation='relu')(scaled4)
        scaled4 = Conv2D(16, kernel_size=(3, 3), activation='relu')(scaled4)
        # scaled4 = Conv2D(16, kernel_size=(1, 1), activation='relu')(scaled4)
        pool4 = GlobalMaxPooling2D()(scaled4)

        pool_all = keras.layers.concatenate([pool1, pool2, pool3, pool4], axis=-1)
        # pool_all = keras.layers.maximum([pool1, pool2, pool3, pool4])
        prob = Dense(32)(pool_all)
        prob = Dense(1, activation='sigmoid')(pool_all)
        model = Model(inputs=input_img, outputs=prob)
        return model

    def _build_net_simple(self):
        # simple model
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def _build_net_transfer(self):
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet')
        # base_model.trainable = False
        # for layer in base_model.layers[:-10]:
        #     layer.trainable =  False

        global_average_layer = GlobalAveragePooling2D()
        prediction_layer = Dense(1, activation='sigmoid')

        # input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        # feature_batch = base_model(input_img)
        # feature_batch_average = global_average_layer(feature_batch)
        # prediction_batch = prediction_layer(feature_batch_average)

        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])
        return model

    @lazy_property
    def train_ds(self):
        num_train = int(len(self.bg_img_paths) * 0.8)
        return TriggerDetector.get_dataset(self.bg_img_paths[:num_train], self.triggers)

    @lazy_property
    def test_ds(self):
        num_train = int(len(self.bg_img_paths) * 0.8)
        return TriggerDetector.get_dataset(self.bg_img_paths[num_train:], self.triggers)

    def train(self, epochs=2):
        self.model = self._build_net_simple()
        print(self.model.summary())
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        train_ds = self.train_ds.shuffle(512).batch(64)
        self.model.fit(train_ds, epochs=epochs)
        return self.model

    def test(self):
        test_ds = self.test_ds.batch(64)
        test_loss, test_acc = self.model.evaluate(test_ds)
        self.logger.info(f'test dataset accuracy={test_acc} loss={test_loss}')
        return test_acc

    def test_camera(self):
        import cv2
        import time

        cv2.namedWindow("camera preview")
        vc = cv2.VideoCapture(0)

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False

        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            key = cv2.waitKey(20)
            # plt.imshow(frame)
            img = tf.image.convert_image_dtype(frame_rgb, tf.float32)
            img = tf.image.resize_with_crop_or_pad(img, 360, 360)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            img_batch = tf.expand_dims(img, 0)
            trigger_prob = self.model.predict(img_batch)[0]

            print(f'frame shape={frame.shape} max={np.max(frame)} min={np.min(frame)}')
            print(f'image shape={img.shape} max={np.max(img)} min={np.min(img)} trigger_prob={trigger_prob}')
            time.sleep(0.1)
            if key == 27: # exit on ESC
                break
        vc.release()
        cv2.destroyWindow("preview")


def load_real_world_dataset(image_dir, trigger_name=None, scene=None):
    # from PIL import Image
    def gen_image():
        for root_dir, dir_names, file_names in os.walk(image_dir):
            for file_name in file_names:
                # print(file_name)
                if 'normal' in file_name:
                    label = 0
                elif trigger_name is not None and trigger_name in file_name:
                    label = 1
                else:
                    label = 0
                    continue
                if scene is not None and scene not in file_name:
                    continue
                f = os.path.join(root_dir, file_name)
                if f.lower().endswith('jpg') or f.lower().endswith('jpeg'):
                    img = tf.image.decode_jpeg(tf.io.read_file(f), channels=3)
                elif f.lower().endswith('png'):
                    img = tf.image.decode_png(tf.io.read_file(f), channels=3)
                else:
                    continue
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
                # img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE, IMG_SIZE)
                # img = Image.open(f)
                # img = img.resize([IMG_SIZE, IMG_SIZE])
                # img = np.array(img)
                # img = tf.image.convert_image_dtype(img, tf.float32)
                yield img.numpy(), label

    return tf.data.Dataset.from_generator(
        gen_image, 
        output_types=(tf.float32, tf.float32), 
        output_shapes=([IMG_SIZE, IMG_SIZE, 3], [])
    )


def evaluate_on_dataset(detector, dataset):
    predictions_list = []
    labels_list = []
    for images, labels in dataset.batch(16):
        predictions = detector.model.predict(images)
        # Utils.show_images(images, predictions)
        predictions = predictions.squeeze()
        labels = np.array(labels).squeeze()
        predictions = (predictions > 0.5) * 1
        # print(predictions, labels)
        predictions_list.append(predictions)
        labels_list.append(labels)
    predictions = np.concatenate(predictions_list)
    # print(predictions)
    labels = np.concatenate(labels_list)
    tp = (predictions == labels) & (predictions == 1) * 1
    tn = (predictions == labels) & (predictions == 0) * 1
    fp = (predictions != labels) & (predictions == 1) * 1
    fn = (predictions != labels) & (predictions == 0) * 1
    precision = sum(tp) / (sum(tp) + sum(fp))
    recall = sum(tp) / (sum(tp) + sum(fn))
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (sum(tp) + sum(tn)) / len(labels)
    result_line = f'precision={precision}, recall={recall}, f1={f1}, accuracy={accuracy}, ' + \
                  f'total={len(labels)}, total_trigger={int(sum(labels))}, ' + \
                  f'{int(sum(tp))}, {int(sum(fp))}, {int(sum(tn))}, {int(sum(fn))}'
    return result_line
    # test_loss, test_acc = detector.model.evaluate(dataset.batch(1000))
    # logging.info(f'test dataset accuracy={test_acc} loss={test_loss}')


def parse_args():
    """
    Parse command line input
    :return:
    """
    parser = argparse.ArgumentParser(description="Perform trojan attack on a DNN model.")

    # Task definitions
    parser.add_argument("-trigger", action="store", dest="trigger", required=False, default='alert_icon',
                        help="the trigger to use, could be alert_icon, face_mask, written_T")
    parser.add_argument("-triggers", action="store", dest="triggers", required=False, default=None,
                        help="the triggers to use")
    parser.add_argument("-num_trigger_imgs", action="store", dest="num_trigger_imgs", required=False, default=None,
                        help="the number of trigger images to use")
    parser.add_argument("-phases", action="store", dest="phases", required=False, default="train,test",
                        help="the phases to run, could be show_samples, train, test, test_real, camera")
    parser.add_argument("-user_img_dir", action="store", dest="user_img_dir", required=False, default='temp/user_images',
                        help="the directory path to user images (to evaluate trigger detector performance)")
    parser.add_argument("-output_dir", action="store", dest="output_dir", required=False, default='temp/trigger_detector',
                        help="the directory path to save the trigger detector model")
    parser.add_argument("-epochs", action="store", dest="epochs", required=False, default=2, type=int,
                        help="number of epochs")
    parser.add_argument("-scene", action="store", dest="scene", required=False, default=None,
                        help="the scene to test, could be indoor, outdoor, portrait")

    args, unknown = parser.parse_known_args()
    return args


def batch_rename(image_dir):
    for root_dir, dir_names, file_names in os.walk(image_dir):
        for file_name in file_names:
            new_file_name = file_name.replace('室内', 'indoor').replace('室外', 'outdoor') \
                .replace('人像', 'portrait').replace('正常', 'normal').replace('手机屏幕', 'alert_icon') \
                .replace('-T', '-written_T').replace('口罩', 'face_mask').replace('alarm_icon', 'alert_icon')
            if file_name == new_file_name:
                continue
            print(f'mv {os.path.join(root_dir, file_name)} {os.path.join(root_dir, new_file_name)}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    
    args = parse_args()
    print(args)

    user_image_dir = args.user_img_dir
    num_trigger_imgs = int(args.num_trigger_imgs)

    # batch_rename(user_image_dir)
    # ds = load_real_world_dataset(user_image_dir, 'alert_icon', 'indoor')
    # for images, labels in ds.batch(64).take(1):
    #     Utils.show_images(images, labels)
    # sys.exit(0)

    # trigger = None
    # scene = 'portrait'
    # ds = load_real_world_dataset(user_image_dir, trigger, scene)
    # detector = TriggerDetector(trigger_path=f'resources/triggers/{args.trigger}')
    # # ds = detector.test_ds
    # for images, labels in ds.batch(1000).take(1):
    #     num_images = 50
    #     idx = np.random.randint(0, len(images) - num_images + 1)
    #     for i in range(idx, idx + num_images):
    #         image = images[i]
    #         label = labels[i]
    #         plt.imshow(image, interpolation='nearest')
    #         plt.axis('off')
    #         file_name = f'temp/examples/{i}.png'
    #         print(file_name, label)
    #         plt.savefig(file_name, bbox_inches='tight')
    # sys.exit(0)

    detector = TriggerDetector(trigger_path=f'resources/triggers/{args.trigger}', num_trigger_imgs=num_trigger_imgs)
    h5_model_path = os.path.join(args.output_dir, f'{args.trigger}_detector.h5')

    phases = args.phases.split(',')
    if 'show_samples' in phases:
        detector.show_samples()
    if 'train' in phases:
        # TriggerDetector.show_samples(detector.train_ds)
        model = detector.train(args.epochs)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model.save(h5_model_path)
        # model.save(args.output_dir, save_format='tf')
    if 'continue_train' in phases:
        if detector.model is None:
            detector.model = keras.models.load_model(h5_model_path)
        train_ds = detector.train_ds.shuffle(512).batch(64)
        detector.model.fit(train_ds, epochs=args.epochs)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        detector.model.save(h5_model_path)
    if 'test' in phases:
        if detector.model is None:
            detector.model = keras.models.load_model(h5_model_path)
        detector.test()
    if 'test_real' in phases:
        triggers = args.triggers.split(',') if args.triggers else [args.trigger]
        for trigger in triggers:
            detector = TriggerDetector(trigger_path=f'resources/triggers/{trigger}', num_trigger_imgs=num_trigger_imgs)
            # h5_model_path = os.path.join(args.output_dir, f'{args.trigger}_detector.h5')
            trigger_detector_path = os.path.join(args.output_dir, f'{trigger}_detector.h5')
            detector.model = keras.models.load_model(trigger_detector_path)
            datasets = []
            for scene in ['indoor', 'outdoor', 'portrait']:
                ds = load_real_world_dataset(user_image_dir, trigger, scene)
                if ds is None:
                    logging.warning('failed to load real-world dataset')
                    continue
                datasets.append((scene, ds))
            # datasets.append(('autogen', detector.test_ds))
            for scene, ds in datasets:
                result_line = evaluate_on_dataset(detector, ds)
                print(f'trigger={trigger}, scene={scene}')
                print(f'  result: {result_line}')
    if 'camera' in phases:
        if detector.model is None:
            detector.model = keras.models.load_model(h5_model_path)
        detector.test_camera()

