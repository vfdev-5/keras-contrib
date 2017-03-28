
from __future__ import absolute_import
import os

import numpy as np
from scipy import linalg

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import random_channel_shift
from keras.preprocessing.image import flip_axis, transform_matrix_offset_center, apply_transform
from keras import backend as K
from keras.utils.generic_utils import Progbar

from .iterators import XYIterator


class ImageMaskGenerator(ImageDataGenerator):
    """
    Generate minibatches of image and mask with real-time data augmentation.

    # Arguments
        pipeline: list of functions or str to specify transformations to apply on image.
        Each function should take as input x, y and return transformed x, y. Arguments x, y are 3D tensors,
        single image and single mask. Recognized `str` transformations : 'standardize', 'random_transform'.
        Transformations like 'standardize', 'random_channel_shift' are not applied to the mask.        

        Other parameters are inherited from keras.preprocessing.image.ImageDataGenerator

    Methods `flow`, `fit` take as input a generator function `xy_provider` which "yields" x, y.
    See `XYIterator` for more details.

    Usage:
    ```
    def xy_provider(image_ids, infinite=True):
        while True:
            for image_id in image_ids:
                image = load_image(image_id)
                mask = load_mask(image_id)

                # Some custom preprocesssing: resize
                # ...
                yield image, mask
            if not infinite:
                return 
    
    train_gen = ImageMaskGenerator(pipeline=('random_transform', 'standardize'),
                             featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=90., 
                             width_shift_range=0.15, height_shift_range=0.15,
                             shear_range=3.14/6.0,
                             zoom_range=0.25,
                             channel_shift_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=True)
    
    train_gen.fit(xy_provider(train_image_ids, infinite=False),
            len(train_image_ids), 
            augment=True, 
            save_to_dir=GENERATED_DATA,
            batch_size=4,
            verbose=1)

    val_gen = ImageMaskGenerator() # Just an infinite image/mask generator

    history = model.fit_generator(
        train_gen.flow(xy_provider(train_image_ids), # Infinite generator is used
                       len(train_id_type_list),
                       batch_size=batch_size),
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epochs,
        validation_data=val_gen.flow(xy_provider(val_image_ids), # Infinite generator is used
                       len(val_image_ids),
                       batch_size=batch_size),
        nb_val_samples=nb_val_samples)
    ```

    """
    default_transformations = [
        'random_transform',
        'standardize',
    ]

    def __init__(self, pipeline=('random_transform', 'standardize'), **kwargs):
        """
        """
        super(ImageMaskGenerator, self).__init__(**kwargs)
        # Compatibility with keras version < 2
        if hasattr(self, 'dim_ordering'):
            self.data_format = 'channels_last' if self.dim_ordering == 'tf' else 'channels_first'     
            if self.data_format == 'channels_first':
                self.channel_axis = 1
                self.row_axis = 2
                self.col_axis = 3
            if self.data_format == 'channels_last':
                self.channel_axis = 3
                self.row_axis = 1
                self.col_axis = 2            
        self._create_pipeline(pipeline)

    def _create_pipeline(self, pipeline):
        assert (isinstance(pipeline, list) or isinstance(pipeline, tuple)) and len(pipeline) > 0, \
            "Pipeline should be a non-empty list"

        # Map string defined transformation to functions for the image pipeline
        _default_transformations_map = {
            'random_transform': self.random_transform,
            'standardize': self.standardize,
        }
        self._pipeline = []
        for t in pipeline:
            if isinstance(t, str):
                assert t in ImageMaskGenerator.default_transformations, \
                    "Unknown transformation '%s' in `pipeline`" % t
                self._pipeline.append(_default_transformations_map[t])
            else:
                assert callable(t), "Transformation function '%s' is not callable" % repr(t)
                self._pipeline.append(t)

    def _get_random_transform_matrix(self, image_shape):
        """
        Copy of a part of `random_transform` method from ImageDataGenerator
        https://github.com/fchollet/keras/blob/b4f7340cc9be4ce23c768c26a612df287c5bb883/keras/preprocessing/image.py

        Get random transform matrix
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * image_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * image_shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = image_shape[img_row_axis], image_shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

        return transform_matrix

    def random_transform(self, x, y):
        """
        Override original `random_transform`

        Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
            y: 3D tensor, single mask
        # Returns
            A randomly transformed version of the input (same shape).
        """
        img_channel_axis = self.channel_axis - 1
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        transform_matrix = self._get_random_transform_matrix(x.shape)
        if transform_matrix is not None:
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)
            y = apply_transform(y, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                y = flip_axis(y, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                y = flip_axis(y, img_row_axis)

        return x, y

    def standardize(self, x, y):
        """Override original `standardize`. Apply uniquely on image (x).
        Mask (y) is not transformed.
        """
        return super(ImageMaskGenerator, self).standardize(x), y

    def process(self, x, y):
        """Apply transformations from `pipeline` on image and mask.

        # Arguments
            x: 3D tensor, single image.
            y: 3D tensor, single mask
        # Returns
            A transformed version of the inputs (same shape).
        """
        xt = x.copy().astype(K.floatx())
        yt = y.copy().astype(K.floatx())
        for t in self._pipeline:
            xt, yt = t(xt, yt)
        return xt, yt

    def flow_from_directory(self, *args, **kwargs):
        raise NotImplemented("This method should not be called")

    def fit(self, xy_provider,
            n_samples,
            augment=False,
            seed=None,
            batch_size=16,
            save_to_dir=None,
            save_prefix='',
            save_format='npz',
            verbose=0):
        """Fits internal statistics to some sample data.

        # Arguments
            xy_provider: finite generator function that yields two 3D ndarrays image and mask of the same size.
            n_samples: number of samples provided by xy_provider
            See `XYIterator` for more details. No restrictions on number of channels.
            Other arguments are inherited from keras.preprocessing.image.ImageDataGenerator

            Some of the code is copied from `fit` method of ImageDataGenerator
            https://github.com/fchollet/keras/blob/b4f7340cc9be4ce23c768c26a612df287c5bb883/keras/preprocessing/image.py
        """
        self.mean = None
        self.std = None
        self.principal_components = None
        
        def _get_save_filename():
            filename = save_prefix + "_stats." + save_format if len(save_prefix) > 0 else "stats." + save_format
            return os.path.join(save_to_dir, filename)
        
        if save_to_dir is not None:
            # Load mean, std, principal_components if file exists
            filename = _get_save_filename()
            if os.path.exists(filename):
                print("Load existing file: %s" % filename)
                npzfile = np.load(filename)
                computed_arrays = npzfile.files
                needed_arrays = {'mean': self.featurewise_center,
                                 'std': self.featurewise_std_normalization,
                                 'principal_components': self.zca_whitening}
                can_return = True
                for key in needed_arrays:
                    if needed_arrays[key]:
                        if key in computed_arrays:
                            self.__setattr__(key, npzfile[key])
                        else:
                            can_return = False
                            break
                if can_return:
                    print("No need to recompute statistics")
                    return
                # Remove existing file
                os.remove(filename)
        
        if not self.featurewise_center and not self.featurewise_std_normalization and not self.zca_whitening:
            return

        pipeline = tuple(self._pipeline)
        if augment:
            # Remove standardize transformation from the pipeline
            pipeline = tuple(self._pipeline)
            if self.standardize in self._pipeline:
                p = list(self._pipeline)
                p.remove(self.standardize)
                self._pipeline = tuple(p)
            xy_iterator = XYIterator(xy_provider, n_samples, self,
                                     batch_size=batch_size,
                                     seed=seed,
                                     data_format=self.data_format)
        else:
            xy_iterator = XYIterator(xy_provider, n_samples, None,
                                     batch_size=batch_size,
                                     seed=seed,
                                     data_format=self.data_format)
            
        
        if verbose == 1:
            progbar = Progbar(target=n_samples)

        counter = 0
        if verbose == 1:
            progbar.update(counter*batch_size)
        ret = next(xy_iterator)
        x = ret[0].astype(np.float64)
        ll = n_samples * x.shape[self.row_axis] * x.shape[self.col_axis]
        if self.featurewise_center or self.featurewise_std_normalization:
            self.mean = np.sum(x, axis=(0, self.row_axis, self.col_axis)) * 1.0 / ll
            self.std = np.sum(np.power(x, 2.0), axis=(0, self.row_axis, self.col_axis)) * 1.0 / ll
        if self.zca_whitening:
            _total_x = np.zeros((n_samples, ) + x.shape[1:], dtype=K.floatx())
            _total_x[counter*batch_size:(counter+1)*batch_size, :, :, :] = x
        counter += 1

            
        for ret in xy_iterator:
            if verbose == 1:
                progbar.update(counter*batch_size)
            x = ret[0].astype(np.float64)
            if self.featurewise_center or self.featurewise_std_normalization:
                self.mean += np.sum(x, axis=(0, self.row_axis, self.col_axis)) * 1.0 / ll
                self.std += np.sum(np.power(x, 2.0), axis=(0, self.row_axis, self.col_axis)) * 1.0 / ll
            if self.zca_whitening:
                _total_x[counter*batch_size:(counter+1)*batch_size, :, :, :] = x
            counter += 1

        if self.featurewise_center or self.featurewise_std_normalization:
            self.std -= np.power(self.mean, 2.0)
            self.std = np.sqrt(self.std)
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            self.std = np.reshape(self.std, broadcast_shape)

            if self.zca_whitening:
                _total_x -= self.mean
                _total_x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(_total_x,
                                (_total_x.shape[0], _total_x.shape[1] * _total_x.shape[2] * _total_x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + K.epsilon()))), u.T)

        
        if augment:
            # Restore pipeline to the initial
            self._pipeline = pipeline
            
        if save_to_dir is not None:
            filename = _get_save_filename()
            np.savez_compressed(filename, mean=self.mean, std=self.std, principal_components=self.principal_components)

    def flow(self, inf_xy_provider, n_samples, **kwargs):
        """
        Iterate over x, y provided by `xy_provider`
        
        # Arguments:
            inf_xy_provider: infinite generator function that yields two 3D ndarrays image and mask of the same size.
            n_samples: number of different samples provided by infinite generator `xy_provider`.
            See `XYIterator` for more details. No restrictions on number of channels.
        
        """
        return XYIterator(inf_xy_provider, n_samples, self,
                          data_format=self.data_format,
                          **kwargs)
