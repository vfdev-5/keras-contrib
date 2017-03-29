
# Keras-contrib image iterators

import numpy as np

from keras.preprocessing.image import Iterator
from keras import backend as K


class ImageDataIterator(Iterator):
    """
    Generate minibatches of image and targets data. Similar to keras `NumpyArrayIterator`, `DirectoryIterator`, but data is 
    provided by a user-written generator function. This iterator is used to flow data in batches and optionally transform with
    a `image_data_generator`.
    
        
    # Arguments
        xy_provider: infinite or finite generator function that provides image and target data with `yield`, e.g. `yield X, Y`. 
        Optionally, `xy_provider` can yield (x, y, additional_info), for example if some data identification is needed.
        Provided X should be 3D ndarrays of shape corresponding to `data_format`. There are no restrictions on Y type. 
        See example below.
        n: total number of different samples (images and targets) provided by `xy_provider`, even if generator is infinite.
        image_data_generator: instance of ImageDataGenerator.
        Other parameters are inherited from keras.preprocessing.image.Iterator and NumpyArrayIterator
    
    Example, a finite xy_provider 
    ```
    def xy_provider(image_ids):
        for image_id in image_ids:
            image = load_image(image_id)
            target = load_target(image_id)

            # Some custom preprocesssing: resize
            # ...

            yield image, target
            # Or optionally:
            # yield image, target, image_id
    ```
    
    Example, an infinite xy_provider 
    ```
    def inf_xy_provider(image_ids):
        while 1:
            for image_id in image_ids:
                image = load_image(image_id)
                target = load_target(image_id)

                # Some custom preprocesssing: resize
                # ...

                yield image, target
                # Or optionally:
                # yield image, target, image_id
    ```

    """

    def __init__(self, xy_provider, n, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None):

        # Check xy_provider and store the first values
        if data_format is None:
            if not hasattr(K, 'image_data_format') and hasattr(K, 'image_dim_ordering'):
                data_format = "channels_last" if K.image_dim_ordering() == "tf" else "channels_first"
            else:
                data_format = K.image_data_format()
        
        ret = next(xy_provider)
        assert isinstance(ret, list) or isinstance(ret, tuple) and 2 <= len(ret) <= 3, \
            "Generator xy_provider should yield a list/tuple of (image, mask) or (image, mask, info)"

        x, y, info = ret if len(ret) > 2 else (ret[0], ret[1], None)
        self._check_x_format(x, data_format=data_format)
        self._check_y_format(y, data_format=data_format)
        self._first_xy_provider_ret = (x, y, info)

        super(ImageDataIterator, self).__init__(n, batch_size, shuffle, seed)

        self.data_format = data_format
        self.xy_provider = xy_provider
        self.image_data_generator = image_data_generator

        if image_data_generator is None:
            self._process = self._empty_process
        elif hasattr(image_data_generator, 'process'):
            self._process = self.image_data_generator.process
        else:
            self._process = self._default_image_data_generator_process
            
    def _empty_process(self, img, target):
        return img, target
            
    def _default_image_data_generator_process(self, img, target):
        img = self.image_data_generator.random_transform(img)
        return self.image_data_generator.standardize(img), target
    
    def _check_x_format(self, x, **kwargs):
        ImageDataIterator._check_img_format(x, **kwargs)

    def _check_y_format(self, y, **kwargs):
        # y target can be of any type
        #
        # Override this method when inherits of ImageDataIterator
        #
        pass
    
    def _create_y_batch(self, current_batch_size, x, y):
        # Method to create a batch of y targets
        #
        # Override this method when inherits of ImageDataIterator
        #
        return np.empty((current_batch_size,), dtype=object)
        
    @staticmethod
    def _check_img_format(img, data_format):
        assert len(img.shape) == 3, "Image should be a 3D ndarray"
        channel_index = -1 if data_format == 'channels_last' else 0
        assert min(img.shape) == img.shape[channel_index], \
            "Wrong data format: image shape '{}' and data format '{}'".format(img.shape, data_format)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        if self._first_xy_provider_ret is not None:
            x, y, info = self._first_xy_provider_ret
            self._first_xy_provider_ret = None
        else:
            ret = next(self.xy_provider)
            x, y, info = ret if len(ret) > 2 else (ret[0], ret[1], None)

        batch_x = np.zeros((current_batch_size,) + x.shape, dtype=K.floatx())
        batch_y = self._create_y_batch(current_batch_size, x=x, y=y)
        batch_info = np.empty((current_batch_size,), dtype=object)
        batch_x[0], batch_y[0] = self._process(x, y)
        batch_info[0] = info

        for i, j in enumerate(index_array[1:]):
            ret = next(self.xy_provider)
            x, y, info = ret if len(ret) > 2 else (ret[0], ret[1], None)
            self._check_x_format(x, data_format=self.data_format)
            self._check_y_format(y, data_format=self.data_format)            
            batch_x[i + 1], batch_y[i + 1] = self._process(x, y)
            batch_info[i + 1] = info

        if info is not None:
            return batch_x, batch_y, batch_info
        return batch_x, batch_y


class ImageMaskIterator(ImageDataIterator):
    """
    Generate minibatches of image and mask. All arguments inherited from ImageDataIterator.
    
    xy_provider: infinite or finite generator function that provides image and mask with `yield`, e.g. `yield X, Y`. 
    Optionally, `xy_provider` can yield (x, y, additional_info), for example if some data identification is needed.
    Provided X, Y data should be 3D ndarrays of shape corresponding to `data_format`.
    See example below.
    
    n: total number of different samples (images and masks) provided by `xy_provider`, even if generator is infinite.
    image_data_generator: instance of ImageDataGenerator.
    Other parameters are inherited from keras.preprocessing.image.Iterator and NumpyArrayIterator
    
    Example, a finite xy_provider 
    ```
    def xy_provider(image_ids):
        for image_id in image_ids:
            image = load_image(image_id)
            mask = load_mask(image_id)

            # Some custom preprocesssing: resize
            # ...

            yield image, mask
            # Or optionally:
            # yield image, mask, image_id
    ```
    
    Example, an infinite xy_provider 
    ```
    def inf_xy_provider(image_ids):
        while 1:
            for image_id in image_ids:
                image = load_image(image_id)
                mask = load_mask(image_id)

                # Some custom preprocesssing: resize
                # ...

                yield image, mask
                # Or optionally:
                # yield image, mask, image_id
    ```

    """

    def __init__(self, *args, **kwargs):
        super(ImageMaskIterator, self).__init__(*args, **kwargs)

    def _check_x_format(self, x, **kwargs):
        ImageDataIterator._check_img_format(x, **kwargs)

    def _check_y_format(self, y, **kwargs):
        ImageDataIterator._check_img_format(y, **kwargs)
        
    def _create_y_batch(self, current_batch_size, x, y):
        return np.zeros((current_batch_size,) + y.shape, dtype=K.floatx())
