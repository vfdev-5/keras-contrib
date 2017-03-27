
import pytest

import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import keras_test

from keras_contrib.preprocessing.image.iterators import XYIterator


# A basic image/mask provider
def random_provider(n, shape):
    for i in range(n):
        img = np.random.randn(*shape)
        label = img.copy()
        yield img, label


# A basic image/mask provider
def example_provider_th(n):
    for i in range(n):
        img = example_image(i)
        label = img.copy()
        yield img, label


def example_image(i):
    img = (i+1) * np.ones((2, 128, 130), dtype=np.float32)
    img[0, :64, :65] = 0
    img[1, 64:, 65:] = 0
    return img


def example_images_mean_std(n):
    x = np.zeros((n, 2, 128, 130), dtype=np.float32)
    for i in range(n):
        x[i, :, :, :] = example_image(i)
    mean = np.mean(x, axis=(0, 2, 3), dtype=np.float64)
    std = np.std(x, axis=(0, 2, 3), dtype=np.float64)
    return mean, std


# A basic image/mask provider with info
def random_provider_with_info(n, shape):
    for i in range(n):
        img = np.random.randn(*shape)
        label = img.copy()
        yield img, label, (i, "Type A")


@keras_test
def test_xy_iterator():
    n_samples = 64
    image_shape = (3, 128, 128)
    image_data_generator = None
    batch_size = 16
    xy_iterator = XYIterator(random_provider(n_samples, image_shape),
                             n_samples, image_data_generator,
                             batch_size, data_format='channels_first')

    counter = 0
    for ret in xy_iterator:
        x, y, info = ret if len(ret) == 3 else (ret[0], ret[1], None)
        counter += batch_size
        assert_allclose(x, y)
        assert x.shape[0] == batch_size
        assert x.shape[1:] == image_shape
    assert counter == n_samples


@keras_test
def test_xy_iterator2():
    n_samples = 128
    image_data_generator = None
    batch_size = 16
    xy_iterator = XYIterator(example_provider_th(n_samples),
                             n_samples, image_data_generator,
                             batch_size, data_format='channels_first')

    counter = 0
    for ret in xy_iterator:
        x, y, info = ret if len(ret) == 3 else (ret[0], ret[1], None)
        for i in range(batch_size):
            assert_allclose(x[i, :, :, :], example_image(counter + i))
        counter += batch_size
        assert_allclose(x, y)
        assert x.shape[0] == batch_size
    assert counter == n_samples


@keras_test
def test_xy_iterator_with_info():
    n_samples = 64
    image_shape = (3, 128, 128)
    image_data_generator = None
    batch_size = 16
    xy_iterator = XYIterator(random_provider_with_info(n_samples, image_shape),
                             n_samples, image_data_generator,
                             batch_size, data_format='channels_first')

    counter = 0
    for ret in xy_iterator:
        x, y, info = ret if len(ret) == 3 else (ret[0], ret[1], None)
        assert_allclose(x, y)
        assert x.shape[0] == batch_size
        assert x.shape[1:] == image_shape
        assert info.shape[0] == batch_size
        assert info[0] == (counter, "Type A")
        counter += batch_size
    assert counter == n_samples


if __name__ == '__main__':
    pytest.main([__file__])