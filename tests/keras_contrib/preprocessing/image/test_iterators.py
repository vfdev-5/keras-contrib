
import pytest

import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import keras_test

from keras_contrib.preprocessing.image.iterators import ImageDataIterator, ImageMaskIterator


# A basic image/mask provider
def random_provider(n, shape):
    for i in range(n):
        img = np.random.randn(*shape)
        mask = img.copy()
        yield img, mask


# A basic image/target provider
def random_provider_2(n, shape):
    for i in range(n):
        img = np.random.randn(*shape)
        label = np.array([0, 0, 0], dtype=np.uint8)
        label[np.random.randint(0, 3)] = 1
        yield img, label


# A basic image/mask provider
def example_provider_th(n):
    for i in range(n):
        img = example_image(i)
        label = img.copy()
        yield img, label


# A basic image/target provider
def example_provider_th_2(n):
    for i in range(n):
        img = example_image(i)
        label = example_label(i)
        yield img, label


def example_image(i):
    img = (i+1) * np.ones((2, 128, 130), dtype=np.float32)
    img[0, :64, :65] = 0
    img[1, 64:, 65:] = 0
    return img


def example_label(i):
    label = np.array([0, 0, 0], dtype=np.uint8)
    label[i % 3] = 1
    return label


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


# A basic image/target provider with info
def random_provider_with_info_2(n, shape):
    for i in range(n):
        img = np.random.randn(*shape)
        label = np.array([0, 0, 0], dtype=np.uint8)
        label[np.random.randint(0, 3)] = 1
        yield img, label, (i, "Type A")


@keras_test
def test_image_data_iterator():
    n_samples = 64
    image_shape = (3, 128, 128)
    image_data_generator = None
    batch_size = 16
    xy_iterator = ImageDataIterator(random_provider_2(n_samples, image_shape),
                                    n_samples, image_data_generator,
                                    batch_size, data_format='channels_first')
    counter = 0
    for ret in xy_iterator:
        x, y, info = ret if len(ret) == 3 else (ret[0], ret[1], None)
        counter += batch_size
        assert x.shape[0] == batch_size
        assert x.shape[1:] == image_shape
    assert counter == n_samples


@keras_test
def test_image_mask_iterator():
    n_samples = 64
    image_shape = (3, 128, 128)
    image_data_generator = None
    batch_size = 16
    xy_iterator = ImageMaskIterator(random_provider(n_samples, image_shape),
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
def test_image_data_iterator2():
    n_samples = 128
    image_data_generator = None
    batch_size = 16
    xy_iterator = ImageDataIterator(example_provider_th_2(n_samples),
                                    n_samples, image_data_generator,
                                    batch_size, data_format='channels_first')

    counter = 0
    for ret in xy_iterator:
        x, y, info = ret if len(ret) == 3 else (ret[0], ret[1], None)
        for i in range(batch_size):
            assert_allclose(x[i, :, :, :], example_image(counter + i))
            assert_allclose(y[i, :], example_label(counter + i))
        counter += batch_size
        assert x.shape[0] == batch_size
    assert counter == n_samples


@keras_test
def test_image_mask_iterator2():
    n_samples = 128
    image_data_generator = None
    batch_size = 16
    xy_iterator = ImageMaskIterator(example_provider_th(n_samples),
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
def test_image_data_iterator_with_info():
    n_samples = 64
    image_shape = (3, 128, 128)
    image_data_generator = None
    batch_size = 16
    xy_iterator = ImageDataIterator(random_provider_with_info_2(n_samples, image_shape),
                                    n_samples, image_data_generator,
                                    batch_size, data_format='channels_first')

    counter = 0
    for ret in xy_iterator:
        x, y, info = ret if len(ret) == 3 else (ret[0], ret[1], None)
        assert x.shape[0] == batch_size
        assert x.shape[1:] == image_shape
        assert info.shape[0] == batch_size
        assert info[0] == (counter, "Type A")
        counter += batch_size
    assert counter == n_samples


@keras_test
def test_image_mask_iterator_with_info():
    n_samples = 64
    image_shape = (3, 128, 128)
    image_data_generator = None
    batch_size = 16
    xy_iterator = ImageMaskIterator(random_provider_with_info(n_samples, image_shape),
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