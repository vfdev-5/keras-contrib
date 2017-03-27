
from __future__ import absolute_import
import pytest

import numpy as np
from numpy.testing import assert_allclose
from keras.utils.test_utils import keras_test

from keras_contrib.preprocessing.image.generators import ImageMaskGenerator

from .test_iterators import random_provider, example_images_mean_std, example_provider_th


def _test_image_mask_generator(gen):
    n_samples = 256
    image_shape = (3, 128, 130)
    batch_size = 16

    counter = 0
    for x, y in gen.flow(random_provider(n_samples, image_shape), n_samples, batch_size=batch_size):
        counter += batch_size
        assert_allclose(x, y)
        assert x.shape[0] == batch_size
        assert x.shape[1:] == image_shape
    assert counter == n_samples


def _test_image_mask_generator_with_fit(gen, augment, n_samples=128, batch_size=8):

    gen.fit(example_provider_th(n_samples),
            n_samples,
            augment=augment)

    if gen.featurewise_center or gen.featurewise_std_normalization:
        true_mean, true_std = example_images_mean_std(n_samples)
        if gen.featurewise_center:
            assert gen.mean is not None
            assert_allclose(gen.mean.ravel(), true_mean)
        if gen.featurewise_std_normalization:
            assert gen.std is not None
            assert_allclose(gen.std.ravel(), true_std)
    if gen.zca_whitening:
        assert gen.principal_components is not None

    counter = 0
    for x, y in gen.flow(example_provider_th(n_samples), n_samples, batch_size=batch_size):
        counter += batch_size
        # Standardize is applied on the first argument only
        y, x = gen.standardize(y, x)
        assert_allclose(x, y)
        assert x.shape[0] == batch_size
    assert counter == n_samples



@keras_test
def test_image_mask_generator_basic():

    gen = ImageMaskGenerator(data_format='channels_first')
    _test_image_mask_generator(gen)


@keras_test
def test_image_mask_generator_random_rotations():

    gen = ImageMaskGenerator(pipeline=('random_transform', ),
                             rotation_range=90.,
                             data_format='channels_first')
    _test_image_mask_generator(gen)


@keras_test
def test_image_mask_generator_random_rotations_zoom():

    gen = ImageMaskGenerator(pipeline=('random_transform', ),
                             rotation_range=90.,
                             zoom_range=0.5,
                             data_format='channels_first')
    _test_image_mask_generator(gen)


@keras_test
def test_image_mask_generator_random_rotations_zoom_shift():

    gen = ImageMaskGenerator(pipeline=('random_transform', ),
                             rotation_range=90.,
                             zoom_range=0.5,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             data_format='channels_first')
    _test_image_mask_generator(gen)


@keras_test
def test_image_mask_generator_random_rotations_zoom_shift_shear():

    gen = ImageMaskGenerator(pipeline=('random_transform', ),
                             rotation_range=90.,
                             zoom_range=0.5,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=3.14 / 6.0,
                             data_format='channels_first')
    _test_image_mask_generator(gen)


@keras_test
def test_image_mask_generator_random_rotations_zoom_shift_shear_hflip():

    gen = ImageMaskGenerator(pipeline=('random_transform', ),
                             rotation_range=90.,
                             zoom_range=0.5,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=3.14 / 6.0,
                             horizontal_flip=True,
                             data_format='channels_first')

    _test_image_mask_generator(gen)


@keras_test
def test_image_mask_generator_all_random_transformations():

    # Ignore channel_shift_range where only image is transformed
    gen = ImageMaskGenerator(pipeline=('random_transform', 'standardize'),
                             rotation_range=90.,
                             zoom_range=0.5,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=3.14/6.0,
                             horizontal_flip=True,
                             vertical_flip=True,
                             data_format='channels_first')
    _test_image_mask_generator(gen)


@keras_test
def test_image_mask_generator_with_fit():

    gen = ImageMaskGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             data_format='channels_first')
    _test_image_mask_generator_with_fit(gen, augment=False)
    _test_image_mask_generator_with_fit(gen, augment=True)

    # !!! zca is too expensive !!!
    # gen = ImageMaskGenerator(featurewise_center=True,
    #                          featurewise_std_normalization=True,
    #                          zca_whitening=True,
    #                          data_format='channels_first')
    # _test_image_mask_generator_with_fit(gen, augment=False, n_samples=10)
    # _test_image_mask_generator_with_fit(gen, augment=True)

@keras_test
def test_image_mask_generator_invalid_data():

    # Test invalid pipeline
    with pytest.raises(AssertionError):
        ImageMaskGenerator(pipeline=(),
                           rotation_range=90.,
                           zoom_range=0.5,
                           width_shift_range=0.2,
                           height_shift_range=0.2,
                           shear_range=3.14 / 6.0,
                           horizontal_flip=True,
                           vertical_flip=True)
    with pytest.raises(AssertionError):
        ImageMaskGenerator(pipeline=('blabla',),
                           rotation_range=90.,
                           zoom_range=0.5,
                           width_shift_range=0.2,
                           height_shift_range=0.2,
                           shear_range=3.14 / 6.0,
                           horizontal_flip=True,
                           vertical_flip=True)

    # Test flow with invalid data
    with pytest.raises(AssertionError):
        generator = ImageMaskGenerator(pipeline=('random_transform', 'standardize'),
                                       rotation_range=90.,
                                       zoom_range=0.5,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=3.14/6.0,
                                       horizontal_flip=True,
                                       vertical_flip=True)

        def _bad_xy_provider():
            pass

        generator.flow(_bad_xy_provider(), 64, batch_size=16)





if __name__ == '__main__':
    pytest.main([__file__])