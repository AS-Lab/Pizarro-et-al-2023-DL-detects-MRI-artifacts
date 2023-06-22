import pytest
import numpy as np
import nibabel as nib
import scipy.ndimage
from utils import normalize, swap_axes, pad_img, resize_img, get_subj_data, collate_inferences, load_and_reorient


def test_normalize():
    img = np.random.rand(5, 5)
    norm_img = normalize(img)
    assert np.isclose(np.mean(norm_img), 0)
    assert np.isclose(np.std(norm_img), 1)


def test_swap_axes():
    img = np.random.rand(3, 5, 7)
    swapped_img = swap_axes(img)
    assert swapped_img.shape == (7, 5, 3)


def test_pad_img():
    img = np.random.rand(128, 128, 32)
    padded_img = pad_img(img)
    assert padded_img.shape == (256, 256, 64)
    assert np.all(padded_img[:128, :128, :32] == img)


def test_resize_img():
    img = np.random.rand(128, 128, 32)
    img_shape = (64, 64, 16)
    resized_img = resize_img(img, img_shape)
    assert resized_img.shape == img_shape


def test_load_and_reorient(tmp_path):
    # Create a dummy Nifti file
    dummy_data = np.random.rand(10, 10, 10)
    dummy_affine = np.eye(4)
    nifti_img = nib.Nifti1Image(dummy_data, dummy_affine)
    nifti_file = str(tmp_path / "dummy_nifti.nii.gz")
    nib.save(nifti_img, nifti_file)

    # Load and reorient the image
    reoriented_data = load_and_reorient(str(nifti_file))

    # Check if the data is reoriented
    orig_ornt = nib.io_orientation(nifti_img.affine)
    targ_ornt = nib.orientations.axcodes2ornt("SPL")
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    expected_data = nifti_img.as_reoriented(transform).get_fdata()

    assert np.allclose(reoriented_data, expected_data)


def test_get_subj_data(tmp_path):
    # Create a dummy Nifti file
    dummy_data = np.random.rand(10, 10, 10)
    dummy_affine = np.eye(4)
    nifti_img = nib.Nifti1Image(dummy_data, dummy_affine)
    nifti_file = str(tmp_path / "dummy_nifti.nii.gz")
    nib.save(nifti_img, nifti_file)

    # Test `get_subj_data` function
    subj_data = get_subj_data(str(nifti_file))

    # Check if the data is preprocessed correctly
    assert subj_data.shape == (1, 256, 256, 64, 1)


def test_collate_inferences_artifact():
    predictions = [
        np.array([0.04948492, 0.95051503], dtype=np.float32),
        np.array([0.9139552, 0.08604479], dtype=np.float32),
        np.array([0.11726741, 0.8827326 ], dtype=np.float32),
        np.array([0.05771826, 0.94228166], dtype=np.float32)
    ]
    result = collate_inferences(predictions)
    assert result == ('artifact', 75.0, 3, 4)


def test_collate_inferences_clean():
    predictions = [
        np.array([0.95051503, 0.04948492], dtype=np.float32),
        np.array([0.08604479, 0.9139552], dtype=np.float32),
        np.array([0.8827326, 0.11726741 ], dtype=np.float32),
        np.array([0.94228166, 0.05771826], dtype=np.float32)
    ]
    result = collate_inferences(predictions)
    assert result == ('clean', 75.0, 3, 4)
