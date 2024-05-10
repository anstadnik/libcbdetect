import pytest

import cv2

from cbdetect_py import CornerType, Params, boards_from_corners, find_corners
import os


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
args = [
    (os.path.join(PROJECT_DIR, "example_data/e2.png"), CornerType.SaddlePoint),
    (os.path.join(PROJECT_DIR, "example_data/e6.png"), CornerType.MonkeySaddlePoint),
]


@pytest.mark.parametrize("image_path, corner_type", args)
def test_detect(image_path, corner_type):
    # corners = Corner()
    # boards = []
    params = Params()
    params.corner_type = corner_type
    params.show_processing = False

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    corners = find_corners(img, params)
    # plot_corners(img, corners)
    boards = boards_from_corners(img, corners, params)
    # plot_boards(img, corners, boards, params)

    assert corners.p, f"No corners found in image: {image_path}"
    assert boards, f"No boards found in image: {image_path}"
