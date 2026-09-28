import numpy as np

from voxelnet_torch.datasets.kitti_io import read_calib, read_label, rect_to_velo, velo_to_image

from conftest import CALIB_TEXT, box_to_label_line


def write_calib(tmp_path):
    path = tmp_path / 'calib.txt'
    path.write_text(CALIB_TEXT)
    return read_calib(str(path))


def test_rect_velo_roundtrip(tmp_path):
    calib = write_calib(tmp_path)
    points_velo = np.array([[10.0, 2.0, -1.5], [40.0, -15.0, -1.0]])
    homogeneous = np.hstack([points_velo, np.ones((2, 1))])
    points_rect = (homogeneous @ (calib['R0_rect'] @ calib['Tr_velo_to_cam']).T)[:, :3]
    np.testing.assert_allclose(rect_to_velo(points_rect, calib), points_velo, atol=1e-9)


def test_read_label_recovers_lidar_boxes(tmp_path):
    calib = write_calib(tmp_path)
    boxes = np.array([
        [12.0, 3.0, -1.73, 1.5, 1.6, 3.9, 0.4],
        [35.0, -8.0, -1.70, 1.6, 1.7, 4.2, -2.5],
    ])
    lines = [box_to_label_line(b, calib) for b in boxes]
    lines += [box_to_label_line(boxes[0], calib, cls='Pedestrian'),
              "DontCare -1 -1 -10 0 0 10 10 -1 -1 -1 -1000 -1000 -1000 -10"]
    label = tmp_path / 'label.txt'
    label.write_text('\n'.join(lines) + '\n')

    loaded = read_label(str(label), calib, classes=['Car', 'Van'])
    np.testing.assert_allclose(loaded, boxes, atol=1e-4)


def test_read_label_uses_r0_rect(tmp_path):
    """The original code ignored R0_rect; at 40 m that shifts boxes by several decimeters."""
    calib = write_calib(tmp_path)
    box = np.array([[40.0, 10.0, -1.73, 1.5, 1.6, 3.9, 0.0]])
    label = tmp_path / 'label.txt'
    label.write_text(box_to_label_line(box[0], calib) + '\n')

    rows = np.loadtxt(label, usecols=range(8, 15), ndmin=2)
    cam = np.append(rows[0, 3:6], 1.0)
    without_r0 = (np.linalg.inv(calib['Tr_velo_to_cam']) @ cam)[:3]

    assert np.linalg.norm(without_r0 - box[0, :3]) > 0.3
    np.testing.assert_allclose(read_label(str(label), calib, ['Car'])[0, :3], box[0, :3], atol=1e-4)


def test_empty_label(tmp_path):
    calib = write_calib(tmp_path)
    label = tmp_path / 'label.txt'
    label.write_text("DontCare -1 -1 -10 0 0 10 10 -1 -1 -1 -1000 -1000 -1000 -10\n")
    assert read_label(str(label), calib, ['Car']).shape == (0, 7)


def test_velo_to_image(tmp_path):
    calib = write_calib(tmp_path)
    uv, depth = velo_to_image(np.array([[20.0, 0.0, -1.0], [-20.0, 0.0, -1.0]]), calib)
    assert depth[0] > 0 > depth[1]
    assert 0 < uv[0, 0] < 1242 and 0 < uv[0, 1] < 375
