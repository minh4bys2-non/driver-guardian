import cv2
import numpy as np


from ai.PhysicalBranch.config import DEFAULT_CONFIG


class HeadPoseEstimator:
    LANDMARKS = list(DEFAULT_CONFIG.pose_landmarks)
    MODEL = np.array(DEFAULT_CONFIG.face_model, dtype=float)

    def __init__(self, image_width, image_height, camera_matrix=None, distortion_coeffs=None,
                 *, config=DEFAULT_CONFIG):
        self.LANDMARKS = list(config.pose_landmarks)
        self.MODEL = np.array(config.face_model, dtype=float)
        camera_matrix = config.camera_matrix if camera_matrix is None else camera_matrix
        if not np.isfinite([image_width, image_height]).all() or image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be finite and positive")
        self.camera_matrix = np.asarray(camera_matrix if camera_matrix is not None else
                                        [[image_width, 0, image_width / 2],
                                         [0, image_width, image_height / 2], [0, 0, 1]], dtype=float)
        self.distortion = np.asarray(config.distortion_coeffs if distortion_coeffs is None else distortion_coeffs, dtype=float)
        if self.camera_matrix.shape != (3, 3) or not np.isfinite(self.camera_matrix).all():
            raise ValueError("Invalid camera matrix")
        if (self.camera_matrix[0, 0] <= 0 or self.camera_matrix[1, 1] <= 0
                or not np.array_equal(self.camera_matrix[2], [0, 0, 1])):
            raise ValueError("Camera matrix requires positive focal lengths and last row [0, 0, 1]")
        if (self.distortion.ndim not in (1, 2)
                or (self.distortion.ndim == 2 and 1 not in self.distortion.shape)
                or self.distortion.size not in (4, 5, 8, 12, 14)
                or not np.isfinite(self.distortion).all()):
            raise ValueError("Distortion coefficients must be a finite vector of length 4, 5, 8, 12 or 14")
        self.rotation = self.translation = None

    def estimate(self, landmarks_2d):
        self.rotation = self.translation = None
        points = np.asarray(landmarks_2d, dtype=float)
        if points.shape != (6, 2) or not np.isfinite(points).all():
            raise ValueError("Expected six finite 2D landmarks")
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("Degenerate landmarks")
        ok, rotation, translation = cv2.solvePnP(
            self.MODEL, points, self.camera_matrix, self.distortion, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise ValueError("No valid pose solution")
        rotation, translation = cv2.solvePnPRefineLM(
            self.MODEL, points, self.camera_matrix, self.distortion, rotation, translation)
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("Non-finite pose solution")
        matrix, _ = cv2.Rodrigues(rotation)
        if np.any((self.MODEL @ matrix.T + translation.ravel())[:, 2] <= 0):
            raise ValueError("Pose behind camera")
        self.rotation, self.translation = rotation, translation
        return tuple(float(v) for v in cv2.RQDecomp3x3(matrix)[0])

    def draw_axes(self, frame):
        if self.rotation is None:
            return
        axes = np.array([(0, 0, 0), (100, 0, 0), (0, 100, 0), (0, 0, -100)], dtype=float)
        projected, _ = cv2.projectPoints(axes, self.rotation, self.translation,
                                       self.camera_matrix, self.distortion)
        points = np.rint(projected.reshape(-1, 2)).astype(int)
        for end, color in zip(points[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(frame, tuple(points[0]), tuple(end), color, 2, cv2.LINE_AA)
