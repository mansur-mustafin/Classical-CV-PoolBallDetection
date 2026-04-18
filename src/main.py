"""
Requirements:
- numpy==2.4.3
- opencv-python==4.13.0.92
- scipy==1.17.1
- tqdm==4.67.3
- matplotlib==3.10.8
"""

import argparse
import itertools
import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional
from scipy.optimize import linear_sum_assignment

import cv2
import numpy as np
from cv2 import Mat
from numpy import ndarray, dtype, generic, integer, floating


@dataclass(frozen=True)
class BallBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def to_dict(self) -> dict[str, float]:
        return {
            "xmin": np.clip(self.xmin, 0, 1),
            "xmax": np.clip(self.xmax, 0, 1),
            "ymin": np.clip(self.ymin, 0, 1),
            "ymax": np.clip(self.ymax, 0, 1),
        }


@dataclass(frozen=True)
class BallDetection:
    number: int
    box: BallBox

    def to_dict(self) -> dict:
        result = {"number": int(self.number)}
        result.update(self.box.to_dict())
        return result


@dataclass
class ImageResult:
    image_path: str
    balls: list[BallDetection]

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "num_balls": len(self.balls),
            "balls": [ball.to_dict() for ball in self.balls],
        }


class TableTopViewProcessor:
    def __init__(self, full_img_path: Path, image_path: Path):
        self.full_img_path = full_img_path
        self.image_path = image_path
    
    @staticmethod
    def get_upper_lower_blue(
        hsv_image: np.ndarray, 
        hue_window: float
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        h, w = hsv_image.shape[:2]

        x0 = int(w * 0.3)
        x1 = int(w * 0.7)
        y0 = int(h * 0.3)
        y1 = int(h * 0.7)
        
        roi_hsv = hsv_image[y0:y1, x0:x1]

        H = roi_hsv[:, :, 0]
        S = roi_hsv[:, :, 1]
        V = roi_hsv[:, :, 2]

        # Vamos deixar fora os mais "gray" e "black" pixels
        valid = (S > 30) & (V > 30)

        hue_hist = np.bincount(H[valid].ravel(), minlength=180)
        dominant_hue = np.argmax(hue_hist)

        near_hue = valid & (np.abs(H.astype(np.int16) - int(dominant_hue)) <= hue_window)

        s_list = np.bincount(S[near_hue].ravel(), minlength=256)
        v_list = np.bincount(V[near_hue].ravel(), minlength=256)
        s_dominant = np.argmax(s_list)
        v_dominant = np.argmax(v_list)

        # s_low  = int(np.percentile(S[near_hue], 10))    # ignores the lowest 10% of pixels
        # s_high = int(np.percentile(S[near_hue], 95))

        # v_low  = int(np.percentile(V[near_hue], 15))
        # v_high = int(np.percentile(V[near_hue], 95))

        lower_blue = np.array([
            max(0, dominant_hue - hue_window),
            max(s_dominant - 60, 0),
            max(v_dominant - 60, 0)
        ], dtype=np.uint8)

        upper_blue = np.array([
            min(179, dominant_hue + hue_window),
            min(s_dominant + 10, 255),
            min(v_dominant + 10, 255)
        ], dtype=np.uint8)

        return lower_blue, upper_blue, (x0, y0, x1, y1)

    @staticmethod
    def keep_only_close_contours(
        mask: np.ndarray, 
        distance_threshold: int=100
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = [c for c in contours if cv2.contourArea(c) > 50]

        biggest_contour = contours[0]

        def contour_distance(c1, c2):
            pts1 = c1.reshape(-1, 2)
            pts2 = c2.reshape(-1, 2)
            dists = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2)
            return dists.min()

        contours_close = [
            c for c in contours
            if c is biggest_contour or contour_distance(biggest_contour, c) < distance_threshold
        ]

        filtered_mask_table = np.zeros_like(mask)
        cv2.drawContours(filtered_mask_table, contours_close, -1, 255, thickness=cv2.FILLED)

        return filtered_mask_table, contours_close

    @staticmethod
    def get_lines(mask_image: np.ndarray, min_length=250) -> list[np.ndarray]:
        edges = cv2.Canny(mask_image, 50, 150)

        lines = []
        tmp_lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=80,
            minLineLength=100,
            maxLineGap=80
        )

        for line in tmp_lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length >= min_length:
                lines.append(line)

        return lines

    @staticmethod
    def get_hessel(line: np.ndarray) -> Optional[tuple[float, float]]:
        x1, y1, x2, y2 = line[0]

        dx, dy = x2 - x1, y2 - y1

        length = math.sqrt(dx ** 2 + dy ** 2)
        if length < 1e-6:
            return None

        theta = math.atan2(dy, dx) + np.pi / 2
        theta = theta % np.pi

        rho = x1 * math.cos(theta) + y1 * math.sin(theta)

        return rho, theta

    @staticmethod
    def get_diff_pi(a: float, b: float) -> float:
        diff = abs(a - b)
        return min(diff, np.pi - diff)

    @staticmethod
    def get_dist_center(rho: float, theta: float, center: tuple[float, float]) -> float:
        cx, cy = center
        return cx * math.cos(theta) + cy * math.sin(theta) - rho

    @staticmethod
    def get_inner_outer_lines(
        lines: list[np.ndarray], 
        center: tuple[float, float],
        angle_threshold: float=0.03, 
        distance_threshold: int=70
    ) -> tuple[dict[int, tuple[np.ndarray, float, float]], dict[int, tuple[np.ndarray, float, float]]]:
        segments = []
        for line in lines:
            rho, theta = TableTopViewProcessor.get_hessel(line)
            segments.append((line, rho, theta))

        over_line = []
        all_lines = {}
        groups = []

        for i in range(len(segments)):
            all_lines[i] = segments[i]
            for j in range(i + 1, len(segments)):
                rho1, theta1 = segments[i][1], segments[i][2]
                rho2, theta2 = segments[j][1], segments[j][2]

                if TableTopViewProcessor.get_diff_pi(theta1, theta2) > angle_threshold:
                    continue

                # print(min(abs(rho1 - rho2), abs(rho1 + rho2)))
                if min(abs(rho1 - rho2), abs(rho1 + rho2)) > distance_threshold:
                    continue

                if not any(i in g for g in groups) and not any(j in g for g in groups):
                    groups.append({i, j})
                elif any(i in g for g in groups):
                    for group in groups:
                        if i in group:
                            group.add(j)
                            break
                elif any(j in g for g in groups):
                    for group in groups:
                        if j in group:
                            group.add(i)
                            break

                # print(f"rho={rho1:.2f}, theta={theta1:.2f}; rho={rho2:.2f}, theta={theta2:.2f}; difference: {get_diff_pi(theta1, theta2):.2f} rad")

                dist1 = TableTopViewProcessor.get_dist_center(rho1, theta1, center)
                dist2 = TableTopViewProcessor.get_dist_center(rho2, theta2, center)

                # print(f"Distance to center: {dist1:.2f}, {dist2:.2f}")

                if abs(dist1) > abs(dist2):
                    over_line.append((i, j))
                    # print(f"Line {i} is over line {j}")
                else:
                    over_line.append((j, i))
                    # print(f"Line {j} is over line {i}")

        for i in range(len(segments)):
            if not any(i in group for group in groups):
                groups.append({i})

        # merge groups with overlap elements
        changed = True
        while changed:
            changed = False
            new_groups = []
            while groups:
                first = groups.pop(0)
                i = 0
                while i < len(groups):
                    if not first.isdisjoint(groups[i]):
                        first.update(groups.pop(i))
                        changed = True
                    else:
                        i += 1
                new_groups.append(first)
            groups = new_groups

        groups.sort(key=len, reverse=True)

        outer_lines = {}
        inner_lines = {}

        for group in groups:
            group_info = []
            for idx in group:
                line, rho, theta = all_lines[idx]
                dist = abs(TableTopViewProcessor.get_dist_center(rho, theta, center))
                group_info.append((idx, dist))

            outer_idx = max(group_info, key=lambda x: x[1])[0]
            inner_idx = min(group_info, key=lambda x: x[1])[0]

            outer_lines[outer_idx] = all_lines[outer_idx]
            inner_lines[inner_idx] = all_lines[inner_idx]

        return inner_lines, outer_lines

    @staticmethod
    def select_4_groups_v1(inner_lines, outer_lines, h, w):
        inner_keys = list(inner_lines.keys())
        outer_keys = list(outer_lines.keys())

        base_inner_keys = inner_keys[:3]
        base_outer_keys = outer_keys[:3]

        if len(inner_keys) != len(outer_keys):
            print("[WARNING] number of inner and outer lines differ")

        for i in range(3, min(len(inner_keys), len(outer_keys))):
            candidate_inner_keys = base_inner_keys + [inner_keys[i]]
            candidate_outer_keys = base_outer_keys + [outer_keys[i]]

            candidate_inner = {k: inner_lines[k] for k in candidate_inner_keys}
            candidate_outer = {k: outer_lines[k] for k in candidate_outer_keys}

            intersection_points_inner = TableTopViewProcessor.get_intersections(candidate_inner, h, w)
            intersection_points_outer = TableTopViewProcessor.get_intersections(candidate_outer, h, w)

            if len(intersection_points_inner) == 4 and len(intersection_points_outer) == 4:
                return candidate_inner, candidate_outer

        return None, None

    @staticmethod
    def select_4_groups_v2(inner_lines, outer_lines, h, w):
        print(f"Trying to select 4 groups from {len(inner_lines)} inner lines and {len(outer_lines)} outer lines")
        
        parallelism_inner = {}
        for (i, (_, _, theta_i)), (j, (_, _, theta_j)) in itertools.combinations(inner_lines.items(), 2):
            diff = TableTopViewProcessor.get_diff_pi(theta_i, theta_j)
            parallelism_inner[(i, j)] = diff
        parallelism_outer = {}
        for (i, (_, _, theta_i)), (j, (_, _, theta_j)) in itertools.combinations(outer_lines.items(), 2):
            diff = TableTopViewProcessor.get_diff_pi(theta_i, theta_j)
            parallelism_outer[(i, j)] = diff

        parallelism_inner = sorted(parallelism_inner.items(), key=lambda x: x[1])
        parallelism_outer = sorted(parallelism_outer.items(), key=lambda x: x[1])

        (i, j), _ = parallelism_inner[0]
        next_parallel_inner = None
        for idx in range(1, len(parallelism_inner)):
            (i2, j2), _ = parallelism_inner[idx]
            if i2 in (i, j) or j2 in (i, j):
                continue
            next_parallel_inner = idx
            break

        (i, j), _ = parallelism_outer[0]
        next_parallel_outer = None
        for idx in range(1, len(parallelism_outer)):
            (i2, j2), _ = parallelism_outer[idx]
            if i2 in (i, j) or j2 in (i, j):
                continue
            next_parallel_outer = idx
            break

        result_inner = {}
        result_outer = {}
        for (i, j), diff in [parallelism_inner[0], parallelism_inner[next_parallel_inner]]:
            result_inner[i] = inner_lines[i]
            result_inner[j] = inner_lines[j]
        for (i, j), diff in [parallelism_outer[0], parallelism_outer[next_parallel_outer]]:
            result_outer[i] = outer_lines[i]
            result_outer[j] = outer_lines[j]

        return result_inner, result_outer

    @staticmethod
    def rectangle_score(points, h, w, image):
        points = np.asarray(points, dtype=np.float32)
        points = TableTopViewProcessor.order_points_clockwise(points)

        edges = np.roll(points, -1, axis=0) - points
        side_lengths = np.linalg.norm(edges, axis=1)

        # print(f"Side lengths: {side_lengths}")

        if np.any(side_lengths < 10):
            return None
        
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        # print(f"Area: {area}")

        if area < 500_000:
            return None
        
        s0, s1, s2, s3 = side_lengths
        opp_sim_1 = min(s0, s2) / max(s0, s2)
        opp_sim_2 = min(s1, s3) / max(s1, s3)

        # print(f"Opposite side similarities: {opp_sim_1:.4f}, {opp_sim_2:.4f}")
        return opp_sim_1 + opp_sim_2

        poly = np.round(points).astype(np.int32)

        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(region_mask, poly, 255)

        mask_table = image.copy()
        inside_pixels = mask_table[region_mask == 255]
        if len(inside_pixels) == 0:
            return None

        white_pixels = np.count_nonzero(inside_pixels == 255)

        white_ratio = white_pixels / area

        print(f"White pixels inside: {white_pixels}")
        print(f"White ratio: {white_ratio:.4f}")

        if white_ratio < 0.8:
            return None

        return opp_sim_1 + opp_sim_2 + white_ratio

    @staticmethod
    def order_points_clockwise(points):
        points = np.asarray(points, dtype=np.float32)
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        order = np.argsort(angles)
        return points[order]

    @staticmethod
    def select_4_groups_v3(lines, image):
        h, w = image.shape[:2]
        keys = list(lines.keys())
        best_lines = None
        best_points = None
        best_score = -np.inf

        for combo in itertools.combinations(keys, 4):
            candidate = {k: lines[k] for k in combo}

            points = TableTopViewProcessor.get_intersections(candidate, h, w)
            if len(points) != 4:
                continue

            score = TableTopViewProcessor.rectangle_score(points, h, w, image)
            if score is None:
                continue

            # print(f"combo={combo}, score={score:.4f}")

            if score > best_score:
                best_score = score
                best_lines = candidate
                best_points = points

        return best_lines, best_points, best_score

    @staticmethod
    def line_intersection_infinite(l1, l2, eps=1e-9):
        x1, y1, x2, y2 = map(float, l1[0])
        x3, y3, x4, y4 = map(float, l2[0])

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < eps:
            return None

        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
        return float(px), float(py)

    @staticmethod
    def get_intersections(lines, h, w):
        if isinstance(lines, dict):
            line_segments = [seg[0] for seg in lines.values()]
        else:
            line_segments = [seg[0] for seg in lines]

        raw_points = []
        for l1, l2 in itertools.combinations(line_segments, 2):
            p = TableTopViewProcessor.line_intersection_infinite(l1, l2)
            if p is None:
                continue
            x, y = p

            if -250 <= x <= (w - 1) + 250 and -250 <= y <= (h - 1) + 250:
                raw_points.append((x, y))

        intersection_points = np.array(list(raw_points), dtype=np.float32)
        return intersection_points

    @staticmethod
    def order_inner_outer_lines(inner_lines, outer_lines, center):
        inner_lines_keys = np.array(list(inner_lines.keys()))
        outer_lines_keys = np.array(list(outer_lines.keys()))
        centers_inner = []
        for i in inner_lines.values():
            x1, y1, x2, y2 = i[0][0]
            centers_inner.append([(x1 + x2) / 2, (y1 + y2) / 2])
        centers_inner = np.array(centers_inner)

        center_pts = np.mean(centers_inner, axis=0)
        angles = np.arctan2(centers_inner[:, 1] - center_pts[1], centers_inner[:, 0] - center_pts[0])
        order = np.argsort(angles)

        inner_lines_ordered = {inner_lines_keys[i]: inner_lines[inner_lines_keys[i]] for i in order}
        outer_lines_ordered = {outer_lines_keys[i]: outer_lines[outer_lines_keys[i]] for i in order}

        return inner_lines_ordered, outer_lines_ordered, angles, order

    @staticmethod
    def get_intersections_ordered(lines, h, w):
        """
        lines are ordered by clockwise angle around the center
        """
        if isinstance(lines, dict):
            line_segments = [seg[0] for seg in lines.values()]
        else:
            line_segments = [seg[0] for seg in lines]

        raw_points = []
        for l1, l2 in zip(line_segments, line_segments[1:] + line_segments[:1]):
            p = TableTopViewProcessor.line_intersection_infinite(l1, l2)
            if p is None:
                continue
            x, y = p

            pad = 250
            if -pad <= x <= (w - 1) + pad and -pad <= y <= (h - 1) + pad:
                raw_points.append((x, y))

        intersection_points = np.array(list(raw_points), dtype=np.float32)
        return intersection_points

    @staticmethod
    def order_points_by_angle(points_a, points_b):
        """
        Order points in points_a by angle from center, 
        and find closest point in points_b for each point in points_a
        """
        center_pts = np.mean(points_a, axis=0)

        angles = np.arctan2(points_a[:, 1] - center_pts[1], points_a[:, 0] - center_pts[0])

        order = np.argsort(angles)
        points_a_ord = points_a[order]
        points_b_ord = []

        for i, (x1, y1) in enumerate(points_a_ord):
            min_dist = float('inf')
            closest_point = None
            for j, (x2, y2) in enumerate(points_b):
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (x2, y2)
            points_b_ord.append(closest_point)
            # print(f"Point {i}: ({x1:.2f}, {y1:.2f}), closest Point: ({closest_point[0]:.2f}, {closest_point[1]:.2f}), distance: {min_dist:.2f}")

        return points_a_ord, points_b_ord, angles, order

    @staticmethod
    def got_angle(points) -> tuple[bool, bool, bool, tuple]:
        """
        (t1, t2, b1, b2), where t1, t2 are top points, b1, b2 are bottom points
        """
        is_top, is_front, is_angle = False, False, False

        points_by_y = points[np.argsort(points[:, 1])]
        t1, t2 = points_by_y[:2]
        b1, b2 = points_by_y[2:4]

        t_w = abs(t1[0] - t2[0])
        b_w = abs(b1[0] - b2[0])
        # print(f"[DEBUG] Top width: {t_w:.2f}, Bottom width: {b_w:.2f}, difference: {abs(t_w - b_w):.2f}")

        # in top-view top_width and bottom_width should be close
        if abs(t_w - b_w) < 30:
            is_top = True
            return is_top, is_front, is_angle, (t1, t2, b1, b2)

        # in front-view top_line and bottom_line are parallel
        dy1 = abs(t1[1] - b1[1])
        dy2 = abs(t2[1] - b2[1])
        # print(f"[DEBUG] dy1: {dy1:.2f}, dy2: {dy2:.2f}, difference: {abs(dy1 - dy2):.2f}")

        if abs(dy1 - dy2) < 20:
            is_front = True
            return is_top, is_front, is_angle, (t1, t2, b1, b2)

        is_angle = True
        return is_top, is_front, is_angle, (t1, t2, b1, b2)

    @staticmethod
    def warp_table_variant(image, pts, pad=0, w=2000, h=2000, inner_color=(0, 255, 0), pad_color=(255, 0, 0)):
        pts = np.array(pts, dtype=np.float32)

        tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
        rect = np.array([tl, tr, br, bl], dtype=np.float32)

        out_w = w
        out_h = h

        dst = np.array([
            [pad, pad],
            [out_w - 1 - pad, pad],
            [out_w - 1 - pad, out_h - 1 - pad],
            [pad, out_h - 1 - pad]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)

        if image.ndim == 2:
            zero_border = 0
        else:
            zero_border = tuple([0] * image.shape[2])

        warped = cv2.warpPerspective(
            image,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=zero_border
        )

        src_valid = np.ones(image.shape[:2], dtype=np.uint8) * 255
        valid_mask = cv2.warpPerspective(
            src_valid,
            M,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        ) > 0

        x0, y0 = pad, pad
        x1, y1 = out_w - pad, out_h - pad

        inner_mask = np.zeros((out_h, out_w), dtype=bool)
        inner_mask[y0:y1, x0:x1] = True

        if image.ndim == 2:
            result = np.full((out_h, out_w), pad_color[0], dtype=image.dtype)
            result[valid_mask] = warped[valid_mask]
            result[inner_mask & (~valid_mask)] = inner_color[0]
        else:
            result = np.full((out_h, out_w, image.shape[2]), pad_color, dtype=image.dtype)
            result[valid_mask] = warped[valid_mask]
            result[inner_mask & (~valid_mask)] = inner_color

        # keep result horizontal
        if result.shape[0] > result.shape[1]:
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

        return result, pts

    @staticmethod
    def find_dot(image, lower_blue, upper_blue):
        has_dot = False
        has_boll = False
        is_empty = False

        h, w, _ = image.shape
        patch_area = h * w

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        lower_blue = np.array([
            lower_blue[0],
            lower_blue[1],
            lower_blue[2] + 30
        ])
        upper_blue = np.array([
            upper_blue[0],
            upper_blue[1],
            upper_blue[2]
        ])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.count_nonzero(blue_mask) / patch_area

        debug_img = image.copy()
        if debug_img.dtype != np.uint8:
            debug_img = np.clip(debug_img, 0, 255).astype(np.uint8)

        candidates_info = []
        result_countours = []

        # almost entirely blue => nothing here
        if blue_ratio > 0.99:
            return False, False, True, debug_img, candidates_info, result_countours

        # non-blue regions = candidates
        non_blue = cv2.bitwise_not(blue_mask)

        # clean noise but keep blurred objects
        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        non_blue = cv2.morphologyEx(non_blue, cv2.MORPH_OPEN, k3, iterations=1)
        non_blue = cv2.morphologyEx(non_blue, cv2.MORPH_CLOSE, k5, iterations=1)

        # white center of the dot: low saturation + high value
        white_mask = np.zeros((h, w), dtype=np.uint8)
        white_mask[(hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 150)] = 255

        contours, _ = cv2.findContours(non_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 80:
                continue

            result_countours.append(cnt)

            x, y, bw, bh = cv2.boundingRect(cnt)
            touches_edge = (x <= 1 or y <= 1 or x + bw >= w - 2 or y + bh >= h - 2)

            # TODO: make decision here
            if touches_edge:
                has_boll = True
            if not touches_edge and area > 2400:
                has_boll = True
            if not touches_edge and area < 2400:
                has_dot = True

        return has_dot, has_boll, is_empty, debug_img, candidates_info, result_countours

    def get_top_view(self):
        image = cv2.imread(str(self.full_img_path))
        scale_factor = 3360 / image.shape[1]
        new_dimensions = (3360, int(image.shape[0] * scale_factor))
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_blur = cv2.GaussianBlur(image, (15, 15), 0)

        hsv_image = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

        lower_blue, upper_blue, _ = TableTopViewProcessor.get_upper_lower_blue(hsv_image, hue_window=1.5)

        mask_table = cv2.inRange(hsv_image, lower_blue, upper_blue)

        kernel = np.ones((7, 7), np.uint8)
        mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_OPEN, kernel)
        mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_CLOSE, kernel)

        mask_table, _ = TableTopViewProcessor.keep_only_close_contours(mask_table, distance_threshold=50)

        lines = TableTopViewProcessor.get_lines(mask_table, min_length=200)

        edge_points = [line[0] for line in lines]
        center = [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2) in edge_points]
        center = np.mean(center, axis=0).astype(int)

        inner_lines, outer_lines = TableTopViewProcessor.get_inner_outer_lines(lines, center, angle_threshold=0.07, distance_threshold=80)

        if len(outer_lines) < 4 or len(inner_lines) < 4:
            print("[DEBUG] Not enough lines found, retrying with wider hue window.")
            lower_blue, upper_blue, _ = TableTopViewProcessor.get_upper_lower_blue(hsv_image, hue_window=5)
            mask_table = cv2.inRange(hsv_image, lower_blue, upper_blue)
            lines = TableTopViewProcessor.get_lines(mask_table, min_length=250)
            inner_lines, outer_lines = TableTopViewProcessor.get_inner_outer_lines(lines, center, angle_threshold=0.07, distance_threshold=80)

        if len(outer_lines) > 4 or len(inner_lines) > 4:
            selected_inner = TableTopViewProcessor.select_4_groups_v3(inner_lines, mask_table)[0]
            selected_outer = TableTopViewProcessor.select_4_groups_v3(outer_lines, mask_table)[0]

            if selected_inner is not None and selected_outer is not None:
                inner_lines = selected_inner
                outer_lines = selected_outer

        inner_lines, outer_lines, _, _ = TableTopViewProcessor.order_inner_outer_lines(inner_lines, outer_lines, center)

        intersection_points_outer = TableTopViewProcessor.get_intersections_ordered(outer_lines, image.shape[0], image.shape[1])
        intersection_points_inner = TableTopViewProcessor.get_intersections_ordered(inner_lines, image.shape[0], image.shape[1])

        if len(intersection_points_outer) != 4:
            intersection_points_outer = sorted(
                intersection_points_outer, 
                key=lambda p: math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)
            )
            intersection_points_outer = np.array(intersection_points_outer[:4], dtype=np.float32)
        if len(intersection_points_inner) != 4:
            intersection_points_inner = sorted(
                intersection_points_inner, 
                key=lambda p: math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)
            )
            intersection_points_inner = np.array(intersection_points_inner[:4], dtype=np.float32)


        if len(intersection_points_outer) > 4 or len(intersection_points_inner) > 4:
            print("[DEBUG] Too many intersection points found.")

        pts_ordered_outer, pts_ordered_inner, _, _ = TableTopViewProcessor.order_points_by_angle(
            intersection_points_outer, intersection_points_inner)

        is_top, is_front, is_angle, (t1, t2, b1, b2) = TableTopViewProcessor.got_angle(pts_ordered_outer)
        start_point = None
        if is_top:
            start_point = t1 if t1[0] < t2[0] else t2
        elif is_front:
            start_point = t1 if t1[0] > t2[0] else t2
        else:
            start_point = t1 if t1[1] < t2[1] else t2

        start_by_angle = np.where((pts_ordered_outer == start_point).all(axis=1))[0][0]

        img_warped = TableTopViewProcessor.warp_table_variant(
            image, 
            pts_ordered_inner, 
            pad=0, w=2000, h=2000,
            inner_color=(74, 158, 229), pad_color=(85, 90, 85)
        )[0]

        w, h = img_warped.shape[1], img_warped.shape[0]
        pad = 80
        centers = [
            (0.75 * w, 0.5 * h),
            (0.5 * w, 0.75 * h),
            (0.25 * w, 0.5 * h),
            (0.5 * w, 0.25 * h)
        ]
        sub_img_warped = []
        for i, center in enumerate(centers):
            sub_img_warped.append(img_warped[int(center[1]-pad):int(center[1]+pad), int(center[0]-pad):int(center[0]+pad)])

        dots = []
        for i in range(4):
            has_dot, has_boll, is_empty, _, _, countours = TableTopViewProcessor.find_dot(sub_img_warped[i], lower_blue, upper_blue)
            total_area = 0
            for cnt in countours:
                area = cv2.contourArea(cnt)
                total_area += area
            dots.append((has_dot, has_boll, is_empty, total_area))

        start_by_dot = None
        start_by_dot_and_ball = None
        start_by_ball = None

        max_area = -1
        for i, (has_dot, has_boll, is_empty, total_area) in enumerate(dots):
            if is_empty:
                continue
            if has_dot and not has_boll:
                start_by_dot = i
            if has_dot and has_boll:
                start_by_dot_and_ball = i
            if has_boll:
                if total_area > max_area:
                    max_area = total_area
                    start_by_ball = i

        if start_by_dot is not None:
            start = start_by_dot
        elif start_by_dot_and_ball is not None:
            start = start_by_dot_and_ball
        elif start_by_ball is not None:
            start = start_by_ball
        else:
            start = start_by_angle

        if start != start_by_angle:
            print(f"[DEBUG] {start_by_angle=}, {start=}.")

        pts_ordered_outer_correct = np.roll(pts_ordered_outer, -start, axis=0)

        img_warped = TableTopViewProcessor.warp_table_variant(
            image,
            pts_ordered_outer_correct,
            pad=75, w=1600, h=1000,
            inner_color=(74, 158, 229), pad_color=(85, 90, 85)
        )[0]

        return img_warped


class BallsDetector:

    REFERENCE_COLORS = { 
        "Yellow": np.array([214, 147, 65]),
        "Blue": np.array([24, 67, 115]),
        "Blue_Dark": np.array([25, 64, 100]),
        "Red": np.array([180, 55, 61]),
        "Red_Dark": np.array([138, 52, 56]),
        "Purple": np.array([74, 74, 111]),
        "Orange": np.array([180, 86, 50]),
        "Orange_Bright": np.array([235, 110, 53]),
        "Green":np.array([40, 92, 85]),
        "Brown": np.array([123, 79, 53]),
        "Brown_alt": np.array([80, 78, 62]),
        "White": np.array([247, 250, 187]),
        "Black": np.array([9, 32, 48]),
    }

    SOLID_NUMBERS = {
    "Yellow": 1, "Blue": 2, "Red": 3, "Red_Dark": 3,
    "Purple": 4, "Orange": 5, "Orange_Bright": 5,
    "Green": 6, "Brown": 7, "White": 0, "Black": 8, "Blue_Dark": 2, "Brown_alt": 7
    }
    STRIPED_NUMBERS = {
        "Yellow": 9, "Blue": 10, "Red": 11, "Red_Dark": 11,
        "Purple": 12, "Orange": 13, "Orange_Bright": 13,
        "Green": 14, "Brown": 15, "Blue_Dark": 10, "Brown_alt": 15
    }
    
    def __init__(self, full_image_path: Path, image_path: Path):
        self.image_path = image_path
        self.full_img_path = full_image_path
        self.image_w = None
        self.image_h = None

        # Images
        self.hsv_image: Any = None
        self.hsv_image_shadow_masked: Any = None
        
        # Median values
        self.h_med: Any = None
        self.s_med: Any = None
        self.v_med: Any = None
        
        # Angles
        self.is_top: Any = None
        self.is_front: Any = None
        self.is_angle: Any = None      
        self.is_angle_r: Any = None
        self.is_angle_l: Any = None
        
        # Lines
        self.top_line_outer_full: Any = None
        self.top_line_inner_full: Any = None
        self.side_line_l_inner_full: Any = None
        self.side_line_l_outer_full: Any = None
        self.side_line_r_inner_full: Any = None
        self.side_line_r_outer_full: Any = None
        self.bot_line_inner_full: Any = None
        
        # Corner points
        self.tl_inner: Any = None
        self.tl_outer: Any = None
        self.tr_inner: Any = None
        self.tr_outer: Any = None
        self.br_inner: Any = None
        self.br_outer: Any = None
        self.bl_inner: Any = None
        self.bl_outer: Any = None

    @staticmethod
    def get_center_mode_hsv(hsv_img, area_ratio=0.20):
        """
        Crops the center of the image to the specified area ratio 
        and returns the most frequent H, S, and V values.
        """
        h, w = hsv_img.shape[:2]
    
        scale = np.sqrt(area_ratio)
        crop_w = int(w * scale)
        crop_h = int(h * scale)
        cx, cy = w // 2, h // 2
        start_x = cx - (crop_w // 2)
        start_y = cy - (crop_h // 2)
        center_crop = hsv_img[start_y:start_y + crop_h, start_x:start_x + crop_w]
        
        H, S, V = cv2.split(center_crop)
        
        h_mode = np.argmax(np.bincount(H.flatten()))
        s_mode = np.argmax(np.bincount(S.flatten()))
        v_mode = np.argmax(np.bincount(V.flatten()))
        
        return int(h_mode), int(s_mode), int(v_mode)

    @staticmethod
    def make_line(p1, p2):
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)

        A = p1[1] - p2[1]
        B = p2[0] - p1[0]
        C = p1[0]*p2[1] - p2[0]*p1[1]

        return A, B, C
    
    @staticmethod
    def is_above_line(pt, line):
        x, y = pt
        A, B, C = line
        y_line = -(A * x + C) / B
        return y < y_line
    
    @staticmethod
    def calculate_overlap_area(c1, c2):
        """Calculates the exact area of intersection between two circles."""
        x1, y1, r1 = c1
        x2, y2, r2 = c2
        d = math.hypot(x2 - x1, y2 - y1)
        
        #No overlap
        if d >= r1 + r2:
            return 0.0
            
        # One circle is inside the other
        if d <= abs(r1 - r2):
            return math.pi * min(r1, r2)**2
            
        #  Partial overlap 
        r1_sq, r2_sq = r1**2, r2**2
        d_sq = d**2
        
        #  areas
        part1 = r1_sq * math.acos((d_sq + r1_sq - r2_sq) / (2 * d * r1))
        part2 = r2_sq * math.acos((d_sq + r2_sq - r1_sq) / (2 * d * r2))
        
        # Triangle areas to subtract
        part3 = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        
        return part1 + part2 - part3
    
    @staticmethod
    def classify_ball(hsv_img, rgb_img, cx, cy, r):
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r * 0.86), 255, -1)
        
        hsv_pixels = hsv_img[mask == 255]
        rgb_pixels = rgb_img[mask == 255]
        if len(hsv_pixels) == 0:
            return "Unknown"

        white_rgb_unit8 = np.uint8([[BallsDetector.REFERENCE_COLORS["White"]]])
        white_hsv_sample = cv2.cvtColor(white_rgb_unit8, cv2.COLOR_RGB2HSV)[0][0]
        is_white = (hsv_pixels[:, 1] < (white_hsv_sample[1] + 45)) & (hsv_pixels[:, 2] > 130) 
        white_ratio = np.sum(is_white) / len(hsv_pixels)    

        # BLACK
        black_ref = BallsDetector.REFERENCE_COLORS["Black"]
        dists_to_black = np.linalg.norm(rgb_pixels - black_ref, axis=1)
        is_black = dists_to_black < 45
        black_ratio = np.sum(is_black) / len(rgb_pixels)


       # print(f"Ball at ({cx}, {cy}) | White Ratio: {white_ratio:.4f} | Black Ratio: {black_ratio:.4f}")

        if white_ratio > 0.75:
            return 0
        elif black_ratio > 0.50:
            return 8
        elif white_ratio < 0.2:
            best_color = "Unknown"
            min_dist = float('inf')

            mean_rgb = np.mean(rgb_pixels, axis=0)
            median_rgb = np.median(rgb_pixels, axis=0)

         #   print(f"Ball at ({cx}, {cy})")
         #   print(f"  -> MEAN sees:   {mean_rgb}")
         #   print(f"  -> MEDIAN sees: {median_rgb}")
        #  print(f"Ball at ({cx}, {cy}) actual median RGB is: {median_rgb}")
            for color_name, ref_color in BallsDetector.REFERENCE_COLORS.items():
                if color_name in ["White", "Black"]:
                    continue         
                dist = np.linalg.norm(median_rgb - ref_color)
                if dist < min_dist:
                    min_dist = dist
                    best_color = color_name
            SOLID_NUMBERS = {
                "Yellow": 1,
                "Blue": 2,
                "Red": 3,
                "Red_Dark": 3,
                "Purple": 4,
                "Orange": 5,
                "Orange_Bright": 5, 
                "Green": 6,
                "Brown": 7
            }

            # Return the solid number, or "Unknown" if it somehow missed
            return SOLID_NUMBERS.get(best_color, "Unknown")

        else:
            non_white_rgb = rgb_pixels[~is_white]
            if len(non_white_rgb) == 0:
                return "Unknown"
            
            stripe_median_rgb = np.median(non_white_rgb, axis=0) 
            best_stripe_color = "Unknown"
            min_dist = float('inf')   
            for color_name, ref_color in BallsDetector.REFERENCE_COLORS.items():
                if color_name in ["White", "Black"]:
                    continue         
                dist = np.linalg.norm(stripe_median_rgb - ref_color)
                if dist < min_dist:
                    min_dist = dist
                    best_stripe_color = color_name        

            STRIPED_NUMBERS = {
                "Yellow": 9,
                "Blue": 10,
                "Red": 11,        
                "Red_Dark": 11,
                "Purple": 12,
                "Orange": 13,
                "Orange_Bright": 13,
                "Green": 14,
                "Brown": 15
            }

            return STRIPED_NUMBERS.get(best_stripe_color, "Unknown") 

    def shadow_mask_borders(self):
        cloth_color = np.array([self.h_med, self.s_med, self.v_med], dtype=np.float32)
        threshold = 90 
        height, width = self.hsv_image.shape[:2]
        h_channel = self.hsv_image[:, :, 0]
        s_channel = self.hsv_image[:, :, 1]
        v_channel = self.hsv_image[:, :, 2]

        #Defines sampled values for shadows and cloth with bigger tolerances
        is_dark_shadow = (        
            (h_channel >= 95) & (h_channel <= 115) &  
            (s_channel > 184) &                        
            (v_channel < 114)                          
        )   

        h_tol = 6
        s_tol_up = 10
        s_tol_down = 40
        v_tol_up = 50
        v_tol_down = 200

        lower_blue = np.array([self.h_med - h_tol, self.s_med - s_tol_down, self.v_med - v_tol_down])
        upper_blue = np.array([self.h_med + h_tol, self.s_med + s_tol_up, self.v_med + v_tol_up])

        lower_blue = np.clip(lower_blue, [0, 0, 0], [179, 255, 255])
        upper_blue = np.clip(upper_blue, [0, 0, 0], [179, 255, 255])

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)

        #Defines cut regions between top and side lines (in case of angled views)
        A, B, C = self.top_line_inner_full
        y_line = -(A * coords[:, 0] + C) / B
        mask_line_flat = coords[:, 1] < y_line
        mask_line = mask_line_flat.reshape(height, width)

        mask_x = (xx >= self.tl_outer[0]) & (xx <= self.tr_outer[0])
        top_spatial_region = mask_line & mask_x
        final_side_mask = np.zeros((height, width), dtype=bool)

        if self.is_angle_l:
         #   print("Table is leaning left.")
            A_out, B_out, C_out = self.side_line_r_outer_full
            A_in, B_in, C_in = self.side_line_r_inner_full

            x_line_r_outer = -(B_out * coords[:, 1] + C_out) / A_out
            x_line_r_inner = -(B_in * coords[:, 1] + C_in) / A_in
            
            mask_side_combined_flat = (coords[:, 0] > x_line_r_inner) & (coords[:, 0] < x_line_r_outer)
            final_side_mask = mask_side_combined_flat.reshape(height, width)

        elif self.is_angle_r:
         #   print("Table is leaning right.")
            A_out, B_out, C_out = self.side_line_l_outer_full
            A_in, B_in, C_in = self.side_line_l_inner_full

            x_line_l_outer = -(B_out * coords[:, 1] + C_out) / A_out
            x_line_l_inner = -(B_in * coords[:, 1] + C_in) / A_in
            
            mask_side_combined_flat = (coords[:, 0] > x_line_l_outer) & (coords[:, 0] < x_line_l_inner)
            final_side_mask = mask_side_combined_flat.reshape(height, width)

        #Replace this regions by uniform blue (cloth color) 
        all_spatial_regions = top_spatial_region | final_side_mask
        diff = np.linalg.norm(self.hsv_image.astype(np.float32) - cloth_color, axis=2)
        mask_color_old = diff <= threshold
        mask_strict_new = cv2.inRange(self.hsv_image, lower_blue, upper_blue) > 0
        all_valid_colors = is_dark_shadow | mask_color_old | mask_strict_new
        mask_final = all_spatial_regions & all_valid_colors

        self.hsv_image_shadow_masked = self.hsv_image.copy()
        self.hsv_image_shadow_masked[mask_final] = [self.h_med, self.s_med, self.v_med]

        return self.hsv_image_shadow_masked

    def patch_corner_holes(self):
        distance_tl = np.linalg.norm(self.tl_inner - self.tl_outer)
        distance_tr = np.linalg.norm(self.tr_inner - self.tr_outer)
        distance_br = np.linalg.norm(self.br_inner - self.br_outer)
        distance_bl = np.linalg.norm(self.bl_inner - self.bl_outer)

        tl_hole=(self.tl_inner+self.tl_outer)/2
        tr_hole=(self.tr_inner+self.tr_outer)/2
        br_hole=(self.br_inner+self.br_outer)/2
        bl_hole=(self.bl_inner+self.bl_outer)/2

        circle_hsv_color = (self.h_med, self.s_med, self.v_med) 

        tl = (int(tl_hole[0]), int(tl_hole[1]))
        tr = (int(tr_hole[0]), int(tr_hole[1]))
        br = (int(br_hole[0]), int(br_hole[1]))
        bl = (int(bl_hole[0]), int(bl_hole[1]))

        cv2.circle(self.hsv_image_shadow_masked, tl, int(distance_tl/2), circle_hsv_color, 1) #just line for top
        cv2.circle(self.hsv_image_shadow_masked, tr, int(distance_tr/2), circle_hsv_color, 1)
        cv2.circle(self.hsv_image_shadow_masked, br, int(distance_br/2)+10, circle_hsv_color, -1) #fill for bottom
        cv2.circle(self.hsv_image_shadow_masked, bl, int(distance_bl/2)+10, circle_hsv_color, -1)

        height, width = self.hsv_image_shadow_masked.shape[:2]
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)

        # Line Mask 
        A, B, C = self.top_line_inner_full
        y_line = -(A * coords[:, 0] + C) / B
        mask_line_flat = coords[:, 1] < y_line 

        # Circle Masks
        # Calculate circle radius based on distances between inner and outer points, with some padding
        radius_tl = int(distance_tl/2) +25
        radius_tr = int(distance_tr/2) + 12

        # Create boolean masks for points inside the top-left and top-right circles
        mask_circle_tl_flat = (coords[:, 0] - tl[0])**2 + (coords[:, 1] - tl[1])**2 <= radius_tl**2
        mask_circle_tr_flat = (coords[:, 0] - tr[0])**2 + (coords[:, 1] - tr[1])**2 <= radius_tr**2

        mask_top_circles_flat = mask_circle_tl_flat | mask_circle_tr_flat

        # Combine Line and Circle Conditions
        # The point must be above the line and inside one of the top circles
        final_mask_flat = mask_line_flat & mask_top_circles_flat
        final_mask = final_mask_flat.reshape(height, width)

        self.hsv_image_shadow_masked[final_mask] = [self.h_med, self.s_med, self.v_med]

        #patch holes in top view
        if self.is_top:
            cv2.circle(self.hsv_image_shadow_masked, (int(self.tl_outer[0]), int(self.tl_outer[1])), int(distance_tl/2)+50, circle_hsv_color, -1)
            cv2.circle(self.hsv_image_shadow_masked, (int(self.tr_outer[0]), int(self.tr_outer[1])), int(distance_tr/2)+50, circle_hsv_color, -1)
            cv2.circle(self.hsv_image_shadow_masked, (int(self.br_outer[0]), int(self.br_outer[1])), int(distance_br/2)+50, circle_hsv_color, -1)
            cv2.circle(self.hsv_image_shadow_masked, (int(self.bl_outer[0]), int(self.bl_outer[1])), int(distance_bl/2)+50, circle_hsv_color, -1)
            mid_point_upper = ((self.tl_outer + self.tr_outer) / 2).astype(int)
            mid_point_lower = ((self.bl_outer + self.br_outer) / 2).astype(int)
            cv2.circle(self.hsv_image_shadow_masked, (mid_point_upper[0], mid_point_upper[1]), int(distance_tl/2)+50, circle_hsv_color, -1)
            cv2.circle(self.hsv_image_shadow_masked, (mid_point_lower[0], mid_point_lower[1]), int(distance_tl/2)+50, circle_hsv_color, -1)
        
        #patch holes and sides in front view
        if self.is_front:
            A_out, B_out, C_out = self.side_line_r_outer_full
            A_in, B_in, C_in = self.side_line_r_inner_full

            y_line_r_outer = -(A_out * coords[:, 0] + C_out) / B_out
            y_line_r_inner = -(A_in * coords[:, 0] + C_in) / B_in
            
            A_out_l, B_out_l, C_out_l = self.side_line_l_outer_full
            A_in_l, B_in_l, C_in_l = self.side_line_l_inner_full

            y_line_l_outer = -(A_out_l * coords[:, 0] + C_out_l) / B_out_l
            y_line_l_inner = -(A_in_l * coords[:, 0] + C_in_l) / B_in_l

            mask_above_inner_flat_l = coords[:, 1] < y_line_l_inner
            mask_below_outer_flat_l = coords[:, 1] > y_line_l_outer

            mask_above_inner_flat = coords[:, 1] < y_line_r_inner
            mask_below_outer_flat = coords[:, 1] > y_line_r_outer
            
            mask_side_combined_flat = (mask_above_inner_flat & mask_below_outer_flat) | (mask_above_inner_flat_l & mask_below_outer_flat_l) 
            final_side_mask = mask_side_combined_flat.reshape(height, width)

            self.hsv_image_shadow_masked[final_side_mask] = [self.h_med, self.s_med, self.v_med]

        return self.hsv_image_shadow_masked
    
    @staticmethod
    def assign_white_and_black_balls(hsv_img, rgb_img, final_balls, reference_colors, white_thresh=0.75, black_thresh=0.50):
        assignment_colors = {}
        
        white_rgb_uint8 = np.uint8([[reference_colors["White"]]])
        white_hsv_sample = cv2.cvtColor(white_rgb_uint8, cv2.COLOR_RGB2HSV)[0][0]
        black_ref = reference_colors["Black"]

        best_white_ratio = 0
        best_white_ball = -1
        
        best_black_ratio = 0
        best_black_ball = -1

        for (cx, cy, r) in final_balls:
            mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx, cy), int(r * 0.86), 255, -1)
            
            hsv_pixels = hsv_img[mask == 255]
            rgb_pixels = rgb_img[mask == 255]
            
            if len(hsv_pixels) == 0:
                continue

            # --- WHITE CHECK ---
            is_white = (hsv_pixels[:, 1] < (white_hsv_sample[1] + 45)) & (hsv_pixels[:, 2] > 130)
            white_ratio = np.sum(is_white) / len(hsv_pixels)    

            # --- BLACK CHECK ---
            dists_to_black = np.linalg.norm(rgb_pixels - black_ref, axis=1)
            is_black = dists_to_black < 45
            black_ratio = np.sum(is_black) / len(rgb_pixels)

            if white_ratio > best_white_ratio:
                best_white_ratio = white_ratio
                best_white_ball = {
                    "cx": cx, "cy": cy, "r": r,
                    "ratio": float(white_ratio),
                    "distance": 0.0  
                }

            if black_ratio > best_black_ratio:
                best_black_ratio = black_ratio
                best_black_ball = {
                    "cx": cx, "cy": cy, "r": r,
                    "ratio": float(black_ratio),
                    "distance": 0.0   
                }

        if best_white_ball and best_white_ratio > white_thresh:
            assignment_colors["White"] = best_white_ball
            
        if best_black_ball and best_black_ratio > black_thresh:
            assignment_colors["Black"] = best_black_ball

        return assignment_colors

    @staticmethod
    def map_balls_to_color_distances(hsv_img, rgb_img, final_balls, reference_colors, solid_dict, striped_dict):
        COLOR_ALIASES = {
            "Orange_Bright": "Orange",
            "Red_Dark": "Red",
            "Brown_alt": "Brown",
            "Blue_Dark": "Blue"
        }

        white_rgb_uint8 = np.uint8([[reference_colors["White"]]])
        white_hsv_sample = cv2.cvtColor(white_rgb_uint8, cv2.COLOR_RGB2HSV)[0][0]

        for (cx, cy, r) in final_balls:
            mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx, cy), int(r * 0.86), 255, -1)
            
            hsv_pixels = hsv_img[mask == 255]
            rgb_pixels = rgb_img[mask == 255]
            
            if len(rgb_pixels) == 0:
                continue
                
            is_white = (hsv_pixels[:, 1] < (white_hsv_sample[1] + 45)) & (hsv_pixels[:, 2] > 130)
            white_ratio = np.sum(is_white) / len(hsv_pixels)

            
            if white_ratio > 0.2:
                non_white_rgb = rgb_pixels[~is_white]
                if len(non_white_rgb) == 0:
                    continue
                stripe_median_rgb = np.median(non_white_rgb, axis=0)
                
                ball_group_distances = {}
                for color_name, ref_color in reference_colors.items():
                    if color_name in ["White", "Black"]:
                        continue
                    
                    dist = float(np.linalg.norm(stripe_median_rgb - ref_color))
                    main_color_name = COLOR_ALIASES.get(color_name, color_name)
                    
                    if main_color_name not in ball_group_distances or dist < ball_group_distances[main_color_name]:
                        ball_group_distances[main_color_name] = dist
                
                for main_color_name, best_dist in ball_group_distances.items():
                    if main_color_name in striped_dict:
                        striped_dict[main_color_name].append({
                            "cx": cx, "cy": cy, "r": r, "distance": best_dist
                        })
            else:
                median_rgb = np.median(rgb_pixels, axis=0)
                
                ball_group_distances = {}
                for color_name, ref_color in reference_colors.items():
                    if color_name in ["White", "Black"]:
                        continue
                    
                    dist = float(np.linalg.norm(median_rgb - ref_color))
                    main_color_name = COLOR_ALIASES.get(color_name, color_name)
                    
                    if main_color_name not in ball_group_distances or dist < ball_group_distances[main_color_name]:
                        ball_group_distances[main_color_name] = dist
                
                for main_color_name, best_dist in ball_group_distances.items():
                    if main_color_name in solid_dict:
                        solid_dict[main_color_name].append({
                            "cx": cx, "cy": cy, "r": r, "distance": best_dist
                        })

        for d in [solid_dict, striped_dict]:
            for color in d:
                d[color].sort(key=lambda b: b["distance"])

        return solid_dict, striped_dict

    @staticmethod
    def build_assignment_map(solid_results, striped_results, SOLID_NUMBERS, STRIPED_NUMBERS): #Greedy method
        assignment_map = {}
        def process_group(results_dict, number_map):
            candidates = []
            for color, balls in results_dict.items():
                if color not in number_map:
                    continue

                ball_number = number_map[color]

                for ball in balls:
                    candidates.append((
                        ball["distance"],
                        ball_number,
                        ball
                    ))
            candidates.sort(key=lambda x: x[0])

    
            for dist, number, ball in candidates:
                if number not in assignment_map:
                    assignment_map[number] = ball

        process_group(solid_results, SOLID_NUMBERS)
        process_group(striped_results, STRIPED_NUMBERS)
        return assignment_map
    
    @staticmethod
    def build_first_ball_map(solid_results, striped_results, SOLID_NUMBERS, STRIPED_NUMBERS): #GREEDY TAKE THE BEST MATCH FIRST SORTED BY DISTANCE
        assignment_map = {}
        

        used_balls = set() 

        def process_group(results_dict, number_map):
            all_potential_matches = []
            for color, balls in results_dict.items():
                if color not in number_map:
                    continue
                for ball in balls:
                    all_potential_matches.append({
                        "color": color,
                        "distance": ball["distance"],
                        "ball_data": ball
                    })

            all_potential_matches.sort(key=lambda x: x["distance"])

            assigned_colors = set()

            for match in all_potential_matches:
                color = match["color"]
                ball = match["ball_data"]
                ball_coords = (ball["cx"], ball["cy"])

                if color in assigned_colors:
                    continue
                if ball_coords in used_balls:
                    continue

                ball_number = number_map[color]
                assignment_map[ball_number] = ball
                assigned_colors.add(color)
                used_balls.add(ball_coords)

        process_group(solid_results, SOLID_NUMBERS)
        process_group(striped_results, STRIPED_NUMBERS)

        return assignment_map
    @staticmethod
    def build_optimal_ball_map(solid_results, striped_results, SOLID_NUMBERS, STRIPED_NUMBERS): #HUNGARIAN ALGORITHM VERSION
        assignment_map = {}
        used_balls = set() 

        def process_group(results_dict, number_map):

            target_colors = [c for c in number_map.keys() if c in results_dict and results_dict[c]]
            if not target_colors:
                return

            unique_balls_dict = {}
            for color in target_colors:
                for b in results_dict[color]:
                    coords = (b["cx"], b["cy"])
                    if coords not in used_balls:
                        unique_balls_dict[coords] = b
                        
            unique_ball_coords = list(unique_balls_dict.keys())
            if not unique_ball_coords:
                return

            cost_matrix = np.full((len(target_colors), len(unique_ball_coords)), 1000.0)

            for i, color in enumerate(target_colors):
                for b in results_dict[color]:
                    coords = (b["cx"], b["cy"])
                    if coords in unique_ball_coords:
                        j = unique_ball_coords.index(coords)
                        cost_matrix[i, j] = b["distance"]

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_ind, col_ind):

                if cost_matrix[i, j] < 1000.0:
                    color = target_colors[i]
                    ball_number = number_map[color]
                    coords = unique_ball_coords[j]
                    
                    assignment_map[ball_number] = unique_balls_dict[coords]
                    used_balls.add(coords)

        process_group(solid_results, SOLID_NUMBERS)
        process_group(striped_results, STRIPED_NUMBERS)
        return assignment_map
    
    @staticmethod
    def get_assignment_data(assignment_map, final_balls):
        coord_to_ids = {}
        for number, ball in assignment_map.items():
            coords = (int(ball["cx"]), int(ball["cy"]))
            if coords not in coord_to_ids:
                coord_to_ids[coords] = []
            coord_to_ids[coords].append(number)
        results = []
        for (cx, cy, r) in final_balls:
            coords = (int(cx), int(cy))
            matches = coord_to_ids.get(coords, [-1])
            for ball_id in matches:
                results.append((ball_id, (int(cx), int(cy)), int(r)))
        return results

    def get_image_result(self) -> ImageResult:
        #image = cv2.imread(self.full_img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #self.hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        image = cv2.imread(str(self.full_img_path))
    
        scale_factor = 3360 / image.shape[1]
        new_dimensions = (3360, int(image.shape[0] * scale_factor))
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        self.image_h, self.image_w = image.shape[:2]


        #Values for second mask
        self.h_med = 103.2
        self.s_med = 167.7   
        self.v_med = 195.2

        # Shadows values sampled
        shadow_h = 106.1
        shadow_s = 202.8
        shadow_v = 97.4

        dark_cloth_h = 101
        dark_cloth_s = 214
        dark_cloth_v = 217


        # Define tolerance around the median values
        h_tol = 8
        s_tol = 20
        v_tol_up = 20      

        v_tol_down = 30

        h_freq, s_freq, v_freq = BallsDetector.get_center_mode_hsv(self.hsv_image, area_ratio=0.20)

       # print(f"Most frequent HSV in the center 20%: H={h_freq}, S={s_freq}, V={v_freq}")
        #lower_blue = np.array([h_freq - h_tol, s_freq- s_tol, v_freq - v_tol_down])
        #upper_blue = np.array([h_freq + h_tol, s_freq + s_tol, v_freq + v_tol_up])
        lower_blue, upper_blue, _ = TableTopViewProcessor.get_upper_lower_blue(self.hsv_image, hue_window=5)
   
        # Clip to valid HSV ranges
        lower_blue = np.clip(lower_blue, [0, 0, 0], [179, 255, 255])
        upper_blue = np.clip(upper_blue, [0, 0, 0], [179, 255, 255])


        mask_table = cv2.inRange(self.hsv_image, lower_blue, upper_blue)

        edges = cv2.Canny(mask_table, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=80,
            minLineLength=100,
            maxLineGap=80
        )

        tmp_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length >= 250:
                tmp_lines.append(line)
        lines = tmp_lines

        pts = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            pts.append((x1, y1))
            pts.append((x2, y2))

        pts = np.array(pts)
        hull = cv2.convexHull(pts)

        #Get lines, edges, corners points, and view angle
        
        lines = TableTopViewProcessor.get_lines(mask_table, min_length=250)
        edge_points = [line[0] for line in lines]
        center = [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2) in edge_points]
        center = np.mean(center, axis=0).astype(int)

        inner_lines, outer_lines = TableTopViewProcessor.get_inner_outer_lines(lines, center, angle_threshold=0.07, distance_threshold=80)

        if len(outer_lines) > 4 or len(inner_lines) > 4:
            selected_inner = TableTopViewProcessor.select_4_groups_v3(inner_lines, mask_table)[0]
            selected_outer = TableTopViewProcessor.select_4_groups_v3(outer_lines, mask_table)[0]

            if selected_inner is not None and selected_outer is not None:
                inner_lines = selected_inner
                outer_lines = selected_outer


        inner_lines, outer_lines, _, _ = TableTopViewProcessor.order_inner_outer_lines(inner_lines, outer_lines, center)

        intersection_points_outer = TableTopViewProcessor.get_intersections_ordered(outer_lines, image.shape[0], image.shape[1])
        intersection_points_inner = TableTopViewProcessor.get_intersections_ordered(inner_lines, image.shape[0], image.shape[1])


        if len(intersection_points_outer) != 4:
            intersection_points_outer = sorted(
                intersection_points_outer, 
                key=lambda p: math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)
            )
            intersection_points_outer = np.array(intersection_points_outer[:4], dtype=np.float32)
        if len(intersection_points_inner) != 4:
            intersection_points_inner = sorted(
                intersection_points_inner, 
                key=lambda p: math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)
            )
            intersection_points_inner = np.array(intersection_points_inner[:4], dtype=np.float32)


        if len(intersection_points_outer) > 4 or len(intersection_points_inner) > 4:
            print("[DEBUG] Too many intersection points found.")

        pts_ordered_outer, pts_ordered_inner, angles, order = TableTopViewProcessor.order_points_by_angle(
            intersection_points_outer, intersection_points_inner)

        self.is_top, self.is_front, self.is_angle, (t1, t2, b1, b2) = TableTopViewProcessor.got_angle(pts_ordered_outer)
       # print(f"Is top view: {self.is_top}, Is front view: {self.is_front}, Is angle view: {self.is_angle}")
       # for i, pt in enumerate(pts_ordered_outer):
            #print(f"Outer Point {i}: {pt}, Angle: {angles[i]:.2f} degrees")

        self.tl_inner = pts_ordered_inner[0]
        self.tr_inner = pts_ordered_inner[1]
        self.br_inner = pts_ordered_inner[2]
        self.bl_inner = pts_ordered_inner[3]
        self.tl_outer = pts_ordered_outer[0]
        self.tr_outer = pts_ordered_outer[1]
        self.br_outer = pts_ordered_outer[2]
        self.bl_outer = pts_ordered_outer[3]

        self.top_line_outer_full = BallsDetector.make_line(self.tl_outer, self.tr_outer)
        self.top_line_inner_full = BallsDetector.make_line(self.tl_inner, self.tr_inner)
        self.side_line_r_outer_full = BallsDetector.make_line(self.tr_outer, self.br_outer)
        self.side_line_r_inner_full = BallsDetector.make_line(self.tr_inner, self.br_inner)
        self.side_line_l_outer_full = BallsDetector.make_line(self.tl_outer, self.bl_outer)
        self.side_line_l_inner_full = BallsDetector.make_line(self.tl_inner, self.bl_inner)
        self.bot_line_inner_full = BallsDetector.make_line(self.bl_inner, self.br_inner)

        self.is_angle_l = self.is_angle and self.tl_outer[0] < self.bl_outer[0]
        self.is_angle_r = self.is_angle and self.tl_outer[0] > self.bl_outer[0]

        self.hsv_image_shadow_masked = self.shadow_mask_borders()
        self.hsv_image_shadow_masked = self.patch_corner_holes()

        rgb_image = cv2.cvtColor(self.hsv_image_shadow_masked, cv2.COLOR_HSV2RGB)

        mask_hull = np.zeros(self.hsv_image_shadow_masked.shape, dtype=np.uint8)
        cv2.fillPoly(mask_hull, [hull], (255, 255, 255))
        masked_image = cv2.bitwise_and(self.hsv_image_shadow_masked, mask_hull)


        dark_cloth_h = 101
        dark_cloth_s = 214
        dark_cloth_v = 217

        dist_to_state = math.hypot(s_freq - self.s_med, v_freq - self.v_med)
        dist_to_dark = math.hypot(s_freq - dark_cloth_s, v_freq - dark_cloth_v)

        # d_h_tol = 6
        # d_s_tol = 35    #TWEAK PARAMETERS HERE
        # d_v_tol_up = 40

        # d_v_tol_down = 60

        d_h_tol = 8
        d_s_tol = 60
        d_v_tol_up = 65

        d_v_tol_down = 60

        lower_dark_blue = np.array([dark_cloth_h - d_h_tol, dark_cloth_s - d_s_tol, dark_cloth_v - d_v_tol_down])
        upper_dark_blue = np.array([dark_cloth_h + d_h_tol, dark_cloth_s + d_s_tol, dark_cloth_v + d_v_tol_up])

        lower_dark_blue = np.clip(lower_dark_blue, [0, 0, 0], [179, 255, 255])
        upper_dark_blue = np.clip(upper_dark_blue, [0, 0, 0], [179, 255, 255])

        mask_dark_cloth = cv2.inRange(masked_image, lower_dark_blue, upper_dark_blue)

        if dist_to_state < dist_to_dark:
            h_tol = 6
            s_tol_up = 10
            s_tol_down = 45
            v_tol_up = 60
            v_tol_down = 80

            lower_blue = np.array([self.h_med - h_tol, self.s_med - s_tol_down, self.v_med - v_tol_down])
            upper_blue = np.array([self.h_med + h_tol, self.s_med + s_tol_up, self.v_med + v_tol_up])

            lower_blue = np.clip(lower_blue, [0, 0, 0], [179, 255, 255])
            upper_blue = np.clip(upper_blue, [0, 0, 0], [179, 255, 255])
            
            mask_cloth = cv2.inRange(masked_image, lower_blue, upper_blue)
            combined_cloth = cv2.bitwise_or(mask_cloth, mask_dark_cloth)
        else:
            combined_cloth = mask_dark_cloth

        lower_any_color = np.array([0, 0, 1])
        upper_any_color = np.array([179, 255, 255])
        mask_not_black = cv2.inRange(masked_image, lower_any_color, upper_any_color)
        mask_inverted_cloth = cv2.bitwise_not(combined_cloth)
        mask = cv2.bitwise_and(mask_inverted_cloth, mask_not_black)

        mask_hull = np.zeros(masked_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_hull, [hull], 255)

        mask = cv2.bitwise_and(mask, mask_hull)

        #gray_mask = cv2.medianBlur(mask, 9)
        #gray_mask = cv2.GaussianBlur(mask, (5, 5), 2)


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        eroded_mask = cv2.dilate(mask, kernel, iterations=6)
        eroded_mask = cv2.erode(eroded_mask, kernel, iterations=6)

        eroded_mask = cv2.erode(mask, kernel, iterations=6)
        opened_mask = cv2.dilate(eroded_mask, kernel, iterations=6)

        opened_mask_blurred = cv2.medianBlur(opened_mask, 7)

        #print(self.is_angle_l, self.is_angle_r)
        if(self.is_angle_l or self.is_angle_r):     #PARAMETERS FOR ANGLED VIEWS
            circles = cv2.HoughCircles(
                opened_mask,          
                cv2.HOUGH_GRADIENT,
                dp=1.4,
                minDist=22, 
                param1=40,
                param2=17.4,
                minRadius=17,
                maxRadius=48
            )
        elif(self.is_top):        #PARAMETERS FOR TOP VIEW
            circles = cv2.HoughCircles(
                opened_mask,          
                cv2.HOUGH_GRADIENT,
                dp=1.4,
                minDist=15, 
                param1=40,
                param2=20,
                minRadius=19,
                maxRadius=52
            )    
        else:                      #PARAMETERS FOR FRONT VIEW
            circles = cv2.HoughCircles(
                opened_mask,          
                cv2.HOUGH_GRADIENT,
                dp=1.4,
                minDist=20, 
                param1=40,
                param2=20,
                minRadius=17,
                maxRadius=52
            )   

        padding_area = 15 
        overlap_threshold = 0.50 # <----THRESHOLD OF OVERLAP

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            #print(f"Found {len(circles)} circle(s) initially")  
            valid_circles = []
            for (cx, cy, r) in circles:
                A, B, C = self.side_line_r_inner_full
                perp_distance = abs(A * cx + B * cy + C) / np.sqrt(A**2 + B**2)
                is_above = BallsDetector.is_above_line((cx, cy), self.side_line_r_inner_full)
                is_below_bot = not BallsDetector.is_above_line((cx, cy), self.bot_line_inner_full) 
                if ((not self.is_top) and is_above and (perp_distance> padding_area)) or is_below_bot:
                    continue      
                valid_circles.append((cx, cy, r))
            
            # check overlaps
            overlapping_indices = set()
            indices_to_drop = set() #smaller circles to remove
            
            for i in range(len(valid_circles)):
                for j in range(i + 1, len(valid_circles)):
                    c1 = valid_circles[i]
                    c2 = valid_circles[j]
                    
                    # intersection
                    intersect_area = BallsDetector.calculate_overlap_area(c1, c2)
                    
                    # calculate area of the circles and determine the smaller
                    area_c1 = math.pi * (c1[2] ** 2)
                    area_c2 = math.pi * (c2[2] ** 2)
                    min_area = min(area_c1, area_c2)

                    if intersect_area > 0: #immediatle remove if one is less than 0.5 of the other
                        if area_c1 < (area_c2 * 0.4):
                            overlapping_indices.update([i, j]) 
                            indices_to_drop.add(i)
                        elif area_c2 < (area_c1 * 0.4):
                            overlapping_indices.update([i, j]) 
                            indices_to_drop.add(j)
                    
                    if intersect_area >= (overlap_threshold * min_area):
                        overlapping_indices.add(i)
                        overlapping_indices.add(j)
                        
                        # compare radius and add the smaller one to the list
                        if c1[2] < c2[2]:
                            indices_to_drop.add(i)
                        elif c2[2] < c1[2]:
                            indices_to_drop.add(j)
                        else:
                            # if they are the same size, drop one at random
                            indices_to_drop.add(j)

            final_count = len(valid_circles) - len(indices_to_drop)
            
           # print(f"Total valid balls drawn on screen: {len(valid_circles)}")
           # print(f"Balls flagged as overlaps: {len(overlapping_indices)}")
           # print(f"--> FINAL BALL COUNT (smaller overlaps removed): {final_count}")
        else:
            print("No circles found")

        final_balls = [c for i, c in enumerate(valid_circles) if i not in indices_to_drop]

        solid_results = {color_name: [] for color_name in self.REFERENCE_COLORS.keys()}
        striped_results = {color_name: [] for color_name in self.REFERENCE_COLORS.keys()}

        self.map_balls_to_color_distances(
            self.hsv_image, 
            rgb_image, 
            final_balls, 
            self.REFERENCE_COLORS, 
            solid_results, 
            striped_results
        )
        solid_results.pop("Red_Dark", None)
        solid_results.pop("Orange_Bright", None)
        solid_results.pop("Brown_alt", None)
        solid_results.pop("Blue_Dark", None)
        striped_results.pop("Red_Dark", None)
        striped_results.pop("Orange_Bright", None)
        striped_results.pop("Brown_alt", None)
        striped_results.pop("Blue_Dark", None)

        special_balls = self.assign_white_and_black_balls(
            hsv_img=self.hsv_image, 
            rgb_img=rgb_image, 
            final_balls=final_balls, 
            reference_colors=self.REFERENCE_COLORS,
            white_thresh=0.75, 
            black_thresh=0.35
        )

        if "White" in special_balls:
            solid_results["White"] = [special_balls["White"]]

        if "Black" in special_balls:
            solid_results["Black"] = [special_balls["Black"]]

        assignment_map = self.build_optimal_ball_map(solid_results,striped_results,self.SOLID_NUMBERS,self.STRIPED_NUMBERS) #change between greedy or hungarian algo
        # detected_balls_info = []
        #for (cx, cy, r) in final_balls:
        #    ball_id = BallsDetector.classify_ball(self.hsv_image, rgb_image, cx, cy, r)

        detected_balls_info = self.get_assignment_data(assignment_map, final_balls)
    # detected_balls_info.append((ball_id, (cx, cy), r))

    # Create BallDetection objects for the output JSON        
        ball_detections = [
            BallDetection(
                number=ball_id,
                box=BallBox(
                    xmin=int(cx - r) / self.image_w,
                    ymin=int(cy - r) / self.image_h,
                    xmax=int(cx + r) / self.image_w,
                    ymax=int(cy + r) / self.image_h
                )
            )
            for ball_id, (cx, cy), r in detected_balls_info
        ]

        return ImageResult(
            image_path=str(self.image_path),
            balls=ball_detections
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline pool-ball detector (heuristic)")
    p.add_argument("--input_json", type=Path, help="Path to data/json/input.json")
    p.add_argument("--output-json", type=Path, help="Path to data/json/output.json")
    p.add_argument("--data-dir", type=Path, default=Path("data/"), help="Directory for input images")
    p.add_argument("--topdown-dir", type=Path, default=Path("data/outputs/"), help="Directory for result images")
    p.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs to run")
    return p.parse_args()


def process_image(
    index: int,
    total: int,
    img_path: str, 
    data_dir: Path, 
    topdown_dir: Path
) -> ImageResult:
    logs = [f"Processing image {index+1}/{total}: {img_path}"]

    full_img_path = data_dir / Path(img_path)

    try:
        image_result = TableTopViewProcessor(full_img_path, Path(img_path)).get_top_view()

        output_image_path = topdown_dir / Path(img_path).name
        cv2.imwrite(str(output_image_path), cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR))
    except Exception as e:
        logs.append(f"[ERROR] processing top-view:\n{e}\n{traceback.format_exc()}")
        #return index, None, logs
    try:
        ball_result = BallsDetector(full_img_path, Path(img_path)).get_image_result()
        return index, ball_result, logs
    except Exception as e:
        logs.append(f"[ERROR] processing balls:\n{e}\n{traceback.format_exc()}")
        return index, None, logs


def main() -> int:
    args = parse_args()

    os.makedirs(args.topdown_dir, exist_ok=True)
    os.makedirs(args.output_json.parent, exist_ok=True)

    with open(args.input_json, "r") as f:
        input_data: dict = json.load(f)
        image_paths: list[str] = input_data.get("image_path", [])

    total = len(image_paths)

    if args.n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    else:
        n_jobs = max(1, args.n_jobs)

    results: list[ImageResult] = [None] * total

    if n_jobs == 1:
        for i, img_path in enumerate(image_paths):
            idx, result, logs = process_image(
                i, total, img_path, args.data_dir, args.topdown_dir
            )
            for msg in logs:
                print(msg)
            results[idx] = result
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    process_image,
                    i, total, img_path, args.data_dir, args.topdown_dir
                )
                for i, img_path in enumerate(image_paths)
            ]

            for future in as_completed(futures):
                try:
                    idx, result, logs = future.result()
                    for msg in logs:
                        print(msg)
                    results[idx] = result
                except Exception as e:
                    print(f"[ERROR] unexpected worker failure:\n{e}")

    output_data = [result.to_dict() for result in results if result is not None]
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("Process interrupted by user.")
