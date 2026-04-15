import argparse
import json
import math
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
import itertools

import cv2
import numpy as np
from tqdm import tqdm


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
    top_down_img: cv2.Mat
    balls: list[BallDetection]

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "num_balls": len(self.balls),
            "balls": [ball.to_dict() for ball in self.balls],
        }


def get_upper_lower_blue(hsv_image, hue_window: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    h, w = hsv_image.shape[:2]

    x0 = int(w * 0.3)
    x1 = int(w * 0.7)
    y0 = int(h * 0.3)
    y1 = int(h * 0.7)
    
    # hsv_image_copy = hsv_image.copy()
    # cv2.rectangle(hsv_image_copy, (x0, y0), (x1, y1), (255, 0, 0), 2)
    # plt.figure(figsize=(12, 8))
    # plt.imshow(cv2.cvtColor(hsv_image_copy, cv2.COLOR_HSV2RGB))
    # plt.axis('off')
    # plt.title('Region of Interest for Color Analysis')
    # plt.show()

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

    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.hist(S[near_hue].ravel(), bins=50, color='green', alpha=0.7)
    # plt.axvline(lower_blue[1], color='red', linestyle='dashed', linewidth=2)
    # plt.axvline(upper_blue[1], color='red', linestyle='dashed', linewidth=2)
    # plt.title('Saturation Histogram of Near-Hue Pixels')
    # plt.subplot(1, 2, 2)
    # plt.hist(V[near_hue].ravel(), bins=50, color='blue', alpha=0.7)
    # plt.axvline(lower_blue[2], color='red', linestyle='dashed', linewidth=2)
    # plt.axvline(upper_blue[2], color='red', linestyle='dashed', linewidth=2)
    # plt.title('Value Histogram of Near-Hue Pixels')
    # plt.tight_layout()
    # plt.show()

    return lower_blue, upper_blue, (x0, y0, x1, y1)


def keep_only_close_contours(mask, distance_threshold=100):
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


def get_lines(mask_image, min_length=250):
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


def get_inner_outer_lines(lines, center, image, angle_threshold=0.03, distance_threshold=70):
    segments = []
    for line in lines:
        rho, theta = get_hessel(line)
        segments.append((line, rho, theta))

    over_line = []
    all_lines = {}
    groups = []

    for i in range(len(segments)):
        all_lines[i] = segments[i]
        for j in range(i + 1, len(segments)):
            rho1, theta1 = segments[i][1], segments[i][2]
            rho2, theta2 = segments[j][1], segments[j][2]

            if get_diff_pi(theta1, theta2) > angle_threshold:
                continue

            # print(min(abs(rho1 - rho2), abs(rho1 + rho2)))
            if min(abs(rho1 - rho2), abs(rho1 + rho2)) > distance_threshold:
                continue

            if not any(i in g for g in groups) and not any(j in g for g in groups):
                groups.append(set([i, j]))
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

            dist1 = get_dist_center(rho1, theta1, center)
            dist2 = get_dist_center(rho2, theta2, center)

            # print(f"Distance to center: {dist1:.2f}, {dist2:.2f}")

            if abs(dist1) > abs(dist2):
                over_line.append((i, j))
                # print(f"Line {i} is over line {j}")
            else:
                over_line.append((j, i))
                # print(f"Line {j} is over line {i}")

    for i in range(len(segments)):
        if not any(i in group for group in groups):
            groups.append(set([i]))

    # merge groups with overlap elements
    changed = True
    while changed:
        changed = False
        new_groups = []
        while groups:
            first = groups.pop(0)
            merged_any = False
            i = 0
            while i < len(groups):
                if not first.isdisjoint(groups[i]):
                    first.update(groups.pop(i))
                    changed = True
                    merged_any = True
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
            dist = abs(get_dist_center(rho, theta, center))
            group_info.append((idx, dist))

        outer_idx = max(group_info, key=lambda x: x[1])[0]
        inner_idx = min(group_info, key=lambda x: x[1])[0]

        outer_lines[outer_idx] = all_lines[outer_idx]
        inner_lines[inner_idx] = all_lines[inner_idx]

    return inner_lines, outer_lines


def get_hessel(line):
    x1, y1, x2, y2 = line[0]

    dx, dy = x2 - x1, y2 - y1

    length = math.sqrt(dx ** 2 + dy ** 2)
    if length < 1e-6:
        return None

    theta = math.atan2(dy, dx) + np.pi / 2
    theta = theta % np.pi

    rho = x1 * math.cos(theta) + y1 * math.sin(theta)

    return (rho, theta)


def get_diff_pi(a, b):
    diff = abs(a - b)
    return min(diff, np.pi - diff)


def get_dist_center(rho, theta, center):
    cx, cy = center
    return cx * math.cos(theta) + cy * math.sin(theta) - rho


def select_4_groups_old(inner_lines, outer_lines, h, w):
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

        intersection_points_inner = get_intersections(candidate_inner, h, w)
        intersection_points_outer = get_intersections(candidate_outer, h, w)

        if len(intersection_points_inner) == 4 and len(intersection_points_outer) == 4:
            return candidate_inner, candidate_outer

    return None, None


def select_4_groups_not_work(inner_lines, outer_lines, h, w):
    print(f"Trying to select 4 groups from {len(inner_lines)} inner lines and {len(outer_lines)} outer lines")
    
    parallelism_inner = {}
    for (i, (line_i, rho_i, theta_i)), (j, (line_j, rho_j, theta_j)) in itertools.combinations(inner_lines.items(), 2):
        diff = get_diff_pi(theta_i, theta_j)
        parallelism_inner[(i, j)] = diff
    parallelism_outer = {}
    for (i, (line_i, rho_i, theta_i)), (j, (line_j, rho_j, theta_j)) in itertools.combinations(outer_lines.items(), 2):
        diff = get_diff_pi(theta_i, theta_j)
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


def rectangle_score(points, h, w, image):
    points = np.asarray(points, dtype=np.float32)
    points = order_points_clockwise(points)

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
    total_pixels = len(inside_pixels)

    white_ratio = white_pixels / total_pixels

    print(f"White pixels inside: {white_pixels}")
    print(f"White ratio: {white_ratio:.4f}")

    if white_ratio < 0.8:
        return None

    # Final score
    score = opp_sim_1 + opp_sim_2 + white_ratio
    return score

def order_points_clockwise(points):
    points = np.asarray(points, dtype=np.float32)
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order]

def select_best_4_line_combo(lines, image):
    h, w = image.shape[:2]
    keys = list(lines.keys())
    best_lines = None
    best_points = None
    best_score = -np.inf

    for combo in itertools.combinations(keys, 4):
        candidate = {k: lines[k] for k in combo}

        points = get_intersections(candidate, h, w)
        if len(points) != 4:
            continue

        score = rectangle_score(points, h, w, image)
        if score is None:
            continue

        # print(f"combo={combo}, score={score:.4f}")

        if score > best_score:
            best_score = score
            best_lines = candidate
            best_points = points

    return best_lines, best_points, best_score


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


def get_intersections(lines, h, w):
    if isinstance(lines, dict):
        line_segments = [seg[0] for seg in lines.values()]
    else:
        line_segments = [seg[0] for seg in lines]

    raw_points = []
    for l1, l2 in itertools.combinations(line_segments, 2):
        p = line_intersection_infinite(l1, l2)
        if p is None:
            continue
        x, y = p

        if -250 <= x <= (w - 1) + 250 and -250 <= y <= (h - 1) + 250:
            raw_points.append((x, y))

    intersection_points = np.array(list(raw_points), dtype=np.float32)
    return intersection_points


def order_inner_outer_lines(inner_lines, outer_lines, center):
    inner_lines_keys = np.array(list(inner_lines.keys()))
    outer_lines_keys = np.array(list(outer_lines.keys()))
    centers_inner = []
    centers_outer = []
    for i in inner_lines.values():
        x1, y1, x2, y2 = i[0][0]
        centers_inner.append([(x1 + x2) / 2, (y1 + y2) / 2])
    for i in outer_lines.values():
        x1, y1, x2, y2 = i[0][0]
        centers_outer.append([(x1 + x2) / 2, (y1 + y2) / 2])
    centers_inner = np.array(centers_inner)
    centers_outer = np.array(centers_outer)

    center_pts = np.mean(centers_inner, axis=0)
    angles = np.arctan2(centers_inner[:, 1] - center_pts[1], centers_inner[:, 0] - center_pts[0])
    order = np.argsort(angles)

    inner_lines_ordered = {inner_lines_keys[i]: inner_lines[inner_lines_keys[i]] for i in order}
    outer_lines_ordered = {outer_lines_keys[i]: outer_lines[outer_lines_keys[i]] for i in order}
    
    return inner_lines_ordered, outer_lines_ordered, angles, order


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
        p = line_intersection_infinite(l1, l2)
        if p is None:
            continue
        x, y = p

        pad = 250
        if -pad <= x <= (w - 1) + pad and -pad <= y <= (h - 1) + pad:
            raw_points.append((x, y))

    intersection_points = np.array(list(raw_points), dtype=np.float32)
    return intersection_points


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


def process_image(full_img_path: Path, image_path: Path) -> ImageResult:
    print(f"Processing image: {image_path}")
    image = cv2.imread(str(full_img_path))
    scale_factor = 3360 / image.shape[1]
    new_dimensions = (3360, int(image.shape[0] * scale_factor))
    image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_blur = cv2.GaussianBlur(image, (15, 15), 0)

    hsv_image = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    lower_blue, upper_blue, _ = get_upper_lower_blue(hsv_image, hue_window=1.5)

    mask_table = cv2.inRange(hsv_image, lower_blue, upper_blue)

    kernel = np.ones((7, 7), np.uint8)
    mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_OPEN, kernel)
    mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_CLOSE, kernel)

    mask_table, _ = keep_only_close_contours(mask_table, distance_threshold=50)

    lines = get_lines(mask_table, min_length=200)

    edge_points = [line[0] for line in lines]
    center = [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2) in edge_points]
    center = np.mean(center, axis=0).astype(int)

    inner_lines, outer_lines = get_inner_outer_lines(lines, center, image, angle_threshold=0.07, distance_threshold=80)

    if len(outer_lines) < 4 or len(inner_lines) < 4:
        print("[DEBUG] Not enough lines found, retrying with wider hue window.")
        lower_blue, upper_blue, _ = get_upper_lower_blue(hsv_image, hue_window=5)
        mask_table = cv2.inRange(hsv_image, lower_blue, upper_blue)
        lines = get_lines(mask_table, min_length=250)
        inner_lines, outer_lines = get_inner_outer_lines(lines, center, image, angle_threshold=0.07, distance_threshold=80)

    if len(outer_lines) > 4 or len(inner_lines) > 4:
        selected_inner = select_best_4_line_combo(inner_lines, mask_table)[0]
        selected_outer = select_best_4_line_combo(outer_lines, mask_table)[0]

        if selected_inner is not None and selected_outer is not None:
            inner_lines = selected_inner
            outer_lines = selected_outer

    inner_lines, outer_lines, _, _ = order_inner_outer_lines(inner_lines, outer_lines, center)

    intersection_points_outer = get_intersections_ordered(outer_lines, image.shape[0], image.shape[1])
    intersection_points_inner = get_intersections_ordered(inner_lines, image.shape[0], image.shape[1])

    if len(intersection_points_outer) != 4:
        intersection_points_outer = sorted(intersection_points_outer, key=lambda p: math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2))
        intersection_points_outer = np.array(intersection_points_outer[:4], dtype=np.float32)
    if len(intersection_points_inner) != 4:
        intersection_points_inner = sorted(intersection_points_inner, key=lambda p: math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2))
        intersection_points_inner = np.array(intersection_points_inner[:4], dtype=np.float32)


    if len(intersection_points_outer) > 4 or len(intersection_points_inner) > 4:
        print("[DEBUG] Too many intersection points found.")

    pts_ordered_outer, pts_ordered_inner, _, _ = order_points_by_angle(
        intersection_points_outer, intersection_points_inner)

    is_top, is_front, is_angle, (t1, t2, b1, b2) = got_angle(pts_ordered_outer)
    start_point = None
    if is_top:
        start_point = t1 if t1[0] < t2[0] else t2
    elif is_front:
        start_point = t1 if t1[0] > t2[0] else t2
    else:
        start_point = t1 if t1[1] < t2[1] else t2

    start_by_angle = np.where((pts_ordered_outer == start_point).all(axis=1))[0][0]

    img_warped = warp_table_variant(image, pts_ordered_inner, pad=0, w=2000, h=2000,
                                    inner_color=(74, 158, 229), pad_color=(85, 90, 85))[0]

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
        has_dot, has_boll, is_empty, _, _, countours = find_dot(sub_img_warped[i], lower_blue, upper_blue)
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

    img_warped = warp_table_variant(
        image,
        pts_ordered_outer_correct,
        pad=75, w=1600, h=1000,
        inner_color=(74, 158, 229), pad_color=(85, 90, 85)
    )[0]

    return ImageResult(
        image_path=str(image_path),
        top_down_img=img_warped,
        balls=[]
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline pool-ball detector (heuristic)")
    p.add_argument("--input_json", type=Path, help="Path to data/json/input.json")
    p.add_argument("--output-json", type=Path, help="Path to data/json/output.json")
    p.add_argument("--data-dir", type=Path, default=Path("data/"), help="Directory for input images")
    p.add_argument("--topdown-dir", type=Path, default=Path("data/outputs/"), help="Directory for result images")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    os.makedirs(args.topdown_dir, exist_ok=True)
    os.makedirs(args.output_json.parent, exist_ok=True)

    # Load input JSON
    with open(args.input_json, "r") as f:
        input_data: dict = json.load(f)
        image_paths: list[str] = input_data.get("image_path", [])

    # Process each image and collect results
    results: list[ImageResult] = []
    for img_path in image_paths:
        full_img_path = args.data_dir / Path(img_path)
        try:
            image_result = process_image(full_img_path, img_path)

            output_image_path = args.topdown_dir / Path(image_result.image_path).name
            cv2.imwrite(str(output_image_path), cv2.cvtColor(image_result.top_down_img, cv2.COLOR_RGB2BGR))

            image_result.top_down_img = None
            results.append(image_result)
        except Exception as e:
            print(f"Error processing {img_path}:\n{e}", "="*40)
            continue

    output_data = [result.to_dict() for result in results]
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
