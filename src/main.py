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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class BallDetectConfig:
    max_balls: int = 16

def get_upper_lower_blue(hsv_image, hue_window: float = 1.5) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
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

    s_low  = int(np.percentile(S[near_hue], 10))    # ignores the lowest 10% of pixels
    s_high = int(np.percentile(S[near_hue], 95))

    v_low  = int(np.percentile(V[near_hue], 5))
    v_high = int(np.percentile(V[near_hue], 95))

    lower_blue = np.array([
        max(0, dominant_hue - hue_window),
        max(0, s_low - 60),
        max(0, v_low - 60)
    ], dtype=np.uint8)

    upper_blue = np.array([
        min(179, dominant_hue + hue_window),
        min(255, s_high + 10),
        min(255, v_high + 10)
    ], dtype=np.uint8)

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
            
            if not any (i in g for g in groups) and not any (j in g for g in groups):
                groups.append(set([i, j]))
            elif any (i in g for g in groups):
                for group in groups:
                    if i in group:
                        group.add(j)
                        break
            elif any (j in g for g in groups):
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

    # select 4 most bigger groups
    # if len(groups) > 4:
    #     groups = groups[:3]

    #     print("TODO: choose between last 2 groups")
    #     for idx, group in enumerate(groups[3:]):
    #         intersection_points = get_intersections(lines, image.shape[0], image.shape[1])

    #         print(f"Group {idx + 1}")
    #         for line_id in group:
    #             line, rho, theta = segments[line_id]
    #             print(f"  Line {line_id}: rho={rho:.2f}, theta={theta:.2f};")

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

def select_4_groups_from_largest_3(inner_lines, outer_lines, h, w):
    inner_keys = list(inner_lines.keys())
    outer_keys = list(outer_lines.keys())

    base_inner_keys = inner_keys[:3]
    base_outer_keys = outer_keys[:3]

    if len(inner_keys) != len(outer_keys):
        print("Warning: number of inner and outer lines differ, cannot reliably select 4 groups")

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

def get_intersections(lines, h, w):
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

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # white center of the dot: low saturation + high value
    white_mask = np.zeros((h, w), dtype=np.uint8)
    white_mask[(hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 150)] = 255

    # dark region around the white center
    dark_mask = cv2.inRange(gray, 0, 110)

    contours, _ = cv2.findContours(non_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    WHITE_R = 3
    DARK_R1 = 4
    DARK_R2 = 11
    BLUE_R1 = 12
    BLUE_R2 = 18

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 80:
            continue

        result_countours.append(cnt)

        x, y, bw, bh = cv2.boundingRect(cnt)
        touches_edge = (x <= 1 or y <= 1 or x + bw >= w - 2 or y + bh >= h - 2)

        # bbox center as fallback
        bx = x + bw // 2
        by = y + bh // 2

        # 1: find a small white blob inside the candidate
        roi_white = white_mask[y:y + bh, x:x + bw]
        white_contours, _ = cv2.findContours(roi_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        chosen_white = None
        chosen_white_area = 0
        best_white_score = -1e9

        for wc in white_contours:
            wa = cv2.contourArea(wc)
            if wa < 2 or wa > 180:
                continue

            wx, wy, ww, wh = cv2.boundingRect(wc)
            wM = cv2.moments(wc)

            if wM["m00"] > 0:
                wcx_local = int(wM["m10"] / wM["m00"])
                wcy_local = int(wM["m01"] / wM["m00"])
            else:
                wcx_local = wx + ww // 2
                wcy_local = wy + wh // 2

            wcx = x + wcx_local
            wcy = y + wcy_local

            # prefer a reasonably sized white blob near the candidate center
            dist_to_bbox_center = np.hypot(wcx - bx, wcy - by)
            score = 2.0 * wa - 0.35 * dist_to_bbox_center

            if score > best_white_score:
                best_white_score = score
                chosen_white = (wcx, wcy)
                chosen_white_area = wa

        # if no white blob found, fall back to blob center
        if chosen_white is not None:
            cx, cy = chosen_white
        else:
            cx, cy = bx, by

        # 2: check dot structure around that center white center + dark ring + blue outside
        pad = BLUE_R2 + 2
        x0 = max(cx - pad, 0)
        y0 = max(cy - pad, 0)
        x1 = min(cx + pad + 1, w)
        y1 = min(cy + pad + 1, h)

        roi_blue = blue_mask[y0:y1, x0:x1]
        roi_white2 = white_mask[y0:y1, x0:x1]
        roi_dark = dark_mask[y0:y1, x0:x1]

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        inner = dist <= WHITE_R
        dark_ring = (dist > DARK_R1) & (dist <= DARK_R2)
        blue_outer = (dist > BLUE_R1) & (dist <= BLUE_R2)

        inner_n = max(np.count_nonzero(inner), 1)
        dark_n = max(np.count_nonzero(dark_ring), 1)
        blue_n = max(np.count_nonzero(blue_outer), 1)

        white_inner_ratio = np.count_nonzero(roi_white2[inner]) / inner_n
        dark_ring_ratio = np.count_nonzero(roi_dark[dark_ring]) / dark_n
        blue_outer_ratio = np.count_nonzero(roi_blue[blue_outer]) / blue_n

        # TODO: make decision here
        if touches_edge:
            has_boll = True
        if not touches_edge and area > 2400:
            has_boll = True

        if not touches_edge and area < 2400:
            has_dot = True

        # ------------------------------------------------------------
        # DEBUG DRAWING
        # ------------------------------------------------------------
        cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(debug_img, (cx, cy), 2, (0, 255, 0), -1)

        # draw sampling rings around tested center
        cv2.circle(debug_img, (cx, cy), WHITE_R, (0, 255, 255), 1)
        cv2.circle(debug_img, (cx, cy), DARK_R2, (255, 0, 255), 1)
        cv2.circle(debug_img, (cx, cy), BLUE_R2, (0, 165, 255), 1)

        candidates_info.append({
            "index": i,
            "bbox": (x, y, bw, bh),
            "center": (cx, cy),
            "area": float(area),
            "white_area": float(chosen_white_area),
            "white_inner_ratio": float(white_inner_ratio),
            "dark_ring_ratio": float(dark_ring_ratio),
            "blue_outer_ratio": float(blue_outer_ratio),
            "touches_edge": bool(touches_edge),
        })

    return has_dot, has_boll, is_empty, debug_img, candidates_info, result_countours


def process_image(full_img_path: Path, image_path: Path, config: BallDetectConfig) -> ImageResult:
    print(f"Processing image: {image_path}")
    image = cv2.imread(str(full_img_path))
    desired_width = 3360
    scale_factor = desired_width / image.shape[1]
    new_dimensions = (desired_width, int(image.shape[0] * scale_factor))
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

    inner_lines, outer_lines = get_inner_outer_lines(lines, center, image)

    if len(outer_lines) < 4 or len(inner_lines) < 4:
        print("[DEBUG] Not enough lines found, retrying with wider hue window.")
        lower_blue, upper_blue, _ = get_upper_lower_blue(hsv_image, hue_window=5)
        mask_table = cv2.inRange(hsv_image, lower_blue, upper_blue)
        lines = get_lines(mask_table, min_length=250)
        inner_lines, outer_lines = get_inner_outer_lines(lines, center, image)
    
    if len(outer_lines) > 4 or len(inner_lines) > 4:
        selected_inner, selected_outer = select_4_groups_from_largest_3(
            inner_lines, outer_lines, image.shape[0], image.shape[1]
        )

        if selected_inner is not None and selected_outer is not None:
            inner_lines = selected_inner
            outer_lines = selected_outer

    intersection_points_outer = get_intersections(outer_lines, image.shape[0], image.shape[1])
    intersection_points_inner = get_intersections(inner_lines, image.shape[0], image.shape[1])

    if len(intersection_points_outer) != 4:
        intersection_points_outer = sorted(intersection_points_outer, key=lambda p: math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2))
        intersection_points_outer = np.array(intersection_points_outer[:4], dtype=np.float32)
    if len(intersection_points_inner) != 4:
        intersection_points_inner = sorted(intersection_points_inner, key=lambda p: math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2))
        intersection_points_inner = np.array(intersection_points_inner[:4], dtype=np.float32)

    if len(intersection_points_outer) > 4 or len(intersection_points_inner) > 4:
        print("[DEBUG] Too many intersection points found.")

    pts_ordered_outer, pts_ordered_inner, angles, order = order_points_by_angle(intersection_points_outer, intersection_points_inner)

    is_top, is_front, is_angle, (t1, t2, b1, b2) = got_angle(pts_ordered_outer)
    start_point = None
    if is_top:
        start_point = t1 if t1[0] < t2[0] else t2
    elif is_front:
        start_point = t1 if t1[0] > t2[0] else t2
    else:
        start_point = t1 if t1[1] < t2[1] else t2

    start_by_angle = np.where((pts_ordered_outer == start_point).all(axis=1))[0][0]

    img_warped = warp_table_variant(image, pts_ordered_inner, pad=0, w=2000, h=2000, inner_color=(74, 158, 229), pad_color=(85, 90, 85))[0]

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
        has_dot, has_boll, is_empty, debug_img, info, countours = find_dot(sub_img_warped[i], lower_blue, upper_blue)
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
        print(f"[DEBUG] Start point determined by angle: {start_by_angle}, but start point determined by dot/ball detection is: {start}.")
    
    pts_ordered_outer_correct = np.roll(pts_ordered_outer, -start, axis=0)

    img_warped = warp_table_variant(image, pts_ordered_outer_correct, pad=75, w=1600, h=1000, inner_color=(74, 158, 229), pad_color=(85, 90, 85))[0]

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

        image_result = process_image(full_img_path, img_path, BallDetectConfig())

        results.append(image_result)

        # Debug
        output_image_path = args.topdown_dir / Path(image_result.image_path).name
        cv2.imwrite(str(output_image_path), cv2.cvtColor(image_result.top_down_img, cv2.COLOR_RGB2BGR))

    # Save output JSON
    output_data = [result.to_dict() for result in results]
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=4)

    # Save result images
    for result in results:
        output_image_path = args.topdown_dir / Path(result.image_path).name
        cv2.imwrite(str(output_image_path), cv2.cvtColor(result.top_down_img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
