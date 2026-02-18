import math
import cv2
import numpy as np
import os
from PIL import Image


def rotate_image(img, angle):
    """Rotates an image by 0, 90, 180, or -90 degrees."""
    if angle == 0: return img
    if angle == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == -90: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
    return img


def detect_direction_automatically(frames):
    print("Detecting camera direction...")
    sample_len = min(len(frames), 60)
    total_tx = 0
    total_ty = 0

    for i in range(sample_len - 1):
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(prev_gray, 500, 0.01, 10)
        if p0 is not None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
            if p1 is not None and len(p1[st == 1]) > 0:
                diffs = (p1[st == 1] - p0[st == 1]).reshape(-1, 2)
                total_tx += np.median(diffs[:, 0])
                total_ty += np.median(diffs[:, 1])

    if abs(total_tx) > abs(total_ty):
        if total_tx > 0:
            return 'R2L', 180
        else:
            return 'L2R', 0
    else:
        if total_ty > 0:
            return 'B2T', 90
        else:
            return 'T2B', -90


def create_perfect_spacetime_video(input_frames_path, convergence_factor=1.0, n_panoramas=30):
    print(f"--- Starting Space-Time Video Creation ---")

    frame_files = sorted([f for f in os.listdir(input_frames_path) if f.lower().endswith('.jpg')])
    frames = [cv2.imread(os.path.join(input_frames_path, f)) for f in frame_files]
    frames = [f for f in frames if f is not None]

    if not frames:
        print(f"Error: No valid jpg images found in {input_frames_path}")
        return []

    print(f"Loaded {len(frames)} source frames.")
    auto_direction, rot_angle = detect_direction_automatically(frames)
    print(f"Detected Direction: {auto_direction}\n")

    frames_norm = [rotate_image(f, rot_angle) for f in frames]
    h, w = frames_norm[0].shape[:2]

    # --- 1. Motion Tracking & Leveling ---
    print("Step 1: Tracking motion and leveling horizon...")
    global_matrices = [np.eye(3)]
    acc_matrix = np.eye(3)
    lk_params = dict(winSize=(51, 51), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    for i in range(len(frames_norm) - 1):
        p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(frames_norm[i], cv2.COLOR_BGR2GRAY), 1000, 0.01, 10)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(frames_norm[i], cv2.COLOR_BGR2GRAY),
                                             cv2.cvtColor(frames_norm[i + 1], cv2.COLOR_BGR2GRAY), p0, None,
                                             **lk_params)
        diffs = (p0[st == 1] - p1[st == 1]).reshape(-1, 2)
        tx, ty = np.median(diffs[:, 0]) * convergence_factor, np.median(diffs[:, 1])
        acc_matrix = acc_matrix @ np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        global_matrices.append(acc_matrix.copy())

    total_ty = global_matrices[-1][1, 2]
    for i, M in enumerate(global_matrices): M[1, 2] -= (i / (len(global_matrices) - 1)) * total_ty

    generate_report_stitching_figure(input_frames_path, global_matrices, 0, 120, "report_figure.png")

    # --- 2. Defining Global Bounds ---
    all_tx, all_ty = [M[0, 2] for M in global_matrices], [M[1, 2] for M in global_matrices]
    canvas_w = int(max(all_tx) - min(all_tx)) + w
    canvas_h = int(max(all_ty) - min(all_ty)) + h
    off_x, off_y = int(-min(all_tx)), int(-min(all_ty))

    # --- 3. Rendering Phase with Feathering ---
    print(f"Step 2: Rendering {n_panoramas} perspective panoramas...")
    offset = math.ceil(n_panoramas * 0.1) * 2
    cut_margin = int(offset / 2)
    slice_offsets = np.linspace(0.1 * w, 0.9 * w, n_panoramas + offset)
    panorama_list = []

    # Feather size in pixels
    feather = 5

    for idx, col_offset in enumerate(slice_offsets):
        # Using float32 for accumulation to avoid rounding errors during blending
        acc_color = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        acc_weights = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

        final_tx = global_matrices[-1][0, 2] + off_x
        viewpoint_max_x = int(final_tx + col_offset)

        for i, frame in enumerate(frames_norm):
            curr_tx, curr_ty = int(global_matrices[i][0, 2] + off_x), int(global_matrices[i][1, 2] + off_y)

            # Standard territory logic
            if i == 0:
                start_x = curr_tx + int(col_offset)
            else:
                start_x = (curr_tx + int(global_matrices[i - 1][0, 2] + off_x)) // 2 + int(col_offset)

            if i == len(frames_norm) - 1:
                end_x = viewpoint_max_x
            else:
                end_x = (curr_tx + int(global_matrices[i + 1][0, 2] + off_x)) // 2 + int(col_offset)

            # Expand the sampling range to create the blend overlap
            target_s = max(curr_tx, start_x - feather)
            target_e = min(curr_tx + w, end_x + feather)

            if target_e > target_s:
                t_w = target_e - target_s
                # Linear alpha ramp: 0 at edges, 1 in the middle
                mask = np.ones((1, t_w), dtype=np.float32)
                if t_w > 2 * feather:
                    ramp = np.linspace(0, 1, feather)
                    mask[0, :feather] = ramp
                    mask[0, -feather:] = ramp[::-1]

                v_s, v_e = max(0, curr_ty), min(canvas_h, curr_ty + h)
                f_v_s, f_v_e = v_s - curr_ty, v_e - curr_ty
                src_x_s, src_x_e = target_s - curr_tx, target_e - curr_tx

                # Weighted accumulation
                slice_data = frame[f_v_s:f_v_e, src_x_s:src_x_e].astype(np.float32)
                weight_data = mask[:, :, np.newaxis]

                acc_color[v_s:v_e, target_s:target_e] += slice_data * weight_data
                acc_weights[v_s:v_e, target_s:target_e] += weight_data

        # Normalize accumulated values to 0-255 uint8
        acc_weights[acc_weights == 0] = 1
        panorama = (acc_color / acc_weights).astype(np.uint8)

        # Smooth seams
        smoothed_pano = cv2.GaussianBlur(panorama, (3, 3), 0)

        final_frame = rotate_image(panorama, {0: 0, 180: 180, -90: 90, 90: -90}.get(rot_angle, 0))
        panorama_list.append(final_frame)


    pil_panorama_list = []
    for frame in panorama_list[cut_margin:-cut_margin]:
        # Convert BGR to RGB for PIL compatibility
        final_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_panorama_list.append(Image.fromarray(final_frame_rgb))

    print(f"--- Process Complete! ---")
    return pil_panorama_list


    # # --- 4. Final Processing & Video Building ---
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # v_h, v_w = panorama_list[0].shape[:2]
    # out_video = cv2.VideoWriter(output_path, fourcc, 120, (v_w, v_h))

    # # Compile Boomerang
    # active_frames = panorama_list[cut_margin:-cut_margin]
    # for frame in active_frames: out_video.write(frame)
    # for frame in reversed(active_frames[1:-1]): out_video.write(frame)
    # out_video.release()
    #
    # # --- 4. Saving Specific Static Images ---
    # print("Step 3: Saving representative static panoramas...")
    #
    # # Save first
    # cv2.imwrite(os.path.join(output_dir, "panorama_first_view.jpg"), panorama_list[0])
    # # Save middle
    # mid_idx = len(panorama_list) // 2
    # cv2.imwrite(os.path.join(output_dir, "panorama_middle_view.jpg"), panorama_list[mid_idx])
    # # Save final
    # cv2.imwrite(os.path.join(output_dir, "panorama_final_view.jpg"), panorama_list[-1])
    #
    # print(f"  Saved first, middle, and final panoramas to {output_dir}")


def visualize_refined_optical_flow(input_frames_path, output_image_path):
    """
    Produces a refined optical flow visualization with thin, anti-aliased arrows
    and a bright green-to-red color gradient.
    """
    frame_files = sorted([f for f in os.listdir(input_frames_path) if f.lower().endswith('.jpg')])
    if len(frame_files) < 2: return

    img1 = cv2.imread(os.path.join(input_frames_path, frame_files[0]))
    img2 = cv2.imread(os.path.join(input_frames_path, frame_files[1]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Track more points for a denser, more detailed look
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, maxCorners=1500, qualityLevel=0.01, minDistance=8)

    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]
    vis = img2.copy()

    # Calculate magnitudes for color mapping
    magnitudes = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
    max_mag = np.max(magnitudes) if len(magnitudes) > 0 else 1

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        # Normalized magnitude (0 to 1)
        norm_mag = magnitudes[i] / max_mag

        # BRIGHT COLOR LOGIC (BGR format)
        # Low movement (norm_mag=0) -> Bright Green (0, 255, 0)
        # High movement (norm_mag=1) -> Bright Red (0, 0, 255)
        color_b = 0
        color_g = int((1 - norm_mag) * 255)
        color_r = int(norm_mag * 255)
        color = (color_b, color_g, color_r)

        # REFINED ARROW SETTINGS
        # thickness=1: Thinner lines
        # tipLength=0.2: Smaller, more refined arrowheads
        # lineType=cv2.LINE_AA: Anti-aliasing for smooth diagonal lines
        cv2.arrowedLine(vis, (int(c), int(d)), (int(a), int(b)), (0,255,0), thickness=1, tipLength=0.2, line_type=cv2.LINE_AA)

    cv2.imwrite(output_image_path, vis)
    print(f"Refined visualization saved to: {output_image_path}")


import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def visualize_exaggerated_drift(input_frames_path):
    # 1. Load subset (every 10th frame)
    frame_files = sorted([f for f in os.listdir(input_frames_path) if f.lower().endswith('.jpg')])
    sample_files = frame_files[::10]
    frames = [cv2.imread(os.path.join(input_frames_path, f)) for f in sample_files]
    frames = [f for f in frames if f is not None]
    if not frames: return

    # 2. Track motion
    raw_tx, raw_ty = [0.0], [0.0]
    curr_tx, curr_ty = 0.0, 0.0

    for i in range(len(frames) - 1):
        p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), 500, 0.01, 10)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
                                             cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY), p0, None)
        diffs = (p0[st == 1] - p1[st == 1]).reshape(-1, 2)
        curr_tx += np.median(diffs[:, 0])
        curr_ty += np.median(diffs[:, 1])
        raw_tx.append(curr_tx)
        raw_ty.append(curr_ty)

    # 3. Apply correction
    total_ty = raw_ty[-1]
    corr_ty = [raw_ty[i] - (i / (len(raw_ty) - 1)) * total_ty for i in range(len(raw_ty))]

    # 4. Plotting - Focusing purely on the "Path" of the camera
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: The Raw Path (Usually a diagonal or a curve)
    ax1.plot(raw_tx, raw_ty, '-o', color='red', markersize=4, label='Camera Path')
    ax1.set_title("Camera Trajectory (With Natural Vertical Drift)", fontsize=14)
    ax1.set_ylabel("Vertical Position (pixels)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: The Corrected Path (Should be a straight horizontal line)
    ax2.plot(raw_tx, corr_ty, '-o', color='green', markersize=4, label='Levelled Path')
    ax2.set_title("Corrected Trajectory (Stabilized Horizon)", fontsize=14)
    ax2.set_ylabel("Vertical Position (pixels)")
    ax2.set_xlabel("Horizontal Position (pixels)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --- THE CRITICAL FIX: Match the Y-limits to make comparison easy ---
    # We force both plots to use the same vertical scale as the raw drift
    y_min, y_max = min(raw_ty), max(raw_ty)
    padding = abs(y_max - y_min) * 0.2
    ax1.set_ylim(y_min - padding, y_max + padding)
    ax2.set_ylim(y_min - padding, y_max + padding)

    plt.tight_layout()
    plt.show()


def generate_panorama(input_frames_path, n_out_frames):
    """
    Main entry point for ex4
    :param input_frames_path : path to a dir with input video frames.
    :param n_out_frames: number of generated panorama frames
    :return: A list of generated panorama frames (PIL images).
    """
    return create_perfect_spacetime_video(input_frames_path, n_panoramas=n_out_frames)


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def generate_report_stitching_figure(input_frames_path, global_matrices, off_x, n_panoramas, output_path):
    # 1. Select two consecutive frames from the middle
    frame_files = sorted([f for f in os.listdir(input_frames_path) if f.lower().endswith('.jpg')])
    idx1, idx2 = 80, 81
    frames = [cv2.imread(os.path.join(input_frames_path, frame_files[i])) for i in [idx1, idx2]]
    h, w = frames[0].shape[:2]

    # Parameters to match your rendering logic
    feather = 50  # Exaggerated for the figure
    col_offset = w // 2

    # Calculate real horizontal shift between these two frames
    tx1 = global_matrices[idx1][0, 2] + off_x
    tx2 = global_matrices[idx2][0, 2] + off_x
    real_shift = int(abs(tx2 - tx1))

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 2)

    slices_rgb = []
    masks = []

    for i, frame in enumerate(frames):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Define the slice territory (Core + Feather)
        s_s, s_e = col_offset - 30, col_offset + 30
        f_s, f_e = s_s - feather, s_e + feather

        # Extract and Create Mask
        slice_img = img_rgb[:, f_s:f_e].astype(np.float32)
        mask = np.ones((h, slice_img.shape[1], 1), dtype=np.float32)
        ramp = np.linspace(0, 1, feather)
        mask[:, :feather, 0] = ramp
        mask[:, -feather:, 0] = ramp[::-1]

        slices_rgb.append(slice_img)
        masks.append(mask)

        # Plot Panel: Segmentation
        ax = fig.add_subplot(grid[0, i])
        overlay = img_rgb.copy()
        cv2.rectangle(overlay, (s_s, 0), (s_e, h), (0, 255, 0), -1)  # Green = Core
        cv2.rectangle(overlay, (f_s, 0), (s_s, h), (255, 165, 0), -1)  # Orange = Feather
        cv2.rectangle(overlay, (s_e, 0), (f_e, h), (255, 165, 0), -1)
        ax.imshow(cv2.addWeighted(overlay, 0.4, img_rgb, 0.6, 0))
        ax.set_title(f"Step 1: Segmenting Frame {idx1 + i}\n(Green=Core, Orange=Feather)")
        ax.axis('off')

    # Plot Panel: The Extracted "Strips" and their weight masks
    for i in range(2):
        ax_slice = fig.add_subplot(grid[1, i])
        # Show the mask as a heatmap to explain the math
        ax_slice.imshow(masks[i].squeeze(), cmap='gray')
        ax_slice.set_title(f"Step 2: Alpha Weight Map (Feathering Ramp)")
        ax_slice.axis('off')

    # Row 3: Final Accurate Stitching using Global Matrices
    ax_final = fig.add_subplot(grid[2, :])

    # Total canvas width for just these two overlapping slices
    canvas_w = real_shift + slices_rgb[0].shape[1]
    acc_color = np.zeros((h, canvas_w, 3), dtype=np.float32)
    acc_weight = np.zeros((h, canvas_w, 1), dtype=np.float32)

    for i in range(2):
        start_x = i * real_shift
        end_x = start_x + slices_rgb[i].shape[1]
        acc_color[:, start_x:end_x] += slices_rgb[i] * masks[i]
        acc_weight[:, start_x:end_x] += masks[i]

    acc_weight[acc_weight == 0] = 1
    stitched = (acc_color / acc_weight).astype(np.uint8)

    ax_final.imshow(stitched)
    ax_final.set_title(f"Step 3: Final Stitch (Shifted by {real_shift}px based on Global Matrices)")
    ax_final.set_xlabel("The orange overlap zones are mathematically merged to hide the seam.")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


# To run this, call it after your create_perfect_spacetime_video has calculated the matrices:
# generate_report_stitching_figure(input_frames_path, global_matrices, off_x, 120, "report_figure.png")

#
# # --- Execution ---
# input_frames_path = r"C:\Users\Ido\Desktop\CS_Projects\IMPR\Ex4\Exercise Inputs-20251220\Iguazu"
# generate_panorama(input_frames_path, 10)
input_path = r"C:\Users\Ido\Desktop\CS_Projects\IMPR\Ex4\Exercise Inputs-20251220\boat"
output_img = "optical_flow_visualization.jpg"
generate_panorama(input_path, 100)