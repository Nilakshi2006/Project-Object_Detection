
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH   = "yolov8n.pt"
CAMERA_INDEX = 0
CONF_THRESH  = 0.40
WINDOW_NAME  = "AI Object Detection"

def class_color(class_id: int) -> tuple:
    np.random.seed(class_id * 3 + 7)
    hue   = int(np.random.randint(0, 180))
    color = np.array([[[hue, 220, 210]]], dtype=np.uint8)
    bgr   = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=2, fill=False):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if fill:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(overlay, (cx, cy), r, color, -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
        cv2.ellipse(img, (x1+r, y1+r), (r,r), 180,  0,  90, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r,r), 270,  0,  90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r,r),  90,  0,  90, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r,r),   0,  0,  90, color, thickness)

def draw_corner_brackets(img, x1, y1, x2, y2, color, length=20, thickness=3):
    pts = [
        [(x1, y1+length), (x1, y1), (x1+length, y1)],
        [(x2-length, y1), (x2, y1), (x2, y1+length)],
        [(x1, y2-length), (x1, y2), (x1+length, y2)],
        [(x2-length, y2), (x2, y2), (x2, y2-length)],
    ]
    for tri in pts:
        for i in range(len(tri)-1):
            cv2.line(img, tri[i], tri[i+1], color, thickness, cv2.LINE_AA)

def draw_label(img, text, x, y, color, font_scale=0.55, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6
    bx1, by1 = x, y - th - 2*pad
    bx2, by2 = x + tw + 2*pad, y
    bx1 = max(bx1, 0); by1 = max(by1, 0)
    draw_rounded_rect(img, (bx1, by1), (bx2, by2), color, radius=6, fill=True)
    cv2.putText(img, text, (bx1+pad, by2-pad), font,
                font_scale, (255,255,255), thickness, cv2.LINE_AA)

def draw_hud(img, fps, count_map, frame_w, frame_h, name_to_id):
    # ── title bar ───────────────────────────────
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, 52), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

    cv2.putText(img, "AI OBJECT DETECTION",
                (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (120, 220, 255), 2, cv2.LINE_AA)

    # ── FPS ─────────────────────────────────────
    fps_text = f"FPS: {fps:5.1f}"
    (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(img, fps_text, (frame_w - fw - 20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (80, 255, 160) if fps >= 20 else (50, 180, 255),
                2, cv2.LINE_AA)

    # ── object list panel ───────────────────────
    if count_map:
        line_h  = 26
        panel_h = len(count_map) * line_h + 20
        px1, py1 = 10, frame_h - panel_h - 10
        px2, py2 = 240, frame_h - 10

        overlay2 = img.copy()
        cv2.rectangle(overlay2, (px1-4, py1-4), (px2+4, py2+4), (15,15,15), -1)
        cv2.addWeighted(overlay2, 0.65, img, 0.35, 0, img)

        for i, (name, cnt) in enumerate(sorted(count_map.items())):
            color = class_color(name_to_id.get(name, 0))
            cy    = py1 + 16 + i * line_h
            cv2.circle(img, (px1 + 10, cy - 4), 5, color, -1, cv2.LINE_AA)
            cv2.putText(img, f"  {name.capitalize()}: {cnt}", (px1 + 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    # ── total count ─────────────────────────────
    tot_text = f"Objects: {sum(count_map.values())}"
    (tw, _), _ = cv2.getTextSize(tot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(img, tot_text, (frame_w - tw - 16, frame_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 210, 80), 2, cv2.LINE_AA)

    # ── hints ────────────────────────────────────
    for text, offset in [("Q / ESC : Quit", 34), ("F : Toggle Fullscreen", 54)]:
        (hw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(img, text, (frame_w - hw - 14, frame_h - offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    start_total = time.time()
    print("\n" + "="*55)
    print("  AI Object Detection — Initializing ...")
    print("="*55)

    # 1. Load Model
    print(f"  [1/3] Loading YOLOv8 model ({MODEL_PATH}) ...", end="", flush=True)
    m_start = time.time()
    model      = YOLO(MODEL_PATH)
    names      = model.names
    name_to_id = {v: k for k, v in names.items()}
    print(f" Done ({time.time() - m_start:.1f}s)")

    # 2. Open Camera (with CAP_DSHOW for Windows speed)
    print(f"  [2/3] Opening Camera (Index {CAMERA_INDEX}) ...", end="", flush=True)
    c_start = time.time()
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"\n ERROR: Could not open camera (index {CAMERA_INDEX}).")
        return
    print(f" Done ({time.time() - c_start:.1f}s)")

    # 3. Configure Resolution
    print(f"  [3/3] Configuring Resolution ...", end="", flush=True)
    r_start = time.time()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" Done ({time.time() - r_start:.1f}s) -> {frame_w}x{frame_h}")

    print(f"\n  ✅ Initialization complete in {time.time() - start_total:.1f}s")
    print("  🚀 Running ... Q/ESC = quit | F = toggle fullscreen\n")

    prev_time     = time.time()
    fps           = 0.0
    count_map     = defaultdict(int)
    is_fullscreen = True

    # ── open fullscreen on launch ────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Frame read error.")
            break

        results = model(frame, conf=CONF_THRESH, verbose=False)[0]
        count_map.clear()

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = names.get(cls_id, str(cls_id))
                color  = class_color(cls_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                draw_rounded_rect(frame, (x1, y1), (x2, y2), color, radius=8, fill=True)
                draw_corner_brackets(frame, x1, y1, x2, y2, color, length=18, thickness=3)
                draw_label(frame, f"{label.upper()}  {conf*100:.0f}%", x1, y1, color)
                count_map[label] += 1

        cur_time  = time.time()
        fps       = 0.9 * fps + 0.1 * (1.0 / max(cur_time - prev_time, 1e-6))
        prev_time = cur_time

        draw_hud(frame, fps, count_map, frame_w, frame_h, name_to_id)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):          # Q or ESC → quit
            break
        if key == ord('f'):                # F → toggle fullscreen
            is_fullscreen = not is_fullscreen
            mode = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Detection stopped. Goodbye!\n")

if __name__ == "__main__":
    main()