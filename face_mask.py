import argparse
from pathlib import Path

import cv2
import dlib
import numpy as np


def read_image(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")

    return image


def write_image(path: str, image: np.ndarray) -> None:
    ext = Path(path).suffix.lower()

    if ext not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Выходной файл должен иметь расширение .jpg, .jpeg или .png")

    ok, encoded = cv2.imencode(ext, image)

    if not ok:
        raise ValueError(f"Не удалось сохранить изображение: {path}")

    encoded.tofile(path)


def get_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return image[:, :, :3].copy()

    return image.copy()


def landmarks_to_np(shape: dlib.full_object_detection) -> np.ndarray:
    points = np.zeros((68, 2), dtype=np.int32)

    for i in range(68):
        points[i] = (shape.part(i).x, shape.part(i).y)

    return points


def detect_all_landmarks(bgr: np.ndarray, predictor_path: str) -> list[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    rects = detector(gray, 1)

    if len(rects) == 0:
        rects = detector(gray, 2)

    if len(rects) == 0:
        raise RuntimeError("Лицо не найдено на изображении")

    all_points = []

    for rect in rects:
        shape = predictor(gray, rect)
        points = landmarks_to_np(shape)
        all_points.append(points)

    return all_points


def clip_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    return clipped.astype(np.int32)


def build_face_region_mask(points: np.ndarray, image_shape) -> np.ndarray:
    height, width = image_shape[:2]
    points = clip_points(points, width, height)

    jaw = points[0:17]

    brow_y = int(np.mean(points[17:27, 1]))
    chin_y = int(points[8, 1])
    face_height = max(1, chin_y - brow_y)

    x_left = int(points[0, 0])
    x_right = int(points[16, 0])

    top_y = brow_y - int(0.55 * face_height)

    forehead = []

    for t in np.linspace(0.0, 1.0, 11):
        x = int(x_left + t * (x_right - x_left))

        # Дуга лба: в центре выше, по краям ниже.
        arch = (2.0 * t - 1.0) ** 2
        y = int(top_y + arch * 0.30 * face_height)

        forehead.append([x, y])

    forehead = np.array(forehead, dtype=np.int32)
    forehead = clip_points(forehead, width, height)

    face_polygon = np.vstack([jaw, forehead[::-1]])

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [face_polygon], 255)

    return mask


def build_excluded_parts_mask(points: np.ndarray, image_shape) -> np.ndarray:
    height, width = image_shape[:2]
    points = clip_points(points, width, height)

    excluded = np.zeros((height, width), dtype=np.uint8)

    face_width = max(1, points[16, 0] - points[0, 0])

    eye_pad = max(1, int(face_width * 0.018))
    mouth_pad = max(1, int(face_width * 0.015))
    brow_thickness = max(2, int(face_width * 0.025))

    left_eye = cv2.convexHull(points[36:42])
    right_eye = cv2.convexHull(points[42:48])
    mouth = cv2.convexHull(points[48:60])

    eyes_mask = np.zeros_like(excluded)
    mouth_mask = np.zeros_like(excluded)
    brows_mask = np.zeros_like(excluded)

    cv2.fillConvexPoly(eyes_mask, left_eye, 255)
    cv2.fillConvexPoly(eyes_mask, right_eye, 255)

    cv2.fillConvexPoly(mouth_mask, mouth, 255)

    cv2.polylines(
        brows_mask,
        [points[17:22]],
        isClosed=False,
        color=255,
        thickness=brow_thickness,
        lineType=cv2.LINE_AA,
    )

    cv2.polylines(
        brows_mask,
        [points[22:27]],
        isClosed=False,
        color=255,
        thickness=brow_thickness,
        lineType=cv2.LINE_AA,
    )

    eye_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * eye_pad + 1, 2 * eye_pad + 1),
    )

    mouth_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * mouth_pad + 1, 2 * mouth_pad + 1),
    )

    eyes_mask = cv2.dilate(eyes_mask, eye_kernel)
    mouth_mask = cv2.dilate(mouth_mask, mouth_kernel)

    excluded = cv2.bitwise_or(excluded, eyes_mask)
    excluded = cv2.bitwise_or(excluded, mouth_mask)
    excluded = cv2.bitwise_or(excluded, brows_mask)

    return excluded


def adaptive_skin_refinement(
    bgr: np.ndarray,
    geometry_mask: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    mask_bool = geometry_mask > 0

    if np.count_nonzero(mask_bool) == 0:
        return geometry_mask

    face_saturation = hsv[:, :, 1][mask_bool]

    # Если фото чёрно-белое или почти монохромное,
    # цветовая фильтрация может удалить нормальную кожу.
    if np.median(face_saturation) < 18:
        return geometry_mask

    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    broad_skin = (
        (cr >= 128)
        & (cr <= 178)
        & (cb >= 70)
        & (cb <= 140)
        & mask_bool
    )

    face_width = max(1, points[16, 0] - points[0, 0])
    radius = max(3, int(face_width * 0.035))

    seed_mask = np.zeros(geometry_mask.shape, dtype=np.uint8)

    seed_points = [
        points[30],
        ((points[2] + points[31]) // 2),
        ((points[14] + points[35]) // 2),
        ((points[27] + points[30]) // 2),
    ]

    for p in seed_points:
        cv2.circle(seed_mask, tuple(p), radius, 255, -1)

    seed_mask = (seed_mask > 0) & mask_bool

    seed_cb = cb[seed_mask]
    seed_cr = cr[seed_mask]

    if len(seed_cb) < 30:
        refined = broad_skin
    else:
        med_cb = np.median(seed_cb)
        med_cr = np.median(seed_cr)

        mad_cb = np.median(np.abs(seed_cb - med_cb))
        mad_cr = np.median(np.abs(seed_cr - med_cr))

        cb_delta = max(12, int(3.0 * mad_cb))
        cr_delta = max(12, int(3.0 * mad_cr))

        adaptive = (
            (np.abs(cb.astype(np.int16) - int(med_cb)) <= cb_delta)
            & (np.abs(cr.astype(np.int16) - int(med_cr)) <= cr_delta)
            & mask_bool
        )

        refined = adaptive | broad_skin

    # Защита от слишком агрессивной цветовой фильтрации.
    if np.count_nonzero(refined) < 0.35 * np.count_nonzero(mask_bool):
        return geometry_mask

    refined = refined.astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, open_kernel)

    return refined


def apply_mask(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(original)

    if original.ndim == 2:
        result[mask > 0] = original[mask > 0]
        return result

    if original.shape[2] == 4:
        result[:, :, :3][mask > 0] = original[:, :, :3][mask > 0]
        result[:, :, 3][mask > 0] = original[:, :, 3][mask > 0]
        return result

    result[mask > 0] = original[mask > 0]
    return result


def create_face_skin_mask(image: np.ndarray, predictor_path: str) -> np.ndarray:
    bgr = get_bgr(image)
    all_points = detect_all_landmarks(bgr, predictor_path)

    final_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)

    for points in all_points:
        face_mask = build_face_region_mask(points, bgr.shape)
        excluded_mask = build_excluded_parts_mask(points, bgr.shape)

        geometry_mask = face_mask.copy()
        geometry_mask[excluded_mask > 0] = 0

        skin_mask = adaptive_skin_refinement(bgr, geometry_mask, points)
        skin_mask[excluded_mask > 0] = 0

        final_mask = cv2.bitwise_or(final_mask, skin_mask)

    return apply_mask(image, final_mask)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Создание маски кожных покровов лица по dlib landmarks"
    )

    parser.add_argument(
        "input",
        help="Путь к входному изображению .jpg/.jpeg/.png",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Путь к выходному изображению. По умолчанию: имя_файла_skin.ext",
    )

    parser.add_argument(
        "-p",
        "--predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="Путь к файлу shape_predictor_68_face_landmarks.dat",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")

    predictor_path = Path(args.predictor)

    if not predictor_path.exists():
        raise FileNotFoundError(
            "Не найден файл модели landmarks: "
            f"{predictor_path}\n"
            "Положите shape_predictor_68_face_landmarks.dat рядом со скриптом "
            "или передайте путь через параметр --predictor."
        )

    if args.output is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_skin{input_path.suffix}"
        )
    else:
        output_path = Path(args.output)

    image = read_image(str(input_path))
    result = create_face_skin_mask(image, str(predictor_path))
    write_image(str(output_path), result)

    print(f"Готово: {output_path}")


if __name__ == "__main__":
    main()