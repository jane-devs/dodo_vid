"""
dodo_vid — инструмент мониторинга столов на видео
Python 3.13
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
PERSON_CLASS_ID = 0    # индекс класса «человек» в COCO
FRAME_SKIP = 2         # обрабатывать каждый N-й кадр
DEBOUNCE_FRAMES = 8    # кадров подряд для смены состояния
ROI_WINDOW = "Выделите область стола"

STATE_EMPTY = "пусто"
STATE_OCCUPIED = "занято"
EVENT_APPROACH = "подход"
EVENT_OCCUPIED = "занято"
EVENT_EMPTY = "пусто"


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Инструмент мониторинга столов на видео"
    )
    parser.add_argument(
        "--video", required=True, help="Путь к входному видеофайлу"
    )
    return parser.parse_args()


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(
            f"Ошибка: не удалось открыть видео '{path}'", file=sys.stderr
        )
        sys.exit(1)
    return cap


def select_roi(
    cap: cv2.VideoCapture,
) -> tuple[int, int, int, int]:
    """Считывает первый кадр и запрашивает у пользователя ROI."""
    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр.", file=sys.stderr)
        cap.release()
        sys.exit(1)

    roi = cv2.selectROI(ROI_WINDOW, first_frame, fromCenter=False)
    cv2.destroyWindow(ROI_WINDOW)

    x, y, w, h = (int(v) for v in roi)
    if w == 0 and h == 0:
        print("Ошибка: ROI не выбрана (нажат Escape).", file=sys.stderr)
        cap.release()
        sys.exit(1)

    return x, y, w, h


def is_center_in_roi(
    cx: float, cy: float,
    x: int, y: int, w: int, h: int,
) -> bool:
    return x <= cx <= x + w and y <= cy <= y + h


def detect_person_in_roi(
    model: YOLO,
    frame,
    x: int, y: int, w: int, h: int,
) -> bool:
    """Возвращает True, если центр хотя бы одного bbox попадает в ROI."""
    results = model(frame, classes=[PERSON_CLASS_ID], verbose=False)
    boxes = results[0].boxes
    if boxes is None:
        return False
    for box in boxes.xyxy.tolist():
        x1, y1, x2, y2 = box
        if is_center_in_roi((x1 + x2) / 2, (y1 + y2) / 2, x, y, w, h):
            return True
    return False


def frame_to_sec(frame_idx: int, fps: float) -> float:
    """Переводит номер кадра в метку времени в секундах."""
    return frame_idx / fps


def fmt_timestamp(timestamp_sec: float) -> str:
    minutes, seconds = divmod(int(timestamp_sec), 60)
    return f"{minutes:02d}:{seconds:02d}"


def draw_overlay(
    frame,
    x: int, y: int, w: int, h: int,
    state: str,
    timestamp_sec: float,
) -> None:
    """Рисует прямоугольник ROI и текстовый оверлей на кадре."""
    color = (0, 255, 0) if state == STATE_EMPTY else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"{state}  {fmt_timestamp(timestamp_sec)}"
    cv2.putText(
        frame, label, (x + 4, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
    )


def compute_wait_times(events: list[dict]) -> list[float]:
    """Для каждого события EVENT_EMPTY находит следующее EVENT_APPROACH
    и вычисляет дельту времени в секундах."""
    df = pd.DataFrame(events, columns=["frame", "timestamp_sec", "event"])
    empty_times = (
        df.loc[df["event"] == EVENT_EMPTY, "timestamp_sec"].tolist()
    )
    approach_times = (
        df.loc[df["event"] == EVENT_APPROACH, "timestamp_sec"].tolist()
    )

    deltas: list[float] = []
    cursor = 0
    for t_empty in empty_times:
        # Пропускаем события «подход», случившиеся до или в момент «пусто».
        while (
            cursor < len(approach_times)
            and approach_times[cursor] <= t_empty
        ):
            cursor += 1
        if cursor < len(approach_times):
            deltas.append(approach_times[cursor] - t_empty)
            cursor += 1  # потребляем пару, чтобы не использовать повторно
    return deltas


def create_writer(
    path: str,
    fps: float,
    frame_w: int,
    frame_h: int,
) -> cv2.VideoWriter:
    """Создаёт VideoWriter с кодеком mp4v.

    Файл корректно финализируется благодаря try/finally в main(),
    который гарантирует вызов writer.release() даже при исключении.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (frame_w, frame_h))


def write_report(
    path: str,
    num_events: int,
    num_cycles: int,
    wait_deltas: list[float],
    mean_wait: float,
    min_wait: float,
    max_wait: float,
) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("=== отчёт сессии dodo_vid ===\n\n")
        fh.write(f"Всего событий        : {num_events}\n")
        fh.write(f"Циклов занятости     : {num_cycles}\n")
        if wait_deltas:
            fh.write(f"Пар для статистики   : {len(wait_deltas)}\n")
            fh.write(f"Среднее ожидание     : {mean_wait:.2f}с\n")
            fh.write(f"Мин. ожидание        : {min_wait:.2f}с\n")
            fh.write(f"Макс. ожидание       : {max_wait:.2f}с\n")
        else:
            fh.write("Время ожидания       : недостаточно данных\n")


# ---------------------------------------------------------------------------
# Основной цикл детекции
# ---------------------------------------------------------------------------

def run_detection_loop(
    cap: cv2.VideoCapture,
    model: YOLO,
    roi: tuple[int, int, int, int],
    fps: float,
    writer: cv2.VideoWriter,
    total_frames: int = 0,
) -> list[dict]:
    """Основной цикл: инференс, дебаунс состояний, рисование, запись кадров."""
    x, y, w, h = roi

    events: list[dict] = []

    state = STATE_EMPTY      # текущее подтверждённое состояние
    candidate = STATE_EMPTY  # состояние, к которому идёт дебаунс
    debounce_count = 0       # число подряд идущих кадров с одним сигналом

    frame_idx = 0

    while True:
        # Пропускаемые кадры: grab() захватывает без декодирования в RAM —
        # экономит CPU и исключает выделение numpy-массива на каждый кадр.
        if frame_idx % FRAME_SKIP != 0:
            if not cap.grab():
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        person_in_roi = detect_person_in_roi(model, frame, x, y, w, h)

        # --- дебаунс состояний ---
        # Состояние обновляется до рисования, чтобы прямоугольник
        # отражал новое состояние уже на кадре перехода.
        raw = STATE_OCCUPIED if person_in_roi else STATE_EMPTY
        if raw == candidate:
            debounce_count += 1
        else:
            candidate = raw
            debounce_count = 1

        timestamp_sec = frame_to_sec(frame_idx, fps)

        if debounce_count >= DEBOUNCE_FRAMES and candidate != state:
            ev = {"frame": frame_idx, "timestamp_sec": timestamp_sec}
            if state == STATE_EMPTY and candidate == STATE_OCCUPIED:
                events.append({**ev, "event": EVENT_APPROACH})
                events.append({**ev, "event": EVENT_OCCUPIED})
            elif state == STATE_OCCUPIED and candidate == STATE_EMPTY:
                events.append({**ev, "event": EVENT_EMPTY})
            state = candidate
            debounce_count = 0

        # --- рисование и запись кадра ---
        draw_overlay(frame, x, y, w, h, state, timestamp_sec)
        writer.write(frame)

        # --- прогресс ---
        processed = frame_idx // FRAME_SKIP + 1
        if processed % 100 == 0:
            pct = f"{frame_idx / total_frames * 100:.1f}%" if total_frames else f"кадр {frame_idx}"
            print(f"\r  Обработано кадров: {processed}  ({pct})", end="", flush=True)

        frame_idx += 1

    print()  # перенос строки после последнего \r
    return events


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def run_analytics(events: list[dict], report_path: str) -> None:
    """Считает статистику ожидания, выводит сводку и пишет отчёт."""
    df = pd.DataFrame(events, columns=["frame", "timestamp_sec", "event"])
    num_cycles = len(df[df["event"] == EVENT_OCCUPIED])
    wait_deltas = compute_wait_times(events)

    if wait_deltas:
        wait_series = pd.Series(wait_deltas)
        mean_wait = float(wait_series.mean())
        min_wait = float(wait_series.min())
        max_wait = float(wait_series.max())

        summary = pd.DataFrame(
            {"ожидание_сек": wait_deltas},
            index=pd.RangeIndex(1, len(wait_deltas) + 1, name="пара"),
        )
        print("\n--- Пары ожидания (пусто → следующий подход) ---")
        print(summary.to_string())
        print(f"\nВсего событий        : {len(df)}")
        print(f"Циклов занятости     : {num_cycles}")
        print(f"Среднее ожидание     : {mean_wait:.2f}с")
        print(f"Мин. ожидание        : {min_wait:.2f}с")
        print(f"Макс. ожидание       : {max_wait:.2f}с")
    else:
        print("Недостаточно данных для расчёта времени ожидания.")
        mean_wait = min_wait = max_wait = float("nan")

    write_report(
        report_path, len(df), num_cycles,
        wait_deltas, mean_wait, min_wait, max_wait,
    )
    print(f"\nОтчёт записан в {report_path}")


def main() -> None:
    args = parse_args()

    # SECTION: ROI SELECTION
    cap = open_video(args.video)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Видео : {args.video}")
    print(f"Кадры : {total_frames}")
    print(f"FPS   : {fps:.2f}")

    x, y, w, h = select_roi(cap)
    print(f"ROI   : x={x}, y={y}, w={w}, h={h}")

    # Перемотка в начало — select_roi сдвинул позицию на 1 кадр вперёд.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # SECTION: DETECTION LOOP
    stem = Path(args.video).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{stem}_{ts}.mp4"
    report_path = f"{stem}_{ts}_report.txt"

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer = create_writer(output_path, fps, frame_w, frame_h)
    print(f"Вывод : {output_path}")

    model = YOLO("yolov8n.pt")
    try:
        events = run_detection_loop(cap, model, (x, y, w, h), fps, writer, total_frames)
    finally:
        # release вызывается в любом случае — иначе moov-атом не записывается
        # и файл оказывается нечитаемым при аварийном завершении
        cap.release()
        writer.release()

    # SECTION: ANALYTICS + REPORT
    run_analytics(events, report_path)


if __name__ == "__main__":
    main()
