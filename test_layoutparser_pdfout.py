import os
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import layoutparser as lp
import numpy as np
from PIL import Image
import cv2

INPUT_FOLDER = "/mnt/c/Users/nikdu/my_python_projects/layout_parser_linux/test_files/inpuf_files"
OUTPUT_FOLDER = "/mnt/c/Users/nikdu/my_python_projects/layout_parser_linux/test_files/output_layouts"
DPI = 200
DETECTION_THRESHOLD = 0.3

COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 255),  # Magenta
    (255, 128, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 255),  # Deep Sky Blue
    (0, 0, 0),      # Black
]

def draw_colored_boxes(image, layout, box_width=3):
    img = image.copy()
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i, block in enumerate(layout):
        x1, y1, x2, y2 = map(int, block.coordinates)
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_width)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

model = lp.Detectron2LayoutModel(
    "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", DETECTION_THRESHOLD],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    device="cpu"
)

pdf_files = [
    os.path.join(INPUT_FOLDER, f)
    for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith('.pdf')
]

for pdf_path in pdf_files:
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"\nPDF: {basename} | Total pages: {num_pages}")

    pil_imgs = convert_from_path(pdf_path, dpi=DPI)
    annotated_imgs = []

    for pg, pil_img in enumerate(pil_imgs, start=1):
        img = np.array(pil_img)
        layout = model.detect(img)
        annotated = draw_colored_boxes(img, layout, box_width=3)
        if annotated.dtype != np.uint8:
            annotated = annotated.astype(np.uint8)
        if annotated.shape[2] == 4:
            annotated = annotated[:, :, :3]
        annotated_pil = Image.fromarray(annotated)
        annotated_imgs.append(annotated_pil)
        print(f"  Processed page {pg}/{num_pages}")

    out_pdf_path = os.path.join(OUTPUT_FOLDER, f"{basename}_annotated.pdf")
    annotated_imgs[0].save(
        out_pdf_path,
        save_all=True,
        append_images=annotated_imgs[1:],
        resolution=DPI
    )
    print(f"  → Saved annotated PDF: {out_pdf_path}") 