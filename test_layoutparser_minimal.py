import os
import random
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import layoutparser as lp
import numpy as np
from PIL import Image
import cv2

INPUT_FOLDER = "/mnt/c/Users/nikdu/my_python_projects/layout_parser_linux/test_files/inpuf_files"
OUTPUT_FOLDER = "/mnt/c/Users/nikdu/my_python_projects/layout_parser_linux/test_files/output_layouts"
SAMPLE_SIZE = 3
DPI = 200

# Detection threshold for the model (lower = more sensitive, higher = stricter)
DETECTION_THRESHOLD = 0.3  # Change this value as needed

# Bold, high-contrast color palette (BGR for OpenCV)
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

# If you have PDFs with large watermarks, you may want to post-process detected boxes
# to filter out very large or low-confidence regions, as the model may sometimes pick up
# watermark shapes as layout elements. This can be done by checking the size or aspect ratio
# of detected boxes, or by raising the DETECTION_THRESHOLD.

def draw_colored_boxes(image, layout, box_width=3):
    img = image.copy()
    # Convert RGB to BGR for OpenCV
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i, block in enumerate(layout):
        x1, y1, x2, y2 = map(int, block.coordinates)
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_width)
    # Convert back to RGB for PIL
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Load model once
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
    pages = random.sample(range(1, num_pages + 1), min(SAMPLE_SIZE, num_pages))
    pages.sort()
    print(f"\nPDF: {basename} | Pages: {pages}")

    for pg in pages:
        pil_img = convert_from_path(pdf_path, dpi=DPI, first_page=pg, last_page=pg)[0]
        img = np.array(pil_img)
        print(f"  Page {pg}: img.shape = {img.shape}, dtype = {img.dtype}")

        layout = model.detect(img)
        annotated = draw_colored_boxes(img, layout, box_width=3)
        print(f"    annotated.shape = {annotated.shape}, dtype = {annotated.dtype}")

        # Ensure correct format for PIL
        if annotated.dtype != np.uint8:
            annotated = annotated.astype(np.uint8)
        if annotated.shape[2] == 4:
            annotated = annotated[:, :, :3]

        out_path = os.path.join(OUTPUT_FOLDER, f"{basename}_pg{pg:03d}_test.png")
        Image.fromarray(annotated).save(out_path)
        print(f"    → saved {out_path}") 