import cv2
import torch

from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_image, load_pdf

if __name__ == "__main__":
    filename = "drugstore_flyer"
    pdf_filepath = f"./images/{filename}.pdf"

    image = load_pdf(pdf_filepath)
    analyzer = DocumentAnalyzer(
        configs={},
        visualize=True,
        device='mps'
    )

    results, ocr_vis, layout_vis = analyzer(image[0])

    # to HTML
    # results.to_html(f"./outputs/{filename}.html")

    # to image
    cv2.imwrite(f"./outputs/{filename}_ocr.jpg", ocr_vis)
    cv2.imwrite(f"./outputs/{filename}_layout.jpg", layout_vis)
