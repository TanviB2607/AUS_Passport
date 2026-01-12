from mrz_reader import MRZReader
from tkinter import Tk, filedialog

# Hide main tkinter window
def clean_text(ocr_list):
    if len(ocr_list) == 0:
        return ""

    texts = []
    for item in ocr_list:
        texts.append(item[1])   # only text

    return " ".join(texts)

Tk().withdraw()

# Open file picker
image_path = filedialog.askopenfilename(
    title="Select passport image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("No image selected!")
    exit()

print("Selected image:", image_path)

# Initialize reader
reader = MRZReader(
    easy_ocr_params={"lang_list": ["en"], "gpu": False}
)

# Run pipeline
result, boxes, face = reader.predict(image_path)

print("\nDETECTED BOXES:")
print(boxes)

print("\nOCR RESULTS:")
print(result)

print("\nFINAL EXTRACTED DATA:\n")

final_output = {}

for field, ocr in result.items():
    final_output[field] = clean_text(ocr)

print("\nFINAL EXTRACTED DATA (SORTED):\n")

for k in sorted(final_output):
    print(f"{k} : {final_output[k]}")

