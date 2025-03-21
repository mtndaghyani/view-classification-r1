import json
import random
import os

PATH_TO_DATA = '/arc/project/st-puranga-1/users/matin/datasets/view_classification/train/'  # To the train directory
SAMPLE_PER_CLASS = 100  # How many samples per class
rules = "Justify your classification based on the following key elements:\n" \
        "- Key Structures: Identify which anatomical structures are visible in the image. Options include:\n" \
        "  - Left Atrium (LA)\n" \
        "  - Right Atrium (RA)\n" \
        "  - Left Ventricle (LV)\n" \
        "  - Right Ventricle (RV)\n" \
        "  - Mitral Valve\n" \
        "  - Tricuspid Valve\n" \
        "  - Aortic Valve (AV)\n" \
        "  - Left Ventricular Outflow Tract (LVOT)\n" \
        "  - Inferior Vena Cava (IVC)\n" \
        "  - Papillary Muscles\n" \
        "  - Aortic Arch and its branches\n" \
        "- Orientation: Determine the orientation of the heart in the image. Options include:\n" \
        "  - Apex of the heart at the top\n" \
        "  - Circular cross-sectional view of the heart\n" \
        "  - Long axis of the heart\n" \
        "  - Subcostal view (heart viewed from beneath)\n" \
        "  - Superior view (viewing the aortic arch from above)\n" \
        "- Distinguishing Features: Highlight distinguishing features that help identify the view. Options include:\n" \
        "  - Presence of two, three, or four chambers\n" \
        "  - Visibility of valves (e.g., mitral, tricuspid, aortic)\n" \
        "  - Visibility of specific vessels (e.g., IVC, aortic arch)\n" \
        "  - Presence of papillary muscles or LVOT\n" \
        "  - Relative sizes and positions of the visible structures"

prompt = "Classify the view of this echocardiography image into: AP2, AP3, AP4, AP5," \
         " PLAX, RVIF, SUBC4, SUBC5, SUBIVC, PSAXAo PSAXM, PSAXPM, PSAXAp, SUPRA.\n" \
         f" {rules}\n" \
         "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:" \
         "<think> ... </think> <answer>view</answer>" \
         "Please strictly follow the format."

random.seed(42)

labels = [
    "AP2", "AP3", "AP4", "AP5",
    "PLAX", "RVIF",
    "SUBC4", "SUBC5", "SUBIVC",
    "PSAXAo", "PSAXM", "PSAXPM", "PSAXAp",
    "SUPRA",
    "OTHER"
]

result = []
for filename in os.listdir(PATH_TO_DATA):
    label = labels[int(filename)]  # Ground Truth Label
    if label == "OTHER":
        continue
    directory = os.path.join(PATH_TO_DATA, filename)
    class_data = os.listdir(directory)
    random.shuffle(class_data)
    num_samples = SAMPLE_PER_CLASS
    if len(class_data) < SAMPLE_PER_CLASS:
        num_samples = len(class_data)
        print(f"{label} has {num_samples} samples.")
    for i in range(num_samples):
        result.append({
            "image_path": os.path.join(directory, class_data[i]),
            "problem": prompt,
            "solution": f"<answer>{label}</answer>"
        })

print(f'Length = {len(result)}')
with open('view_data.json', 'w') as file:
    json.dump(result, file)
