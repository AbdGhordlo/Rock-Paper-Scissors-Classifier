import gradio as gr
from fastai.vision.all import *
import pathlib
import platform

# Define all custom functions used in the original DataBlock
def get_x_from_dict(x): return x['image']
def get_y_from_dict(x): return x['label']

# Adjust pathlib for Windows if necessary
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# Load the exported learner
path = Path()
learn = load_learner(path/'rps_model.pkl', cpu=True)

# Get the class labels from the learner's DataLoaders
labels = learn.dls.vocab

def predict_image(img):
    """Predicts the class of an input image."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)  # Convert numpy to PIL
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Create examples list
example_files = ['Rock.png', 'Paper.png', 'Scissors.png']
examples = [[f"examples/{f}"] for f in example_files if Path(f"examples/{f}").exists()]

# Gradio Interface
title = "Rock, Paper, Scissors Classifier"
description = (
    "Upload an image of a hand gesture (rock, paper, or scissors), "
    "and this model will predict which one it is. "
    "Model based on ResNet18, trained with fastai."
)
article = "<p style='text-align: center'><a href='https://www.tensorflow.org/datasets/catalog/rock_paper_scissors' target='_blank'>TensorFlow Rock, Paper, Scissors Dataset</a> | <a href='https://github.com/fastai/fastai' target='_blank'>fastai Library</a></p>"

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Hand Gesture Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging='never',
    analytics_enabled=True,
    theme=gr.themes.Soft()
)

if __name__ == '__main__':
    iface.launch()