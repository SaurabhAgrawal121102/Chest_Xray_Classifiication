from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
import io
from pathlib import Path


# =========================================================
# 1. DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("======================================")
print("Chest X-Ray Classification Backend")
print("======================================")
print("Using device:", device)


# =========================================================
# 2. MODEL PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "chest_xray_model.pth"


# =========================================================
# 3. CREATE EFFICIENTNET-B0 MODEL
# =========================================================

model = efficientnet_b0(weights=None)

# Original EfficientNet-B0 classifier:
# Linear(1280, 1000)
#
# We need 2 classes:
# NORMAL
# PNEUMONIA

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)


# =========================================================
# 4. LOAD CHECKPOINT
# =========================================================

print("Loading model from:", MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# Your .pth file contains:
#
# {
#     "model_state_dict": ...,
#     "class_names": ...,
#     "best_val_loss": ...,
#     "epochs_completed": ...,
#     "image_size": ...
# }


model.load_state_dict(
    checkpoint["model_state_dict"]
)


# =========================================================
# 5. GET INFORMATION FROM CHECKPOINT
# =========================================================

classes = checkpoint.get(
    "class_names",
    ["NORMAL", "PNEUMONIA"]
)

image_size = checkpoint.get(
    "image_size",
    224
)

print("Classes:", classes)
print("Image size:", image_size)

if "best_val_loss" in checkpoint:
    print(
        "Best validation loss:",
        checkpoint["best_val_loss"]
    )

if "epochs_completed" in checkpoint:
    print(
        "Epochs completed:",
        checkpoint["epochs_completed"]
    )


# =========================================================
# 6. MOVE MODEL TO DEVICE
# =========================================================

model.to(device)

model.eval()

print("Model loaded successfully!")
print("======================================")


# =========================================================
# 7. IMAGE TRANSFORMATION
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (image_size, image_size)
    ),

    transforms.ToTensor(),

    # ImageNet normalization
    # commonly used when training EfficientNet
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 8. FASTAPI APPLICATION
# =========================================================

app = FastAPI(

    title="Chest X-Ray Classification API",

    description=(
        "EfficientNet-B0 based chest X-ray "
        "classification API for Normal and Pneumonia cases."
    ),

    version="1.0.0"
)


# =========================================================
# 9. HOME ROUTE
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Chest X-Ray Classification API is running",
        "model": "EfficientNet-B0",
        "classes": classes,
        "device": str(device)
    }


# =========================================================
# 10. HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


# =========================================================
# 11. PREDICTION ROUTE
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Check file type
    # -----------------------------------------------------

    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )


    try:

        # -------------------------------------------------
        # Read uploaded image
        # -------------------------------------------------

        image_bytes = await file.read()


        # -------------------------------------------------
        # Convert bytes -> PIL image
        # -------------------------------------------------

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        # -------------------------------------------------
        # Preprocess image
        # -------------------------------------------------

        image_tensor = transform(image)


        # -------------------------------------------------
        # Add batch dimension
        # [3, 224, 224]
        #
        # becomes
        #
        # [1, 3, 224, 224]
        # -------------------------------------------------

        image_tensor = image_tensor.unsqueeze(0)


        # -------------------------------------------------
        # Move image to GPU / CPU
        # -------------------------------------------------

        image_tensor = image_tensor.to(device)


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        with torch.no_grad():

            outputs = model(
                image_tensor
            )


            # Convert logits to probabilities
            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            # Get highest probability
            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )


        # -------------------------------------------------
        # Get prediction
        # -------------------------------------------------

        predicted_index = predicted.item()

        predicted_class = classes[
            predicted_index
        ]


        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        confidence_value = (
            confidence.item() * 100
        )


        # -------------------------------------------------
        # Get probabilities for both classes
        # -------------------------------------------------

        class_probabilities = {}

        for i, class_name in enumerate(classes):

            class_probabilities[class_name] = round(
                probabilities[0][i].item() * 100,
                2
            )


        # -------------------------------------------------
        # Return response
        # -------------------------------------------------

        return {

            "filename": file.filename,

            "prediction": predicted_class,

            "confidence": round(
                confidence_value,
                2
            ),

            "probabilities": class_probabilities

        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )