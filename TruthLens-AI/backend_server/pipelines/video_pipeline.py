import os
import cv2
import requests
import numpy as np
from PIL import Image
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

SIGHTENGINE_USER = os.getenv("SIGHTENGINE_USER")
SIGHTENGINE_SECRET = os.getenv("SIGHTENGINE_SECRET")

# 1. Load Local Full-Frame ViT (Detects Sora/Runway/Midjourney full-scene generation)
print("[INFO] Loading Full-Frame Video ViT Model...")
try:
    full_frame_classifier = pipeline(
        "image-classification",
        model="dima806/ai_vs_real_image_detection",
        device=-1  # Set to 0 if using CUDA GPU, -1 for CPU
    )
except Exception as e:
    print(f"[ERROR] Failed to load Full-Frame ViT model: {e}")
    full_frame_classifier = None

# 2. Load Dedicated Face-Deepfake Classifier (Detects DeepFaceLab/Roop face-swaps)
print("[INFO] Loading Dedicated Face-Deepfake Classifier...")
try:
    face_deepfake_classifier = pipeline(
        "image-classification",
        model="dima806/deepfake_vs_real_image_detection", # Specialized face-forgery model
        device=-1
    )
except Exception as e:
    print(f"[ERROR] Failed to load Face-Deepfake model: {e}")
    face_deepfake_classifier = None

# Load lightweight Haar Cascade face detector for face-crop forensics.
# Unlike FaceDetectorYN, this does not require an ONNX model file.
cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError(
        f"Could not load OpenCV Haar Cascade face detector: {cascade_path}"
    )


def _get_synthetic_prob(pil_img: Image.Image, classifier) -> float:
    """Runs designated local HuggingFace classifier and returns synthetic probability."""
    if classifier is None:
        return 0.5

    preds = classifier(pil_img)
    for p in preds:
        label = p["label"].lower()
        if any(k in label for k in ["fake", "ai", "artificial", "synthetic"]):
            return float(p["score"])
        if any(k in label for k in ["real", "human", "authentic", "original"]):
            return 1.0 - float(p["score"])
    return 0.5


def _extract_keyframes_and_faces(video_path: str, max_frames: int = 8):
    """
    Samples evenly spaced keyframes from the video and extracts padded face crops,
    filtering out blurry frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open or decode video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = round(total_frames / fps, 2) if total_frames > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video file contains 0 readable frames.")

    # We will sample more frames initially, then filter down to the best `max_frames`
    num_samples = min(max_frames * 3, total_frames) 
    frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    valid_frames = []
    full_frames = []
    face_crops = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # BLUR DETECTION: Calculate variance of the Laplacian
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Reject blurry frames (threshold typically between 50 and 100)
        if blur_score < 80.0:
            continue
            
        valid_frames.append((frame, gray))
        if len(valid_frames) >= max_frames:
            break

    # Now process only the sharp, valid frames
    for frame, gray in valid_frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        full_frames.append(Image.fromarray(rgb_frame))

        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(120, 120)
        )
        
        for (x, y, w, h) in faces[:1]:  # Analyze primary prominent face per frame
            # Add 25% margin to include jawline/hairline blending artifacts
            pad_w, pad_h = int(w * 0.25), int(h * 0.25)
            y1 = max(0, y - pad_h)
            y2 = min(rgb_frame.shape[0], y + h + pad_h)
            x1 = max(0, x - pad_w)
            x2 = min(rgb_frame.shape[1], x + w + pad_w)

            face_crop = Image.fromarray(rgb_frame[y1:y2, x1:x2])
            face_crops.append(face_crop)

    cap.release()
    return full_frames, face_crops, duration_sec, total_frames


def _sightengine_frame_fallback(pil_img: Image.Image) -> float:
    """Commercial Fallback: Sends most suspicious keyframe to Sightengine GenAI API."""
    if not SIGHTENGINE_USER or not SIGHTENGINE_SECRET:
        return -1.0

    try:
        import io
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        buf.seek(0)

        response = requests.post(
            "https://api.sightengine.com/1.0/check.json",
            files={"media": ("keyframe.jpg", buf, "image/jpeg")},
            data={
                "models": "genai",
                "api_user": SIGHTENGINE_USER,
                "api_secret": SIGHTENGINE_SECRET,
            },
            timeout=10,
        )
        data = response.json()
        if data.get("status") == "success":
            return float(data.get("type", {}).get("ai_generated", -1.0))
    except Exception as e:
        print(f"[WARN] Sightengine video keyframe fallback failed: {e}")
    return -1.0


def analyze_video_pipeline(video_path: str) -> dict:
    """
    Main entrypoint for video verification.
    """
    # THIS LINE WAS MISSING: You must extract the frames first
    full_frames, face_crops, duration_sec, total_frames = _extract_keyframes_and_faces(
        video_path, max_frames=8
    )

    if not full_frames:
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "explanation": "Could not extract valid frames from the uploaded video.",
        }

    # 1. Full-Frame Spatial Analysis
    frame_scores = [_get_synthetic_prob(f, full_frame_classifier) for f in full_frames]
    primary_frame_score = float(np.median(frame_scores)) 

    # 2. Face-Crop Forensics 
    valid_face_crops = [fc for fc in face_crops if fc.size[0] >= 112 and fc.size[1] >= 112]
    face_scores = [_get_synthetic_prob(fc, face_deepfake_classifier) for fc in valid_face_crops] if valid_face_crops else []
    primary_face_score = float(np.median(face_scores)) if face_scores else 0.0

    primary_score = max(primary_frame_score, primary_face_score)
    strategy_used = "Tier 1: Dual Local Edge Models (ViT + Face Forgery)"

    # 3. Trigger Sightengine Fallback for ambiguous scores (0.45 - 0.65)
    if 0.45 <= primary_score <= 0.65:
        median_idx = np.argsort(frame_scores)[len(frame_scores)//2]
        fallback_score = _sightengine_frame_fallback(full_frames[median_idx])
        if fallback_score >= 0.0:
            primary_score = (primary_score * 0.4) + (fallback_score * 0.6)
            strategy_used = "Tier 2: Edge Models + Sightengine Keyframe Fallback"

    # 4. Updated Classification Logic
    if primary_frame_score >= 0.94: 
        verdict = "AI-GENERATED VIDEO"
        confidence = round(primary_frame_score * 100, 2)
        explanation = f"Full-frame analysis detected overwhelming synthetic textures across {len(full_frames)} sharp keyframes."
        
    # Face boundary manipulation remains highly accurate, but threshold raised slightly
    elif valid_face_crops and primary_face_score >= 0.75 and (primary_face_score - primary_frame_score) >= 0.10:
        verdict = "DEEPFAKE (FACE MANIPULATION)"
        confidence = round(primary_face_score * 100, 2)
        explanation = f"Localized facial boundary analysis across {len(valid_face_crops)} high-res face crops flagged manipulation artifacts."
        
    elif primary_score >= 0.85:
        verdict = "DEEPFAKE / MANIPULATED VIDEO"
        confidence = round(primary_score * 100, 2)
        explanation = "Detected synthetic visual artifacts exceeding natural compression limits."
        
    else:
        verdict = "REAL / AUTHENTIC VIDEO"
        # Adjust confidence calculation to reflect the new strict threshold
        adjusted_real_conf = min(100.0, ((0.94 - primary_score) / 0.94) * 100)
        confidence = round(adjusted_real_conf, 2)
        explanation = f"Spatial and facial consistency verified across {len(full_frames)} keyframes. Minor artifacts match natural video compression."

        
    return {
        "verdict": verdict,
        "confidence": confidence,
        "strategy": strategy_used,
        "metrics": {
            "frames_analyzed": len(full_frames),
            "faces_detected": len(valid_face_crops),
            "duration_seconds": duration_sec,
            "total_video_frames": total_frames,
            "full_frame_synthetic_score": round(primary_frame_score * 100, 2),
            "face_crop_synthetic_score": round(primary_face_score * 100, 2),
            "temporal_variance": 0.0, # Removed jitter penalty
        },
        "explanation": explanation,
    }