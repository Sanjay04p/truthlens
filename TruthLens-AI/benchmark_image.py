import os
import glob
from backend_server.pipelines.image_pipeline_local import ImageVerificationPipeline

pipeline = ImageVerificationPipeline()

real_images = glob.glob("test_data/images/real/*.*")
fake_images = glob.glob("test_data/images/fake/*.*")

dataset = [(p, "Real") for p in real_images] + [(p, "Fake") for p in fake_images]

def evaluate(mode="hybrid"):
    tp, tn, fp, fn = 0, 0, 0, 0
    fallbacks = 0

    for path, ground_truth in dataset:
        if mode == "siglip_only":
            try:
                score = pipeline._primary_hf_analysis(path)
                pred = "Fake" if score >= 50.0 else "Real"
            except Exception:
                pred = "Real"
        elif mode == "sightengine_only":
            try:
                score = pipeline._fallback_sightengine_analysis(path)
                pred = "Fake" if score >= 50.0 else "Real"
            except Exception:
                pred = "Real"
        else:  # Hybrid: SigLIP2 + Fallback
            res = pipeline.verify_image(path)
            pred = res.get("label", "Real")
            if "Sightengine" in res.get("provider_used", ""):
                fallbacks += 1

        if ground_truth == "Fake" and pred == "Fake":
            tp += 1
        elif ground_truth == "Real" and pred == "Real":
            tn += 1
        elif ground_truth == "Real" and pred == "Fake":
            fp += 1
        elif ground_truth == "Fake" and pred == "Real":
            fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fallback_rate = (fallbacks / total * 100) if total else 0.0

    return acc, f1, fpr, fallback_rate

# Run evaluations across all three rows
acc_s, f1_s, fpr_s, _ = evaluate("siglip_only")
acc_e, f1_e, fpr_e, _ = evaluate("sightengine_only")
acc_h, f1_h, fpr_h, fb_rate = evaluate("hybrid")

print("\n" + "="*65)
print("TABLE II: IMAGE DETECTION PERFORMANCE (MEASURED)")
print("="*65)
print(f"{'Method':<22} | {'Acc.':<8} | {'F1':<8} | {'FPR (real)':<12} | {'Fallback Rate':<12}")
print("-" * 65)
print(f"{'SigLIP2 only':<22} | {acc_s:<8.3f} | {f1_s:<8.3f} | {fpr_s:<12.3f} | {'--':<12}")
print(f"{'Sightengine only':<22} | {acc_e:<8.3f} | {f1_e:<8.3f} | {fpr_e:<12.3f} | {'--':<12}")
print(f"{'SigLIP2 + fallback':<22} | {acc_h:<8.3f} | {f1_h:<8.3f} | {fpr_h:<12.3f} | {f'{fb_rate:.1f}%':<12}")
print("="*65)