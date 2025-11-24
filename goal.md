# High-Performance Visual Place Verification System

This final iteration incorporates all the best practices we have discussed: the superior **EfficientNet** architecture, the robust **Combined Metric** scoring, and the crucial **Human-in-the-Loop** system for continuous improvement.

## Project Goal

To deploy a high-confidence, low-latency verification system that confirms a user's presence at a specific landmark (e.g., Wawel Castle). The system will utilize a modern backbone and a robust combined metric, with a self-improving loop to ensure long-term accuracy against evolving data conditions.

---

## Technology Stack (Optimized)

- **Core Framework:** PyTorch
- **Backbone Model:** **EfficientNet (e.g., B4/B5)** (Pre-trained on **Google Landmarks Dataset v2 - GLDv2**)
- **Feature Vector Dimension:** 1280 or 1536 (Model dependent, the output of the Global Average Pooling layer)
- **Feature Extraction:** Siamese Metric Learning Architecture
- **Verification Metric:** **Combined Similarity Score** (Weighted average of Cosine Similarity and Normalized Euclidean Distance)

---

## Architectural Pipeline

### Stage 1: Pre-Processing and Occlusion Handling (CRITICAL)

The user's selfie must be segmented to isolate the landmark features.

1. **Semantic Segmentation:** Implement a segmentation model (e.g., DeepLabV3) to detect and mask the **"person"** class in the user's selfie.
2. **Mask Application:** Set the person's pixels to a neutral value (e.g., black) to prevent the feature extractor from being biased by the face.
3. **Output:** A clean image tensor containing only the background landmark.

---

### Stage 2: Reference Gallery Construction (Offline Batch Processing)

This step runs only once upon system initialization for maximum speed.

1. **Gallery Data Collection:** Gather *N* diverse reference images (Positive Anchors) covering different viewpoints and lighting conditions.
2. **Batch Preparation:** Use `torch.stack()` to combine all *N* pre-processed image tensors into **one single batch tensor** (Shape: *N × 3 × H × W*).
3. **Feature Extraction:** Pass the single batch tensor through the EfficientNet model in one fast inference step.
4. **Storage:** Store the resulting embedding matrix (Shape: *N × D*, where *D* is the feature dimension) in memory. This single matrix is the final Reference Gallery.

---

### Stage 3: Verification and Combined Scoring (Online Step)

This is the real-time core loop for verification.

1. **Query Feature Extraction:** Pass the processed user selfie through the EfficientNet model to generate **Vector A** (Shape: *1 × D*).
2. **Metric Calculation:** Compare **Vector A** against the entire Reference Gallery Matrix to simultaneously calculate:
   - **Cosine Similarity (S_cos):** Measures angle/direction.
   - **Euclidean Distance (D_euc):** Measures physical closeness/magnitude.
3. **Combined Score:** Normalize *D_euc* and calculate a final weighted score (*S_combined*) that must satisfy both directional and magnitude closeness requirements.
4. **Max-Similarity Decision:** Take the **Maximum S_combined Score** from all gallery comparisons.

| Score | Decision |
|:------|:---------|
| Score > T_verify | **TRUE** (Location Verified) |
| Score ≤ T_verify | **FALSE** (Verification Failed) |

---

### Stage 4: Continuous Improvement and Training Data Acquisition

This implements the essential feedback loop for system improvement.

1. **Acquisition Gate:** Flag the original (unmasked) user image if the verification score is significantly high (≥ *T_acquire*, e.g., **0.95**).
2. **Human Verification:** Place the flagged image into a **Review Queue** where a human confirms the photo's authenticity and novelty (new angle, unique light). This step prevents **Poisoning Attacks**.
3. **Data Labeling:** The human-approved image is permanently labeled and saved as a **Positive Anchor** for Wawel Castle.
4. **Re-training Cycle:** Periodically (e.g., monthly), the EfficientNet model is **re-trained** using the augmented dataset and a **Learned Metric Loss** (e.g., ArcFace, Triplet Loss) to explicitly force better separation in the embedding space.
5. **Deployment:** The new model version and its updated Reference Gallery replace the old one in production.

---

## Deliverables

| ID | Task Description | Status   |
|:---|:-----------------|:---------|
| **P-1** | Research and acquire **EfficientNet** (B4/B5) weights pre-trained on GLDv2. | Done     |
| **P-2** | Implement the pre-processing module, including semantic segmentation for person removal. | Done     |
| **G-2** | Implement the `load_gallery` function using `torch.stack()` for efficient batching and storing the final *N × D* embedding matrix. | Done     |
| **V-1** | Implement the core verification function using **Combined Similarity** (*S_combined*) against the Reference Matrix. | Done     |
| **V-2** | Establish and document a reliable baseline for the **Verification Threshold** (*T_verify*). | Pending  |
| **V-3** | Implement optional security check using **Negative Anchors** (other landmarks/castles) for added robustness. | Optional |
| **CI-1** | Define the **Acquisition Threshold** (*T_acquire*) and implement the logic to flag high-confidence user photos for future training. | Pending  |
| **CI-2** | Design the **Human-in-the-Loop** review pipeline and the periodic **Model Re-training** and **Deployment** strategy. | Pending  |