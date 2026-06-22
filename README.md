# TRAFSBAR
Leveraging Trajectory Cues for Few-Shot Basketball Action Recognition
<img width="3652" height="2109" alt="Image" src="https://github.com/user-attachments/assets/2a01fa0c-07fc-4f29-a1f5-7748368f8491" />


Conventional action recognition methods usually rely on large-scale annotated data, which limits their generalization ability when only a few samples are available for target classes. Few-shot action recognition addresses this issue by recognizing novel action categories with limited support samples. Compared with daily-life actions, basketball videos are more challenging due to rapid scene changes, frequent multi-person interactions, severe occlusion, and subtle visual differences between action categories.

This study proposes a few-shot action recognition method that incorporates trajectory information. We adopt an object-guided point sampling strategy to focus trajectory points on action-relevant regions, such as players and the basketball. In addition, a trajectory branch is designed as a plug-in auxiliary signal, enabling the model to consider both appearance and motion information. Experiments on FineSports and MultiSports using TAMT, TEAM, and D2ST-Adapter show that trajectory information improves accuracy in most datasets and sample settings, demonstrating its practical value for fine-grained sports video classification.
