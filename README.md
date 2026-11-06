# Coral Reef Health Monitor: live demo

Classifies an underwater coral photograph as **healthy** or **bleached**
(MobileNetV2, ImageNet transfer learning) and shows a **Grad-CAM** heat map of
the regions that drove the decision. Underwater images are first restored
(Gray-World, CLAHE, bilateral filter, unsharp mask).

This repository holds only what the hosted Streamlit demo needs. Research code,
training pipeline and full results are in a separate private repository.

- 5-fold cross-validation accuracy of this model: 74.1 % ± 3.2 on 896 images
- Research prototype: not a substitute for an expert reef survey
- Sample photographs: from the Kaggle dataset
  `vencerlanz09/healthy-and-bleached-corals-image-classification` (Flickr)

Run locally: `pip install -r requirements.txt && streamlit run streamlit_cloud/streamlit_app.py`
