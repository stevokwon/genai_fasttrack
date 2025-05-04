# 🧠 `DCGAN-MNIST`: Generating Handwritten Digits with GANs

> Part of the **GenAI FastTrack** series – building foundational generative AI capabilities from scratch.

---

## 📌 Overview

This project implements a Deep Convolutional GAN (DCGAN) to generate synthetic handwritten digits using the MNIST dataset.  
It integrates real-world GAN training techniques to overcome instability, improve generator learning, and visualize progress during training — even on CPU.

---

## 🚀 Highlights

- ✅ Generator & Discriminator fully built from scratch using PyTorch
- ✅ Dynamic label smoothing, dropout, and input noise to stabilize training
- ✅ Generator progress saved every 200 steps for **live visual feedback**
- ✅ Fully CPU-compatible for resource-constrained training
- ⚠️ Training run demonstrated until ~step 400 (partial run)

---

## 🗂️ Project Structure

```bash
genai_rl_agents/
├── dcgan_mnist.py                # 🔧 Main training script
├── output/                       # 📸 Generated samples
│   ├── fake_samples_epoch0_step0.png
│   └── ...
├── data/                         # (excluded) MNIST raw dataset
├── .gitignore
├── README.md
```
---

## 📦 Setup
Install required packages:
```bash
pip install -r requirements.txt
```
---
## ▶️ Run Training
```bash
python dcgan_mnist.py
```
Training will:
* Print `Loss_D` and `Loss_G` every 100 batches
* Save generated image samples every 200 steps
* Output saved in `output/` folder
> ⏳ CPU mode is supported (used in this run), but training may take longer.

> ⚠️ Note: Training on CPU may take 5–10× longer than on GPU.
---
## 📈 Sample Results
Here are generated images from an early-stage training run (~step 400):
```markdown
| Step | Generated Image |
|------|-----------------|
| Step 0 | ![](output/fake_samples_epoch0_step0.png) |
| Step 400 | ![](output/fake_samples_epoch0_step400.png) |
```
> ⚠️ Full training was not completed due to CPU limitations.
> Generator quality improves further beyond ~5 epochs on GPU.
---
## 🧪 Training Tricks Used
| Technique | Purpose |
|------|-----------------|
| Label Smoothing (0.8-1.0) | Prevent overconfident Discriminator |
| Gaussian Noise on Real Inputs | Regularize Discriminator |
| Dropout in D (0.3) | Reduce overfitting |
| Fixed Noise | Consistent image comparison over time |
---
## 💡 What I Learned
* How GANs pit two networks (G vs D) in adversarial training
* Common training imbalances and stabilization tricks
* Live-saving of fake image samples for monitoring progress
* How to build and train deep learning models even without GPU
---
## 📚 References
```markdown
- [DCGAN Paper (Radford et al., 2015)](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
```
---
## 🏁 Next Steps
* Complete full training (5–10 epochs on GPU)
* Create animated GIF showing Generator improvement
* Add model checkpoint saving
* Log loss curves over training
* Try Conditional GAN (CGAN) on digit labels
