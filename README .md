# Facial Emotion Classification

A machine learning course project that compares PCA + KNN, custom convolutional neural networks, and a pretrained ResNet18 for facial expression classification on FER2013.

The project covers image preprocessing, data augmentation, model training, hyperparameter experiments, and evaluation with learning curves and confusion matrices. It classifies facial expressions in images; it does not identify people.

## Methods

- **PCA + KNN:** Standardize flattened grayscale images, reduce their dimensionality with PCA, and compare different PCA dimensions and neighbor counts.
- **Custom CNNs:** Compare a one-convolution baseline, a two-convolution model, and a small VGG-style network with three convolutional blocks.
- **ResNet18:** Fine-tune an ImageNet-pretrained model with a replacement classification head for six emotion classes.
- **Training techniques:** Use image augmentation, normalization, dropout, and experiments with Adam, SGD, and learning-rate scheduling.
- **Evaluation:** Inspect accuracy, loss curves, confusion matrices, and classification reports.

## Dataset

The notebook uses FER2013 in an image-folder format. The original dataset has seven emotion classes. This experiment removes the `disgust` class from both training and test data and uses the remaining six classes:

| Label | Emotion |
| --- | --- |
| `angry` | Anger |
| `fear` | Fear |
| `happy` | Happiness |
| `neutral` | Neutral expression |
| `sad` | Sadness |
| `surprise` | Surprise |

After filtering, the notebook records **28,273 training images** and **7,067 test images**. The custom CNN section splits the filtered training data into 25,445 training images and 2,828 validation images.

The original images are 48 × 48 pixels. Custom CNNs use 48 × 48 inputs, while the ResNet18 pipeline uses 224 × 224 inputs and three-channel grayscale images.

Dataset images and pretrained model weights are not included in this repository. Obtain FER2013 separately from a source that permits your intended use. The exact download source used for the original experiment is not recorded in the supplied notebook.

## Recorded Results

These values come from the saved outputs in the notebook, not from a new training run.

| Model | Recorded accuracy |
| --- | ---: |
| PCA + KNN, 50 components and K = 3 | 42.30% |
| One-convolution CNN | 44.01% |
| Two-convolution CNN | 54.31% |
| Small VGG-style CNN | 65.13% |
| Pretrained ResNet18 | 70.37% |

**Evaluation scope:** These are exploratory course-project results. The KNN search uses test-set accuracy to compare hyperparameters. The ResNet18 section uses the test loader for learning-rate scheduling and checkpoint selection, then evaluates on that same data. Its 70.37% accuracy is therefore not an independent held-out test estimate. Different model sections also use different training setups, so this table is a descriptive comparison rather than a controlled benchmark.

The poster summarizes earlier experiments and contains some differences from the notebook, including the KNN accuracy and CNN epoch count. The table above follows the supplied notebook's saved outputs. These six-class results should not be directly compared with standard seven-class FER2013 benchmarks.

## Repository Files

| File | Description |
| --- | --- |
| [facial_emotion_classification.ipynb](facial_emotion_classification.ipynb) | Training code, saved experiment outputs, and visualizations |
| [project_poster.pdf](project_poster.pdf) | Original team project poster |
| [README.md](README.md) | Project overview and setup instructions |

## Running the Notebook

The notebook was written for a Google Colab environment and uses `/content` paths. A GPU is recommended for the neural-network experiments. The KNN search runs on the CPU and may take substantial time.

1. Open `facial_emotion_classification.ipynb` in Google Colab.
2. Select a GPU runtime for the CNN and ResNet18 sections.
3. Prepare a FER2013 image archive named `archive.zip` and upload it to `/content/archive.zip`. The archive must contain `train/` and `test/` directly at its root, with images grouped into emotion subfolders, such as `train/angry/` and `test/happy/`. A CSV-only download will need conversion first.
4. If dependencies are missing, run this command in a new notebook cell:

   ```python
   %pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow
   ```

5. Run the notebook from top to bottom. The first cell extracts the archive into the working directory; the CNN and ResNet18 sections expect `/content/train` and `/content/test`.
6. The KNN section separately extracts the same archive into the relative directory `./content/fer2013_data`. Keep the original path settings when running the notebook as supplied.

The ResNet18 section downloads ImageNet-pretrained weights on first use, so network access is required. To run locally, update all data paths and replace or adapt the Colab shell commands.

Package versions were not pinned in the original notebook, and the setup instructions have not been validated through a fresh end-to-end training run. Results may vary with random initialization, augmentation, library versions, and hardware.

## Limitations and Next Steps

- Use a separate validation set for all hyperparameter tuning, scheduler decisions, and checkpoint selection. Evaluate the final model on an untouched test set once.
- Give the custom CNN validation split deterministic preprocessing. In the current notebook, it shares the training dataset's random augmentation pipeline.
- Use `copy.deepcopy(model.state_dict())` when saving the best custom CNN checkpoint. The current custom CNN code stores a state-dictionary reference; the ResNet18 section already uses a deep copy.
- Set random seeds consistently and record package versions to improve reproducibility.
- Address class imbalance and report per-class precision, recall, and F1 alongside accuracy.
- Align the poster with the final notebook and compare models under a consistent evaluation protocol.

## Team

This is a team course project by **Xun Lu, Austin He, and Keyu Zhu**, as credited on the project poster.

The repository presents the shared project materials and does not imply sole authorship by the repository owner.
