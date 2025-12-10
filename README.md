## Project Overview

This project is a deep learning-based system for classifying Arabic and English alphabet characters. It involves a complete machine learning pipeline, from data loading and visualization to data augmentation and training of multiple deep learning models. The goal is to develop a robust classifier that can recognize characters from both languages.

The project uses a variety_ of popular Python libraries for data science and deep learning, including:
- **`TensorFlow` and `PyTorch`**: For building and training the deep learning models.
- **`Pandas`**: For data manipulation and loading CSV files.
- **`Matplotlib` and `Seaborn`**: For data visualization.
- **`OpenCV`**: For image processing and data augmentation.
- **`Scikit-learn`**: For machine learning utilities and metrics.

## Project Structure

The project is organized into several key components:

- **`Data/`**: Contains the raw datasets for Arabic and English characters, likely in CSV format.
- **`augmented_images/`**: Directory for storing images created during the data augmentation process.
- **`Final Model NoteBooks/`**: Contains Jupyter notebooks for training different deep learning models, including InceptionV1, MobileNet, and ResNet.
- **Jupyter Notebooks**: The root directory contains several notebooks that define the project's workflow:
    - `Visualize_raw_data.ipynb`: For initial data exploration and visualization.
    - `preprocessing-and-merge.ipynb`: For preprocessing the raw data and merging the datasets.
    - `augmentation.ipynb`: For augmenting the training data to improve model performance.

## Building and Running

This project is based on Jupyter Notebooks, so there is no single command to run the entire project. Instead, the notebooks should be run in a specific order to follow the machine learning pipeline.

### 1. Setup the Environment

To run this project, you need to have Python and the required libraries installed. You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
pip install tensorflow torch torchvision opencv-python scikit-learn
```

### 2. Running the Notebooks

The notebooks should be run in the following order:

1.  **`Visualize_raw_data.ipynb`**: To inspect the raw data and understand its characteristics.
2.  **`preprocessing-and-merge.ipynb`**: To preprocess the data and save it into a format suitable for training (e.g., `.npy` files).
3.  **`augmentation.ipynb`**: To augment the training data and create a larger, more diverse dataset.
4.  **`Final Model NoteBooks/*.ipynb`**: To train the deep learning models on the augmented data. You can run any of the notebooks in this directory to train a specific model (e.g., `resnet-alphabet-classification.ipynb`).

## Development Conventions

- **Notebook-driven workflow**: The project is organized as a series of Jupyter notebooks, each responsible for a specific part of the machine learning pipeline.
- **Data augmentation**: The project heavily relies on data augmentation to improve model accuracy and generalization.
- **Experimentation with multiple models**: The `Final Model NoteBooks` directory suggests that the project encourages experimenting with different model architectures to find the best solution.
- **Use of `.npy` files**: The project uses `.npy` files to store preprocessed and augmented data, which is an efficient way to handle large datasets in NumPy.

## Notebook Outputs (Images)

This section contains key visualizations and outputs generated from the Jupyter notebooks.

### Data Visualization (`Visualize_raw_data.ipynb`)

*   **Data Distribution (Pre-balancing)**
    *   ![Data Distribution Pre Balancing](./assets/Data%20Dist%20Pre%20Balancing.png)
    *   *Description: Shows the initial distribution of labels in the datasets before any balancing is applied.*

*   **Data Distribution (After-balancing)**
    *   ![Data Distribution After Balancing](./assets/Data%20Dist%20After%20Balancing.png)
    *   *Description: Shows the distribution of labels after balancing the datasets.*

### Augmentation (`augmentation.ipynb`)

*   **Data Pre-Augmentation**
    *   ![Data Pre Augmentation](./assets/Data%20Pre%20Augmentaion.png)
    *   *Description: A sample of the original data before augmentation.*

*   **Data Post-Augmentation**
    *   ![Data Post Augmentation](./assets/Data%20Post%20Augmentation.png)
    *   *Description: A sample of the data after applying augmentation techniques.*

*   **ResNet Data Augmentation Example**
    *   ![Resnet Data Augmentation](./assets/Resnet%20Data%20Augmentation.png)
    *   *Description: An example of the augmented data used for training the ResNet model.*

### Model Performance (`Final Model NoteBooks/*.ipynb`)

*   **InceptionV1 - ROC Curve**
    *   ![InceptionV1 ROC](./assets/InceptionV1%20ROC.png)
    *   *Description: The ROC curve for the InceptionV1 model, showing its performance at various classification thresholds.*

*   **InceptionV1 - Test Samples**
    *   ![InceptionV1 Test Samples](./assets/InceptionV1%20Test%20Samples.png)
    *   *Description: Examples of test samples with their predicted labels from the InceptionV1 model.*

*   **MobileNet - Accuracy Distribution**
    *   ![Mobile Net Accurarcy Distribution](./assets/Mobile%20Net%20Accurarcy%20Distribution.png)
    *   *Description: The distribution of accuracies for the MobileNet model across different classes.*

*   **MobileNet - Confusion Matrix**
    *   ![Mobilenet Confusion matrix](./assets/Mobilenet%20Confusion%20matrix.png)
    *   *Description: The confusion matrix for the MobileNet model, illustrating the true vs. predicted labels.*

---

## Model Accuracies

This section is for documenting the performance metrics, particularly the accuracy, of the trained deep learning models.

*   **ResNet Model Accuracies**:
    *   Arabic Characters Accuracy: 87.32%
    *   English Characters Accuracy: 82.12%
    *   Overall Test Accuracy: 84.81%
    *   _Notes: Best performance achieved after hyperparameter tuning._

    ### Top 10 Best Performing Classes (ResNet)
    | Class | Accuracy | Samples |
    |---|---|---|
    | V | 90.00% | 120 |
    | د | 91.67% | 120 |
    | ر | 91.67% | 120 |
    | Z | 91.67% | 120 |
    | م | 92.50% | 120 |
    | S | 92.50% | 120 |
    | هـ | 93.33% | 120 |
    | أ | 96.67% | 120 |
    | X | 96.67% | 120 |
    | O | 98.33% | 120 |

    ### Top 10 Worst Performing Classes (ResNet)
    | Class | Accuracy | Samples |
    |---|---|---|
    | G | 55.83% | 120 |
    | ت | 57.50% | 120 |
    | ف | 57.50% | 120 |
    | L | 61.67% | 120 |
    | B | 62.50% | 120 |
    | Q | 68.33% | 120 |
    | ق | 71.67% | 120 |
    | خ | 72.50% | 120 |
    | H | 73.33% | 120 |
    | Y | 74.17% | 120 |

*   **MobileNet Model Accuracy**:
    *   Overall Test Accuracy: 74.21%
    *   _Notes: Achieved using transfer learning._

*   **InceptionV1 Model Accuracy**:
    *   Final Test Accuracy: 74.21%
    *   _Notes: Best Validation Accuracy: 66.50%; Test Loss: 0.8628._

---