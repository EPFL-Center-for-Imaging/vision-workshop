import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo


@app.cell(hide_code=True)
async def _(Path, dataset_path, public_folder_path, requests, zipfile):
    import sys
    import os

    # Seaborn isn't installed by default in Pyodide, so we install it here (only if the notebook runs on WebAssembly):
    if "pyodide" in sys.modules:
        import micropip
        await micropip.install("seaborn")
        import seaborn as sns

        # Create the `public` folder
        if not public_folder_path.exists():
            os.mkdir(public_folder_path)

        # Download and unzip the dataset from the repository
        if not dataset_path.exists():
            with mo.status.spinner(title="⏬ Downloading the dataset...") as _spinner:
                zip_path = Path("public") / "dataset.zip"
                url = mo.notebook_location() / "public" / "dataset.zip"
                r = requests.get(str(url))
                r.raise_for_status()
                zip_path.write_bytes(r.content)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(public_folder_path)
                print(f"Extracted dataset in: {public_folder_path}")
                zip_path.unlink(missing_ok=True)  # Delete the zipped dataset
    else:
        import seaborn as sns
    return (sns,)


@app.cell(hide_code=True)
def _():
    import base64
    import time
    import zipfile
    import requests
    from pathlib import Path
    from io import BytesIO
    from pickle import dump

    import numpy as np
    import pandas as pd
    from PIL import Image

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    import altair as alt

    from skimage.io import imread
    from skimage.measure import regionprops_table
    from skimage.exposure import rescale_intensity

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    return (
        BytesIO,
        ConfusionMatrixDisplay,
        Image,
        ListedColormap,
        Path,
        alt,
        base64,
        confusion_matrix,
        dump,
        imread,
        np,
        pd,
        plt,
        requests,
        rescale_intensity,
        zipfile,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ![logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
    # Introduction to Machine Learning for Vision Applications

    **Mallory Wittwer** (Presenter)

    [↗️ Workshop repository (GitHub)](https://github.com/EPFL-Center-for-Imaging/vision-workshop)

    ---

    This workshop takes a hands-on approach to exploring some of the fundamental concepts of machine learning. We will introduce the [Scikit-learn](https://scikit-learn.org/stable/) library and use it to create a simple **image classification** system. Along the way, we will also see how [Marimo](https://marimo.io/) can help us create interactive elements and data visualizations to deepen our understanding.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Outline

    1. Using Marimo
    2. Working with an image dataset
    3. Training an image classifier
    4. Comparing different models
    5. Running a model live on a camera feed
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Using Marimo

    This is a [Marimo](https://marimo.io/) notebook. It contains alternating **Python code** cells and **Markdown** cells (text cells). You can edit the notebook's content to experiment with the code. The output of each code cell is displayed at the top of the cell.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ### Reactive execution

    In contrast to Jupyter notebooks, Marimo notebooks are **reactive**: updating a cell's state automatically re-runs all other cells that depend on it. As a result, code cells *do not* necessarily need to run from the beginning to the end of the notebook (they can be be placed in any order).

    #### Running cells

    When you edit a cell, it becomes *stale*; you have to click the yellow "Run" buton next to the cell to execute it (this will also update all dependent cells). You can also click the "Play" button in the bottom-right corner of the interface to run and update all *stale* cells at once.
    """)
    return


@app.cell
def _():
    example_slider = mo.ui.slider(start=1, stop=10, value=3, label="Select a value", show_value=True, full_width=True)

    example_slider
    return (example_slider,)


@app.cell
def _(example_slider):
    mo.md(f"""
    The selected value is: **{example_slider.value}**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ### Side bar

    Take a look at the side bar on the left side of the screen; you will see options to *View files*, *Explore variables*, *Manage packages*, *View outline*, *View live docs*, and more.

    ### Toggle app view

    Click "Toggle app view" in the lower-right corner of the screen, or use the shortcut `Ctrl + .` to hide or show the code below the code cells.

    ### Running Python in the browser

    Marimo notebooks can be used locally or exported to WebAssembly to run directly in the browser via [Pyodide](https://pyodide.org/en/stable/). If you prefer to set up and run this notebook locally, you can refer to the [setup instructions](https://github.com/EPFL-Center-for-Imaging/vision-workshop).

    ---
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Working with an image dataset

    We will work with a dataset of images organized as follows, which is typical of a dataset intended for image classification:

    ```
    dataset
        |---- 0  <- Subfolder is the class
            |---- 2026-01-01.png
            |---- 2026-01-02.png
            |---- *.png
        |---- 1
        |---- 2
        |---- ...
    ```

    Let's start by making sure we have access to the dataset:
    """)
    return


@app.cell
def _(Path):
    public_folder_path = Path("./public")

    dataset_path = public_folder_path / "dataset"

    path_exists = "✅ yes" if dataset_path.exists() else "❌ no"

    # We use Marimo's "vstack" to display items stacked vertically:
    mo.vstack([
        mo.md(f"Dataset path: **{str(dataset_path.resolve())}**"),
        mo.md(f"Path exists: **{path_exists}**")
    ])
    return dataset_path, public_folder_path


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Loading the dataset

    The first step in our analysis is to load the dataset into a [Pandas DataFrame](https://pandas.pydata.org/), which contains one row per image and columns for the image file path and class label (the class names are given by the subfolder names). We also read the image files and add a third "image" column to display an preview of the images.
    """)
    return


@app.cell
def _():
    load_dataset_btn = mo.ui.run_button(label="Load dataset")

    load_dataset_btn
    return (load_dataset_btn,)


@app.cell
def _(Path, dataset_path, img2url, imread, load_dataset_btn, pd):
    # We use `mo.stop` so that this cell only executes when the button is clicked:
    mo.stop(not load_dataset_btn.value, mo.md("Click the button to load the dataset."))

    def read_image(image_path) -> str:
        """Read an image file and encode its contents as a URL."""
        img = imread(image_path)
        return img2url(img)

    def load_dataset(dataset_path: Path):
        """Load an image classification dataset into a Pandas DataFrame and add an image preview."""
        class_labels = [subdir.name for subdir in list(dataset_path.glob("*"))]

        rows = []
        for class_label in class_labels:
            for img_file in (dataset_path / class_label).glob("*.png"):
                rows.append({
                    "image_path": str(img_file),
                    "class": class_label,
                    "image": read_image(img_file)
                })

        df = pd.DataFrame(rows)

        return df, class_labels

    df, class_labels = load_dataset(dataset_path)

    num_labels = len(class_labels)  # Number of classes

    df  # Display the DataFrame
    return class_labels, df, num_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We also load all the images into a single Numpy array, which we will use extensively in the downstream analysis:
    """)
    return


@app.cell
def _(df, imread, np):
    images = np.array([imread(f) for f in df['image_path']])

    mo.md(f"✅ Read all the images into a Numpy array of shape **{images.shape}** corresponding to (N, X, Y).")
    return (images,)


@app.cell(hide_code=True)
def _(num_labels):
    mo.md(f"""
    ### Exploring the dataset

    We can see that our dataset contains {num_labels} different classes. To examine them more closely, we create a *dropdown* element that allows us to select classes. We connect the selection to a figure displaying the corresponding images.
    """)
    return


@app.cell
def _(class_labels):
    class_dropdown = mo.ui.dropdown(options=class_labels, value=class_labels[0], label="Class ")
    return (class_dropdown,)


@app.cell
def _(class_dropdown, df, images, np, plt):
    def show_images_selection(images_selection, ncols=7):
        nrows = int(np.ceil(len(images_selection) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))
        for idx, im in enumerate(images_selection):
            ax = axes.ravel()[idx]
            ax.imshow(im, cmap="gray")
        for ax in axes.ravel():
            ax.set_axis_off()
        plt.tight_layout()
        return fig

    # Select images based on the dropdown value:
    images_selection = images[list( df[df["class"] == class_dropdown.value].index)]

    mo.vstack([class_dropdown, show_images_selection(images_selection)], align="center")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Adding measurements

    To add a few numerical measurements to our dataset, we compute the mean intensity value of the images, as well as the standard deviation of intensity values. To do this, we use the `.apply` method of Pandas DataFrames, which applies a Python function to every row in the DataFrame.
    """)
    return


@app.cell(hide_code=True)
def _():
    measurements_btn = mo.ui.run_button(label="Compute measurements")

    measurements_btn
    return (measurements_btn,)


@app.cell
def _(df, measurements_btn, np, url2img):
    mo.stop(not measurements_btn.value, mo.md("Click the button to compute the measurements."))

    def compute_mean_intensity(base64_image: str):
        """Compute the mean gray level intensity of an image."""
        img = url2img(base64_image)
        mean_intensity = np.mean(img)
        return mean_intensity

    df["mean_intensity"] = df["image"].apply(compute_mean_intensity)

    def compute_std_intensity(base64_image: str):
        """Compute the standard deviation of the intensity values of an image."""
        img = url2img(base64_image)
        std_intensity = np.std(img)
        return std_intensity

    df["std_intensity"] = df["image"].apply(compute_std_intensity)

    # We display the first few rows to confirm that we have our measurements:
    df[["image", "mean_intensity", "std_intensity"]].head()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Using the DataFrame viewer

    Marimo's built-in DataFrame viewer is a powerful tool for exploring datasets. It offers a **search** functionality which can be used to filter data, and a **chart builder** that can be used to create plots without writing any code.


    /// admonition | Exercise

    1. Can you search the DataFrame and display only classes *3* and *4*, sorted by *mean_intensity* ?

    2. Can you use the chart builder (*Visualize*) to plot

        i. A 2D scatter plot of the mean versus standard deviation intensity, colored by class?

        ii. A pie chart respresenting the counts in each class?

        iii. A histogram of the mean intensity values?

    ///
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <!-- Upon examining the dataset, we find that intensity-based features (mean, standard deviation) do not effective distinguish between the different classes of numbers. -->

    Next, we will proceed to develop our image classification algorithm.

    <!-- To inspect these measurements, we create an interactive plot for ourselves: -->
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training an image classifier

    An *classifier* model is a type of machine learning model designed to assign an input to one of several predefined classes. Here, we will train an image classifier to regognize numbers in a supervised way, using the ground truth labels from our dataset.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Image preprocessing

    There are a few steps to take before training the classifier itself. First, we **rescale the intensity** values of the images to the range [0-1] using min/max normalization. Applying this normalization will help our system handle different levels of image brightness, such as those resulting from different image acquisition conditions.
    """)
    return


@app.cell
def _(images, np, rescale_intensity):
    images_normed = np.array([rescale_intensity(img, out_range=(0, 1)) for img in images])

    mo.md(f"➡️ Shape of images (normalized): **{images_normed.shape}**")
    return (images_normed,)


@app.cell(hide_code=True)
def _(images, images_normed, np, plot_image_before_after_normalization):
    random_idx = np.random.randint(len(images))

    img_before = images[random_idx]
    img_after = images_normed[random_idx]

    mo.vstack([plot_image_before_after_normalization(img_before, img_after)], align="center")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, we **reshape** the images into a 2D matrix, where each row corresponds to an image and each column respresents an individual pixel value.
    """)
    return


@app.cell
def _(images_normed, np):
    pixel_features = np.reshape(images_normed, (len(images_normed), -1))

    mo.md(f"➡️ Shape of pixel value array: **{pixel_features.shape}**")
    return (pixel_features,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Splitting the data

    After that, we randomly split the data into a **training** and a **validation** set. The algorithms will be fitted on the training set, and evaluated on the validation set.
    """)
    return


@app.cell
def _():
    valid_fract_slider = mo.ui.slider(start=0, stop=1, value=0.25, step=0.05, show_value=True, label="Validation fraction ", debounce=True)
    return (valid_fract_slider,)


@app.cell
def _(df, pd, pixel_features, valid_fract_slider):
    from sklearn.model_selection import train_test_split

    X = pixel_features
    y = df["class"].values

    test_size = float(valid_fract_slider.value)

    x_train, x_val, y_train, y_val= train_test_split(X, y, test_size=test_size)

    _df = pd.DataFrame({
        "Split": ["Train", "Validation", "Total"],
        "N": [len(x_train), len(x_val), len(X)],
        "Data shape": [x_train.shape, x_val.shape, X.shape],
    })

    mo.vstack([valid_fract_slider, _df])
    return X, x_train, x_val, y, y_train, y_val


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Featurization

    Featurization (or feature extraction) is the process of converting raw pixel values into a *feature vector* - a compact numerical representation of the data from which a machine learning model can learn relevant patterns. Features can be engineered manually, computed from statistical algorithms, or extracted from pretrained deep neural networks. Here, we will apply the PCA algorithm for featurization.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Principal Component Analysis (PCA)

    [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) is a well-known **dimensionality reduction** technique that allows a high-dimensional dataset (such as our array of flattened pixel values) to be projected onto a lower-dimensional space while retaining most of the variance in the data. The PCA components are ranked according to the proportion of variance they explain; thus, the first few components can often serve as effective **features** for classification.

    Let's apply PCA to our dataset and limit the decomposition to **2** components:
    """)
    return


@app.cell(hide_code=True)
def _():
    compute_pca_btn = mo.ui.run_button(label="Compute PCA")

    compute_pca_btn
    return (compute_pca_btn,)


@app.cell
def _(X, compute_pca_btn, x_train, x_val):
    mo.stop(not compute_pca_btn.value, mo.md("Click the button to run PCA decomposition."))

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)

    pca.fit(x_train)  # "Fit" the PCA estimator to the training data

    x_train_projected = pca.transform(x_train)  # "Transform" (=project) the data into the PCA feature space
    x_val_projected = pca.transform(x_val)
    X_projected = pca.transform(X)

    mo.md(f"➡️ PCA-projected data shape: **{X_projected.shape}**")
    return PCA, X_projected, x_train_projected, x_val_projected


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can add the first two PCA features as additional columns to our dataset:
    """)
    return


@app.cell(hide_code=True)
def _(X_projected, df):
    df["PCA-0"] = X_projected[:, 0]
    df["PCA-1"] = X_projected[:, 1]

    df[["image", "PCA-0", "PCA-1"]].head(3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's plot the `PCA-0` and `PCA-1` features on a two-dimensional chart, in a slightly nicer way than the built-in *Chart builder* allows.
    """)
    return


@app.cell
def _():
    switch_show_images = mo.ui.switch(label="Show images")
    return (switch_show_images,)


@app.cell
def _(create_altair_chart, df, switch_show_images):
    chart_pca = create_altair_chart(df, x_col="PCA-0", y_col="PCA-1", show_images=switch_show_images.value, size=600)

    chart_pca = mo.ui.altair_chart(chart_pca)  # To make the plot interactive
    return (chart_pca,)


@app.cell
def _(chart_pca, switch_show_images):
    # The chart's "value" is the data in the selected region
    _selection = chart_pca.value[['class', 'image', 'PCA-0', 'PCA-1']]

    mo.vstack([switch_show_images, chart_pca, _selection])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This time, we can see that images belonging to the same class tend to cluster together. That's a good  sign; it probably means we're on the right track to solving our classification problem.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Training a model

    In Scikit-learn, training a model generally involves selecting a type of [model](https://scikit-learn.org/stable/supervised_learning.html) for a given task (in this case, classification), instantiating the model, and then calling the `.fit` method using the input data (`x_train`) and corresponding target labels (`y_train`) from the training set. Once the model is trained, it can be applied to new data using the `.predict` method, and evaluated using the `.score` method (typically on the validation set: `x_val`, `y_val`).
    """)
    return


@app.cell
def _(ConfusionMatrixDisplay, confusion_matrix):
    def fit_and_evaluate_classifier(model, x_train, y_train, x_val, y_val):
        """Fit a Scikit-Learn classifier on the training set and report evaluation metrics on the validation set."""
        # Fit on the training set
        model.fit(X=x_train, y=y_train)

        # Score on the training set
        acc_train = model.score(X=x_train, y=y_train)

        # Score on the validation set
        acc_val = model.score(X=x_val, y=y_val)

        # Predictions on the training set
        y_pred_train = model.predict(x_train)

        # Predictions on the validation set
        y_pred_val = model.predict(x_val)

        # Confusion matrices
        cm_train = confusion_matrix(y_train, y_pred_train, labels=model.classes_) 
        disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=model.classes_)

        cm_val = confusion_matrix(y_val, y_pred_val, labels=model.classes_)
        disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=model.classes_)

        return {
            "accuracy_training": acc_train, 
            "accuracy_validation": acc_val, 
            "confusion_matrix_training": disp_train,
            "confusion_matrix_validation": disp_val,
            "n_train": len(x_train),
            "n_validation": len(x_val),
            "x_training": x_train,
            "x_validation": x_val,
            "y_training": y_train,
            "y_validation": y_val,
        }

    return (fit_and_evaluate_classifier,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Baseline model

    Before training a "real" model, it is generally recommended to establish a **baseline performance** that will serve as a reference for comparisons. In Scikit-learn, this can be achieved by fitting a *DummyClassifier* to the training data. Below, we use a dummy classifier to classify the inputs based simply on the most frequent class (the model systematicall predicts that class) and calculate the corresponding accuracy.
    """)
    return


@app.cell
def _():
    from sklearn.dummy import DummyClassifier

    return (DummyClassifier,)


@app.cell(hide_code=True)
def _():
    fit_baseline_btn = mo.ui.run_button(label="Fit baseline model")

    fit_baseline_btn
    return (fit_baseline_btn,)


@app.cell
def _(
    DummyClassifier,
    fit_and_evaluate_classifier,
    fit_baseline_btn,
    x_train_projected,
    x_val_projected,
    y_train,
    y_val,
):
    mo.stop(not fit_baseline_btn.value, mo.md("Click the button to fit the baseline model."))

    baseline_model = DummyClassifier(strategy="most_frequent")  # always predicts the most frequent class

    baseline_results = fit_and_evaluate_classifier(baseline_model, x_train_projected, y_train, x_val_projected, y_val)

    mo.vstack([
        mo.md("""### Baseline model"""),
        mo.hstack([print_classification_results(baseline_results)])
    ], align="center")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Logistic regression

    A [Logistic regression model](https://en.wikipedia.org/wiki/Logistic_regression) is a common type of linear model that can be used for classification tasks. It is particularly well-suited for problems in which the classes are *linearly separable* in the feature space.

    Let's fit a logistic regression model to our training data and evaluate its accuracy. Moreover, since we are working with a two-dimensional feature space, we can plot the model's **classification boundaries**.
    """)
    return


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression

    return (LogisticRegression,)


@app.cell(hide_code=True)
def _():
    # Fit model button
    fit_linear_model_btn = mo.ui.run_button(label="Fit linear model")

    fit_linear_model_btn
    return (fit_linear_model_btn,)


@app.cell
def _(
    LogisticRegression,
    fit_and_evaluate_classifier,
    fit_linear_model_btn,
    x_train_projected,
    x_val_projected,
    y_train,
    y_val,
):
    mo.stop(not fit_linear_model_btn.value, mo.md("Click the button to fit the logistic regression model."))

    linear_model = LogisticRegression(C=1.0, max_iter=1000)

    linear_results = fit_and_evaluate_classifier(linear_model, x_train_projected, y_train, x_val_projected, y_val)

    mo.vstack([
        mo.md("""### Logistic regression"""),
        mo.hstack([print_classification_results(linear_results)])
    ], align="center")
    return linear_model, linear_results


@app.cell
def _():
    split_dropdown = mo.ui.dropdown(options=["training", "validation"], value="validation", label="Show on split ")
    return (split_dropdown,)


@app.cell
def _(
    linear_model,
    linear_results,
    plot_classification_results,
    split_dropdown,
):
    _fig = plot_classification_results(linear_model, linear_results, split=split_dropdown.value)

    mo.vstack([split_dropdown, _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Comparing models

    Our choice of classification model can influence performance in terms of accuracy, robustness, and other factors (such as fitting time or prediction time). Scikit-learn offers many different types of [classifiers](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) that all implement a common set of methods (`.fit`, `.predict`, `.score`, etc.), making them easily interchangeable in the code.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Decision tree classifier

    A [decision tree classifier](https://scikit-learn.org/stable/modules/tree.html#tree) is a type of model available in Scikit-learn that can be fitted on our training data. Decision trees infer simple rules from the input features in order to predict class labels.

    Let's replace our logistic regression model with a decision tree classifier and compare the two models:
    """)
    return


@app.cell
def _():
    from sklearn.tree import DecisionTreeClassifier

    return (DecisionTreeClassifier,)


@app.cell(hide_code=True)
def _():
    fit_decision_tree_btn = mo.ui.run_button(label="Fit decision tree model")

    fit_decision_tree_btn
    return (fit_decision_tree_btn,)


@app.cell(hide_code=True)
def _(
    DecisionTreeClassifier,
    fit_and_evaluate_classifier,
    fit_decision_tree_btn,
    x_train_projected,
    x_val_projected,
    y_train,
    y_val,
):
    mo.stop(not fit_decision_tree_btn.value, mo.md("Click the button to fit the decision tree model."))

    _model = DecisionTreeClassifier()

    _decision_tree_results = fit_and_evaluate_classifier(_model, x_train_projected, y_train, x_val_projected, y_val)

    mo.vstack([
        mo.md("""### Decision tree"""),
        mo.hstack([print_classification_results(_decision_tree_results)])
    ], align="center")
    return


@app.cell(hide_code=True)
def _():
    # We also create a slider to tweak the model's `max_depth` parameter:
    max_depth_selector = mo.ui.slider(start=1, stop=15, value=5, show_value=True, label="Max depth ", debounce=True)
    return (max_depth_selector,)


@app.cell(hide_code=True)
def _():
    split_dropdown_dt = mo.ui.dropdown(options=["training", "validation"], value="validation", label="Show on split ")
    return (split_dropdown_dt,)


@app.cell(hide_code=True)
def _(
    DecisionTreeClassifier,
    fit_and_evaluate_classifier,
    max_depth_selector,
    plot_classification_results,
    split_dropdown_dt,
    x_train_projected,
    x_val_projected,
    y_train,
    y_val,
):
    decision_tree_model = DecisionTreeClassifier(max_depth=max_depth_selector.value)

    decision_tree_results = fit_and_evaluate_classifier(decision_tree_model, x_train_projected, y_train, x_val_projected, y_val)

    _fig = plot_classification_results(decision_tree_model, decision_tree_results, split=split_dropdown_dt.value)

    mo.vstack([max_depth_selector, split_dropdown_dt, _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | Exercise

    Can you fit one of these models to the data? What do the classification boundaries look like?

    - [Support Vector Classifier (SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    - [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    - [Multi-layer Perceptron (MLP)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

    ///

    /// details | Solution

    SVC:

    ```python
    from sklearn.svm import SVC

    svc_model = SVC(C=1.0)  # You can tweak the C parameter to control regularization

    svc_results = fit_and_evaluate_classifier(svc_model, x_train_projected, y_train, x_val_projected, y_val)

    mo.vstack([
        plot_classification_results(svc_model, svc_results, split="training"),
        plot_classification_results(svc_model, svc_results, split="validation")
    ])
    ```

    Random forest:

    ```python
    from sklearn.ensemble import RandomForestClassifier

    tree_model = RandomForestClassifier(n_estimators=100)

    tree_results = fit_and_evaluate_classifier(tree_model, x_train_projected, y_train, x_val_projected, y_val)

    mo.vstack([
        plot_classification_results(tree_model, tree_results, split="training"),
        plot_classification_results(tree_model, tree_results, split="validation")
    ])
    ```

    MLP:

    ```python
    from sklearn.neural_network import MLPClassifier

    nn_model = MLPClassifier(hidden_layer_sizes=(100,))

    nn_results = fit_and_evaluate_classifier(nn_model, x_train_projected, y_train, x_val_projected, y_val)

    mo.vstack([
        plot_classification_results(nn_model, nn_results, split="training"),
        plot_classification_results(nn_model, nn_results, split="validation")
    ])
    ```
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Increasing PCA dimensionality

    So far, we have applied our classifiers only to the two-dimensional feature space corresponding to the first two PCA components. Working with two components is practical (and pedagogical) because it allows us to plot and interpret the classification boundaries, but it may not be the most effective solution. What happens if we increase the number of components in our PCA decomposition?
    """)
    return


@app.cell(hide_code=True)
def _():
    n_components_slider = mo.ui.slider(start=1, stop=20, value=2, label="PCA components", show_value=True)
    return (n_components_slider,)


@app.cell(hide_code=True)
def _(
    LogisticRegression,
    PCA,
    fit_and_evaluate_classifier,
    n_components_slider,
    x_train,
    x_val,
    y_train,
    y_val,
):
    def fit_model_with_pca(model, x_train, y_train, x_val, y_val, n_components):
        """Fit the PCA estimator and the classifier on the input data."""
        pca = PCA(n_components)
        x_train_projected = pca.fit_transform(x_train)
        x_val_projected = pca.transform(x_val)
        return fit_and_evaluate_classifier(model, x_train_projected, y_train, x_val_projected, y_val)

    _model = LogisticRegression(max_iter=1000)

    _results = fit_model_with_pca(
        model=_model,
        x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
        n_components=n_components_slider.value,
    )

    mo.vstack([
        mo.md(f"Model: **{_model.__class__.__name__}**"),
        n_components_slider, 
        mo.hstack([print_classification_results(_results)])
    ], align="center")
    return (fit_model_with_pca,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It appears that accuracy does indeed increase as the number of PCA features increases. We can compare performance on the training and on the validation set in order to determine the number of PCA components to use for our application.
    """)
    return


@app.cell(hide_code=True)
def _():
    pca_grid_search_btn = mo.ui.run_button(label="Optimize PCA")

    pca_grid_search_btn
    return (pca_grid_search_btn,)


@app.cell(hide_code=True)
def _(
    LogisticRegression,
    pca_grid_search_btn,
    plot_accuracy_vs_pca_components,
    x_train,
    x_val,
    y_train,
    y_val,
):
    mo.stop(not pca_grid_search_btn.value, mo.md("Click the button to run the computation."))

    _model = LogisticRegression(max_iter=1000)

    pca_grid_search_btn
    with mo.status.spinner(title="🚀 Working on it...") as _spinner:
        _fig, best_n_components = plot_accuracy_vs_pca_components(
            model=_model,
            x_train=x_train, 
            x_val=x_val, 
            y_train=y_train, 
            y_val=y_val,
        )

    _fig
    return (best_n_components,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exporting an end-to-end pipeline

    Scikit-learn provides a [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) class that allows us to apply multiple processing steps in sequence. Here, we create a *Pipeline* to fit and apply the PCA and classifier estimators to our data at once.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    PCA,
    best_n_components,
    x_train,
    x_val,
    y_train,
    y_val,
):
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("pca", PCA(n_components=best_n_components)),  # We use the best number of components from our parameter search
        ("model", LogisticRegression(max_iter=1000)),
    ])

    pipe.fit(x_train, y_train)

    pipe_score_train = pipe.score(x_train, y_train)
    pipe_score_val = pipe.score(x_val, y_val)

    mo.vstack([
        mo.md(f"➡️ Pipeline accuracy (training): **{pipe_score_train:.2f}**"),
        mo.md(f"➡️ Pipeline accuracy (validation): **{pipe_score_val:.2f}**")
    ])
    return (pipe,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Is that final score acceptable? Should we work more on improving the feature extraction, the model, or other parts of the pipeline to try to improve it?

    Taking a look at the misclassified data will give us a hint:
    """)
    return


@app.cell(hide_code=True)
def _():
    show_misclassified_btn = mo.ui.run_button(label="Show misclassified images")

    show_misclassified_btn
    return (show_misclassified_btn,)


@app.cell(hide_code=True)
def _(X, pipe, show_misclassified, show_misclassified_btn, y):
    mo.stop(not show_misclassified_btn.value, mo.md("Click the button to show misclassified images."))

    # Show misclassified examples
    x_pred = pipe.predict(X)

    misclassified = []
    for _x, _p, _y in zip(X, x_pred, y):
        if _p != _y:
            misclassified.append({"image": _x, "pred": _p, "true": _y})

    show_misclassified(misclassified)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finally, we can save the pipeline as a Pickle file so that we can reload it in other programs.
    """)
    return


@app.cell(hide_code=True)
def _():
    save_btn = mo.ui.run_button(label="Save the pipeline")

    save_btn
    return (save_btn,)


@app.cell(hide_code=True)
def _(Path, dump, pipe, save_btn):
    mo.stop(not save_btn.value, mo.md("Click the button to save the pipeline."))

    pipeline_file = Path("./pipeline.pkl")

    with open(pipeline_file, "wb") as f:
        dump(pipe, f, protocol=5)

    mo.md(f"✔️ Saved pipeline file: **{pipeline_file.resolve()}**")
    return


@app.cell(hide_code=True)
def _(inference_link, inference_screenshot):
    mo.md(f"""
    ## Running a model live on a camera feed

    Now that our pipeline is saved, we can reload it in other Python programs and apply our classifier to new images, for example those coming from a live camera feed.

    For the demonstration, we have put together a Python script that captures video frames from a USB camera device, identifies a bounding box around each object in the image, and applies our Scikit-learn pipeline to each of object to classify them.

    {inference_link}

    {inference_screenshot}

    /// note
    You don't have to run the inference script yourself during the workshop (unless you have a camera and setup ready for it!)
    ///

    /// admonition | Discussion

    How robust is our model to

    - variations in the illumination conditions?
    - changes in the distance or orientation of the camera?
    - handwriting among different people?

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mermaid_chart):
    mo.md(f"""
    ## Conclusion

    In this workshop, we have been exposed to some of the main steps of a computer vision project: 

    {mermaid_chart}

    Thanks to this, we have managed to develop a working image classification application.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Feedback

    If you have any feedback you'd like to share with us about this workshop, feel free to use the following form. Thank you!

    - [↗️ Feedback form](https://forms.gle/YwuoduC7azgQnPRX7)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Utility functions

    The Python functions in this section are used throughout the notebook, but they are not essential to understand the concepts; so, for the sake of clarity, we have grouped them together here.
    """)
    return


@app.cell(hide_code=True)
def _(BytesIO, Image, alt, base64, np):
    def url2img(base64_image: str) -> np.ndarray:
        """Convert an image encoded as a base64 URL string to a Numpy array."""
        header, encoded = base64_image.split(",", 1)
        img = Image.open(BytesIO(base64.b64decode(encoded)))
        img_arr = np.array(img)
        return img_arr


    def img2url(img_arr: np.ndarray) -> str:
        """Convert an image array to a base64-encoded URL string, so that the image can be rendered in a DataFrame column."""
        pil_img = Image.fromarray(img_arr)
        output = BytesIO()
        pil_img.save(output, format='PNG')
        base64_image = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        return base64_image


    def compute_thumbnail(path):
        """Compute a 15x15px image thumbnail that can be displayed in Altair charts."""
        pil_img = Image.open(path)

        output_thumb = BytesIO()
        pil_img.resize((15, 15)).save(output_thumb, format='PNG')
        base64_thumb = "data:image/png;base64," + base64.b64encode(output_thumb.getvalue()).decode()

        return base64_thumb


    def create_altair_chart(df, x_col, y_col, show_images, size=700):
        """Create a 2D scatter plot with Altair, with the option to show image thumbnails directly on the plot."""
        x_domain = [df[x_col].min(), df[x_col].max()]
        y_domain = [df[y_col].min(), df[y_col].max()]

        if show_images is True:
            df_altair = df.copy()
            df_altair["thumbnail"] = df["image_path"].apply(compute_thumbnail)

            # Altair chart with image thumbnails
            chart = (
                alt.Chart(df_altair, height=size, width=size)
                .mark_image(width=15, height=15).encode(
                    x=alt.X(x_col, scale=alt.Scale(domain=x_domain)), 
                    y=alt.Y(y_col, scale=alt.Scale(domain=y_domain)), 
                    url="thumbnail", 
                    tooltip=["image", "class"],
                )
            )
        else:
            # Altair scatter plot with colors representing classes
            chart = (
                alt.Chart(df, height=size, width=size)
                .mark_circle(size=60, opacity=0.8)
                .encode(
                    x=alt.X(x_col, scale=alt.Scale(domain=x_domain)),
                    y=alt.Y(y_col, scale=alt.Scale(domain=y_domain)),
                    color=alt.Color("class:N", title="Class"),
                    tooltip=["image", "class"],
                )
            )

        return chart

    return create_altair_chart, img2url, url2img


@app.cell(hide_code=True)
def _(extent, np):
    def predict_on_regular_grid(model, nx=200, ny=200):
        """Run a classifier over a regular 2D-XY grid and return class indices."""
        global extent
        xx, yy = np.meshgrid(
            np.linspace(extent[0], extent[1], nx), 
            np.linspace(extent[2], extent[3], ny),
        )
        grid_yx = np.c_[xx.ravel(), yy.ravel()]
        preds_grid = model.predict(grid_yx)
        label2id = {cls: i for i, cls in enumerate(model.classes_)}
        preds_grid_idx = np.array([label2id[c] for c in preds_grid]).reshape(ny, nx)

        return preds_grid_idx

    return (predict_on_regular_grid,)


@app.cell(hide_code=True)
def _(ListedColormap, extent, pd, plt, predict_on_regular_grid, sns):
    def plot_classification_results(model, results, split="validation", palette_name="pastel"):
        """Display a confusion matrix and a 2D scatter plot of PCA-0/PCA-1 coordinates colored according to class indices."""
        global extent
        # Prediction on a regular grid
        preds_grid_idx = predict_on_regular_grid(model)

        X = results[f"x_{split}"]
        y = results[f"y_{split}"]

        df_train = pd.DataFrame({"PCA-0": X[:, 0], "PCA-1": X[:, 1], "class": y})

        palette = sns.color_palette(palette=palette_name, n_colors=len(model.classes_))

        cmap = ListedColormap(palette)
        color_mapping = dict(zip(model.classes_, palette))

        fig, (ax_cm, ax_plot) = plt.subplots(1, 2, figsize=(14, 6))

        disp_train = results[f"confusion_matrix_{split}"]
        disp_train.plot(ax=ax_cm, colorbar=False, cmap="inferno")

        acc_train = results[f"accuracy_{split}"]
        ax_cm.set_title(f"Confusion Matrix ({split})\nAccuracy={acc_train:.2f}")

        ax_plot.imshow(
            preds_grid_idx, 
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="nearest",
            alpha=0.6,
            aspect="auto",
        )

        # Scatter plot
        sns.scatterplot(
            data=df_train, 
            x="PCA-0", 
            y="PCA-1", 
            hue="class", 
            hue_order=model.classes_,
            ax=ax_plot, 
            palette=color_mapping,
            edgecolor="k",
        )

        ax_plot.set_xlabel("PCA-0")
        ax_plot.set_ylabel("PCA-1")

        ax_plot.set_xlim(extent[0], extent[1])
        ax_plot.set_ylim(extent[2], extent[3])

        ax_plot.legend(
            title="class",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0
        )
        ax_plot.set_title(f"Classification boundaries ({model.__class__.__name__})")

        plt.tight_layout()

        return fig

    return (plot_classification_results,)


@app.function(hide_code=True)
def print_classification_results(results: dict):
    """Print a small classification report in Marimo markdown format."""
    accuracy_train = results["accuracy_training"]
    accuracy_val = results["accuracy_validation"]

    return mo.md(f"""
    | Split | N | Accuracy |
    | ----- | --- | ------ |
    | Training | {results["n_train"]} | {accuracy_train:.2f} |
    | Validation | {results["n_validation"]} | {accuracy_val:.2f} |
    """)


@app.cell(hide_code=True)
def _(plt):
    def plot_image_before_after_normalization(img_before, img_after):
        """Display a side-by-side comparison of an image before and after min/max normalization."""
        fig, axes = plt.subplots(ncols=2)
        im0 = axes[0].imshow(img_before, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Original image")
        axes[0].set_axis_off()
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(img_after, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Intensity rescaled")
        axes[1].set_axis_off()
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        return fig

    return (plot_image_before_after_normalization,)


@app.cell(hide_code=True)
def _(fit_model_with_pca, np, plt):
    def plot_accuracy_vs_pca_components(model, x_train, x_val, y_train, y_val, max_components=20):
        """Plot training and validation accuracies against the number of PCA components."""
        train_accs = []
        valid_accs = []
        for n_components in range(1, max_components+1):
            results = fit_model_with_pca(model, x_train, y_train, x_val, y_val, n_components)
            train_accs.append(results["accuracy_training"])
            valid_accs.append(results["accuracy_validation"])

        best_valid_acc = float(np.max(valid_accs))
        pca_components_best_valid_acc = int(np.argmax(valid_accs)) + 1    

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.arange(1, max_components+1), train_accs, label=f"Training")
        ax.plot(np.arange(1, max_components+1), valid_accs, label=f"Validation")
        ax.set_title(f"Best validation accuracy: {best_valid_acc:.2f} ({pca_components_best_valid_acc} components)")
        ax.set_xlabel("PCA components")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_xlim(1, max_components)
        ax.set_xticks(np.arange(1, max_components+1, 2))
        plt.legend()

        return fig, pca_components_best_valid_acc

    return (plot_accuracy_vs_pca_components,)


@app.cell(hide_code=True)
def _(images, np, plt):
    def show_misclassified(misclassified, ncols=7):
        """Display misclassified examples."""
        nrows = int(np.ceil(len(misclassified) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))
        for idx, item in enumerate(misclassified):
            ax = axes.ravel()[idx]
            im = item["image"].reshape((images.shape[1], images.shape[2]))
            ax.imshow(im, cmap="gray")
            ax.set_title(f"Predicted: {item['pred']}\nTrue: {item['true']}")
        for ax in axes.ravel():
            ax.set_axis_off()
        plt.tight_layout()
        return fig

    return (show_misclassified,)


@app.cell(hide_code=True)
def _(x_train_projected):
    # The bounds of the projected training data - we keep this as a global variable
    extent = [x_train_projected[:, 0].min(), x_train_projected[:, 0].max(), x_train_projected[:, 1].min(), x_train_projected[:, 1].max()]
    return (extent,)


@app.cell(hide_code=True)
def _(public_folder_path):
    _img_src = public_folder_path / "assets" / "inference.png"

    inference_screenshot = mo.vstack([mo.image(src=_img_src, alt="screenshot", height=400, caption="Live inference (screenshot)")], align="center")
    return (inference_screenshot,)


@app.cell(hide_code=True)
def _():
    # We interpolate this markdown so that it renders as centered text:
    inference_link = mo.vstack([mo.md("[↗️ Inference script](https://github.com/EPFL-Center-for-Imaging/vision-workshop/blob/main/inference.py)")], align="center")
    return (inference_link,)


@app.cell(hide_code=True)
def _():
    mermaid_chart = mo.mermaid("""
    flowchart LR
        A(🔬 Data inspection) --> B(⚙️ Preprocessing)
        B --> C(🛠️ Featurization)
        C --> D(🔍 Model selection)
        D --> E(🎓 Training)
        E --> F(⚖️ Evaluation)
        F --> G(🔋 Inference)
    """)
    return (mermaid_chart,)


if __name__ == "__main__":
    app.run()
