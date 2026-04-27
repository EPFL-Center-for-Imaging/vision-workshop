![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# Introduction to Machine Learning for Vision Applications

This workshop takes a hands-on approach to exploring some of the fundamental concepts of machine learning applied to computer vision. It walks through:

- Using Marimo
- Working with an image dataset
- Training an image classifier
- Comparing different models
- Running a model live on a camera feed

## Quick start

You can access the workshop material directly in your web browser at this link:

> ➡️ https://epfl-center-for-imaging.github.io/vision-workshop/

### Local installation

Make sure to have a working [Python setup](https://epfl-center-for-imaging.github.io/python-setup/). Install the packages listed in [requirements.txt](./requirements.txt):

```bash
pip install -r requirements.txt
```

Then, from the command-line, run the Marimo notebook in *edit* mode:

```bash
marimo edit intro_to_ml.py
```

You should be able to access the notebook locally at: http://localhost:2718/.

## Scripts

**Acquiring a training set of images**

To extract a training dataset of labelled digits from images captured with a camera device, use:

- [`preprocessing.py`](./preprocessing.py)

This script will save the training dataset in a local `dataset` folder. The class labels are inferred from the predetermined positions of the digits in a 3 X 4 grid.

**Testing the model live on a camera**

To extract digits and classify them based on the trained Scikit-learn pipeline, use:

- [`inference.py`](./inference.py)

This script will reload a Scikit-learn `pipeline.pkl` file, run the pipeline on the video feed from a camera device and display the results.

## License

This workshop material is distributed under the terms of the [BSD-3](http://opensource.org/licenses/BSD-3-Clause) license.
