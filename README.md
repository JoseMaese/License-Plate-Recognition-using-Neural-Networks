# License Plate Recognition using Neural Networks

Trabajo de Fin de Grado. José Enrique Maese Álvarez

Dpto. de Ingeniería de Sistemas y Automatización

Escuela Técnica Superior de Ingeniería

Universidad de Sevilla

### Objective

The primary goal of this project is the extraction of characters from vehicle license plates using neural networks. The study involves exploring different neural network models, implementing the solution in Python, and applying computer vision techniques learned during the course.

### Considerations

- Standardized conditions: distance, height, and lighting.
- Prior segmentation of characters.
- Training on numerical part and extrapolation.
  
### Image Preprocessing

#### Creation of the Database

- 125 images
- 875 characters
  - 500 numbers (50 examples/number)
  - 375 letters (18.75 examples/letter)

#### License Plate Cropping Algorithm

1. Convert to grayscale
2. Apply Gaussian filter
3. Binarize
4. Edge extraction using gradient calculation
5. Closing of edges
6. Locate closed surfaces
7. Select the top 10 areas
8. Select contour meeting specifications: 
   - Area
   - Proportion

#### Optimal Cropping - EU Symbol

- Appearance of edges
- Excessive brightness

#### Character Extraction Algorithm

1. Convert to grayscale
2. Apply Gaussian filter
3. Binarize: darker elements to high level
4. Remove surfaces below a threshold area
5. Create column-wise histogram to locate characters
6. Variable lower limit to eliminate errors from edge appearance
7. Initial high-level strip to eliminate errors from the EU symbol

### Neural Network Fundamentals

#### General Concepts

- Perceptron
  - Weight coefficients: w
  - Bias: b
  - Input: X, Output: Y

#### General Concepts: Neural Network

- Multilayer perceptron
- 5 layers
- Different number of perceptrons per layer

#### Training

1. Forward-propagation
   - Introduce input data
   - Traverse the network
   - Obtain the cost function (to minimize)
2. Back-propagation
   - Traverse the network in reverse
   - Update parameters

#### Training: Hyperparameters

- Network structure
- Learning rate
- Database split
- Batch training
- Epochs

#### Problems

- Overfitting
- Underfitting
- Oscillations around equilibrium point

### Neural Network Models

1. LeNet-5
2. AlexNet
3. ResNet50

### Results

- Model accuracies on training and validation sets:

  | Model      | Training Set | Validation Set |
  |------------|--------------|----------------|
  | LeNet-5    | 99.13%       | 75.01%          |
  | AlexNet    | 98.27%       | 94.74%          |
  | ResNet50   | 99.57%       | 96.03%          |

### Conclusion

The project successfully achieved high accuracies in license plate character recognition using various neural network models. The experimentation with LeNet-5, AlexNet, and ResNet50 demonstrated the effectiveness of deep learning in this application.

### Possible Improvements

1. Increase the size of the database for better generalization.
2. Enhance the license plate cropping algorithm.
3. Explore more complex neural network models for improved accuracy.

