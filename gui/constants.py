from statistical_feature_extraction.sfe import LOGISTIC_REGRESSION
from statistical_feature_extraction.sfe import RANDOM_FOREST
from statistical_feature_extraction.sfe import SVM
from statistical_feature_extraction.sfe import LINEAR_SVM
from statistical_feature_extraction.sfe import K_NEAREST_NEIGHBORS
from statistical_feature_extraction.sfe import NAIVE_BAYES
from statistical_feature_extraction.sfe import MULTILAYER_PERCEPTRON
from statistical_feature_extraction.sfe import DECISION_TREE
from statistical_feature_extraction.sfe import GAUSSIAN_PROCESS
from statistical_feature_extraction.sfe import ADABOOST
from statistical_feature_extraction.sfe import RADIAL_BASIS_FUNCTION

# Images
image_plot = '.\\images\\tmp.png'
image_reading_data = '.\\images\\S1-ReadingData.jpg'
image_normalizing = '.\\images\\S2-Normalizing.jpg'
image_shuffling = '.\\images\\S3-Shuffling.jpg'
image_reducing_dimensions = '.\\images\\S4-ReducingDimensions.jpg'
image_testing_model = '.\\images\\S5-TestingModel.jpg'

# Combobox Values
statistical = 'Statistical'
cnn = 'Convolutional NN'
METHODS = [statistical, cnn]

raw = 'Raw'
processed = 'Processed'
DATA_TYPE = [raw, processed]

yes = 'Yes'
no = 'No'
YES_NO = [yes, no]

no_dimensionality_reduction = 'No Dimensionality Reduction'
pca = 'Principal Component Analysis'
lda = 'Linear Discriminant Analysis'
DIMENSIONALITY_REDUCTION = [no_dimensionality_reduction, pca, lda]

logistic_regression = 'Logistic Regression'
random_forest = 'Random Forest'
svm = 'SVM'
linear_svm = 'Linear SVM'
k_nearest_neighbors = 'K Nearest Neighbors'
naive_bayes = 'Naive Bayes'
multilayer_perceptron = 'Multilayer Perceptron'
decision_tree = 'Decision Tree'
gaussian_process = 'Gaussian Process'
adaboost = 'AdaBoost'
gaussian_process_with_rbf = 'Gaussian Process with RBF'
MODELS = [LOGISTIC_REGRESSION, RANDOM_FOREST, SVM, LINEAR_SVM, K_NEAREST_NEIGHBORS, NAIVE_BAYES,
          MULTILAYER_PERCEPTRON, DECISION_TREE, GAUSSIAN_PROCESS, ADABOOST, RADIAL_BASIS_FUNCTION]

# constants
width = 1230
height = 493
margin = 10

control_component_width = 200
control_parameter_width = 150
control_component_height = 22
control_component_all_height = 2 * control_component_height + margin
control_address_width_acc = 100

frame_control_width = 6 * margin // 2 + control_component_width + control_parameter_width
button_width = (frame_control_width - 5 * margin) // 4
