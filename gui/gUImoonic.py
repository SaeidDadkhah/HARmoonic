import tkinter.filedialog as filedialog
from tkinter import *
from tkinter import ttk
from PIL import ImageTk
from PIL import Image

from convolutional_neural_network.cnn import CNN
from statistical_feature_extraction import test
from statistical_feature_extraction import sfe
from util import files
from gui import plot

# Images
image_plot = '.\\images\\tmp.png'
image_reading_data = '.\\images\\S1-ReadingData.jpg'
image_normalizing = '.\\images\\S2-Normalizing.jpg'
image_shuffling = '.\\images\\S3-Shuffling.jpg'
image_reducing_dimensions = '.\\images\\S4-ReducingDimensions.jpg'
image_testing_model = '.\\images\\S5-TestingModel.jpg'
image_class = {
    'a01': '.\\images\\A1-Sitting.jpg',
    'a02': '.\\images\\A2-Standing.jpg',
    'a03': '.\\images\\A3-LyingOnBack.jpg',
    'a04': '.\\images\\A4-LyingOnRightSide.jpg',
    'a05': '.\\images\\A5-AscendingStairs.jpg',
    'a06': '.\\images\\A6-DescendingStairs.jpg',
    'a07': '.\\images\\A7-StandingInElevator.jpg',
    'a08': '.\\images\\A8-MovingAroundInElevator.jpg',
    'a09': '.\\images\\A9-WalkingInParkingLot.jpg',
    'a10': '.\\images\\A10-WalkingOnTreadmill4-0.jpg',
    'a11': '.\\images\\A11-WalkingOnTreadmill4-15.jpg',
    'a12': '.\\images\\A12-RunningOnTreadmill8.jpg',
    'a13': '.\\images\\A13-ExercisingOnStepper.jpg',
    'a14': '.\\images\\A14-ExercisingOnCrossTrainer.jpg',
    'a15': '.\\images\\A15-CyclingOnExerciseBikeHorizontal.jpg',
    'a16': '.\\images\\A16-CyclingOnExerciseBikeVertical.jpg',
    'a17': '.\\images\\A17-Rowing.jpg',
    'a18': '.\\images\\A18-Jumping.jpg',
    'a19': '.\\images\\A19-PlayingBasketball.jpg',
}

# Inputs
method = 'Method'
input_data_type = 'Input Data Type'
load_address = 'Load Address'
save_data = 'Save Data'
save_address = 'Save Address'
normalize = 'Normalize'
shuffle = 'Shuffle'
seed = 'Seed'
dimensionality_reduction = 'Dimensionality Reduction'
dimensions = 'Dimensions'
model = 'Model'
model_parameter = 'Model Parameter'
test_strategy = 'Test Strategy'
test_parameter = 'Test Parameter'

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
MODELS = [sfe.LOGISTIC_REGRESSION, sfe.RANDOM_FOREST, sfe.SVM, sfe.LINEAR_SVM, sfe.K_NEAREST_NEIGHBORS, sfe.NAIVE_BAYES,
          sfe.MULTILAYER_PERCEPTRON, sfe.DECISION_TREE, sfe.GAUSSIAN_PROCESS, sfe.ADABOOST, sfe.RADIAL_BASIS_FUNCTION]

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
button_width_sfe = (frame_control_width - 5 * margin) // 4
button_width_cnn = (frame_control_width - 4 * margin) // 3


class GUImoonic:
    def __init__(self):
        self.__state = dict()
        self.__current_load_address = None
        self.__current_save_address = None
        self.__current_data_address = None
        self.__current_model_address = None
        self.__current_result = None
        self.__current_image = None
        self.__har = sfe.HAR()
        self.__cnn = CNN()

        # Making root window
        self.root = Tk()
        self.root.title("HARmoonic")

        # Controlling size of root window
        # self.root.resizable(0, 0)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry("{width}x{height}+{x}+{y}".format(width=width,
                                                             height=height,
                                                             x=int((screen_width - width) / 2),
                                                             y=int((screen_height - height - 50) / 2)))

        # Making plot frame
        frame_plot = LabelFrame(self.root, text="Plot")
        frame_plot.place(width=-margin - frame_control_width - 2 * margin,
                         height=-margin * 3 // 2,
                         x=margin,
                         y=margin // 2,
                         relheight=1,
                         relwidth=1,
                         relx=0,
                         rely=0)

        frame_control = LabelFrame(self.root, text="Toolbox")
        frame_control.place(width=frame_control_width,
                            height=-margin * 3 // 2,
                            x=-margin - frame_control_width,
                            y=margin // 2,
                            relheight=1,
                            relwidth=0,
                            relx=1,
                            rely=0)

        # Plot
        self.label_plot = Label(frame_plot)
        self.label_plot.place(relwidth=1, relheight=1)
        self.label_plot.bind('<Configure>', lambda event: self.__resize_callback())

        # Method
        self.label_method = Label(frame_control, text=method + ':', anchor=W, bg='red')
        self.label_method.place(width=control_component_width,
                                height=control_component_height,
                                x=margin // 2,
                                y=0)
        self.combobox_method = ttk.Combobox(frame_control, state='readonly')
        self.combobox_method['values'] = METHODS
        self.combobox_method.current(1)
        self.combobox_method.bind('<<ComboboxSelected>>', lambda event: self.__select_method())
        self.combobox_method.place(width=control_component_width,
                                   height=control_component_height,
                                   x=margin,
                                   y=control_component_height)

        # Control Frame
        self.frame_statistical = Frame(frame_control, bg='green')
        self.frame_statistical.place(width=0,
                                     height=-control_component_all_height,
                                     relwidth=0,
                                     relheight=0,
                                     x=0,
                                     y=control_component_all_height)

        self.frame_cnn = Frame(frame_control, bg='blue')
        self.frame_cnn.place(width=0,
                             height=-control_component_all_height,
                             relwidth=0,
                             relheight=0,
                             x=0,
                             y=control_component_all_height)

        # Statistical Method Frame
        # Input Data Type
        self.label_data_type_sfe = Label(self.frame_statistical, text=input_data_type + ':', anchor=W, bg='red')
        self.label_data_type_sfe.place(width=control_component_width - control_address_width_acc,
                                       height=control_component_height,
                                       x=margin // 2,
                                       y=0)
        self.combobox_data_type_sfe = ttk.Combobox(self.frame_statistical, state='readonly')
        self.combobox_data_type_sfe['values'] = DATA_TYPE
        self.combobox_data_type_sfe.current(1)
        self.combobox_data_type_sfe.place(width=control_component_width - control_address_width_acc,
                                          height=control_component_height,
                                          x=margin,
                                          y=control_component_height)

        # Load Address
        self.label_load_address_sfe = Label(self.frame_statistical, text=load_address + ':', anchor=W, bg='red')
        self.label_load_address_sfe.place(width=control_parameter_width - 60,
                                          height=control_component_height,
                                          x=3 * margin // 2 + control_component_width - control_address_width_acc,
                                          y=0)
        self.button_load_address_sfe = Button(self.frame_statistical,
                                              text='Browse...',
                                              command=lambda: self.__browse_load_address())
        x = 3 * margin // 2 \
            + control_component_width \
            + control_parameter_width \
            - control_address_width_acc - 55
        self.button_load_address_sfe.place(width=55,
                                           height=control_component_height,
                                           x=x,
                                           y=0)
        self.label_current_load_address_sfe = Label(self.frame_statistical, text="No file/folder", anchor=W, bg='red')
        self.label_current_load_address_sfe.place(width=control_parameter_width + 100,
                                                  height=control_component_height,
                                                  x=2 * margin + control_component_width - control_address_width_acc,
                                                  y=control_component_height)

        # Save Data
        self.label_save_data = Label(self.frame_statistical, text=save_data + ':', anchor=W, bg='red')
        self.label_save_data.place(width=control_component_width - control_address_width_acc,
                                   height=control_component_height,
                                   x=margin // 2,
                                   y=control_component_all_height)
        self.combobox_save_data = ttk.Combobox(self.frame_statistical, state='readonly')
        self.combobox_save_data['values'] = YES_NO
        self.combobox_save_data.current(1)
        self.combobox_save_data.place(width=control_component_width - control_address_width_acc,
                                      height=control_component_height,
                                      x=margin,
                                      y=control_component_all_height + control_component_height)

        # Save Address
        self.label_save_address = Label(self.frame_statistical, text=save_address + ':', anchor=W, bg='red')
        self.label_save_address.place(width=control_parameter_width - 60,
                                      height=control_component_height,
                                      x=3 * margin // 2 + control_component_width - control_address_width_acc,
                                      y=control_component_all_height)
        self.button_save_address = Button(self.frame_statistical,
                                          text='Browse...',
                                          command=lambda: self.__browse_save_address())
        x = 3 * margin // 2 \
            + control_component_width \
            + control_parameter_width \
            - control_address_width_acc - 55
        self.button_save_address.place(width=55,
                                       height=control_component_height,
                                       x=x,
                                       y=control_component_all_height)
        self.label_current_save_address = Label(self.frame_statistical, text="No file", anchor=W, bg='red')
        self.label_current_save_address.place(width=control_parameter_width + 100,
                                              height=control_component_height,
                                              x=2 * margin + control_component_width - control_address_width_acc,
                                              y=control_component_all_height + control_component_height)

        # Normalize
        self.label_normalize = Label(self.frame_statistical, text=normalize + ':', anchor=W, bg='red')
        self.label_normalize.place(width=control_component_width,
                                   height=control_component_height,
                                   x=margin // 2,
                                   y=2 * control_component_all_height)
        self.combobox_normalize = ttk.Combobox(self.frame_statistical, state='readonly')
        self.combobox_normalize['values'] = YES_NO
        self.combobox_normalize.current(0)
        self.combobox_normalize.place(width=control_component_width,
                                      height=control_component_height,
                                      x=margin,
                                      y=2 * control_component_all_height + control_component_height)

        # Shuffle
        self.label_shuffle = Label(self.frame_statistical, text=shuffle + ':', anchor=W, bg='red')
        self.label_shuffle.place(width=control_component_width,
                                 height=control_component_height,
                                 x=margin // 2,
                                 y=3 * control_component_all_height)
        self.combobox_shuffle = ttk.Combobox(self.frame_statistical, state='readonly')
        self.combobox_shuffle['values'] = YES_NO
        self.combobox_shuffle.current(0)
        self.combobox_shuffle.place(width=control_component_width,
                                    height=control_component_height,
                                    x=margin,
                                    y=3 * control_component_all_height + control_component_height)

        # Seed
        self.label_seed = Label(self.frame_statistical, text=seed + ':', anchor=W, bg='red')
        self.label_seed.place(width=control_parameter_width,
                              height=control_component_height,
                              x=3 * margin // 2 + control_component_width,
                              y=3 * control_component_all_height)
        self.entry_seed = Entry(self.frame_statistical)
        self.entry_seed.place(width=control_parameter_width,
                              height=control_component_height,
                              x=2 * margin + control_component_width,
                              y=3 * control_component_all_height + control_component_height)

        # Dimensionality Reduction
        self.label_dimensionality_reduction = Label(self.frame_statistical,
                                                    text=dimensionality_reduction + ':',
                                                    anchor=W,
                                                    bg='red')
        self.label_dimensionality_reduction.place(width=control_component_width,
                                                  height=control_component_height,
                                                  x=margin // 2,
                                                  y=4 * control_component_all_height)
        self.combobox_dimensionality_reduction = ttk.Combobox(self.frame_statistical,
                                                              state='readonly')
        self.combobox_dimensionality_reduction['values'] = DIMENSIONALITY_REDUCTION
        self.combobox_dimensionality_reduction.current(0)
        self.combobox_dimensionality_reduction.place(width=control_component_width,
                                                     height=control_component_height,
                                                     x=margin,
                                                     y=4 * control_component_all_height + control_component_height)

        # Dimensions
        self.label_dimensions = Label(self.frame_statistical, text=dimensions + ':', anchor=W, bg='red')
        self.label_dimensions.place(width=control_parameter_width,
                                    height=control_component_height,
                                    x=3 * margin // 2 + control_component_width,
                                    y=4 * control_component_all_height)
        self.entry_dimensions = Entry(self.frame_statistical)
        self.entry_dimensions.place(width=control_parameter_width,
                                    height=control_component_height,
                                    x=2 * margin + control_component_width,
                                    y=4 * control_component_all_height + control_component_height)

        # Model
        self.label_model = Label(self.frame_statistical, text=model + ':', anchor=W, bg='red')
        self.label_model.place(width=control_component_width,
                               height=control_component_height,
                               x=margin // 2,
                               y=5 * control_component_all_height)
        self.combobox_model = ttk.Combobox(self.frame_statistical, state='readonly')
        self.combobox_model['values'] = MODELS
        self.combobox_model.current(5)
        self.combobox_model.place(width=control_component_width,
                                  height=control_component_height,
                                  x=margin,
                                  y=5 * control_component_all_height + control_component_height)

        # Model Parameter
        self.label_model_parameter = Label(self.frame_statistical, text="Model Parameter:", anchor=W, bg='red')
        self.label_model_parameter.place(width=control_parameter_width,
                                         height=control_component_height,
                                         x=3 * margin // 2 + control_component_width,
                                         y=5 * control_component_all_height)
        self.entry_model_parameter = Entry(self.frame_statistical)
        self.entry_model_parameter.place(width=control_parameter_width,
                                         height=control_component_height,
                                         x=2 * margin + control_component_width,
                                         y=5 * control_component_all_height + control_component_height)

        # Test Method
        self.label_test_method = Label(self.frame_statistical, text=test_strategy + ':', anchor=W, bg='red')
        self.label_test_method.place(width=control_component_width,
                                     height=control_component_height,
                                     x=margin // 2,
                                     y=6 * control_component_all_height)
        self.combobox_test_strategy = ttk.Combobox(self.frame_statistical, state='readonly')
        self.combobox_test_strategy['values'] = sfe.TEST_STRATEGIES
        self.combobox_test_strategy.current(0)
        self.combobox_test_strategy.place(width=control_component_width,
                                          height=control_component_height,
                                          x=margin,
                                          y=6 * control_component_all_height + control_component_height)

        # Test Parameter
        self.label_test_parameter = Label(self.frame_statistical, text=test_parameter + ':', anchor=W, bg='red')
        self.label_test_parameter.place(width=control_parameter_width,
                                        height=control_component_height,
                                        x=3 * margin // 2 + control_component_width,
                                        y=6 * control_component_all_height)
        self.entry_test_parameter = Entry(self.frame_statistical)
        self.entry_test_parameter.place(width=control_parameter_width,
                                        height=control_component_height,
                                        x=2 * margin + control_component_width,
                                        y=6 * control_component_all_height + control_component_height)

        # Buttons
        self.button_test_sfe = Button(self.frame_statistical,
                                      text="Test",
                                      command=lambda: self.__test_sfe())
        self.button_test_sfe.place(width=button_width_sfe,
                                   height=control_component_height,
                                   x=margin,
                                   y=7 * control_component_all_height)

        self.button_confusion_matrix = Button(self.frame_statistical,
                                              text="Test Matrix",
                                              command=lambda: self.__show_confusion_matrix())
        self.button_confusion_matrix.place(width=button_width_sfe,
                                           height=control_component_height,
                                           x=2 * margin + button_width_sfe,
                                           y=7 * control_component_all_height)

        self.button_accuracy_sfe = Button(self.frame_statistical,
                                          text="Test Accuracy",
                                          command=lambda: self.__show_accuracy_sfe())
        self.button_accuracy_sfe.place(width=button_width_sfe,
                                       height=control_component_height,
                                       x=3 * margin + 2 * button_width_sfe,
                                       y=7 * control_component_all_height)

        self.button_predict_sfe = Button(self.frame_statistical,
                                         text="Predict",
                                         command=lambda: self.__predict_sfe())
        self.button_predict_sfe.place(width=button_width_sfe,
                                      height=control_component_height,
                                      x=4 * margin + 3 * button_width_sfe,
                                      y=7 * control_component_all_height)

        # Convolutional Neural Network Method Frame
        # Input Data Type
        self.label_data_type_cnn = Label(self.frame_cnn, text=input_data_type + ':', anchor=W, bg='red')
        self.label_data_type_cnn.place(width=control_component_width - control_address_width_acc,
                                       height=control_component_height,
                                       x=margin // 2,
                                       y=0)
        self.combobox_data_type_cnn = ttk.Combobox(self.frame_cnn, state='readonly')
        self.combobox_data_type_cnn['values'] = DATA_TYPE
        self.combobox_data_type_cnn.current(1)
        self.combobox_data_type_cnn.place(width=control_component_width - control_address_width_acc,
                                          height=control_component_height,
                                          x=margin,
                                          y=control_component_height)

        # Load Address
        self.label_load_address_cnn = Label(self.frame_cnn, text=load_address + ':', anchor=W, bg='red')
        self.label_load_address_cnn.place(width=control_parameter_width - 60,
                                          height=control_component_height,
                                          x=3 * margin // 2 + control_component_width - control_address_width_acc,
                                          y=0)
        self.button_load_address_cnn = Button(self.frame_cnn,
                                              text='Browse...',
                                              command=lambda: self.__browse_data_address())
        x = 3 * margin // 2 \
            + control_component_width \
            + control_parameter_width \
            - control_address_width_acc - 55
        self.button_load_address_cnn.place(width=55,
                                           height=control_component_height,
                                           x=x,
                                           y=0)
        self.label_current_load_address_cnn = Label(self.frame_cnn, text="No folder", anchor=W, bg='red')
        self.label_current_load_address_cnn.place(width=control_parameter_width + 100,
                                                  height=control_component_height,
                                                  x=2 * margin + control_component_width - control_address_width_acc,
                                                  y=control_component_height)

        # Model Address
        self.label_model_address = Label(self.frame_cnn, text=save_address + ':', anchor=W, bg='red')
        self.label_model_address.place(width=control_parameter_width - 60,
                                       height=control_component_height,
                                       x=margin // 2,
                                       y=control_component_all_height)
        self.button_model_address = Button(self.frame_cnn,
                                           text='Browse...',
                                           command=lambda: self.__browse_model_address())
        self.button_model_address.place(width=55,
                                        height=control_component_height,
                                        x=margin // 2 + control_parameter_width - 55,
                                        y=control_component_all_height)
        self.label_current_model_address = Label(self.frame_cnn, text="No folder", anchor=W, bg='red')
        self.label_current_model_address.place(width=-3 * margin // 2,
                                               height=control_component_height,
                                               relwidth=1,
                                               relheight=0,
                                               x=margin,
                                               y=control_component_all_height + control_component_height)

        # Buttons
        self.button_test = Button(self.frame_cnn,
                                  text="Test",
                                  command=lambda: self.__test_cnn())
        self.button_test.place(width=button_width_cnn,
                               height=control_component_height,
                               x=margin,
                               y=2 * control_component_all_height)

        self.button_accuracy_sfe = Button(self.frame_cnn,
                                          text="Show Accuracy",
                                          command=lambda: self.__show_accuracy_cnn())
        self.button_accuracy_sfe.place(width=button_width_cnn,
                                       height=control_component_height,
                                       x=2 * margin + button_width_cnn,
                                       y=2 * control_component_all_height)

        self.button_predict_sfe = Button(self.frame_cnn,
                                         text="Predict",
                                         command=lambda: self.__predict_cnn())
        self.button_predict_sfe.place(width=button_width_cnn,
                                      height=control_component_height,
                                      x=3 * margin + 2 * button_width_cnn,
                                      y=2 * control_component_all_height)

        self.__select_method()
        self.root.mainloop()

    def __set_image(self, path):
        self.__current_image = Image.open(path)
        self.__redraw_image()

    def __redraw_image(self):
        img = self.__current_image.resize((self.label_plot.winfo_width(), self.label_plot.winfo_height()),
                                          Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.label_plot.configure(image=img)
        self.label_plot.image = img
        self.root.update()

    def __resize_callback(self):
        if self.__current_image is not None:
            self.__redraw_image()

    def __select_method(self):
        selected_method = self.combobox_method.get()
        print('Select method:', selected_method)
        if selected_method == statistical:
            self.frame_statistical.place(relwidth=1, relheight=1)
            self.frame_cnn.place(relwidth=0, relheight=0)
        elif selected_method == cnn:
            self.frame_statistical.place(relwidth=0, relheight=0)
            self.frame_cnn.place(relwidth=1, relheight=1)
        self.root.update()

    # Statistical Methods
    def __browse_load_address(self):
        file_name_len = 41
        data_type = self.combobox_data_type_sfe.get()
        if data_type == raw:
            address = filedialog.askdirectory()
        elif data_type == processed:
            address = filedialog.askopenfilename()
        else:
            return
        if address == '':
            return
        self.__current_load_address = address
        address = files.shrink_file_name(address, file_name_len)
        self.label_current_load_address_sfe.config(text=address)

    def __browse_save_address(self):
        file_name_len = 41
        address = filedialog.asksaveasfilename()
        if address == '':
            return
        self.__current_save_address = address
        address = files.shrink_file_name(address, file_name_len)
        self.label_current_save_address.config(text=address)

    def __test_sfe(self):
        self.__set_image(image_reading_data)
        self.__state[input_data_type] = self.combobox_data_type_sfe.get()
        print('Data type:', self.__state[input_data_type])
        if self.__state[input_data_type] == raw:
            self.__har.load_data(self.__current_load_address)
        elif self.__state[input_data_type] == processed:
            self.__har.load_pickle(self.__current_load_address)
        else:
            return

        self.__state[save_data] = self.combobox_save_data.get()
        print('Save data:', self.__state[save_data])
        if self.__state[save_data] == yes:
            self.__har.save_pickle(self.__current_save_address)
        elif self.__state[save_data] == no:
            pass
        else:
            return

        self.__set_image(image_normalizing)
        self.__state[normalize] = self.combobox_normalize.get()
        print('Normalize:', self.__state[normalize])
        if self.__state[normalize] == yes:
            self.__har.normalize()
        elif self.__state[normalize] == no:
            pass
        else:
            return

        self.__set_image(image_shuffling)
        self.__state[shuffle] = self.combobox_shuffle.get()
        print('Shuffle:', self.__state[shuffle])
        if self.__state[shuffle] == yes:
            self.__state[seed] = self.entry_seed.get()
            try:
                self.__state[seed] = int(self.__state[seed])
            except ValueError:
                self.__state[seed] = 0
            if seed != 0:
                print('\tSeed:', self.__state[seed])
                sfe.set_seed(self.__state[seed])
            self.__har.shuffle()
        elif self.__state[shuffle] == no:
            pass
        else:
            return

        self.__set_image(image_reducing_dimensions)
        self.__state[dimensionality_reduction] = self.combobox_dimensionality_reduction.get()
        print('Dimensionality reduction:', self.__state[dimensionality_reduction])
        if self.__state[dimensionality_reduction] == no_dimensionality_reduction:
            pass
        elif self.__state[dimensionality_reduction] == pca:
            self.__state[dimensions] = self.entry_dimensions.get()
            try:
                self.__state[dimensions] = int(self.__state[dimensions])
            except ValueError:
                self.__state[dimensions] = 30
            print('\tDimensions:', self.__state[dimensions])
            self.__har.pca(self.__state[dimensions])
        elif self.__state[dimensionality_reduction] == lda:
            self.__state[dimensions] = self.entry_dimensions.get()
            try:
                self.__state[dimensions] = int(self.__state[dimensions])
            except ValueError:
                self.__state[dimensions] = 18
            print('\tDimensions:', self.__state[dimensions])
            self.__har.lda(self.__state[dimensions])
        else:
            return

        self.__set_image(image_testing_model)
        self.__har.drop_extra_features()

        self.__state[model] = self.combobox_model.get()
        print('Model:', model)
        self.__har.select_models([self.__state[model]])

        self.__state[test_strategy] = self.combobox_test_strategy.get()
        print('Test strategy:', self.__state[test_strategy])
        result = self.__har.test(self.__state[test_strategy])
        self.__current_result = result[0]
        self.__show_confusion_matrix()
        self.__har.split_data()
        self.__har.train()

    def __show_confusion_matrix(self):
        plot.confusion_matrix(
            self.__current_result[test.constants.CONFUSION_MATRIX],
            classes=['a{}'.format(i) for i in range(1, 20)],
            normalize=True)
        self.__set_image(image_plot)

    def __show_accuracy_sfe(self):
        plot.normal(self.__current_result[test.constants.ACCURACY])
        self.__set_image(image_plot)

    def __predict_sfe(self):
        address = filedialog.askopenfilename()
        if address == '':
            return
        address = files.separate_by_os(address)
        if address == '':
            address = None
        self.__har.load_instance(address)

        if self.__state[normalize] == yes:
            self.__har.normalize(train=False)

        if self.__state[dimensionality_reduction] == pca:
            self.__har.pca(train=False)
        elif self.__state[dimensionality_reduction] == lda:
            self.__har.lda(train=False)

        self.__set_image(image_class[self.__har.predict()[0][0]])

    # Convolutional Neural Network Methods
    def __browse_data_address(self):
        file_name_len = 41
        address = filedialog.askdirectory()
        if address == '':
            return
        address = files.separate_by_os(address)
        address = files.end_with_sep(address)
        self.__current_data_address = address
        address = files.shrink_file_name(address, file_name_len)
        self.label_current_load_address_cnn.config(text=address)

    def __browse_model_address(self):
        file_name_len = 58
        address = filedialog.askdirectory()
        if address == '':
            return
        address = files.separate_by_os(address)
        address = files.end_with_sep(address)
        self.__current_model_address = address
        address = files.shrink_file_name(address, file_name_len)
        self.label_current_model_address.config(text=address)

        self.__cnn = CNN(self.__current_model_address + 'model_{}.ckpt')
        self.__current_result = (self.__cnn.training_accuracy, self.__cnn.testing_accuracy)
        self.__show_accuracy_cnn()

    def __test_cnn(self):
        self.__set_image(image_reading_data)
        self.__state[input_data_type] = self.combobox_data_type_cnn.get()
        print('Data type:', self.__state[input_data_type])
        if self.__state[input_data_type] == raw:
            self.__cnn.read_data(self.__current_data_address)
            self.__cnn.save_data(self.__current_model_address)
        elif self.__state[input_data_type] == processed:
            self.__cnn.load_data(self.__current_data_address)
        else:
            return
        self.__set_image(image_testing_model)
        self.__cnn.split_data()
        model_address = self.__current_model_address
        if model_address is not None:
            model_address = model_address + 'model_{}.ckpt'

        checkpoints = files.get_checkpoints_list(model_address[:-13])
        if len(checkpoints) > 0:
            self.__cnn.run(path=model_address, last_checkpoint=checkpoints[-1])
        else:
            self.__cnn.run(path=model_address)
        self.__current_result = (self.__cnn.training_accuracy, self.__cnn.testing_accuracy)
        self.__show_accuracy_cnn()

    def __show_accuracy_cnn(self):
        plot.trend(self.__current_result)
        self.__set_image(image_plot)

    def __predict_cnn(self):
        address = filedialog.askopenfilename()
        if address == '':
            return
        address = files.separate_by_os(address)
        if address == '':
            address = None

        model_address = None
        if self.__current_model_address is not None:
            model_address = self.__current_model_address + 'model_{}.ckpt'
        result = CNN().predict(model=model_address, path=address)[0]
        self.__set_image(image_class['a{:02}'.format(result)])
