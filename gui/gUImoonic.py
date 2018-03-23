import tkinter.filedialog as filedialog
from tkinter import *
from tkinter import ttk
from PIL import ImageTk
from PIL import Image

from statistical_feature_extraction.test import constants
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
button_width = (frame_control_width - 5 * margin) // 4


class GUImoonic:
    def __init__(self):
        self.__state = dict()
        self.__current_load_address = None
        self.__current_save_address = None
        self.__current_result = None
        self.__current_image = None
        self.__har = sfe.HAR()

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
        self.label_method = Label(frame_control, text="Method:", anchor=W, bg='red')
        self.label_method.place(width=control_component_width,
                                height=control_component_height,
                                x=margin // 2,
                                y=0)
        self.combobox_method = ttk.Combobox(frame_control, state='readonly')
        self.combobox_method['values'] = METHODS
        self.combobox_method.current(0)
        self.combobox_method.bind('<<ComboboxSelected>>', lambda: self.__select_method())
        self.combobox_method.place(width=control_component_width,
                                   height=control_component_height,
                                   x=margin,
                                   y=control_component_height)

        # Input Data Type
        self.label_data_type = Label(frame_control, text="Input Data Type:", anchor=W, bg='red')
        self.label_data_type.place(width=control_component_width - control_address_width_acc,
                                   height=control_component_height,
                                   x=margin // 2,
                                   y=control_component_all_height)
        self.combobox_data_type = ttk.Combobox(frame_control, state='readonly')
        self.combobox_data_type['values'] = DATA_TYPE
        self.combobox_data_type.current(1)
        self.combobox_data_type.place(width=control_component_width - control_address_width_acc,
                                      height=control_component_height,
                                      x=margin,
                                      y=control_component_all_height + control_component_height)

        # Load Address
        self.label_load_address = Label(frame_control, text="Load Address:", anchor=W, bg='red')
        self.label_load_address.place(width=control_parameter_width - 60,
                                      height=control_component_height,
                                      x=3 * margin // 2 + control_component_width - control_address_width_acc,
                                      y=control_component_all_height)
        self.button_load_address = Button(frame_control,
                                          text='Browse...',
                                          command=lambda: self.__browse_load_address())
        x = 3 * margin // 2 + control_component_width + control_parameter_width - control_address_width_acc - 55
        self.button_load_address.place(width=55,
                                       height=control_component_height,
                                       x=x,
                                       y=control_component_all_height)
        self.label_current_load_address = Label(frame_control, text="No file/folder", anchor=W, bg='red')
        self.label_current_load_address.place(width=control_parameter_width + 100,
                                              height=control_component_height,
                                              x=2 * margin + control_component_width - control_address_width_acc,
                                              y=control_component_all_height + control_component_height)

        # Save Data
        self.label_save_data = Label(frame_control, text="Save Data:", anchor=W, bg='red')
        self.label_save_data.place(width=control_component_width - control_address_width_acc,
                                   height=control_component_height,
                                   x=margin // 2,
                                   y=2 * control_component_all_height)
        self.combobox_save_data = ttk.Combobox(frame_control, state='readonly')
        self.combobox_save_data['values'] = YES_NO
        self.combobox_save_data.current(1)
        self.combobox_save_data.place(width=control_component_width - control_address_width_acc,
                                      height=control_component_height,
                                      x=margin,
                                      y=2 * control_component_all_height + control_component_height)

        # Save Address
        self.label_save_address = Label(frame_control, text="Save Address:", anchor=W, bg='red')
        self.label_save_address.place(width=control_parameter_width - 60,
                                      height=control_component_height,
                                      x=3 * margin // 2 + control_component_width - control_address_width_acc,
                                      y=2 * control_component_all_height)
        self.button_save_address = Button(frame_control,
                                          text='Browse...',
                                          command=lambda: self.__browse_save_address())
        x = 3 * margin // 2 + control_component_width + control_parameter_width - control_address_width_acc - 55
        self.button_save_address.place(width=55,
                                       height=control_component_height,
                                       x=x,
                                       y=2 * control_component_all_height)
        self.label_current_save_address = Label(frame_control, text="No file", anchor=W, bg='red')
        self.label_current_save_address.place(width=control_parameter_width + 100,
                                              height=control_component_height,
                                              x=2 * margin + control_component_width - control_address_width_acc,
                                              y=2 * control_component_all_height + control_component_height)

        # Normalize
        self.label_normalize = Label(frame_control, text="Normalize:", anchor=W, bg='red')
        self.label_normalize.place(width=control_component_width,
                                   height=control_component_height,
                                   x=margin // 2,
                                   y=3 * control_component_all_height)
        self.combobox_normalize = ttk.Combobox(frame_control, state='readonly')
        self.combobox_normalize['values'] = YES_NO
        self.combobox_normalize.current(0)
        self.combobox_normalize.place(width=control_component_width,
                                      height=control_component_height,
                                      x=margin,
                                      y=3 * control_component_all_height + control_component_height)

        # Shuffle
        self.label_shuffle = Label(frame_control, text="Shuffle:", anchor=W, bg='red')
        self.label_shuffle.place(width=control_component_width,
                                 height=control_component_height,
                                 x=margin // 2,
                                 y=4 * control_component_all_height)
        self.combobox_shuffle = ttk.Combobox(frame_control, state='readonly')
        self.combobox_shuffle['values'] = YES_NO
        self.combobox_shuffle.current(0)
        self.combobox_shuffle.place(width=control_component_width,
                                    height=control_component_height,
                                    x=margin,
                                    y=4 * control_component_all_height + control_component_height)

        # Seed
        self.label_seed = Label(frame_control, text="Seed:", anchor=W, bg='red')
        self.label_seed.place(width=control_parameter_width,
                              height=control_component_height,
                              x=3 * margin // 2 + control_component_width,
                              y=4 * control_component_all_height)
        self.entry_seed = Entry(frame_control)
        self.entry_seed.place(width=control_parameter_width,
                              height=control_component_height,
                              x=2 * margin + control_component_width,
                              y=4 * control_component_all_height + control_component_height)

        # Dimensionality Reduction
        self.label_dimensionality_reduction = Label(frame_control,
                                                    text="Dimensionality Reduction:",
                                                    anchor=W,
                                                    bg='red')
        self.label_dimensionality_reduction.place(width=control_component_width,
                                                  height=control_component_height,
                                                  x=margin // 2,
                                                  y=5 * control_component_all_height)
        self.combobox_dimensionality_reduction = ttk.Combobox(frame_control,
                                                              state='readonly')
        self.combobox_dimensionality_reduction['values'] = DIMENSIONALITY_REDUCTION
        self.combobox_dimensionality_reduction.current(0)
        self.combobox_dimensionality_reduction.place(width=control_component_width,
                                                     height=control_component_height,
                                                     x=margin,
                                                     y=5 * control_component_all_height + control_component_height)

        # Dimensions
        self.label_dimensions = Label(frame_control, text="Dimensions:", anchor=W, bg='red')
        self.label_dimensions.place(width=control_parameter_width,
                                    height=control_component_height,
                                    x=3 * margin // 2 + control_component_width,
                                    y=5 * control_component_all_height)
        self.entry_dimensions = Entry(frame_control)
        self.entry_dimensions.place(width=control_parameter_width,
                                    height=control_component_height,
                                    x=2 * margin + control_component_width,
                                    y=5 * control_component_all_height + control_component_height)

        # Model
        self.label_model = Label(frame_control, text="Model:", anchor=W, bg='red')
        self.label_model.place(width=control_component_width,
                               height=control_component_height,
                               x=margin // 2,
                               y=6 * control_component_all_height)
        self.combobox_model = ttk.Combobox(frame_control, state='readonly')
        self.combobox_model['values'] = MODELS
        self.combobox_model.current(5)
        self.combobox_model.place(width=control_component_width,
                                  height=control_component_height,
                                  x=margin,
                                  y=6 * control_component_all_height + control_component_height)

        # Model Parameter
        self.label_model_parameter = Label(frame_control, text="Model Parameter:", anchor=W, bg='red')
        self.label_model_parameter.place(width=control_parameter_width,
                                         height=control_component_height,
                                         x=3 * margin // 2 + control_component_width,
                                         y=6 * control_component_all_height)
        self.entry_model_parameter = Entry(frame_control)
        self.entry_model_parameter.place(width=control_parameter_width,
                                         height=control_component_height,
                                         x=2 * margin + control_component_width,
                                         y=6 * control_component_all_height + control_component_height)

        # Test Method
        self.label_test_method = Label(frame_control, text="Test Method:", anchor=W, bg='red')
        self.label_test_method.place(width=control_component_width,
                                     height=control_component_height,
                                     x=margin // 2,
                                     y=7 * control_component_all_height)
        self.combobox_test_strategy = ttk.Combobox(frame_control, state='readonly')
        self.combobox_test_strategy['values'] = sfe.TEST_STRATEGIES
        self.combobox_test_strategy.current(0)
        self.combobox_test_strategy.place(width=control_component_width,
                                          height=control_component_height,
                                          x=margin,
                                          y=7 * control_component_all_height + control_component_height)

        # Test Parameter
        self.label_test_parameter = Label(frame_control, text="Test Parameter:", anchor=W, bg='red')
        self.label_test_parameter.place(width=control_parameter_width,
                                        height=control_component_height,
                                        x=3 * margin // 2 + control_component_width,
                                        y=7 * control_component_all_height)
        self.entry_test_parameter = Entry(frame_control)
        self.entry_test_parameter.place(width=control_parameter_width,
                                        height=control_component_height,
                                        x=2 * margin + control_component_width,
                                        y=7 * control_component_all_height + control_component_height)

        # Buttons
        self.button_test = Button(frame_control,
                                  text="Test",
                                  command=lambda: self.__test())
        self.button_test.place(width=button_width,
                               height=control_component_height,
                               x=margin,
                               y=8 * control_component_all_height)

        self.button_confusion_matrix = Button(frame_control,
                                              text="Test Matrix",
                                              command=lambda: self.__show_confusion_matrix())
        self.button_confusion_matrix.place(width=button_width,
                                           height=control_component_height,
                                           x=2 * margin + button_width,
                                           y=8 * control_component_all_height)

        self.button_boxplot = Button(frame_control,
                                     text="Test Accuracy",
                                     command=lambda: self.__show_accuracy())
        self.button_boxplot.place(width=button_width,
                                  height=control_component_height,
                                  x=3 * margin + 2 * button_width,
                                  y=8 * control_component_all_height)

        self.button_predict = Button(frame_control,
                                     text="Predict",
                                     command=lambda: self.__predict())
        self.button_predict.place(width=button_width,
                                  height=control_component_height,
                                  x=4 * margin + 3 * button_width,
                                  y=8 * control_component_all_height)

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

    # noinspection PyUnusedLocal
    def __select_method(self):
        print('Select method:', self.combobox_method.get())

    def __browse_load_address(self):
        file_name_len = 41
        data_type = self.combobox_data_type.get()
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
        self.label_current_load_address.config(text=address)

    def __browse_save_address(self):
        file_name_len = 41
        address = filedialog.asksaveasfilename()
        if address == '':
            return
        self.__current_save_address = address
        address = files.shrink_file_name(address, file_name_len)
        self.label_current_save_address.config(text=address)

    def __test(self):
        self.__set_image(image_reading_data)
        data_type = self.combobox_data_type.get()
        print('Data type:', data_type)
        if data_type == raw:
            self.__har.load_data(self.__current_load_address)
        elif data_type == processed:
            self.__har.load_pickle(self.__current_load_address)
        else:
            return

        save_data = self.combobox_save_data.get()
        print('Save data:', save_data)
        if save_data == yes:
            self.__har.save_pickle(self.__current_save_address)
        elif save_data == no:
            pass
        else:
            return

        self.__set_image(image_normalizing)
        normalize = self.combobox_normalize.get()
        print('Normalize:', normalize)
        if normalize == yes:
            self.__har.normalize()
        elif normalize == no:
            pass
        else:
            return

        self.__set_image(image_shuffling)
        shuffle = self.combobox_shuffle.get()
        print('Shuffle:', shuffle)
        if shuffle == yes:
            seed = self.entry_seed.get()
            try:
                seed = int(seed)
            except ValueError:
                seed = 0
            if seed != 0:
                print('\tSeed:', seed)
                sfe.set_seed(seed)
            self.__har.shuffle()
        elif shuffle == no:
            pass
        else:
            return

        self.__set_image(image_reducing_dimensions)
        dimensionality_reduction = self.combobox_dimensionality_reduction.get()
        print('Dimensionality reduction:', dimensionality_reduction)
        if dimensionality_reduction == no_dimensionality_reduction:
            pass
        elif dimensionality_reduction == pca:
            dimensions = self.entry_dimensions.get()
            try:
                dimensions = int(dimensions)
            except ValueError:
                dimensions = 30
            print('\tDimensions:', dimensions)
            self.__har.dimensionality_reduction(dimensions)
        elif dimensionality_reduction == lda:
            dimensions = self.entry_dimensions.get()
            try:
                dimensions = int(dimensions)
            except ValueError:
                dimensions = 18
            print('\tDimensions:', dimensions)
            self.__har.dimensionality_lda_reduction(dimensions)
        else:
            return

        self.__set_image(image_testing_model)
        self.__har.drop_extra_features()

        model = self.combobox_model.get()
        print('Model:', model)
        self.__har.select_models([model])

        test_strategy = self.combobox_test_strategy.get()
        print('Test strategy:', test_strategy)
        result = self.__har.test(test_strategy)
        self.__current_result = result[0]
        # self.__show_confusion_matrix()
        self.__show_accuracy()

    def __show_confusion_matrix(self):
        plot.confusion_matrix(
            self.__current_result[constants.CONFUSION_MATRIX],
            classes=['a{}'.format(i) for i in range(1, 20)],
            normalize=True)
        self.__set_image(image_plot)

    def __show_accuracy(self):
        plot.normal(self.__current_result[constants.ACCURACY])
        self.__set_image(image_plot)

    def __predict(self):
        pass
