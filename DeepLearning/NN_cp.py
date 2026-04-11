import numpy as np
import cupy as cp
import pickle


class ReLU():
    def forward(self, z):
        return np.maximum(0, z)
    
    #derivative of ReLU is 0(x<0) and 1(x>0)
    def backward(self, z):
        return (z>0).astype(float)
    
class Sigmoid():
    def forward(self, z):
        return 1 / (1 + np.exp(-z))
    
    def backward(self, z):
        s = self.forward(z)
        return s * (1-s)
    
class Softmax():
    def forward(self, z):
        # Minus z with max value of z to avoid overflow of e^z
        # z.shape = (batch, num_class)
        # output.shape = (batch, num_class)
        output = np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims=True) 
        return output

class BinaryCrossEntropy:
    def forward(self, y_pred, y):
        #y_pred.shape    = (1, batch)
        #y.shape         = (1, batch)  (0 or 1)
        #output.shape    = (1, batch)
        return - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

class CategoricalCrossEntropy:
    def forward(self, y_pred, y):
        #y_pred.shape    = (batch, num_class)
        #y.shape         = (batch, num_class)
        #output.shap     = (batch, 1) - num_class has been sum
        return np.sum(-y * np.log(y_pred), 1)

    def backward(self, z):
        return -1/z

class Flatten:
    def __init__(self):
        self.batch = None
        self.channel = None
        self.height = None
        self.width = None

    def forward(self, x):
        # Flatten (batch, channel, height, width) inputs -> (batch, channel, height x width)
        self.batch, self.channel, self.height, self.width = x.shape
        return x.reshape(self.batch, -1)
    
    def backward(self, d_values):
        return d_values.reshape(self.batch, self.channel, self.height, self.width)
    
class MaxPooling:
    def __init__(self, kernel_size = (None, None), stride = 1):
        self.kernel_height = kernel_size[-2]
        self.kernel_width = kernel_size[-1]
        self.stride = stride
        self.outputs = None
        self.inputs = None
        self.back_prop = None
        self.derivative = None

    def calculate_conv2d_MaxPooling(inputs, filter_size, stride, input_shape = ()):
        # TODO: recheck maxpooling
        filter_height = filter_size[-2]
        filter_width = filter_size[-1]
        # inputs.shape = (batch_size, channel, height, width)
        batch_size, channel, height, width = inputs.shape

        # inputs.strides = (batch_strides, row_stride, columns_strides)
        # output is the num of bit need to move to next row/column/depth
        batch_stride, channel_stride, row_stride, column_stride = inputs.strides

        output_height = int((height - filter_height) / stride + 1)
        output_width = int((width - filter_width) / stride + 1)

        # we divided each inputs into new matrix (output_height x output_width) each elements size kernel
        new_shape = (batch_size, channel, output_height, output_width, filter_height, filter_width)

        # bit need to move between row/column/depth of new matrix
        new_stride = (batch_stride, channel_stride, row_stride * stride, column_stride * stride, row_stride, column_stride)

        new_input = np.lib.stride_tricks.as_strided(inputs, new_shape, new_stride)

        # result shape (batch, channe;, height, width)
        result = np.max(new_input, axis=(-2, -1))

        result_2 = np.max(new_input, axis=(-2, -1), keepdims=True)

        # back_propagation of maxpooling now as shape (batch, channel, height, width, kernel_size, kernel_size,) each elements only contain 0,1 | 1 for max
        back_prop = ((new_input - result_2)==0).astype(float)
        
        return result, back_prop
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs, self.back_prop = MaxPooling.calculate_conv2d_MaxPooling(inputs, (self.kernel_height, self.kernel_width), self.stride)
        return self.outputs
    
    def backward(self, d_values):
        # d_values.shape = (batch, channel, height, width)
        self.back_prop = np.einsum("bchwkt,bchw->bchwkt", self.back_prop, d_values, optimize='optimal')

        # create self.derivative as zero matrix (batch, channel, height, width) of self.inputs
        self.derivative = np.zeros(self.inputs.shape).astype(float)

        # inputs.shape = (batch_size, channel, height, width)
        batch_size, channel, height, width = self.inputs.shape

        # inputs.strides = (batch_strides, row_stride, columns_strides)
        # output is the num of bit need to move to next row/column/depth
        batch_stride, channel_stride, row_stride, column_stride = self.inputs.strides

        output_height = int((height - self.kernel_height) / self.stride + 1)
        output_width = int((width - self.kernel_width) / self.stride + 1)

        # we divided each inputs into new matrix (output_height x output_width) each elements size kernel
        conv_shape = (batch_size, channel, output_height, output_width, self.kernel_height, self.kernel_width)

        # bit need to move between row/column/depth of new matrix
        conv_stride = (batch_stride, channel_stride, row_stride * self.stride, column_stride * self.stride, row_stride, column_stride)

        derivative_conv = np.lib.stride_tricks.as_strided(self.derivative, conv_shape, conv_stride, writeable=True)

        np.maximum.at(derivative_conv, tuple(np.ogrid[tuple(slice(s) for s in derivative_conv.shape)]), self.back_prop)

        print(f"self.derivative: {self.derivative.shape}")

        return self.derivative

class Conv2D:
    def __init__(self, num_kernel, kernel_size = (None, None), input_shape = (None, None, None, None), stride = 1, activation = None):
        self.num_kernel = num_kernel
        self.kernel_height = kernel_size[0]
        self.kernel_width = kernel_size[1]
        self.stride = stride
        self.input_channel = input_shape[-3] 
        #we save kernel under weights = (channel, num_filters, height, width)
        self.weights = np.random.randn(self.input_channel, self.num_kernel, self.kernel_height, self.kernel_width) * 0.1
        self.bias = np.zeros((self.input_channel, self.num_kernel, 1))
        self.d_weights = None
        self.activation = activation
        self.inputs = None
        self.outputs = None

    def calculate_conv2d(inputs, filters, stride, padding = 0):
        """
        type 1
            inputs                  = (batch, channel, height, width)
            filters                 = (channel, num_filter, kernel_height, kernel_width)
            result                  = (batch_size, new_channel = num_filter, height, width)

        type 2
            inputs                  = (batch, channel = num_filters_previous_layer, height, width)
            d_values = filters      = (batch, channel = num_filters_current_layer, kernel_height, kernel_width)
        
            d_weights = result      = (batch, num_filters_previous_layer, num_filters_current_layer, kernel_height, kernel_width)

        type 3
            d_values = inputs       = (batch, channel = num_filters_current_layer, height, width)
            self.weights = filters  = (channel = num_filters_previous_layer, num_filters, kernel_height, kernel_width)

            d_input = result        = (batch, num_filters_previous_layer, height, width)
        """

        #create padding = 0 for batch and channel, create padding = padding_size to height and width
        padding_inputs = np.pad(inputs, pad_width = ((0,0), (0,0), (padding, padding), (padding, padding)))
        # filters.shape = (num_filter, channel, kernel_height, kernel_width)
        num_filter = filters.shape[-3]
        kernel_height = filters.shape[-2]
        kernel_width = filters.shape[-1]

        # inputs.shape = (batch_size, channel, height, width)
        batch_size, channel, height, width = padding_inputs.shape

        # inputs.strides = (batch_strides, row_stride, columns_strides)
        # output is the num of bit need to move to next row/column/depth
        batch_stride, channel_stride, row_stride, column_stride = padding_inputs.strides

        output_height = int((height - kernel_height) / stride + 1)
        output_width = int((width - kernel_width) / stride + 1)

        # we divided each inputs into new matrix (output_height x output_width) each elements size kernel
        new_shape = (batch_size, channel, output_height, output_width, kernel_height, kernel_width)

        # bit need to move between row/column/depth of new matrix
        new_stride = (batch_stride, channel_stride, row_stride * stride, column_stride * stride, row_stride, column_stride)

        new_input = np.lib.stride_tricks.as_strided(padding_inputs, new_shape, new_stride)

        return new_input
    
    def forward(self, inputs):
        """
        ==============================
        outputs of forward information
        ==============================

        inputs                              = (batch, channel, height, width)
        ---> conv2d of inputs | bchwkt      = (batch, channel, new_height, new_width, kernel_height, kernel_width)
        weights = filters     | cfkt        = (channel, num_filter, kernel_height, kernel_width)
        outputs               | bfhw        = (batch_size, new_channel = num_filter, height, width)
        """

        self.inputs = inputs
        inputs_conv2d = Conv2D.calculate_conv2d(inputs, self.weights, self.stride)
        self.outputs = np.einsum("bchwkt,cfkt->bfhw", inputs_conv2d, self.weights, optimize='optimal')
        self.outputs = self.activation.forward(self.outputs)

        return self.outputs
    
    #TODO: backpropagation
    def backward(self, d_values):
        """
        =====================
        d_weights information
        =====================
        inputs                              = (batch, channel = num_filters_previous_layer, height, width)
        ---> inputs_conv2d | bchwkt         = (batch, channel = num_filters_previous_layer, new_height, new_width, kernel_height, kernel_width)
        d_values           | bfkt           = (batch, channel = num_filters_current_layer, kernel_height, kernel_width)
        d_weights          | bcfhw          = (batch, num_filters_previous_layer, num_filters_current_layer, kernel_height, kernel_width)


        ===================
        d_input information
        ===================
        d_values                            = (batch, channel = num_filters_current_layer, kernel_height, kernel_width)
        ---> d_values_conv2d | bfhwkt       = (batch, channel = num_filters_current_layer, kernel_height, kernel_width)
        weights_rotated      | bcfkt        = (batch, num_filters_previous_layer, num_filters_current_layer, kernel_height, kernel_width)
        d_input              | bchw         = (batch, channel = num_filters_previous_layer, height, width)
        """
        #dL/dz = d_values * f'(conv_output) 
        d_values = np.einsum("bfkt,bfkt->bfkt",d_values,self.activation.backward(self.outputs), optimize='optimal')

        print(f"ReLU: {self.activation.backward(self.outputs)}")
        print(f"ReLUshape: {self.activation.backward(self.outputs).shape}")

        inputs_conv2d = Conv2D.calculate_conv2d(self.inputs, d_values, self.stride)

        self.d_weights = np.einsum("bchwkt,bfkt->bcfhw", inputs_conv2d, d_values, optimize='optimal')
        self.d_weights = np.sum(self.d_weights, axis = 0)


        column = self.outputs.shape[-1]
        row = self.outputs.shape[-2]
        insert_column = tuple(i for i in range(1, column + 1) for _ in range(self.stride-1))
        insert_row = tuple(i for i in range(1, row + 1) for _ in range(self.stride-1))

        # insert zeros between each elements (for stride>1) 
        d_values = np.insert(d_values, obj = insert_column, values = 0, axis = -1)
        d_values = np.insert(d_values, obj = insert_row, values = 0, axis = -2)

        #self.weights[:,:,::-1,::-1] rotate 180 degree
        rotated_weights = self.weights[:,:,::-1,::-1]
        d_values_conv2d = Conv2D.calculate_conv2d(d_values, self.weights[:,:,::-1,::-1], stride = 1, padding = self.kernel_height - 1)

        d_input = np.einsum("bfhwkt,cfkt->bchw", d_values_conv2d, rotated_weights, optimize='optimal')

        return d_input

class Dense:
    def __init__(self, n_inputs, n_neurons, activation):
        self.weights = np.random.randn(n_neurons, n_inputs) * 0.1  #Create 2D-weights from gaussian disstricution (n_inputs x n_neurons)
        self.d_weights = None
        self.bias = np.zeros((n_neurons,1))   #Create array bias lenth
        self.d_bias = None
        self.activation = activation
        self.z = None
        self.inputs = None
        self.outputs = None

    def forward(self,inputs):
        if isinstance(self.activation, Softmax):
            self.outputs = self.activation.forward(inputs)
        else:
            # inputs.shape  = batch, inputs
            # weights.shape = neurons, inputs
            # bias.shape    = neurons, 1
            # output.shape  = batch, neurons

            self.inputs = inputs
            self.z = np.dot(inputs, self.weights.T) + self.bias.T
            self.outputs = self.activation.forward(self.z)

        return self.outputs
    
    def backward(self, d_values):

        """
        C: Cost
        L: loss
        a(L) = f(z(L))
        z(L) = w(L) * a(L-1) + b(L)
        d_values. shape = m,inputs

        delta   = dL/dz(L) = dL/dz(L+1) * f'(L)     |   (m, n_(L))*(m, n_(L)) = (m, n_(L))
        dC/dW   = (1/m) * w(L+1) * dz(L+1) * f'(L)  |   
                = (1/m) * d_input_pre * w(L-1)T     |   (n_(L), n_(L-1))
                = (1/m) * d_input_pre * w(L-1)T 
        dC/dB   = (1/m) * sum(delta)                |   (n_(L), 1)
        d_input = w(L) * dL/dz(L)
        neuron, inputs * m, inputs
        """
        if not isinstance(self.activation, Softmax):
            #num of batch
            m = d_values.shape[0]

            #delta = dL/dz(L) = dL/dz(L+1) * f'(L) (n_i,m)*(n_i,m) = (n_i,m)
            self.delta =  np.einsum("ml,ml->ml", d_values, self.activation.backward(self.z), optimize='optimal')

            #dC/dw = dz/dw * da/dz * dc/da
            #dC/dw = a * delta
            self.d_weights = (1/m) * np.einsum("pm,ml->pl", self.inputs.T, self.delta, optimize='optimal')
            self.d_weights = self.d_weights.T

            #dC/db = dz/db * da/dz * dc/da
            #dC/dw = delta
            self.d_bias = ((1/m) * np.einsum("mn->n", self.delta, optimize='optimal'))
            self.d_bias = self.d_bias.reshape(self.d_bias.shape[0], 1)
        
            #w(L)T * dL/dz(L)
            self.d_input = np.einsum("mi,in->mn", d_values, self.weights, optimize='optimal')

            return self.d_input

# TODO: thinking about update in each layer def update()?    
class SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.layers = None

    def update(self, layers):
        self.layers = layers
        print("hello")

        for layer in self.layers[:-1]:
            if (isinstance(layer, Conv2D)):
                layer.weights -= layer.d_weights * self.learning_rate
            if (isinstance(layer, Dense)):
                layer.weights -= layer.d_weights * self.learning_rate
                layer.bias -= layer.d_bias * self.learning_rate

        return self.layers

class Model:
    def __init__(self, layers=None, loss_function=None, optimizer=None):
        self.layers = layers
        self.loss = loss_function
        self.optimizer = optimizer
        self.loss_value = None
        self.epoch_loss_value = None
        self.save_loss_value = None

    def fit(self, x_train, y_train, batch_size, epochs, x_validation = None, y_validation = None):
        self.save_loss_value = np.zeros(epochs)
        for epoch in range(1):
            # 1. FORWARD PASS
            print(f"Epochs: {epoch + 1}/{epochs}")
            #for iterations in range(1, x_train.shape[0] // batch_size):
            for iterations in range(1, 2):
                print(f"iterations:{iterations}")
                x = x_train[(iterations - 1) * batch_size : iterations * batch_size]
                y = y_train[(iterations - 1) * batch_size : iterations * batch_size]

                # output.shape = (batch, inputs)
                # train_y = (batch, one-hot encoding)
                output = x
                train_y = y

                for layer in self.layers:
                    print(layer)
                    output = layer.forward(output)

                check_output = self.loss.forward(output,train_y)

                # 2. CALCULATE LOSS
                #recent layer is final layer
                if isinstance(self.loss, CategoricalCrossEntropy) and isinstance(self.layers[-1].activation, Softmax):
                    #Softmax combine with CrossEntropy: dC/da_(L-1)(i) = a_(L)(i) - y_i (value a_(L) is softmax layer, a_(L-1) is previous layer)
                    self.loss_value = output - train_y


                # 3. BACKWARD PASS
                # Start the chain with the loss gradient
                d_values = self.loss_value
                #print(f"self.loss_value: {self.loss_value}")

                for layer in reversed(self.layers):
                    if (isinstance(layer, Flatten)) or (isinstance(layer, MaxPooling)) or (not isinstance(layer.activation, Softmax)):
                        d_values = layer.backward(d_values)

                # 4. UPDATE WEIGHTS AND BIAS
                #self.optimizer.update(self.layers)
                self.layers = self.optimizer.update(self.layers)

            """
            if not x_validation.all():
                output_validation = x_validation
                # y_validation.shape = batch, one-hot

                for layer in self.layers:
                    output_validation = layer.forward(output_validation)
                
                output_validation = np.argmax(output_validation, axis = 1)

                correct_predict = 0
                for i in range(output_validation.shape[0]):
                    if y_validation[i][output_validation[i]] == 1: correct_predict += 1
                
                validation_accuracy = correct_predict / output_validation.shape[0]

            else:
                validation_accuracy = None

            self.epoch_loss_value = (1/batch_size) * np.sum(check_output)

            print(f"Epochs: {epoch + 1}/{epochs}    loss_value: {self.epoch_loss_value}     validation: {validation_accuracy}")
            
            self.save_loss_value[epoch] = (1/batch_size) * np.sum(check_output)
        
        return self.save_loss_value """

    def save(self, filename):
        model_data = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                model_data.append({
                    'weights' : layer.weights ,
                    'bias' : layer.bias 
                })

        with open(f"{filename}.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")

    def load(self, filename):
        with open(f"{filename}.pkl", 'rb') as f:
            model_data = pickle.load(f)

        data_index = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights = model_data[data_index]['weights']
                layer.bias = model_data[data_index]['bias']
                data_index += 1
        
        print(f"Model loaded from {filename}")
        

    #TODO: fix this
    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        
        return {'prediction': np.argmax(output), 
                'confidence': np.max(output)*100}

        




            

            




