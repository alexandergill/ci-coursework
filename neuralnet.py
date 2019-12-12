# definition of NN class
class NeuralNet:

    # initiation
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # set number of nodes for each layer
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        # initialise weights matrices to random numbers
        self.inputToHiddenWeights  = numpy.random.normal(0.0,pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.hiddenToOutputWeights = numpy.random.normal(0.0,pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))
        