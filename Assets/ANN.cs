using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/*
 * Construct all layers and neurons
 * Trains the network
 */

public class ANN  {

    public int inputCount;
    public int outputCount;
    public int hiddenLayerCount;
    public int neuronCount;
    public double alpha; //used as weight - determines how fast the network learns
    List<Layer> layers = new List<Layer>();


    public ANN(int inputCount, int outputCount, int hiddenLayerCount, int neuronCount, double alpha) {
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.hiddenLayerCount = hiddenLayerCount;
        this.neuronCount = neuronCount;
        this.alpha = alpha;

        ConstructInputLayer();
        ConstructHiddenLayers();
        ConstructOutputLayer();
        
    }

    private void ConstructInputLayer() {
        Layer input = new Layer(neuronCount, inputCount);
        layers.Add(input);
    }

    private void ConstructHiddenLayers() {
        for (int i = 0; i < hiddenLayerCount; i++) {
            Layer hidden = new Layer(neuronCount, neuronCount);
            layers.Add(hidden);
        }
    }
    
    private void ConstructOutputLayer() {
        //output takes in hidden layers if they exist 
        int inputs = hiddenLayerCount > 0 ? hiddenLayerCount : inputCount;
        Layer output = new Layer(outputCount, inputs);
        layers.Add(output);
    }

    public List<double> Go(List<double> inputValues, List<double> desiredOutput) {

        List<double> inputs = inputValues;
        List<double> outputs = new List<double>();

        if (inputValues.Count != inputCount) {
            Debug.LogError("Error: Number of inputs must be: " + inputCount);
            return outputs;
        }

        for (int i = 0; i < hiddenLayerCount; i++) {
            
            if (i > 0)
                inputs = outputs;
            outputs.Clear();
            
            for (int j = 0; j < neuronCount; j++) {

                double neuronValue = 0;
                layers[i].neurons[j].inputs.Clear();
                
                for (int k = 0; k < inputCount; k++) {
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    neuronValue += layers[i].neurons[j].weights[k] * inputs[k];
                }

                neuronValue -= layers[i].neurons[j].bias;
                layers[i].neurons[j].output = Activate(neuronValue);
                outputs.Add(layers[i].neurons[j].output);

            }
        }

        UpdateWeights(outputs, desiredOutput);
        return outputs;
        
    }

    private void UpdateWeights(List<double> outputs, List<double> desiredOutput) {

        double error;

        for (int i = hiddenLayerCount; i >= 0; i--) {
            for (int j = 0; j < layers[i].neurons.Count; j++) {

                if (i >= hiddenLayerCount) {
                    error = desiredOutput[j] - outputs[j];
                    double errorGradient = outputs[j] * (1 - outputs[j]) * error; //delta rule
                    layers[i].neurons[j].errorGradient = errorGradient;
                } else {
                    double myOuput = layers[i].neurons[j].output;
                    double errorGradient = myOuput * (1 - myOuput);
                    layers[i].neurons[j].errorGradient = errorGradient;
                    double gradientSum = 0d;
                    for (int k = 0; k < layers[i+1].neurons.Count; k++) {
                        gradientSum += layers[i + 1].neurons[k].errorGradient * layers[i + 1].neurons[k].weights[j];
                    }
                    layers[i].neurons[j].errorGradient *= gradientSum;
                }

                for (int k = 0; k < layers[i].neurons[j].inputCount; k++) {
                    if (i == hiddenLayerCount) {
                        error = desiredOutput[j] - outputs[j];
                        layers[i].neurons[j].weights[k] *= alpha * layers[i].neurons[j].inputs[k] * error;
                    } else {
                        double gradient = layers[i].neurons[j].errorGradient;
                        layers[i].neurons[j].weights[k] *= alpha * layers[i].neurons[j].inputs[k] * gradient;
                    }

                    layers[i].neurons[j].bias += -alpha * layers[i].neurons[j].errorGradient;
                }
            }
        }
    }

    private double Activate(double neuronValue) {
        return Sigmoid(neuronValue);
    }

    double Step(double neuronVal) { // take binary step
        return neuronVal < 0 ? 0 : 1;
    }

    double Sigmoid(double neuronVal) { //take logistic softstep
        double k = (double) System.Math.Exp(neuronVal);
        return k / (1 + k);
    }

}
