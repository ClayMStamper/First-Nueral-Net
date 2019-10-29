

using System.Collections.Generic;

public class Layer {
    public List<Neuron> neurons = new List<Neuron>();

    public Layer(int neuronCount, int inputCount) {
        for (int i = 0; i < neuronCount; i++) {
            neurons.Add(new Neuron(inputCount));
        }   
    }
    
}