using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron  {

    public int inputCount;
    public double bias;
    public double output;
    public double errorGradient;
    public List<double> weights = new List<double>();
    public List<double> inputs = new List<double>();

    public Neuron(int inputCount) {
        this.inputCount = inputCount;
        bias = Random.Range(-1f, 1f);
        InitializeRandomWeights();
    }

    private void InitializeRandomWeights() {
        for (int i = 0; i < inputCount; i++) {
            weights.Add(UnityEngine.Random.Range(-1f, 1f));
        }
    }
    
}
