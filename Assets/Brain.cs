using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour {
    
    private ANN model;
    private double sumSquareError = 0; //how closely the model fits the data

    [SerializeField] 
    private int epochs = 1000;

    private void Start() {
        
        model = new ANN(2,1,1,2,0.8d);
        List<double> result;

        for (int i = 0; i < epochs; i++) {
            sumSquareError = 0;
            //input 1, input 2, expected output for an XOR
            result = Train(1, 1, 0);
            sumSquareError += Mathf.Pow((float) result[0], 2);
            result = Train(1, 0, 1);
            sumSquareError += Mathf.Pow((float) result[0] - 1, 2);
            result = Train(0, 1, 1);
            sumSquareError += Mathf.Pow((float) result[0] - 1, 2);
            result = Train(0, 0, 0);
            sumSquareError += Mathf.Pow((float) result[0], 2);
        }
        
        Debug.Log("Sum square error: " + sumSquareError);
        
        //predict not spereated from "Train" but this is effectively predicting
        result = Train(1, 1, 0);
        Debug.Log("1 1 = " + result);
        result = Train(1, 0, 1);
        Debug.Log("1 0 = " + result);
        result = Train(0, 1, 1);
        Debug.Log("0 1 = " + result);
        result = Train(0, 0, 0);
        Debug.Log("0 0 = " + result);

    }

    private List<double> Train(int i1, int i2, int o) {
        List<double> inputs = new List<double>();
        List<double>  outputs = new List<double>();
        inputs.Add(i1);
        inputs.Add(i2);
        outputs.Add(o);
        return (model.Go(inputs, outputs));
    }
    
}
