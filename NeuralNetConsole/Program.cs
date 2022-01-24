// See https://aka.ms/new-console-template for more information
using MathNet.Numerics.LinearAlgebra;
using NeuralNetLibrary;
using System.Text.Json;

NeuralNet? neuralNet;
string s;
Layer layer;

neuralNet = new NeuralNet(new Random());
neuralNet.AddLayer(3);
neuralNet.AddLayer(2, ActivationTypes.SIGMOID);
//neuralNet.AddLayer(1);

var input = Vector<double>.Build.DenseOfArray(new double[] { 1, 1, -1 });

var output = neuralNet.Predict(input);
Console.WriteLine(output);


s = JsonSerializer.Serialize(neuralNet, new JsonSerializerOptions() { WriteIndented = true }); ;
Console.WriteLine(s);

Console.WriteLine("=====================================");

neuralNet = JsonSerializer.Deserialize<NeuralNet>(s);
s = JsonSerializer.Serialize(neuralNet, new JsonSerializerOptions() { WriteIndented = true }); ;
Console.WriteLine(s);


////layer = Layer.GetInputLayer(3);
//layer = Layer.GetDenseLayer(3,ActivationTypes.SIGMOID,2);
//s = JsonSerializer.Serialize(layer, new JsonSerializerOptions() { WriteIndented = false }); ;
//Console.WriteLine(s);

//layer = JsonSerializer.Deserialize<Layer>(s) ?? new Layer();
//s = JsonSerializer.Serialize(layer, new JsonSerializerOptions() { WriteIndented = false }); ;
//Console.WriteLine(s);