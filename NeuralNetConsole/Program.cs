// See https://aka.ms/new-console-template for more information
using MathNet.Numerics.LinearAlgebra;
using NeuralNetLibrary;
using System.Text.Json;

NeuralNet? neuralNet;
string s;
Layer layer;

Random random = new Random();
neuralNet = new NeuralNet(random);
neuralNet.AddLayer(1);
neuralNet.AddLayer(2, ActivationTypes.SIGMOID);
neuralNet.AddLayer(1);

var input = Vector<double>.Build.DenseOfArray(new double[] { 1 });
var outputToBe = Vector<double>.Build.DenseOfArray(new double[] { 1 });

for (int i = 0; i < 100; i++)
{
    input[0] = random.NextDouble();
    outputToBe[0] = input[0];

    var output = neuralNet.Predict(input);
    Console.WriteLine($"{i:00} {outputToBe[0] - output[0]}");
}




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