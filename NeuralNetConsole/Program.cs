// See https://aka.ms/new-console-template for more information
using MathNet.Numerics.LinearAlgebra;
using NeuralNetLibrary;
using System.Text.Json;

Random random = new Random();
NeuralNet? neuralNet;
string s;
//Layer layer;


int inputVectorSize = 1;
neuralNet = new NeuralNet(random);
neuralNet.AddLayer(inputVectorSize);
neuralNet.AddLayer(1, ActivationTypes.LeakyReLU);

var input = Vector<double>.Build.DenseOfArray(new double[] { inputVectorSize });
var outputToBe = Vector<double>.Build.DenseOfArray(new double[] { inputVectorSize });
for (int i = 0; i < 100; i++)
{
    input[0] = random.NextDouble();
    outputToBe[0] = input[0];

    neuralNet.Train(input, outputToBe, 0.1);
    Console.WriteLine($"{i:00} Weight={neuralNet.Layers.Last().WeightsMatrixByRows[0][0]} Error={outputToBe[0] - neuralNet.Predict(input)}");
}




s = JsonSerializer.Serialize(neuralNet, new JsonSerializerOptions() { WriteIndented = true }); ;
Console.WriteLine(s);

Console.WriteLine("=================================================");

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