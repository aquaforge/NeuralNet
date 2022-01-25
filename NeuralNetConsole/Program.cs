// See https://aka.ms/new-console-template for more information
using MathNet.Numerics.LinearAlgebra;
using NeuralNetLibrary;
using System.Text;
using System.Text.Json;

Random random = new Random();
//NeuralNet? neuralNet;
//string s;
//Layer layer;

//var lengths = new int[] { 2, 2, 1 };
//Matrix<double>[] weightsArray = new Matrix<double>[]
//{
//                Matrix<double>.Build.DenseOfArray(new double[,] { { 0.45, -0.12 },{ 0.78, 0.13 } }),
//                Matrix<double>.Build.DenseOfArray(new double[,] { { 1.5, -2.3} })
//};

//NeuralNet neuralNet = NeuralNet.Build(lengths, weightsArray, ActivationTypes.Sigmoid);

//var input = Vector<double>.Build.DenseOfArray(new double[] { 1.0, 0.0 });
//var outputToBe = Vector<double>.Build.DenseOfArray(new double[] { 1.0 });
//neuralNet.Train(input, outputToBe);
//NeuralNet.SaveJson(@"D:\Temp\net.json", neuralNet);
//Console.WriteLine(NeuralNet.SerializeJsonIndented(neuralNet));

NeuralNet neuralNet = new(random);
neuralNet.AddLayer(1);
neuralNet.AddLayer(1, ActivationTypes.LeakyReLU);
for (int i = 0; i < 200; i++)
{
    double d = random.NextDouble();
    var input = Vector<double>.Build.DenseOfArray(new double[] { d });
    var outputToBe = Vector<double>.Build.DenseOfArray(new double[] { d });

    neuralNet.Train(input, outputToBe);

    StringBuilder sb = new();
    sb.AppendLine($"TrainNo={i}: Weight={neuralNet.Layers.Last().WeightsMatrixByRows[0][0]}");
    sb.AppendLine($" Input={input[0]} Output={neuralNet.Layers.Last().OutputVector[0]}");
    sb.AppendLine($" ErrorMSE={neuralNet.ErrorMSE(outputToBe) * 100.0:0.00}%");
    sb.AppendLine();
    Console.WriteLine(sb.ToString());
}
NeuralNet.SaveJson(@"D:\Temp\net.json", neuralNet);
Console.WriteLine(NeuralNet.SerializeJsonIndented(neuralNet));

