// See https://aka.ms/new-console-template for more information
using MathNet.Numerics.LinearAlgebra;
using NeuralNetLibrary;
using System.Text;
using System.Text.Json;

//Random random = new Random();
//NeuralNet? neuralNet;
//string s;
//Layer layer;

//https://habr.com/ru/post/313216/
var lengths = new int[] { 2, 2, 1 };
Matrix<double>[] weightsArray = new Matrix<double>[]
{
                Matrix<double>.Build.DenseOfArray(new double[,] { { 0.45, -0.12 },{ 0.78, 0.13 } }),
                Matrix<double>.Build.DenseOfArray(new double[,] { { 1.5, -2.3} })
};

NeuralNet neuralNet = NeuralNet.Build(lengths, weightsArray, ActivationTypes.Sigmoid);

var input = Vector<double>.Build.DenseOfArray(new double[] { 1.0, 0.0 });
var outputToBe = Vector<double>.Build.DenseOfArray(new double[] { 1.0 });

neuralNet.Train(input, outputToBe);
NeuralNet.SaveJson(@"D:\Temp\net.json", neuralNet);

StringBuilder sb = new();
sb.Append($" Input={input[0]} Output={neuralNet.Layers.Last().OutputVector[0]}");
sb.AppendLine();
sb.Append($" ErrorMSE={neuralNet.ErrorMSE(outputToBe) * 100.0:0.00}%");
sb.AppendLine();
Console.WriteLine(sb.ToString());
Console.WriteLine();

//int inputVectorLenght = 1;
//neuralNet = new NeuralNet(random);
//neuralNet.AddLayer(lenght: inputVectorLenght);
////neuralNet.AddLayer(lenght: 2, activationType: ActivationTypes.Sigmoid);
//neuralNet.AddLayer(lenght: 1, activationType: ActivationTypes.LeakyReLU);

//var input = Vector<double>.Build.DenseOfArray(new double[] { inputVectorLenght });
//var outputToBe = Vector<double>.Build.DenseOfArray(new double[] { inputVectorLenght });
//StringBuilder sb = new();
//int maxTrain = 1000;
//for (int i = 0; i < maxTrain; i++)
//{
//    input[0] = (i == maxTrain - 1) ? 0.99 : (i == maxTrain - 2) ? 0.01 : random.NextDouble();
//    outputToBe[0] = input[0];
//    neuralNet.Train(input, outputToBe, alpha: 0.3);
//    var output = neuralNet.Forward(input);

//    sb.Clear();

//    sb.Append($"TrainNo={i}: Weight={neuralNet.Layers.Last().WeightsMatrixByRows[0][0]}");
//    sb.Append($" Input={input[0]} Output={output[0]}");
//    sb.AppendLine();
//    sb.Append($" AbsError={neuralNet.AbsError(outputToBe):0.0000}");
//    sb.Append($" QuadraticError={neuralNet.QuadraticError(outputToBe) * 100.0:0.00}%");
//    sb.AppendLine();
//    Console.WriteLine(sb.ToString());

//}

//s = JsonSerializer.Serialize(neuralNet, new JsonSerializerOptions() { WriteIndented = true }); ;
//Console.WriteLine(s);

//Console.WriteLine("=================================================");

//neuralNet = JsonSerializer.Deserialize<NeuralNet>(s);
//s = JsonSerializer.Serialize(neuralNet, new JsonSerializerOptions() { WriteIndented = true }); ;
//Console.WriteLine(s);


////layer = Layer.GetInputLayer(3);
//layer = Layer.GetDenseLayer(3,ActivationTypes.SIGMOID,2);
//s = JsonSerializer.Serialize(layer, new JsonSerializerOptions() { WriteIndented = false }); ;
//Console.WriteLine(s);

//layer = JsonSerializer.Deserialize<Layer>(s) ?? new Layer();
//s = JsonSerializer.Serialize(layer, new JsonSerializerOptions() { WriteIndented = false }); ;
//Console.WriteLine(s);