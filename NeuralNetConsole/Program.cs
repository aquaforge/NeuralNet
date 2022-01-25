using MathNet.Numerics.LinearAlgebra;
using NeuralNetLibrary;
using System.Text;

Random random = new Random();

//function y=x range[0;1]
NeuralNet neuralNet = new(random);
neuralNet.AddLayer(1);
neuralNet.AddLayer(2, ActivationTypes.Sigmoid);
neuralNet.AddLayer(1, ActivationTypes.Sigmoid);


Queue<double> errorsQueue = new();
for (int i = 0; i < 100_000; i++)
{
    double d = random.NextDouble();
    var input = Vector<double>.Build.DenseOfArray(new double[] { d });
    var outputToBe = Vector<double>.Build.DenseOfArray(new double[] { d }); 

    neuralNet.Train(input, outputToBe);

    double errorMSE = neuralNet.ErrorMSE(outputToBe);
    errorsQueue.Enqueue(errorMSE);

    StringBuilder sb = new();
    sb.AppendLine($"TrainNo={i}  ErrorMSE={errorMSE * 100.0:0.00000}%");
    sb.AppendLine($" Input={input[0]} Output={neuralNet.Layers.Last().OutputVector[0]}");
    sb.AppendLine();
    Console.WriteLine(sb.ToString());
    if (errorsQueue.Count >10 && errorsQueue.Average() < 0.001) break;
    if (errorsQueue.Count > 50) errorsQueue.Dequeue();
}
//neuralNet.ClearAllButWeight();
//NeuralNet.SaveJson(@"D:\Temp\net.json", neuralNet);
//Console.WriteLine(NeuralNet.SerializeJsonIndented(neuralNet));

