using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetLibrary;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace NeuralNetLibrary.Tests
{
    [TestClass()]
    public class NeuralNetTests
    {
        double DOUBLE_DELTA = 1e-5;



        private static Vector<double> PredictOneValue(int[] lengths,
            Matrix<double>[] weightsArray, Vector<double> input,
            ActivationTypes activationTypes = ActivationTypes.Identity)
        {

            NeuralNet neuralNet = NeuralNet.Build(lengths, weightsArray, activationTypes);
            JsonCheck(neuralNet);
            return neuralNet.Forward(input);
        }


        [TestMethod()]
        public void PredictLastDouble01()
        {
            var lengths = new int[] { 2, 1 };
            var input = Vector<double>.Build.DenseOfArray(new double[] { 1, 2 });
            Matrix<double>[] weightsArray = new Matrix<double>[]
            {
                Matrix<double>.Build.DenseOfArray(new double[1,2] { { 0.5, 0.5} })
            };

            double d = PredictOneValue(lengths, weightsArray, input)[0];
            Assert.AreEqual(1.5, d, DOUBLE_DELTA);
        }

        [TestMethod()]
        public void PredictLastDouble02()
        {
            var lengths = new int[] { 1, 1 };
            var input = Vector<double>.Build.DenseOfArray(new double[] { 0.6 });
            Matrix<double>[] weightsArray = new Matrix<double>[]
            {
                Matrix<double>.Build.DenseOfArray(new double[,] { { 0.5} })
            };

            double d = PredictOneValue(lengths, weightsArray, input)[0];
            Assert.AreEqual(0.3, d, DOUBLE_DELTA);
        }

        [TestMethod()]
        public void PredictLastDouble03()
        {
            var lengths = new int[] { 2, 2, 1 };
            var input = Vector<double>.Build.DenseOfArray(new double[] { 1, 1 });
            Matrix<double>[] weightsArray = new Matrix<double>[]
            {
                Matrix<double>.Build.DenseOfArray(new double[2,2] { { 0.5, 0.5 },{ 0.5, 0.5 } }),
                Matrix<double>.Build.DenseOfArray(new double[1,2] { { 0.5, 0.5} })
            };

            double d = PredictOneValue(lengths, weightsArray, input)[0];
            Assert.AreEqual(1.0, d, DOUBLE_DELTA);
        }



        private static void JsonCheck(NeuralNet neuralNet)
        {
            string s = NeuralNet.SerializeJsonIndented(neuralNet); 
            NeuralNet? neuralNet2 = JsonSerializer.Deserialize<NeuralNet?>(s);
            if (neuralNet2 == null) Assert.Fail("JsonSerializer.Deserialize failed", s);
            string s2 = NeuralNet.SerializeJsonIndented(neuralNet2);
            Assert.AreEqual(s, s2);
        }


        [TestMethod()]
        public void TrainLastDouble02()
        {
            var lengths = new int[] { 1, 1 };
            Matrix<double>[] weightsArray = new Matrix<double>[]
            {
                Matrix<double>.Build.DenseOfArray(new double[,] { { 0.5} })
            };

            NeuralNet neuralNet = NeuralNet.Build(lengths, weightsArray, ActivationTypes.Identity);

            var input = Vector<double>.Build.DenseOfArray(new double[] { 1.0 });
            var outputToBe = Vector<double>.Build.DenseOfArray(new double[] { 1.0 });

            neuralNet.Train(input, outputToBe);

            JsonCheck(neuralNet);
            Assert.AreEqual(0.45, neuralNet.Layers.Last().WeightsMatrixByRows[0][0], DOUBLE_DELTA);
        }


        public void TrainLastDouble03()
        {
            //https://habr.com/ru/post/313216/
            //var lengths = new int[] { 2, 2, 1 };
            //Matrix<double>[] weightsArray = new Matrix<double>[]
            //{
            //    Matrix<double>.Build.DenseOfArray(new double[,] { { 0.45, -0.12 },{ 0.78, 0.13 } }),
            //    Matrix<double>.Build.DenseOfArray(new double[,] { { 1.5, -2.3} })
            //};

            //NeuralNet neuralNet = NeuralNet.Build(lengths, weightsArray, ActivationTypes.Sigmoid);

            //var input = Vector<double>.Build.DenseOfArray(new double[] { 1.0, 0.0 });
            //var outputToBe = Vector<double>.Build.DenseOfArray(new double[] { 1.0 });

            //neuralNet.Train(input, outputToBe);
            //neuralNet.Forward(input);
            //NeuralNet.SaveJson(@"D:\Temp\net.json", neuralNet);

            //StringBuilder sb = new();
            //sb.Append($" Input={input[0]} Output={neuralNet.Layers.Last().OutputVector[0]}");
            //sb.AppendLine();
            //sb.Append($" ErrorMSE={neuralNet.ErrorMSE(outputToBe) * 100.0:0.00}%");
            //sb.AppendLine();
            //Console.WriteLine(sb.ToString());
            //Console.WriteLine();
        }


    }
}