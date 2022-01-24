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

        private static NeuralNet PrepareNet(int[] lengths, Matrix<double>[] weightsArray,
            ActivationTypes activationTypes = ActivationTypes.Identity)
        {
            NeuralNet neuralNet = new();
            for (int i = 0; i < lengths.Length; i++)
            {
                if (i == 0)
                    neuralNet.AddLayer(lengths[i]);
                else
                    neuralNet.AddLayer(lengths[i], activationTypes, weightsArray[i - 1]);
            }
            JsonCheck(neuralNet);
            return neuralNet;
        }

        private static Vector<double> PredictOneValue(int[] lengths,
            Matrix<double>[] weightsArray, Vector<double> input,
            ActivationTypes activationTypes = ActivationTypes.Identity)
        {

            NeuralNet neuralNet = PrepareNet(lengths, weightsArray, activationTypes);
            JsonCheck(neuralNet);
            return neuralNet.Predict(input);
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

        private static string SerializeIndented(object o)
        {
            return JsonSerializer.Serialize(o, new JsonSerializerOptions() { WriteIndented = true });
        }

        private static void JsonCheck(NeuralNet neuralNet)
        {
            string s = SerializeIndented(neuralNet); 
            NeuralNet? neuralNet2 = JsonSerializer.Deserialize<NeuralNet?>(s);
            if (neuralNet2 == null) Assert.Fail("JsonSerializer.Deserialize failed", s);
            string s2 = SerializeIndented(neuralNet2);
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

            NeuralNet neuralNet = PrepareNet(lengths, weightsArray, ActivationTypes.Identity);

            var input = Vector<double>.Build.DenseOfArray(new double[] { 1.0 });
            var outputToBe = Vector<double>.Build.DenseOfArray(new double[] { 1.0 });

            neuralNet.Train(input, outputToBe);

            string s = JsonSerializer.Serialize(neuralNet, new JsonSerializerOptions() { WriteIndented = true }); ;
            File.WriteAllText(@"D:\Temp\net.json", s);

            JsonCheck(neuralNet);
            Assert.AreEqual(0.45, neuralNet.Layers.Last().WeightsMatrixByRows[0][0], DOUBLE_DELTA);
        }

    }
}