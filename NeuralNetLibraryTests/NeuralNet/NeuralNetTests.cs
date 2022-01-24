using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetLibrary;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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
            return neuralNet;
        }

        private Vector<double> PredictOneValue(int[] lengths,
            Matrix<double>[] weightsArray, Vector<double> input,
            ActivationTypes activationTypes = ActivationTypes.Identity)
        {

            NeuralNet neuralNet = PrepareNet(lengths, weightsArray, activationTypes);
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

            Assert.AreEqual(0.45, neuralNet.Layers.Last().WeightsMatrixByRows[0][0],DOUBLE_DELTA);
        }

    }
}