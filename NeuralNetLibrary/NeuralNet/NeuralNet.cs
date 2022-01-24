﻿using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    [Serializable]
    public class NeuralNet
    {
        private Random _random;
        private LinkedList<Layer> _layers = new();
        public int LayersCount => _layers.Count;

        public Vector<double> _input;
        public int InputVectorSize { get; set; } = -1;

        public double[] InputVector
        {
            get { return _input.ToArray(); }
            set { _input = Vector<double>.Build.DenseOfArray(value); }
        }

        public LinkedList<Layer> Layers
        {
            get { return _layers; }
            set { _layers = value; }
        }

        public NeuralNet() => _random = new Random();

        public NeuralNet(Random random) => _random = random;


        public void AddLayer(int lenght, ActivationTypes activationType = ActivationTypes.Sigmoid, Matrix<double>? weights = null)
        {
            if (_layers.Count == 0 && InputVectorSize < 0)
            {
                _input = Vector<double>.Build.Dense(lenght);
                InputVectorSize = lenght;
                return;
            }

            int prevLayerLenght = _layers.Count == 0 ? InputVectorSize : _layers.Last().Lenght;
            _layers.AddLast(new Layer(prevLayerLenght, lenght, activationType, _random, weights));
        }


        public void Clear()
        {
            _input.Clear();
            foreach (var layer in _layers)
                layer.Clear();
        }



        public Vector<double> Predict(Vector<double> input)
        {
            if (_input is null) throw new ArgumentNullException($"Net has no Layers");
            if (_input.Count != input.Count) throw new ArgumentException($"Predict vector length {input.Count},expected {_input.Count}");

            Clear();

            _input = Vector<double>.Build.DenseOfVector(input);
            for (int i = 0; i < _layers.Count; i++)
            {
                Layer layer = _layers.ElementAt(i);

                Vector<double> vect;
                if (i == 0)
                    vect = _input;
                else
                    vect = _layers.ElementAt(i - 1)._input;
                layer._input = layer._weights.Multiply(vect);
                layer._output = layer.GetActivationFunction().Activate(layer._input);

            }

            return Vector<double>.Build.DenseOfVector(_layers.Last()._output);

        }


        public double QuadraticError(Vector<double> outputToBe)
        {
            return Layers.Last().QuadraticError(outputToBe);
        }

        public double AbsError(Vector<double> outputToBe)
        {
            return Layers.Last().AbsError(outputToBe);
        }


        public void TrainEpoch(Vector<double>[] arrayOutput, Vector<double>[] arrayOutputToBe, double alpha = 0.1)
        {
            if (arrayOutput == null) throw new ArgumentNullException(nameof(arrayOutput));
            if (arrayOutputToBe == null) throw new ArgumentNullException(nameof(arrayOutputToBe));
            if (arrayOutput.Length != arrayOutputToBe.Length)
                throw new ArgumentException($"TrainEpoch in {arrayOutput.Length}/ToBe {arrayOutputToBe.Length} vectors are not equal");
            if (arrayOutput.Length == 0) throw new ArgumentException("Epoch: nothing to train");

            for (int i = 0; i < arrayOutput.Length; i++)
            {
                Train(arrayOutput[i], arrayOutputToBe[i], alpha);
            }
        }


        public void Train(Vector<double> input, Vector<double> outputToBe, double alpha = 0.1)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (outputToBe == null) throw new ArgumentNullException(nameof(outputToBe));
            if (input.Count != outputToBe.Count)
                throw new ArgumentException($"Lenght of in {input.Count}/ToBe {outputToBe.Count} vectors are not equal");


            Predict(input);
            _layers.Last()._error = outputToBe - _layers.Last()._output;
            Console.WriteLine();
            for (int i = _layers.Count - 2; i >= 0; i--)
            {
                Layer layer = _layers.ElementAt(i);
                Layer layerNext = _layers.ElementAt(i + 1);
                layer._error = layerNext._weights.TransposeThisAndMultiply(layerNext._error);
            }

            for (int i = 0; i < _layers.Count; i++)
            {
                Vector<double> outputPrev;
                if (i == 0)
                    outputPrev = _input.Clone();
                else
                    outputPrev = _layers.ElementAt(i - 1)._output.Clone();

                _layers.ElementAt(i).UpdateWeights(outputPrev, alpha);
            }

        }
    }
}