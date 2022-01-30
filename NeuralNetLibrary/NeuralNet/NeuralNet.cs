using MathNet.Numerics.LinearAlgebra;
using System.Text.Json;

namespace NeuralNetLibrary
{
    [Serializable]
    public class NeuralNet : IEquatable<NeuralNet>, ICloneable
    {
        double DOUBLE_DELTA = 1e-9;

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


        /// <summary>
        /// Direct ref link to layer weights matrix
        /// </summary>
        /// <param name="layerNo"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public Matrix<double> WeightsList(int layerNo)
        {
            if (layerNo < 0 || layerNo>=_layers.Count) throw new ArgumentOutOfRangeException(nameof(layerNo));
            return _layers.ElementAt(layerNo)._weights;
        }


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


        public void ClearAllButWeight()
        {
            _input.Clear();
            foreach (var layer in _layers)
                layer.ClearAllButWeight();
        }
        public void ClearPrediction()
        {
            _input.Clear();
            foreach (var layer in _layers)
                layer.ClearPrediction();
        }

        public Vector<double> Forward(Vector<double> input)
        {
            if (_input is null) throw new ArgumentNullException($"Net has no Layers");
            if (_input.Count != input.Count) throw new ArgumentException($"Input vector length {input.Count}, expected {_input.Count}");

            ClearPrediction();

            _input = Vector<double>.Build.DenseOfVector(input);
            for (int i = 0; i < _layers.Count; i++)
            {
                Layer layer = _layers.ElementAt(i);

                Vector<double> vect;
                if (i == 0)
                    vect = _input;
                else
                    vect = _layers.ElementAt(i - 1)._output;
                layer._input = layer._weights.Multiply(vect);
                layer._output = layer.GetActivationFunction().Activate(layer._input);

            }

            return Vector<double>.Build.DenseOfVector(_layers.Last()._output);
        }

        public double ErrorMSE(Vector<double> outputToBe) => Layers.Last().ErrorMSE(outputToBe);

        public void TrainEpoch(Vector<double>[] arrayOutput, Vector<double>[] arrayOutputToBe, double alpha = 0.1)
        {
            if (arrayOutput == null) throw new ArgumentNullException(nameof(arrayOutput));
            if (arrayOutputToBe == null) throw new ArgumentNullException(nameof(arrayOutputToBe));
            if (arrayOutput.Length != arrayOutputToBe.Length)
                throw new ArgumentException($"TrainEpoch in {arrayOutput.Length}/ToBe {arrayOutputToBe.Length} vectors are not equal");
            if (arrayOutput.Length == 0) throw new ArgumentException("Epoch: nothing to train");

            for (int i = 0; i < arrayOutput.Length; i++)
                Train(arrayOutput[i], arrayOutputToBe[i], alpha);
        }


        public void Train(Vector<double> input, Vector<double> outputToBe, double learningVelocityEpsilon = 0.7, double momentAlpha = 0.3)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (outputToBe == null) throw new ArgumentNullException(nameof(outputToBe));
            if (_layers.Last()._output.Count != outputToBe.Count)
                throw new ArgumentException($"Lenght of out {_layers.Last()._output.Count}/ToBe {outputToBe.Count} vectors are not equal");

            #region Forward
            Forward(input);
            #endregion

            #region Backward For Errors
            Layer layer;
            layer = _layers.Last();
            layer._delta = (outputToBe - layer._output)
                .PointwiseMultiply(layer.GetActivationFunction().Deactivate(layer._input, layer._output)); ;

            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                layer = _layers.ElementAt(i);
                Vector<double> err;
                if (i == _layers.Count - 1)
                    err = outputToBe - layer._output;
                else
                {
                    Layer layerNext = _layers.ElementAt(i + 1);
                    err = layerNext._weights.TransposeThisAndMultiply(layerNext._delta);
                }
                layer._delta = err.PointwiseMultiply(layer.GetActivationFunction().Deactivate(layer._input, layer._output));
            }
            #endregion

            #region UpdateWeights
            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                layer = _layers.ElementAt(i);
                Vector<double> outputPrev;
                if (i == 0)
                    outputPrev = _input;
                else
                    outputPrev = _layers.ElementAt(i - 1)._output;

                layer._deltaWeightsPrev = learningVelocityEpsilon * layer._delta.ToColumnMatrix() * outputPrev.ToRowMatrix();
                layer._deltaWeightsPrev += momentAlpha * layer._deltaWeightsPrev;

                layer._weights += layer._deltaWeightsPrev;
            }
            #endregion
        }

        public static void SaveJson(string path, object o, bool indented = true)
            => File.WriteAllText(path, SerializeJsonIndented(o, indented));

        public static string SerializeJsonIndented(object o, bool indented = true)
            => JsonSerializer.Serialize(o, new JsonSerializerOptions() { WriteIndented = indented });

        public static NeuralNet Build(int[] lengths, Matrix<double>[] weightsArray,
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

        override public bool Equals(object? obj)
        {
            if (obj is not NeuralNet other) return false;
            return Equals(other);
        }

        public bool Equals(NeuralNet? other)
        {
            if (other == null) return false;

            if (_input.Count != other._input.Count) return false;
            if (_layers.Count != other._layers.Count) return false;
            for (int k = 0; k < _layers.Count; k++)
            {
                if (_layers.ElementAt(k).ActivationType != other._layers.ElementAt(k).ActivationType) return false;
                Matrix<double> w1 = _layers.ElementAt(k)._weights;
                Matrix<double> w2 = other._layers.ElementAt(k)._weights;
                if (w1 != null || w2 != null)
                {
                    if (w1 == null || w2 == null) return false;
                    if (w1.RowCount != w2.RowCount) return false;
                    if (w1.ColumnCount != w2.ColumnCount) return false;
                    for (int i = 0; i < w1.RowCount; i++)
                        for (int j = 0; j < w1.ColumnCount; j++)
                            if (Math.Abs(w1[i, j] - w2[i, j]) > DOUBLE_DELTA) return false;
                }
            }
            return true;
        }



        /// <summary>
        /// provides deep copy
        /// </summary>
        /// <returns></returns>
        public NeuralNet Copy()
        {
            NeuralNet n = new NeuralNet(_random);
            n.AddLayer(_input.Count);
            for (int i = 0; i < _layers.Count; i++)
            {
                Layer layer = _layers.ElementAt(i);
                n.AddLayer(layer.Lenght, layer.ActivationType, layer._weights);
            }
            return n;

        }
        public object Clone() => (object)this.Copy();

        public override string ToString() => JsonSerializer.Serialize(this);

    }
}