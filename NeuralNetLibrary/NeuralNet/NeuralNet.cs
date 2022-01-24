using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    [Serializable]
    public class NeuralNet
    {
        private Random _random;
        private LinkedList<Layer> _layers = new();
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


        public void AddLayer(int size, ActivationTypes activationType = ActivationTypes.SIGMOID)
        {
            int prevLayerSize = -1;
            if (_layers.Count == 0 && InputVectorSize <= 0)
            {
                _input = Vector<double>.Build.Dense(size);
                InputVectorSize = size;
                return;
            }

            prevLayerSize = _layers.Count == 0 ? InputVectorSize : _layers.Last().Lenght;
            _layers.AddLast(Layer.GetDenseLayer(size, activationType, prevLayerSize, _random));
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

            _input = Vector<double>.Build.DenseOfVector(input);
            for (int i = 0; i < _layers.Count; i++)
            {
                Layer layer = _layers.ElementAt(i);
                //Matrix<double> m = Matrix<double>.Build.DenseOfColumnVectors(_input);

                Vector<double> vect;
                if (i == 0)
                    vect = _input;
                else
                    vect = _layers.ElementAt(i - 1)._input;
                layer._input = layer._weights.Multiply(vect);

                IActivation activation;
                switch (layer.ActivationType)
                {
                    case ActivationTypes.NO:
                        activation = new NoActivation();
                        break;
                    case ActivationTypes.SIGMOID:
                        activation = new SigmoidActivation();
                        break;
                    default:
                        throw new ArgumentException($"Unknown ActivationType: [{layer.ActivationType}]");
                        //break;
                }
                layer._output = Vector<double>.Build.Dense(layer._input.Count, (k) => activation.Activate(layer._input[k]));
                //for (int k = 0; k < layer._input.Count; k++)
                //    layer._output[k] = activation.Activate(layer._input[k]);
            }

            return Vector<double>.Build.DenseOfVector(_layers.Last()._output);

        }


        public void Train() { }

    }
}