using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    [Serializable]
    public class Layer
    {
        Random _random;

        public Vector<double> _input;
        public Vector<double> _output;
        public Vector<double> _error;
        internal Matrix<double> _weights;

        public ActivationTypes ActivationType { get; set; }
        public int Lenght => _input.Count;

        public double[][]? WeightsMatrixByRows
        {
            get { return _weights.ToRowArrays(); }
            set
            {
                if (value == null) throw new ArgumentNullException(nameof(value));
                _weights = Matrix<double>.Build.DenseOfRowArrays(value);
            }
        }

        public double[] InputVector
        {
            get { return _input.ToArray(); }
            set { _input = Vector<double>.Build.DenseOfArray(value); }
        }

        public double[] OutputVector
        {
            get { return _output.ToArray(); }
            set { _output = Vector<double>.Build.DenseOfArray(value); }
        }

        public double[] ErrorVector
        {
            get { return _error.ToArray(); }
            set { _error = Vector<double>.Build.DenseOfArray(value); }
        }



        public static Layer GetInputLayer(int length, Random? random = null) => new(length, ActivationTypes.NO, random);
        public static Layer GetDenseLayer(int length, ActivationTypes activationType, int prevLayerSize, Random? random = null)
        {
            Layer layer = new(length, activationType, random);
            //layer._weights = Matrix<double>.Build.Dense(length, prevLayerSize, (i, j) => (layer._random.NextDouble() - 0.5));
            layer._weights = Matrix<double>.Build.Dense(length, prevLayerSize, (i, j) => i + j + 1);
            Console.WriteLine(layer._weights);
            return layer;
        }

        public Layer() { }



        protected Layer(int length, ActivationTypes activationType, Random? random = null)
        {
            _input = Vector<double>.Build.Dense(length);
            _output = Vector<double>.Build.Dense(length);
            _error = Vector<double>.Build.Dense(length);

            ActivationType = activationType;
            _random = random ?? new Random();
        }

        public void Clear()
        {
            _input.Clear();
            _output.Clear();
            _error.Clear();
        }


        public IActivation GetActivationFunction()
        {

            IActivation activation;
            switch (ActivationType)
            {
                case ActivationTypes.NO:
                    activation = new NoActivation();
                    break;
                case ActivationTypes.SIGMOID:
                    activation = new SigmoidActivation();
                    break;
                default:
                    throw new ArgumentException($"Unknown ActivationType: [{ActivationType}]");
                    //break;
            }
            return activation;
        }



        public void UpdateWeights(double alpha = 0.1)
        {
            Matrix<double> delta;
            //TODO
            //delta = alpha * _error * GetActivationFunction().Deactivate(_output) * _output;
            _weights += delta;

        }

    }
}