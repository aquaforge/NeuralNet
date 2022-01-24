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

        public double[][] WeightsMatrixByRows
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

        public Layer() { }



        public Layer(int prevLayerLength, int length, ActivationTypes activationType, Random? random = null, Matrix<double>? weights = null)
        {
            _random = random ?? new Random();
            ActivationType = activationType;

            _input = Vector<double>.Build.Dense(length);
            _output = Vector<double>.Build.Dense(length);
            _error = Vector<double>.Build.Dense(length);


            if(weights == null)
                _weights = Matrix<double>.Build.Dense(length, prevLayerLength, (i, j) => (_random.NextDouble() - 0.5));
            else
                _weights = Matrix<double>.Build.DenseOfMatrix(weights);
        }

        public void Clear()
        {
            _input.Clear();
            _output.Clear();
            _error.Clear();
        }


        public IActivation GetActivationFunction()
        {
            return ActivationType switch
            {
                ActivationTypes.Identity => new IdentityActivation(),
                ActivationTypes.Sigmoid => new SigmoidActivation(),
                ActivationTypes.LeakyReLU => new LeakyReLUActivation(),
                _ => throw new ArgumentException($"Unknown ActivationType: [{ActivationType}]"),
            };
        }



        public void UpdateWeights(double alpha = 0.1)
        {
            Matrix<double> delta;
            //TODO
            //delta = alpha * _error * GetActivationFunction().Deactivate(_output) * _output;
            //_weights += delta;

        }

    }
}