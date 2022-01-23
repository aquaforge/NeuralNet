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
        private Matrix<double>? _weights;

        public ActivationTypes ActivationType { get; set; }
        public int Size => _input.Count;

        public double[][]? WeightsMatrixByRows
        {
            get { return _weights?.ToRowArrays(); }
            set
            {
                if (value == null)
                    _weights = null;
                else
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



        public static Layer GetInputLayer(int size, Random? random = null) => new(size, ActivationTypes.NO, random);
        public static Layer GetDenseLayer(int size, ActivationTypes activationType, int prevLayerSize, Random? random = null)
        {
            Layer layer = new(size, activationType, random);
            layer._weights = Matrix<double>.Build.Dense(prevLayerSize, size, (i, j) => (layer._random.NextDouble()-0.5));
            return layer;
        }

        public Layer() { }

        protected Layer(int size, ActivationTypes activationType, int prevLayerSize, Random? random = null) : this(size, activationType, random)
        {

        }


        protected Layer(int size, ActivationTypes activationType, Random? random = null)
        {
            _input = Vector<double>.Build.Dense(size);
            _output = Vector<double>.Build.Dense(size);
            _error = Vector<double>.Build.Dense(size);

            ActivationType = activationType;
            _random = random ?? new Random();
        }

        public void Clear()
        {
            _input.Clear();
            _output.Clear();
            _error.Clear();
        }
    }
}