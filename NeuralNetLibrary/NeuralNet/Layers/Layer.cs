using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    [Serializable]
    public class Layer
    {
        Random _random;

        internal Vector<double> _input;
        internal Vector<double> _output;
        internal Vector<double> _delta;
        internal Matrix<double> _weights;
        internal Matrix<double> _deltaWeightsPrev;

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
        public double[][] DeltaWeightsPrevMatrixByRows
        {
            get { return _deltaWeightsPrev.ToRowArrays(); }
            set
            {
                if (value == null) throw new ArgumentNullException(nameof(value));
                _deltaWeightsPrev = Matrix<double>.Build.DenseOfRowArrays(value);
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

        public double[] DeltaVector
        {
            get { return _delta.ToArray(); }
            set { _delta = Vector<double>.Build.DenseOfArray(value); }
        }

        public Layer() { }



        public Layer(int prevLayerLength, int length, ActivationTypes activationType, Random? random = null, Matrix<double>? weights = null)
        {
            _random = random ?? new Random();
            ActivationType = activationType;

            _input = Vector<double>.Build.Dense(length);
            _output = _input.Clone();
            _delta = _input.Clone();


            if (weights == null)
                _weights = Matrix<double>.Build.Dense(length, prevLayerLength, (i, j) => (_random.NextDouble() - 0.5));
            else
                _weights = Matrix<double>.Build.DenseOfMatrix(weights);
            _deltaWeightsPrev = Matrix<double>.Build.Dense(length, prevLayerLength);
        }

        public void ClearAllButWeight()
        {
            ClearPrediction();
            _deltaWeightsPrev.Clear();
        }

        public void ClearPrediction()
        {
            _input.Clear();
            _output.Clear();
            _delta.Clear();
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

        internal double ErrorMSE(Vector<double> outputToBe) => (outputToBe - _output).PointwisePower(2).Sum() / _output.Count;
        //{
        //    //for testing
        //    Vector<double> v = (outputToBe - _output).PointwisePower(2);
        //    return v.Sum() / _output.Count;
        //}




        internal double AbsError(Vector<double> outputToBe) => (outputToBe - _output).PointwiseAbs().Sum();


        public void UpdateWeights(Vector<double> outputPrev, double alpha = 0.1)
        {
            Matrix<double> delta;
            Vector<double> v;
            v = _delta.PointwiseMultiply(GetActivationFunction().Deactivate(_input, _output));

            Matrix<double> m1 = v.ToColumnMatrix();
            Matrix<double> m2 = outputPrev.ToRowMatrix();

            //TODO
            delta = alpha * m1 * m2;

            _weights -= delta;


        }

    }
}