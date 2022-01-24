using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    [Serializable]
    public class NoActivation : IActivation
    {
        public double Activate(double d) => d;
        public double Deactivate(double d) => 1.0;

        public Vector<double> Activate(Vector<double> v) => Vector<double>.Build.DenseOfVector(v);
        public Vector<double> Deactivate(Vector<double> v) => Vector<double>.Build.DenseOfVector(v);

    }


}

