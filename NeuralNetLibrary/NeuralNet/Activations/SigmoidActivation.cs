using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    [Serializable]
    public class SigmoidActivation : IActivation
    {
        public double Activate(double d) => 1 / (1 + Math.Exp(-d));
        public double Deactivate(double d) => d * (1 - d);

        public Vector<double> Activate(Vector<double> v) => Vector<double>.Build.Dense(v.Count, (i) => Activate(v[i]));
        public Vector<double> Deactivate(Vector<double> v) => Vector<double>.Build.Dense(v.Count, (i) => Deactivate(v[i]));

    }


}

