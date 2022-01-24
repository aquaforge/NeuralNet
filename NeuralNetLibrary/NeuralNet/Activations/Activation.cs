using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    [Serializable]
    public abstract class Activation : IActivation
    {
        public abstract double Activate(double d);
        public abstract double Deactivate(double input, double output);

        public Vector<double> Activate(Vector<double> v) 
            => Vector<double>.Build.Dense(v.Count, (i) => Activate(v[i]));

        public Vector<double> Deactivate(Vector<double> input, Vector<double> output)
            => Vector<double>.Build.Dense(input.Count, (i) => Deactivate(input[i], output[i]));

    }


}

