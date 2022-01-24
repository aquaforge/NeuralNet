using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    [Serializable]
    public class NoActivation : Activation, IActivation
    {
        public override double Activate(double d) => d;
        public override double Deactivate(double input, double output) => 1.0;
    }
}

