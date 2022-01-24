using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    [Serializable]
    public class LeakyReLUActivation : Activation, IActivation
    {
        public override double Activate(double d) => d < 0 ? 0.01 * d : d;
        public override double Deactivate(double input, double output) => input < 0 ? 0.01 : 1;
    }
}

