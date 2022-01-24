using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    [Serializable]
    public class SigmoidActivation : Activation, IActivation
    {
        public override double Activate(double d) => 1 / (1 + Math.Exp(-d));
        public override double Deactivate(double input, double output) => output * (1 - output);
    }
}

