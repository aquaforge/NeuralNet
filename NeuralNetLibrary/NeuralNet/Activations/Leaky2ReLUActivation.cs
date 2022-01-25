using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    [Serializable]
    public class Leaky2ReLUActivation : Activation, IActivation
    {
        public override double Activate(double d)
        {
            if (d < 0) return 0.01 * d;
            if (d > 1) return 1 + 0.01 * (d - 1);
            return d;
        }
        
        public override double Deactivate(double input, double output)
        {
            if (input < 0) return 0.01;
            if (input > 1) return 0.01;
            return 1;
        }

    }
}

