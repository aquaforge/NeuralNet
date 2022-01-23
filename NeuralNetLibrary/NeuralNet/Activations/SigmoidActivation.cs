namespace NeuralNetLibrary
{
    [Serializable]
    public class SigmoidActivation : IActivation
    {
        public double Activate(double d) => 1 / (1 + Math.Exp(-d));
        public double Deactivate(double d) => d * (1 - d);
    }


}

