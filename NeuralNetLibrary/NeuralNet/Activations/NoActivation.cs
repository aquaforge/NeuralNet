namespace NeuralNetLibrary
{
    [Serializable]
    public class NoActivation : IActivation
    {
        public double Activate(double d) => d;
        public double Deactivate(double d) => 1.0;
    }


}

