namespace NeuralNetLibrary
{
    [Serializable]
    public abstract class Layer
    {
        public Vector Neurons;

        protected Layer(Vector neurons)
        {
            Neurons = neurons;
        }
    }
}