namespace NeuralNetLibrary
{
    [Serializable]
    public class NeuralNet
    {
        List<Layer> _layers = new();
        public void AddLayer(Layer layer)
        {
            _layers.Add(layer);
        }
    }
}