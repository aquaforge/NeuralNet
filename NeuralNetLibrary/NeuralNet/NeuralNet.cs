namespace NeuralNetLibrary
{
    [Serializable]
    public class NeuralNet
    {
        private Random _random;
        private LinkedList<Layer> _layers = new();

        public LinkedList<Layer> Layers
        {
            get { return _layers; }
            set { _layers = value; }
        }

        public NeuralNet() => _random = new Random();

        public NeuralNet(Random random) => _random = random;


        public void AddLayer(int size, IActivation activation)
        {
            if (_layers.Count == 0)
                _layers.AddLast(Layer.GetInputLayer(size));
            else
                _layers.AddLast(Layer.GetDenseLayer(size, ActivationTypes.SIGMOID, _layers.Last().Size, _random));
        }


        public void Clear()
        {
            foreach (var layer in _layers)
                layer.Clear();
        }
    }
}