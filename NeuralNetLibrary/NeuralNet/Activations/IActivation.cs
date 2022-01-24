using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    public interface IActivation
    {
        public double Activate(double d);
        public double Deactivate(double input, double output);
        
        public Vector<double> Activate(Vector<double> vector);
        public abstract Vector<double> Deactivate(Vector<double> input, Vector<double> output);




    }
}